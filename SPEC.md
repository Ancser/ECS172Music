# ECS172 Final Project Spec
# Emotion-Aware Music Recommendation with LLM Lyric Analysis & Knowledge Distillation

## 1. Problem Statement

Build an emotion-aware music recommendation system that predicts the next songs in a playlist by combining:
- **Long-term preference** from playlist-level song co-occurrence (collaborative filtering)
- **Short-term mood** from recent emotional trajectory within the playlist (LLM-derived emotion tags)

**Evaluation setup**: Use the first 80% of each playlist as input, hold out the last 20% as ground truth. Recommend songs and measure whether they match the held-out 20%.

**Metrics**: Recall@10, NDCG@10

---

## 2. Datasets

| Dataset | Source | Contents | Role |
|---------|--------|----------|------|
| Spotify Million Playlist Dataset (MPD) | Kaggle (himanshuwagh/spotify-million) | ~1M playlists, track lists, artist/album/track metadata | Main interaction data; each playlist = proxy user |
| Spotify Million Song Dataset | Kaggle (notshrirang/spotify-million-song-dataset) | Song titles, artist names, links, **lyrics** | Lyrics source for emotion tagging |

**Data join**: Match MPD tracks to Song Dataset by (song title, artist name) to get lyrics. Tracks without lyrics fallback to metadata-only features.

**Subsample strategy** (for Colab feasibility):
- Sample ~50K-100K playlists from MPD (not all 1M)
- Keep only tracks that have lyrics matches
- Filter playlists with >= 10 songs (so 80/20 split is meaningful)

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    OFFLINE PIPELINE                      │
│                                                         │
│  Lyrics ──► LLM Emotion Tagger ──► Mood Tags per Song  │
│              (Llama3 / Gemini)      (stored in DB)      │
│                                                         │
│  MPD ──► Song Co-occurrence Matrix ──► Item Embeddings  │
│           (collaborative filtering)                     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   RECOMMENDATION MODEL                   │
│                                                         │
│  Input: Playlist (first 80% of songs)                   │
│                                                         │
│  ┌──────────────┐    ┌──────────────────┐               │
│  │  Long-Term   │    │   Short-Term     │               │
│  │  Preference  │    │   Mood Signal    │               │
│  │  Encoder     │    │   Encoder        │               │
│  │ (full 80%    │    │ (last N songs    │               │
│  │  co-occur)   │    │  emotion trend)  │               │
│  └──────┬───────┘    └────────┬─────────┘               │
│         │                     │                         │
│         └──────┬──────────────┘                         │
│                ▼                                        │
│     ┌──────────────────┐                                │
│     │  Adaptive Fusion │  (learned weight α)            │
│     │  score = α·long  │                                │
│     │  + (1-α)·short   │                                │
│     └────────┬─────────┘                                │
│              ▼                                          │
│     Ranked candidate songs ──► Top-10 recommendations   │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Pipeline Stages

### Stage 0: Data Preparation
- Download and parse MPD JSON slices
- Load Million Song lyrics CSV
- Join datasets on (title, artist) with fuzzy matching
- Filter: playlists >= 10 songs, tracks with lyrics
- Split each playlist: first 80% = train, last 20% = test

### Stage 1: LLM Emotion Tagging (Offline, One-Time)
**Goal**: For each song with lyrics, produce structured emotion tags.

**LLM Choice**: Llama3-8B (via Ollama local or HuggingFace) or Gemini 1.5 Flash (free API tier)

**Prompt Template** (factorization prompting, inspired by KAR):
```
Given the following song lyrics, analyze the emotional content.

Song: "{title}" by {artist}
Lyrics: "{lyrics_text}"

Respond in JSON:
{
  "primary_emotion": one of [joy, sadness, anger, fear, love, nostalgia, energy, calm, melancholy, hope],
  "secondary_emotion": one of the same list or null,
  "valence": float from -1.0 (negative) to 1.0 (positive),
  "arousal": float from 0.0 (calm) to 1.0 (energetic),
  "mood_summary": one sentence describing the emotional arc
}
```

**Output**: A lookup table `song_id -> {primary_emotion, secondary_emotion, valence, arousal, mood_summary}`

**Efficiency**: Pre-generate all tags, store as CSV/parquet. No LLM at inference time (KAR prestore pattern).

### Stage 2: Baseline — Collaborative Filtering Only
**Purpose**: Establish baseline Recall@10 and NDCG@10 without emotion features.

**Method**: Song co-occurrence matrix from playlists
- Build item-item co-occurrence: for each pair of songs appearing in the same playlist, increment count
- Normalize by song popularity (PMI or cosine)
- For a test playlist's 80% input, score candidates by sum of co-occurrence similarity to input songs
- Return top-10

**Alternative baseline**: Item2Vec (Word2Vec trained on playlist sequences, treating songs as "words")

### Stage 3: Long-Term Preference Encoder
- Aggregate co-occurrence scores across all songs in the 80% input
- Produces a score for each candidate song: `score_long(candidate) = Σ sim(candidate, song_i)` for all song_i in input

### Stage 4: Short-Term Mood Encoder
- Take the **last N songs** (e.g., N=5) from the 80% input
- Extract their emotion tags (from Stage 1)
- Compute **recent mood vector**: average (valence, arousal) of last N songs
- Detect **mood drift**: compare average emotion of first half vs second half of input
  - If diverging (e.g., calm → energetic), weight the trend direction
- Score candidates by emotional similarity to the recent mood:
  `score_short(candidate) = cosine_sim(candidate_emotion_vec, recent_mood_vec)`

**Mood drift detection**:
```python
first_half_mood = avg(valence, arousal of songs 0..len/2)
second_half_mood = avg(valence, arousal of songs len/2..len)
drift_vector = second_half_mood - first_half_mood
projected_mood = recent_mood + β * drift_vector  # extrapolate the trend
```

### Stage 5: Adaptive Fusion & Ranking
Combine long-term and short-term scores:

```
final_score(candidate) = α · score_long(candidate) + (1 - α) · score_short(candidate)
```

**Option A (Simple)**: Tune α as a hyperparameter on validation set (grid search 0.0 to 1.0)

**Option B (Learned, for bonus points)**: Train a small MLP or logistic regression:
- Input features: [score_long, score_short, mood_drift_magnitude, playlist_length, candidate_popularity]
- Label: 1 if candidate is in held-out 20%, 0 otherwise
- This learns a per-context α

### Stage 6 (Stretch): Knowledge Distillation Auxiliary Task
*Inspired by the YouTube Music multi-task ranker architecture*

If the co-occurrence model is the "student" on our small Spotify dataset:
- Use LLM emotion predictions as **distillation labels** (soft targets)
- Add an auxiliary task head: given a song's co-occurrence embedding, predict its emotion vector
- Multi-task loss: `L = L_retrieval + λ · L_emotion_distill`
- The shared layers learn representations that capture both collaborative and emotional signals

This is directly analogous to the YouTube diagram:
- Pre-existing task (hard label) = playlist co-occurrence prediction
- Teacher prediction task (distillation label) = LLM emotion vector prediction
- Shared layers = song embedding layer

---

## 5. Experiments & Comparisons

### Models to Compare

| Model | Description |
|-------|-------------|
| **Baseline 1: Random** | Random top-10 from catalog (sanity check) |
| **Baseline 2: Popularity** | Recommend globally most popular songs |
| **Baseline 3: Co-occurrence CF** | Stage 2 only, no emotion |
| **Baseline 4: Item2Vec** | Word2Vec on playlist sequences |
| **Ours (no mood drift)** | CF + emotion similarity, static α |
| **Ours (with mood drift)** | CF + emotion with drift extrapolation |
| **Ours (learned fusion)** | CF + emotion with learned MLP fusion |
| **Ours (KD multi-task)** | Stage 6 multi-task with emotion distillation |

### Ablation Studies
1. **Emotion source**: LLM tags vs no emotion → measures value of lyric analysis
2. **Short-term window**: vary N (last 3, 5, 10 songs) → find optimal mood window
3. **Fusion weight α**: sweep 0.0–1.0 → show optimal blend
4. **Mood drift**: with vs without drift extrapolation
5. **LLM choice**: Compare Llama3 vs Gemini emotion tags (if time permits)

### Expected Results Table (template)

| Model | Recall@10 | NDCG@10 | Notes |
|-------|-----------|---------|-------|
| Random | ~0.001 | ~0.001 | lower bound |
| Popularity | ? | ? | |
| Co-occurrence CF | ? | ? | baseline |
| Item2Vec | ? | ? | baseline |
| Ours (static α) | ? | ? | expect improvement |
| Ours (drift) | ? | ? | expect further gain |
| Ours (learned) | ? | ? | best expected |

---

## 6. Report Structure (8 pages, ACM format)

The course requires a detailed report exceeding the example. Plan:

### Abstract (~150 words)
Problem, approach (LLM emotion tags + dual-temporal fusion), key result (X% improvement over CF baseline).

### 1. Introduction (1 page)
- Music recommendation limitations (semantic blindness, cold-start)
- Why lyrics carry emotional signal that audio features miss
- Our contribution: LLM-based emotion tagging + adaptive short/long-term fusion
- Preview of results

### 2. Related Work (1 page)
- Traditional CF for music (playlist-based, Item2Vec)
- Emotion in music recommendation (MER, audio valence/arousal)
- LLMs for recommendation (LEMON, KAR, ROEGEN)
- Knowledge distillation in recommendation (YouTube Music multi-task)
- Position our work: we combine LLM emotion extraction + dual-temporal + adaptive fusion

### 3. Methodology (2 pages)
- 3.1 Problem formulation (playlist continuation task)
- 3.2 LLM emotion tagging pipeline (prompt design, emotion taxonomy, prestore)
- 3.3 Long-term preference encoder (co-occurrence / Item2Vec)
- 3.4 Short-term mood encoder (recent window, drift detection)
- 3.5 Adaptive fusion mechanism
- 3.6 Knowledge distillation variant (multi-task auxiliary head)
- Architecture diagram (full pipeline figure)

### 4. Experiments (1.5 pages)
- 4.1 Dataset statistics (after filtering/joining)
- 4.2 Implementation details (hyperparameters, LLM config, compute)
- 4.3 Baseline descriptions
- 4.4 Results table (Recall@10, NDCG@10 for all models)
- 4.5 Ablation results

### 5. Discussion & Analysis (1.5 pages)
- When does emotion help most? (mood-coherent playlists vs random collections)
- Failure cases (playlists for archiving vs listening)
- Effect of mood drift detection
- Qualitative examples (show a playlist, its mood trajectory, and recommendations)
- Comparison: LLM emotion tags vs Spotify audio features (valence/energy)

### 6. Limitations & Ethics (0.5 page)
- Playlist ≠ user (proxy assumption)
- LLM bias in lyric interpretation (cultural, multilingual)
- Emotional spiraling risk
- Dataset temporal mismatch

### 7. Conclusion (0.5 page)

### Contribution Statement
One sentence per team member.

### References
3+ papers from required venues (LEMON-KDD25, KAR-RecSys24, ROEGEN-RecSys24) plus additional.

---

## 7. Implementation Plan & Timeline

Assume ~3 weeks remaining (current: Week 7ish, presentation: Week 9, paper: Week 10).

### Week 7-8: Core Implementation
- [ ] Download and parse both datasets
- [ ] Join datasets, filter, create train/test splits
- [ ] Run LLM emotion tagging on matched songs (batch processing)
- [ ] Implement co-occurrence CF baseline → get baseline metrics
- [ ] Implement Item2Vec baseline
- [ ] Implement short-term mood encoder
- [ ] Implement adaptive fusion

### Week 8-9: Experiments & Analysis
- [ ] Run all model variants, collect metrics
- [ ] Ablation studies (window size, α sweep, emotion source)
- [ ] Generate visualizations (mood trajectories, result tables, feature importance)
- [ ] Try KD multi-task variant if time permits
- [ ] Prepare presentation slides (6 min)

### Week 9-10: Report Writing
- [ ] Draft full paper in ACM template
- [ ] Create architecture diagram
- [ ] Write qualitative analysis with examples
- [ ] Final editing and formatting

---

## 8. Technical Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.10+ |
| Compute | Google Colab (free GPU for embeddings) |
| LLM for emotion | Gemini 1.5 Flash (free tier) or Llama3-8B via Ollama |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| ML Framework | PyTorch or scikit-learn |
| Data Processing | pandas, numpy |
| Evaluation | custom (Recall@K, NDCG@K) |
| Visualization | matplotlib, seaborn |
| Report | LaTeX (ACM template) or Overleaf |

---

## 9. Key Design Decisions

### Why LLM for lyrics → mood tags (not a classifier)?
- Traditional emotion classifiers use fixed categories and miss metaphor/irony
- LLMs understand multilingual lyrics, slang, indirect expression
- KDD 2025 LEMON paper validates this approach
- One-time cost (prestore pattern from KAR) — no inference latency

### Why 80/20 playlist split?
- Course proposal already committed to this
- Naturally tests both long-term (full input) and short-term (end of input) signals
- Standard in playlist continuation literature

### Why co-occurrence CF as baseline (not matrix factorization)?
- Simpler, more interpretable, easier to compare against
- Can always add Item2Vec or ALS as a second baseline for richer comparison
- MF (Funk SVD) is rating-based — playlists are implicit feedback (presence, not rating)

### Knowledge Distillation connection (from YouTube lecture)
The YouTube Music architecture uses a multi-task model where:
- Hard labels come from existing tasks (click, play, etc.)
- Distillation labels come from a teacher model (larger model trained on YouTube's massive data)
- Shared bottom layers learn from both signals

Our adaptation:
- Hard labels = whether a song appears in the held-out 20%
- Distillation labels = LLM-generated emotion vectors (the LLM is our "teacher")
- Shared layers = song embeddings that capture both co-occurrence and emotional patterns
- This is zero-shot cross-domain KD: the LLM (trained on internet text) transfers knowledge to our small music domain

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Low lyrics coverage (MPD songs not in Song Dataset) | Fuzzy matching + fallback to metadata-only features |
| LLM cost too high for all songs | Process only songs appearing in sampled playlists |
| Emotion tags too noisy | Validate with Spotify audio features (valence/energy) as sanity check |
| Metrics too low overall | This is expected for playlist continuation on large catalogs; focus on relative improvement over baseline |
| Colab compute limits | Subsample to 50K playlists; precompute all embeddings |
| Not enough novelty for grading | KD multi-task + mood drift detection are novel combinations not in any single reference paper |

---

## 11. Novel Contributions (for grading justification)

1. **LLM-as-emotion-teacher**: Using LLM emotion tags as distillation labels for a lightweight recommender (combines LEMON's emotion extraction with YouTube's multi-task KD pattern)
2. **Mood drift extrapolation**: Detecting emotional trajectory shifts within playlists and extrapolating to predict next-song emotion (not done in any of the three reference papers)
3. **Adaptive dual-temporal fusion**: Learned blending of long-term co-occurrence preference with short-term mood signal (inspired by LEMON but applied to playlist continuation with a simpler, more reproducible architecture)

---

## 12. Reference Papers (meeting course requirement: 3+ from top venues)

1. **LEMON** — Wang et al. "Enhanced Emotion-aware Music Recommendation via Large Language Models." KDD 2025.
2. **KAR** — Xi et al. "Towards Open-World Recommendation with Knowledge Augmentation from LLMs." RecSys 2024 (arXiv:2306.10933).
3. **ROEGEN** — Tekle et al. "Music Recommendation through LLM Song Summary." ROEGEN@RecSys 2024.

Additional references:
- YouTube Music multi-task KD architecture (internal YouTube/Google presentation)
- Item2Vec — Barkan & Koenigstein. "Item2Vec: Neural Item Embedding for Collaborative Filtering." RecSys 2016 Workshop.
- Spotify Million Playlist Dataset documentation
