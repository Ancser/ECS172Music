# ECS172 Music Recommendation Project

## 1. Project Direction

This project builds an emotion-aware playlist continuation recommender.

The proposal direction is:

```text
Input:  all earlier songs in a playlist
Target: up to final 10 songs of that playlist
Output: top-10 recommended songs
Metrics: Recall@10 and NDCG@10
```

The final system should combine:

- long-term preference from playlist-level song co-occurrence
- short-term mood from recent lyric/emotion trajectory
- adaptive fusion between long-term and short-term signals

Current implementation has two paths:

```text
Baseline path: CF + popularity + metadata + weak mood.
LLM path: CF + popularity candidates, then structured semantic-text reranking.
Raw lyrics are not directly compared in the current LLM hybrid path.
```

Core workflow:

```text
data review
  -> Stage 1 candidate retrieval
  -> Stage 2 ranking
  -> validation metrics
  -> ablation and comparison
```

## 2. Current Files

```text
newSpotify.py      main prototype and evaluator
download_data.py   one-time Kaggle data downloader
install_llm.py     optional Gemma 3 LLM downloader/tester
spotify_web.py     local browser demo for playlist continuation
start_spotify_web.bat  Windows launcher for the web demo
skill.md           detailed algorithm workflow notes
SPEC.md            pointer to this README
data/              local datasets
```

Current local data:

```text
data/spotify_millsongdata.csv
```

Playlist slices should also live directly in `data/`:

```text
data/mpd.slice.0-999.json
data/mpd.slice.1000-1999.json
```

## 3. One-Time Data Download

Download the large Spotify Million Playlist Dataset once:

```bat
pushd <project-folder>
python .\download_data.py --playlists
```

The downloader skips work if `data/mpd.slice.*.json` already exists.

If `kagglehub` fails because of package mismatch, repair it with:

```bat
python -m pip install --user --upgrade --force-reinstall kagglehub==0.3.13
```

The lyrics CSV is already in:

```text
data/spotify_millsongdata.csv
```

## 4. Local Spotify Web Demo

Launch the browser prototype:

```bat
pushd <project-folder>
start_spotify_web.bat
```

The web app opens at `http://127.0.0.1:5050`. It indexes lyrics and MPD playlists into `models/web_cache/spotify_web.sqlite`. The first full index can take a while because the Spotify Million Playlist Dataset is large; later launches reuse the SQLite cache.

For a faster classroom/demo index:

```bat
pushd <project-folder>
start_spotify_web.bat --max-playlists 50000 --rebuild
```

Default web recommendation controls:

```text
batch size: 10
candidates: 200
```

This means Stage 1 retrieves 200 non-LLM candidates before Stage 2 runs LLM structured semantic text and reranks the final top 10.

The page supports:

```text
Overview and Recommendation tabs
playlist/users list with 5+, 10+, and 50+ filters
compact song list with search, random sampling, and hover JSON details
data overview before recommendation, including sparsity, playlist year range, language, and lyric type
playlist detail view with first 10 songs hidden/gray for evaluation
Train / recommend button that uses song 11 onward as observed history
background progress for candidate generation and LLM structured semantic text
final recommendation table with CF, singer recency, title/context, popularity, and semantic-text scores
```

The web LLM path intentionally does not use raw emotion labels. It asks Gemma to generate structured semantic text, caches it, embeds the text with TF-IDF, and uses it as an auxiliary ranking feature beside CF and popularity.

The web ranking formula explicitly includes artist/singer and title context, but weights are adaptive per playlist:

```text
Stage 1:
  CF + popularity retrieve candidates from the full catalog.

Stage 2:
  LLM structured semantic text is generated only for observed playlist songs
  and candidate songs, then the final top-10 is reranked.

Adaptive playlist personalities:
  singer-led          -> higher singer-recency weight
  playlist-title-led  -> higher playlist-title / song-title context weight
  style-led           -> higher lyric type and semantic-text weight
  mixed-personal      -> higher CF weight
```

## 5. Data Review

Always review data before modeling.

Required sources:

| Dataset | Source | Contents | Role |
|---|---|---|---|
| Spotify Million Playlist Dataset | Kaggle `himanshuwagh/spotify-million` | playlist names, track lists, artists, albums | playlist continuation interactions |
| Spotify Million Song Dataset | Kaggle `notshrirang/spotify-million-song-dataset` | song title, artist, lyrics | lyric content features and future emotion tags |

Join logic:

```text
Normalize artist and song title.
Join playlist tracks to lyrics by (artist_name, track_name).
Keep matched tracks for the lyrics-first prototype.
```

MPD does not provide per-song `date_added`. It provides playlist-level `modified_at` and track-level `pos`, so the prototype uses `pos` as the old-to-new playlist order.

Data review must print:

```text
head 10 songs
head 10 playlists
song lyric rows
playlist count
matched playlist tracks
unique matched tracks
average / median / min / max playlist length
average / median / min / max lyric token count
matrix density
cold playlists with <= 3 observed songs
```

Run current data review:

```bat
pushd <project-folder>
python .\newSpotify.py
```

At the moment, this prints lyric stats even before playlist slices are downloaded.

## 5. Validation

Use fixed last-10 playlist-order holdout:

```text
heldout  = playlist[-min(10, len(playlist)-1):]
observed = all songs before heldout
```

Do not randomly split tracks for the main experiment, because the task is playlist continuation. The goal is: given the earlier playlist history, recommend the next/final songs, up to 10.

Evaluation rules:

```text
recommendations must exclude observed songs
truth = final heldout songs, up to 10
top K = 10 for final recommendations
candidate K = 300 for Stage 1 retrieval
default min playlist length = 2
one observed song is valid
```

If a playlist has only one observed song, it is treated as cold-start and ranked by popularity.

Metrics:

```text
Recall@10 = heldout songs found in top 10 / heldout songs
NDCG@10   = ranking-quality score with higher credit for earlier hits
Proxy     = (Recall@10 + NDCG@10) / 2
```

Stage-specific metrics:

```text
Stage 1: Retrieval@300
Stage 2: Recall@10, NDCG@10, Proxy
```

## 6. Stage 1 Candidate Retrieval

Stage 1 should retrieve a broad candidate pool. It is optimized for recall, not final ordering.

### Stage 0: Song Classification

Current no-LLM song features:

```text
language = simple lyric language heuristic
lyric type = rule-based lyric topic/emotion bucket
valence/arousal = weak emotion vector from lyric lexicon
artist = metadata from MPD/lyrics
```

Lyric type buckets:

```text
love, sadness, energy, anger, calm, hope, nostalgia, other
```

These are placeholders for later LLM emotion tags.

### Route A: Structured Semantic Text

Current implemented LLM route. Raw lyrics are only used as input to produce a cached structured semantic profile; they are not directly compared during hybrid ranking.

Algorithm:

```text
1. Generate or load a structured semantic profile per song.
2. Generate deterministic training_text from structured fields.
3. Tokenize training_text, not raw lyrics.
4. Compute document frequency per token.
   idf(t) = log((1 + num_songs) / (1 + df(t))) + 1
5. Build normalized semantic-text vector per processed song.
6. Stage 1 retrieves candidates with CF + popularity.
7. Stage 2 scores only processed semantic candidates by cosine similarity.
8. Exclude observed songs.
```

Recent semantic profile:

```text
profile_recent = time-decayed average semantic-text vector of last N observed songs with profiles
score_semantic(candidate) = cosine(profile_recent, candidate_semantic_vector)
```

Time decay:

```text
newest observed song weight = 1.0
one song older = time_decay
two songs older = time_decay^2
default time_decay = 0.90
```

Cold-start shortcut:

```text
if observed length <= cold_start_threshold:
  skip CF profile scoring
  rank candidates with popularity fallback
default cold_start_threshold = 1
```

Speedup:

```text
candidate scores are prepared once
semantic reranking reuses cached CF/popularity candidates
progress prints every --progress-interval eval playlists
```

### Route B: Co-Occurrence CF

Implemented long-term preference route.

Algorithm:

```text
1. Treat each playlist as a basket of songs.
2. Count song-song co-occurrence in observed playlists.
3. Normalize by popularity using cosine or PPMI.
4. Score candidates by summed similarity to observed songs.
```

Cosine co-occurrence:

```text
sim(i, j) = co_count(i, j) / sqrt(pop(i) * pop(j))
```

PPMI:

```text
pmi(i, j) = log((co_count(i, j) * num_playlists) / (pop(i) * pop(j)))
ppmi(i, j) = max(pmi(i, j), 0)
```

### Route C: Popularity

Popularity is a real baseline and fallback:

```text
score_pop(song) = number of training playlists containing song
```

Use it for:

```text
cold playlists
candidate-pool stabilization
tie breaking
baseline comparison
```

### Stage 1 Fusion

Current candidate pool combines:

```text
stage1_score =
  w_lyrics * lyrics_score_norm
+ w_cf     * cf_score_norm
+ w_pop    * popularity_score_norm
```

Tune weights by:

```text
Retrieval@300
```

## 7. Stage 2 Ranking

Stage 2 reranks candidates into the final top 10.

Primary metrics:

```text
Recall@10
NDCG@10
Proxy
```

Current prototype ranker:

```text
final_score =
  lyrics_weight * (alpha * lyrics_long_norm + (1-alpha) * lyrics_short_norm)
+ cf_weight * cf_score_norm
+ mood_weight * mood_drift_score_norm
+ artist_weight * same_artist_score_norm
+ type_weight * lyric_type_score_norm
+ language_weight * language_score_norm
+ pop_weight * popularity_norm
```

Grid search:

```text
alpha = 0.0, 0.1, ..., 1.0
```

Current Stage 2 features:

```text
lyrics_long_norm
lyrics_short_norm
lyrics_delta
cf_score_norm
track_popularity_norm
same_artist_count
lyric_type_overlap
language_match
mood_drift_score
```

Do not assume a learned classifier is better. Logistic regression or MLP should only be added after manual fusion baselines are measured, and must be validated by Recall@10/NDCG@10 rather than accuracy.

## 8. Future LLM Emotion Stage

Do not add LLM calls until the TF-IDF, popularity, and CF baselines are measured.

### Option 1: Small Gemma 3 Semantic Tagger

Use this as the first LLM test path. The local machine has limited GPU memory, so the default is the smallest Gemma 3 instruction model:

```text
google/gemma-3-270m-it
```

Reference links:

```text
Google Gemma 3 docs: https://ai.google.dev/gemma/docs/core
Gemma 3 270M announcement: https://developers.googleblog.com/introducing-gemma-3-270m/
Hugging Face model: https://huggingface.co/google/gemma-3-270m-it
```

Purpose:

```text
title + artist + lyrics -> controlled semantic JSON
controlled JSON -> programmatic training_text
compare CF+popularity vs CF+popularity+Gemma semantic labels
check whether controlled semantic compression helps more than weak keyword tags
```

Install/download/test:

```bat
pushd <project-folder>
python .\install_llm.py
```

For NVIDIA GPU acceleration on Windows, install CUDA PyTorch once:

```bat
pushd <project-folder>
python .\install_llm.py --cuda-torch --device cuda
```

Gemma models on Hugging Face may require license acceptance before download:

```text
1. Open https://huggingface.co/google/gemma-3-270m-it
2. Accept the Google usage license
3. Run: python -m huggingface_hub.cli.hf auth login
4. Run python .\install_llm.py again
```

The model cache is stored under:

```text
models/llm_cache
```

When `newSpotify.py` uses `--emotion-source llm`, per-song semantic profiles are cached under:

```text
models/semantic_cache
```

Cached fields:

```text
song_id
language
semantic_summary
themes
lyrical_narrative
listening_context
playlist_function
transition_note
keywords
training_text
```

The LLM is not allowed to output numeric emotion scores or recommendations. It returns compact structured meaning fields. `training_text` is generated by code:

```text
Semantic profile: themes=...; narrative=...; context=...; playlist_function=...; transition=...; keywords=....
```

Run a small Gemma comparison first:

```bat
pushd <project-folder>
python .\newSpotify.py --emotion-source llm --lyrics-csv .\data\spotify_millsongdata.csv --mpd-path .\data --max-playlists 1000 --max-eval-cases 100 --min-playlist-len 20 --holdout-k 10 --pool-size 200 --emotion-limit 200 --semantic-weight 0.15 --progress-interval 25
```

Use `--llm-device cuda` after CUDA PyTorch is installed:

```bat
pushd <project-folder>
python .\newSpotify.py --emotion-source llm --llm-device cuda --llm-batch-size 10 --require-semantic-coverage --lyrics-csv .\data\spotify_millsongdata.csv --mpd-path .\data --max-playlists 1000 --max-eval-cases 100 --min-playlist-len 20 --holdout-k 10 --pool-size 200 --emotion-limit 0 --semantic-weight 0.15 --progress-interval 25
```

For pilot runs, keep `--emotion-limit` small and omit `--require-semantic-coverage`. For final LLM semantic experiments, use `--emotion-limit 0 --require-semantic-coverage` so every selected eval song has a structured semantic profile. Cached profiles are reused on later runs. The current semantic cache uses `structured_semantic_v3`, which stores structured semantic text and does not reuse older raw emotion caches. Raw lyrics are not directly compared in the current console hybrid path; Stage 1 uses CF + popularity candidates, and Stage 2 reranks with metadata, weak mood, and structured semantic-text similarity only where profiles exist. LLM profiling output is kept to one clear semantic progress line with rate, elapsed time, and ETA. On a 4 GB GPU, batch 10 worked locally; if CUDA runs out of memory, lower it.

Later embedding table:

```text
song_id
training_text
semantic_embedding
semantic_cluster_id
```

Prompt target:

```text
Given lyrics, return JSON with:
semantic_summary, themes, lyrical_narrative, listening_context, playlist_function, transition_note, keywords
```

Short-term mood algorithm:

```text
recent_mood = average valence/arousal of last N observed songs
drift = second_half_mood - first_half_mood
projected_mood = recent_mood + beta * drift
score_emotion(candidate) = cosine(candidate_emotion_vector, projected_mood)
```

This supports the proposal novelty:

- LLM-as-emotion-teacher
- mood drift extrapolation
- adaptive dual-temporal fusion
- optional knowledge distillation auxiliary task

## 9. Experiments

Required baselines:

```text
Random
Popularity
Co-occurrence CF
CF + popularity baseline
Hybrid CF + metadata + mood + structured semantic text
```

Later LLM variants:

```text
CF + Gemma 3 semantic training_text embedding
CF + Gemma 3 Semantic ID / cluster
with vs without mood drift
learned fusion
knowledge distillation
```

Ablations:

```text
short_window = 3, 5, 10
candidate_pool = 100, 300, 500
min_df = 1, 2, 5
max_features = 10k, 30k, 50k
alpha = 0.0 to 1.0
CF similarity = raw, cosine, PPMI
```

Result table:

```text
Comparison baseline: CF + popularity baseline
Model                         Retrieval     Recall    dRecall       NDCG      dNDCG      Proxy     dProxy
----------------------------------------------------------------------------------------------------------
Random                                -          ?          ?          ?          ?          ?          ?
Popularity                            -          ?          ?          ?          ?          ?          ?
CF + popularity baseline              ?          ?   +0.00000          ?   +0.00000          ?   +0.00000
Hybrid structured semantic            ?          ?          ?          ?          ?          ?          ?
Best Hybrid structured semantic       ?          ?          ?          ?          ?          ?          ?
No CF                                 ?          ?          ?          ?          ?          ?          ?
No mood/drift                         ?          ?          ?          ?          ?          ?          ?
No metadata                           ?          ?          ?          ?          ?          ?          ?
No structured semantic                ?          ?          ?          ?          ?          ?          ?
```

`dRecall`, `dNDCG`, and `dProxy` are measured against `CF + popularity baseline`.
Positive values mean the added metadata, weak mood, or structured semantic signal helped over the strongest simple recommender.
Negative values mean that signal hurt ranking accuracy on that run.

## 10. Report Direction

Use this report structure:

```text
Abstract
1. Introduction
2. Related Work
3. Methodology
4. Experiments
5. Discussion
6. Limitations and Ethics
7. Conclusion
Contribution Statement
References
```

Important discussion points:

- playlists are proxy users, not real listening histories
- lyric coverage may bias the catalog
- LLM lyric interpretation may contain cultural and language bias
- emotion-aware recommenders can risk emotional spiraling
- metrics may be low on large-catalog playlist continuation, so focus on relative improvement

Reference direction:

```text
LEMON: LLM emotion-aware music recommendation
KAR: LLM knowledge augmentation for recommendation
ROEGEN: LLM song summaries for music recommendation
Item2Vec: playlist sequence embedding baseline
```

## 11. Commands

Run demo pipeline:

```bat
pushd <project-folder>
python .\newSpotify.py --demo
```

Run data review / real pipeline after playlist slices exist:

```bat
pushd <project-folder>
python .\newSpotify.py --lyrics-csv .\data\spotify_millsongdata.csv --mpd-path .\data --max-playlists 1000 --min-playlist-len 2 --holdout-k 10 --pool-size 300 --time-decay 0.9 --max-eval-cases 1000 --progress-interval 100
```

Install and run the smallest Gemma 3 LLM comparison:

```bat
pushd <project-folder>
python -m huggingface_hub.cli.hf auth login
python .\install_llm.py
python .\newSpotify.py --emotion-source llm --lyrics-csv .\data\spotify_millsongdata.csv --mpd-path .\data --max-playlists 1000 --max-eval-cases 100 --min-playlist-len 20 --holdout-k 10 --pool-size 200 --emotion-limit 200 --semantic-weight 0.15 --progress-interval 25
```

Download playlist data once:

```bat
pushd <project-folder>
python .\download_data.py --playlists
```





