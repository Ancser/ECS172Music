# Music Recommendation Algorithm Workflow Skill

## 0. Response Rule And First Run

At the end of every project response, include a short runnable command block that:

1. Changes into the current project folder.
2. Executes the most relevant command for the current state.

Use a generic project folder placeholder in reusable docs:

```powershell
pushd <project-folder>
```

For this active conversation only, the current local folder is:

```text
F:\ancserProject\ECS172Music
```

When answering in this conversation, use the real current folder in the final command block.

First-time setup must download the playlist data before real validation can run:

```powershell
pushd <project-folder>
python .\download_data.py --playlists
```

After data exists, run the real data review / prototype:

```powershell
pushd <project-folder>
python .\newSpotify.py --mpd-path .\data --lyrics-csv .\data\spotify_millsongdata.csv --max-playlists 1000 --max-eval-cases 1000 --holdout-k 10 --min-playlist-len 2 --time-decay 0.9 --progress-interval 100
```

Optional Gemma 3 LLM setup and small comparison run:

```powershell
pushd <project-folder>
python -m huggingface_hub.cli.hf auth login
python .\install_llm.py --cuda-torch --device cuda
python .\newSpotify.py --emotion-source llm --llm-device cuda --llm-batch-size 20 --require-semantic-coverage --mpd-path .\data --lyrics-csv .\data\spotify_millsongdata.csv --max-playlists 1000 --max-eval-cases 100 --holdout-k 10 --min-playlist-len 20 --pool-size 200 --emotion-limit 0 --semantic-weight 0.15 --progress-interval 25
```

Default LLM model: `google/gemma-3-270m-it`.
Use this first because the local GPU memory budget is about 3 GB.
Use `--require-semantic-coverage --emotion-limit 0` for final LLM runs, because this fails instead of comparing unprocessed songs semantically when selected playlist songs are missing structured semantic profiles. Raw lyrics should not be compared directly in the hybrid LLM path. LLM progress should stay readable: one semantic song progress line with cached count, batch size, rate, elapsed time, and ETA. On the local 4 GB GPU, batch 20 worked; lower it if CUDA memory fails.

For the local Spotify-style web demo:

```powershell
pushd <project-folder>
.\start_spotify_web.bat
```

Default web recommendation controls are `batch=20` and `candidates=200`, because the local GPU handled batch 20 well and candidate 200 better demonstrates the two-stage retrieval/reranking design.

For data coverage review:

```powershell
python F:\ancserProject\ECS172Music\dataScan.py
```

Use `python F:\ancserProject\ECS172Music\dataScan.py --max-playlists 1000` for a quick sample. Use `python F:\ancserProject\ECS172Music\dataScan.py --workers 8` for a full multi-threaded MPD scan. The scan should report MPD track coverage against the lyrics catalog as `(matched/total) percent`, playlist full/partial/zero-match counts, playlist filters by matched song count and matched coverage percent, and playlist length bar charts for original MPD length and matched-song count.

To cache coverage into marked MPD JSON copies once:

```powershell
python F:\ancserProject\ECS172Music\dataFilter.py --workers 4
```

This reads original MPD JSON from `data/` and writes marked copies to `dataMarked/`; it must not modify original playlist data. It adds integer fields `matched_song_count` and `matched_coverage_percent` to each playlist object. Use `--dry-run --max-files 1` before a full write if testing.

To extract filtered playlist-track CSV files:

```powershell
python F:\ancserProject\ECS172Music\dataExtract.py
```

This reads `dataMarked/` by default and creates `dataFiltered/playlists_50songs_50coverage.csv`, `dataFiltered/playlists_50songs.csv`, and `dataFiltered/playlists_50coverage.csv`.

For a quick no-data demo:

```powershell
pushd <project-folder>
python .\newSpotify.py --demo
```

If a command cannot run yet because data is missing, still show the command and clearly state what file or folder is needed first.

## 1. Purpose

This skill defines the working algorithm structure for the ECS172 music recommendation project.

The project goal is playlist continuation:

```text
Input:  all earlier songs in a playlist
Target: up to final 10 songs of the same playlist
Output: top-10 recommended songs
Metric: Recall@10 and NDCG@10
```

The current prototype must stay simple and measurable before adding LLMs:

```text
lyrics CSV + playlist data
  -> data review
  -> Stage 1 candidate retrieval
  -> Stage 2 ranking
  -> validation metrics
  -> algorithm comparison
```

Core principle:

```text
Stage 1 optimizes candidate recall.
Stage 2 optimizes top-10 ranking quality.
Validate each stage separately.
Do not add LLM or learned models before the baselines are measured.
```

## 2. Data Review

Always start by understanding the data before modeling.

### 2.1 Required Data Sources

Lyrics data:

```text
data/spotify_millsongdata.csv
columns usually: artist, song, link, text
```

Playlist data:

```text
data/mpd.slice.*.json
source: Spotify Million Playlist Dataset
each playlist is treated as a proxy user/session
```

A simple custom playlist CSV can also be used:

```text
playlist_id, track_name, artist_name, position
```

### 2.2 Join Logic

Join lyrics to playlist tracks by normalized:

```text
(artist_name, track_name)
```

Normalization should:

```text
lowercase text
remove bracketed suffixes like "(Remastered)" or "[Live]"
remove punctuation
collapse whitespace
```

Important join metrics:

```text
raw lyric rows
unique lyric songs
raw playlist count
raw playlist tracks
matched playlist tracks
unique matched tracks
lyrics coverage = matched playlist tracks / raw playlist tracks
playlist retention = playlists after filtering / raw playlists
```

MPD does not provide per-song `date_added`. It provides playlist-level `modified_at` and track-level `pos`, so use `pos` as the old-to-new order inside each playlist.

### 2.3 Filtering

Filter for meaningful evaluation:

```text
playlist length after lyrics join >= 2
one observed song is valid
heldout songs = final songs, up to 10
if observed length <= cold_start_threshold, use popularity-only cold start
deduplicate repeated tracks within a playlist before splitting
```

Use smaller samples first:

```text
max_playlists = 1,000 for debugging
max_playlists = 50,000 for first real experiment
max_playlists = 100,000 if runtime is acceptable
```

### 2.4 Data Statistics To Print

For every run, print:

```text
head 10 songs
head 10 playlists
song lyric rows
playlist count
matched interactions
unique matched tracks
average playlist length
median playlist length
min/max playlist length
average lyric token count
median lyric token count
matrix density
cold playlists with <= 3 observed songs
```

Matrix density:

```text
density = matched_interactions / (num_playlists * unique_tracks)
sparsity = 1 - density
```

Interpretation:

```text
Low density means collaborative filtering will be sparse.
Low lyric coverage means lyrics-only recommendation may bias toward older or more popular songs.
Short playlists make short-term mood estimates unstable.
```

## 3. Validation

### 3.1 Split

Use fixed last-10 playlist-order holdout:

```text
For each playlist:
  heldout  = playlist[-min(10, len(playlist)-1):]
  observed = all songs before heldout
```

This directly tests the top-10 playlist continuation task:

```text
Given previous songs, recommend the next/final songs, up to 10.
```

Do not randomly split tracks inside a playlist, because random split leaks future playlist context into training.

### 3.2 Ground Truth

For each playlist:

```text
truth = set(heldout songs)
recommendations must exclude observed songs
```

### 3.3 Metrics

Recall@K:

```text
Recall@K = number of heldout songs in top K / number of heldout songs
```

Use:

```text
K = 10 for final ranking
K = 300 for Stage 1 retrieval
```

NDCG@K:

```text
DCG@K = sum over relevant recommendations of 1 / log2(rank + 1)
IDCG@K = best possible DCG for that playlist
NDCG@K = DCG@K / IDCG@K
```

Proxy metric for tuning:

```text
Proxy = (Recall@10 + NDCG@10) / 2
```

Stage-specific metrics:

```text
Stage 1: Retrieval@300
Stage 2: Recall@10, NDCG@10, Proxy
```

Optional diagnostic metrics:

```text
catalog coverage = unique recommended tracks / catalog size
artist diversity = unique recommended artists / recommended items
popularity bias = average popularity rank of recommendations
cold-track hit rate = heldout cold tracks recovered / heldout cold tracks
```

## 4. Stage 1 Candidate Retrieval

### 4.1 Goal

Stage 1 should create a broad candidate pool.

It does not need perfect ordering.

Primary metric:

```text
Retrieval@300
```

Definition:

```text
Retrieval@300 = heldout songs appearing anywhere in candidate pool / heldout songs
```

### 4.2 Stage 0: Song Classification

Current no-LLM song features:

```text
language = simple lyric language heuristic
lyric type = rule-based topic/emotion bucket
valence/arousal = weak emotion vector from lyric lexicon
artist = metadata from MPD/lyrics
```

Lyric type buckets:

```text
love, sadness, energy, anger, calm, hope, nostalgia, other
```

These are placeholders for future LLM-generated emotion tags.

### 4.3 Route A: Lyrics TF-IDF Retrieval

Current no-LLM prototype route.

Algorithm:

```text
1. Tokenize each song lyric.
2. Compute document frequency per token.
3. Compute IDF:
   idf(t) = log((1 + num_songs) / (1 + df(t))) + 1
4. Build normalized TF-IDF song vector.
5. Build playlist profile by averaging vectors of observed songs.
6. Score candidate songs with cosine similarity.
7. Exclude observed songs.
```

Long-term profile:

```text
profile_long = time-decayed average TF-IDF vector of all observed songs
score_long(candidate) = cosine(profile_long, candidate_vector)
```

Short-term profile:

```text
profile_short = time-decayed average TF-IDF vector of last N observed songs
score_short(candidate) = cosine(profile_short, candidate_vector)
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
  skip TF-IDF profile scoring
  rank with popularity fallback
default cold_start_threshold = 1
```

Runtime acceleration:

```text
prepare candidate scores once
print progress every --progress-interval eval playlists
reuse cached long/short/popularity scores for every alpha
```

Tune:

```text
min_df
max_features
short_window N
candidate pool size
```

### 4.4 Route B: Co-Occurrence Collaborative Filtering

Playlist co-occurrence route.

Algorithm:

```text
1. Treat each playlist as a basket of songs.
2. For every pair of songs in the same observed playlist, increment co-occurrence count.
3. Normalize raw counts to reduce popularity domination.
4. For a target playlist, sum similarities from observed songs to candidate songs.
5. Exclude observed songs.
```

Useful similarity options:

Cosine co-occurrence:

```text
sim(i, j) = co_count(i, j) / sqrt(pop(i) * pop(j))
```

PMI:

```text
pmi(i, j) = log((co_count(i, j) * num_playlists) / (pop(i) * pop(j)))
ppmi(i, j) = max(pmi(i, j), 0)
```

Playlist score:

```text
score_cf(candidate) = sum(sim(candidate, observed_song) for observed_song in playlist)
```

Tune:

```text
similarity type: raw, cosine, PPMI
max neighbors per song
candidate pool size
```

### 4.5 Route C: Popularity Retrieval

Popularity is a serious baseline, not only a fallback.

Algorithm:

```text
popularity(song) = number of training playlists containing song
score_pop(song) = normalized popularity(song)
```

Use popularity for:

```text
cold playlists with very few observed songs
tie breaking
candidate pool stabilization
baseline comparison
```

Avoid overtrusting it:

```text
High Recall@10 from popularity may indicate dataset popularity bias.
Check catalog coverage and artist diversity.
```

### 4.6 Route D: Structured LLM Semantic Text

Use this only after CF + popularity is measured.

Use an LLM-generated structured song semantic table:

```text
song_id
semantic_summary
themes
lyrical_narrative
listening_context
playlist_function
transition_note
keywords
training_text
```

Candidate score:

```text
recent_semantic = average training_text vector of last N observed songs with profiles
score_semantic(candidate) = cosine(candidate_semantic_text_vector, recent_semantic)
```

This replaces or augments the current short-term lyrics TF-IDF route.

### 4.7 Stage 1 Fusion

Merge retrieval routes with normalized scores:

```text
stage1_score =
  w_lyrics * lyrics_score_norm
+ w_cf     * cf_score_norm
+ w_pop    * popularity_score_norm
```

Grid search weights by Retrieval@300:

```text
w_lyrics + w_cf + w_pop = 1
step = 0.10 for first search
step = 0.05 for refinement
```

Report:

```text
best Stage 1 weights
Retrieval@300
average candidate count
candidate catalog coverage
```

## 5. Stage 2 Ranking

### 5.1 Goal

Stage 2 reranks the Stage 1 candidate pool into final top-10 recommendations.

Primary metrics:

```text
Recall@10
NDCG@10
Proxy = (Recall@10 + NDCG@10) / 2
```

### 5.2 Ranking Features

Start with interpretable features.

Lyrics features:

```text
lyrics_long_norm:
  normalized cosine similarity to full observed playlist TF-IDF profile

lyrics_short_norm:
  normalized cosine similarity to last N songs TF-IDF profile

lyrics_delta:
  lyrics_short_norm - lyrics_long_norm
```

Collaborative features:

```text
cf_score_norm:
  normalized co-occurrence similarity to observed playlist songs

same_artist_count:
  number of observed songs by candidate artist
```

Popularity features:

```text
track_popularity_norm:
  normalized playlist-count popularity of candidate track

artist_popularity_norm:
  normalized playlist-count popularity of candidate artist
```

Metadata / weak emotion features:

```text
artist_overlap_norm:
  candidate artist count in observed playlist

type_overlap_norm:
  candidate lyric type count in observed playlist

language_match_norm:
  candidate lyric language count in observed playlist

mood_drift_score_norm:
  similarity to recent_mood + beta * drift_vector
```

Future emotion features:

```text
emotion_similarity:
  cosine similarity between candidate valence/arousal and recent mood

mood_drift_magnitude:
  norm(second_half_mood - first_half_mood)

mood_projection_score:
  similarity to recent_mood + beta * drift_vector
```

### 5.3 Manual Fusion Ranker

Use a weighted ranker before learned models:

```text
final_score =
  lyrics_weight * (alpha * lyrics_long_norm + (1-alpha) * lyrics_short_norm)
+ cf_weight * cf_score_norm
+ mood_weight * mood_drift_score_norm
+ artist_weight * artist_overlap_norm
+ type_weight * type_overlap_norm
+ language_weight * language_match_norm
+ pop_weight * popularity_norm
```

Grid search:

```text
a + b + c + d + e = 1
step = 0.20 for first search
step = 0.10 for refinement
```

Select by:

```text
highest Proxy
tie-breaker 1: higher Recall@10
tie-breaker 2: higher NDCG@10
tie-breaker 3: less popularity bias
```

### 5.4 Learned Ranker Caution

Learned classifiers are optional later.

Do not assume logistic regression or MLP improves top-10 ranking.

Risks:

```text
negative sampling changes the problem
classification loss does not directly optimize NDCG
class imbalance is severe
probability calibration does not guarantee good ranking
```

If using a learned ranker:

```text
train only on Stage 1 candidates
sample negatives from the same candidate pools
validate by Recall@10 and NDCG@10, not accuracy
compare against manual fusion
```

## 6. Experiment Plan

### 6.1 Required Baselines

Run and report:

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
mood drift ablation
```

### 6.2 Ablations

Short-term window:

```text
N = 3, 5, 10
```

Candidate size:

```text
K = 100, 300, 500
```

Lyrics vector settings:

```text
min_df = 1, 2, 5
max_features = 10k, 30k, 50k
```

Fusion:

```text
alpha from 0.0 to 1.0
alpha = weight on long-term lyrics profile
1 - alpha = weight on short-term lyrics profile
```

CF normalization:

```text
raw co-count
cosine
PPMI
```

### 6.3 Result Table

Use this table structure:

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

Include data stats above the result table so results are interpretable.
Use `dRecall`, `dNDCG`, and `dProxy` to explain whether each added component improves over the strongest simple baseline.

## 7. Cold Start And Bias Checks

### 7.1 Cold Playlists

Definition:

```text
cold playlist = observed songs <= 3
```

Handling:

```text
use lyrics profile if lyrics exist
use popularity fallback
avoid overfitting short-term profile from one song
```

Report:

```text
Recall@10 for cold playlists
Recall@10 for non-cold playlists
```

### 7.2 Cold Tracks

Definition:

```text
cold track = no training playlist interactions
```

Handling:

```text
CF cannot retrieve cold tracks
lyrics TF-IDF can retrieve cold tracks if lyrics exist
LLM emotion can retrieve cold tracks later if tags exist
```

Report:

```text
cold-track heldout count
cold-track hit rate
```

### 7.3 Popularity Bias

Measure:

```text
average popularity rank of recommendations
catalog coverage
artist diversity
```

Interpretation:

```text
If popularity wins but coverage is tiny, the model may be recommending generic hits.
If lyrics improves coverage while keeping Recall@10 close, that is useful for the report.
```

## 8. Implementation Commands

Current data review:

```powershell
pushd <project-folder>
python .\newSpotify.py
```

Demo full pipeline:

```powershell
pushd <project-folder>
python .\newSpotify.py --demo
```

Real MPD run after placing playlist slices in `data/`:

```powershell
pushd <project-folder>
python .\newSpotify.py --lyrics-csv .\data\spotify_millsongdata.csv --mpd-path .\data --max-playlists 1000 --min-playlist-len 2 --holdout-k 10 --pool-size 300 --time-decay 0.9 --max-eval-cases 1000 --progress-interval 100
```

Gemma 3 LLM run:

```powershell
pushd <project-folder>
python -m huggingface_hub.cli.hf auth login
python .\install_llm.py --cuda-torch --device cuda
python .\newSpotify.py --emotion-source llm --llm-device cuda --llm-batch-size 20 --require-semantic-coverage --lyrics-csv .\data\spotify_millsongdata.csv --mpd-path .\data --max-playlists 1000 --max-eval-cases 100 --min-playlist-len 20 --holdout-k 10 --pool-size 200 --emotion-limit 0 --semantic-weight 0.15 --progress-interval 25
```

LLM semantic progress should report `semantic songs done/total`, cached profiles, batch size, speed, elapsed time, and ETA. Keep warning noise hidden; if a warning appears, treat it as an implementation bug to clean up unless it is an actual model loading failure.

## 9. Final Lesson

The project should progress in this order:

```text
1. Verify data and join coverage.
2. Build measurable CF + popularity baseline.
3. Add popularity baseline.
4. Add co-occurrence CF baseline.
5. Tune Stage 1 by Retrieval@300.
6. Tune Stage 2 by Recall@10 and NDCG@10.
7. Only then add structured LLM semantic text and mood drift.
```

Strong algorithms are built by separating the jobs:

```text
data review explains the dataset
Stage 1 finds enough plausible songs
Stage 2 orders the best 10
validation proves whether the change helped
```






