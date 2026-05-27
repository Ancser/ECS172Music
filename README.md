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

Current implementation intentionally starts simpler:

```text
No LLM yet.
Use lyrics TF-IDF/content-IDF first.
Measure baselines before adding emotion tags or learned models.
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
newSpotify.py   main prototype and evaluator
download.py     one-time Kaggle downloader
skill.md        detailed algorithm workflow notes
SPEC.md         pointer to this README
data/           local datasets
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
python .\download.py --playlists
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

## 4. Data Review

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

### Route A: Lyrics TF-IDF

Current implemented no-LLM route.

Algorithm:

```text
1. Tokenize lyrics.
2. Compute document frequency per token.
3. Compute IDF:
   idf(t) = log((1 + num_songs) / (1 + df(t))) + 1
4. Build normalized TF-IDF vector per song.
5. Average observed-song vectors into playlist profile.
6. Score candidate songs by cosine similarity.
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
  rank candidates with popularity fallback
default cold_start_threshold = 1
```

Speedup:

```text
candidate scores are prepared once
alpha sweep reuses cached long/short/popularity scores
progress prints every --progress-interval eval playlists
```

### Route B: Co-Occurrence CF

Next baseline to add.

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

Later combine retrieval routes:

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
  alpha * lyrics_long_norm
+ (1 - alpha) * lyrics_short_norm
+ pop_weight * popularity_norm
```

Grid search:

```text
alpha = 0.0, 0.1, ..., 1.0
```

Future Stage 2 features:

```text
lyrics_long_norm
lyrics_short_norm
lyrics_delta
cf_score_norm
track_popularity_norm
artist_popularity_norm
same_artist_count
playlist_length
observed_unique_artists
```

Do not assume a learned classifier is better. Logistic regression or MLP should only be added after manual fusion baselines are measured, and must be validated by Recall@10/NDCG@10 rather than accuracy.

## 8. Future LLM Emotion Stage

Do not add LLM calls until the TF-IDF, popularity, and CF baselines are measured.

Later offline emotion table:

```text
song_id
primary_emotion
secondary_emotion
valence
arousal
mood_summary
```

Prompt target:

```text
Given lyrics, return JSON with:
primary_emotion, secondary_emotion, valence, arousal, mood_summary
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
Lyrics TF-IDF long-term
Lyrics TF-IDF short-term
Lyrics TF-IDF long/short fusion
Co-occurrence CF
Lyrics TF-IDF + CF + popularity
```

Later LLM variants:

```text
LLM emotion only
CF + LLM emotion
CF + lyrics TF-IDF + LLM emotion
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
| Model | Retrieval | Recall | NDCG | Proxy |
|---|---:|---:|---:|---:|
| Random | - | ? | ? | ? |
| Popularity | - | ? | ? | ? |
| TF-IDF alpha=0.00 | ? | ? | ? | ? |
| TF-IDF alpha=0.50 | ? | ? | ? | ? |
| TF-IDF alpha=1.00 | ? | ? | ? | ? |
| Best TF-IDF alpha=? | ? | ? | ? | ? |
```

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

Download playlist data once:

```bat
pushd <project-folder>
python .\download.py --playlists
```



