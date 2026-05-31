# Music Recommendation Algorithm Workflow Skill

## 0. Response Rule And Current Command

At the end of every project response, include a short runnable command block that:

1. Changes into the current project folder.
2. Executes the most relevant command for the current state.

For reusable docs, use:

```cmd
pushd <project-folder>
```

For this active conversation, use the real local folder:

```cmd
F:
cd \ancserProject\ECS172Music
```

Current main run:

```cmd
python .\recommandation.py --lyrics-csv .\data\spotify_millsongdata.csv --playlist-csv .\dataFiltered\spotify_playlist_50percent_50item.csv --max-playlists 0 --max-eval-cases 1000 --min-playlist-len 20 --holdout-k 10
```

First-time data setup:

```cmd
python .\download_data.py --playlists
python .\playlistMarker.py --workers 4
python .\playlistFilter.py
```

## 1. Current Scope

The active recommender is non-LLM.

Removed from the active algorithm:

```text
songSemantic.py
playlistSemantic.py
generated semantic CSV files
--emotion-source llm
language features and language output
direct raw lyric text similarity
```

Allowed active signals:

```text
song title and artist for joining/display
artist metadata for same-artist retrieval and ranking
playlist order position for observed/heldout split
playlist co-occurrence collaborative filtering
observed-playlist popularity
```

## 2. Data Review

Always understand MPD-to-lyrics coverage before interpreting algorithm quality.

Use:

```cmd
python .\dataScan.py --workers 8
```

Expected coverage outputs:

```text
matched track entries:       (matched/total) percent
playlists fully available:   (count/total) percent
playlists partially matched: (count/total) percent
playlists with zero matches: (count/total) percent
unique MPD songs matched:    (matched/total) percent
playlist length histogram
matched-song count histogram
```

Coverage cache:

```cmd
python .\playlistMarker.py --workers 4
```

Filtered CSV extraction:

```cmd
python .\playlistFilter.py
```

Current filtered datasets:

```text
dataFiltered/spotify_playlist_50percent_50item.csv
dataFiltered/spotify_playlist_50item.csv
dataFiltered/spotify_playlist_50percent.csv
```

## 3. Validation Design

Use playlist-order continuation:

```text
observed = all songs before the final holdout block
heldout  = final min(10, playlist_length - 1) songs
```

Never use random per-track split for the main experiment. The recommendation task is: given the earlier playlist history, predict the next/final songs.

Metrics:

```text
Stage 1: Recall@100, Recall@200, ..., Recall@500
Stage 2: Recall@10, NDCG@10, Proxy@10
Proxy@10 = (Recall@10 + NDCG@10) / 2
```

Candidate pools are fixed at 100, 200, 300, 400, and 500. The ranking step uses the largest listed Stage 1 pool, 500 candidate songs, before final top-10 ranking.

## 4. Stage 1 Candidate Retrieval

Stage 1 is optimized for recall, not final ranking.

Current Stage 1 method:

```text
same artist + CF + popularity
```

Mechanism:

```text
1. Build co-occurrence CF neighbors from observed playlist songs.
2. Rank CF candidates by summed neighbor similarity.
3. Fill missing candidate slots with globally popular observed songs.
4. Test same-artist quotas at 25%, 50%, 75%, and 100%.
5. Same artists from the most recent observed songs receive the strongest boost.
```

Console output must show the same-artist Stage 1 recall grid:

```text
Candidate method                                  R@100 R@200 R@300 R@400 R@500
CF + artist25% + popularity                       ...
CF + artist50% + popularity                       ...
CF + artist75% + popularity                       ...
CF + artist100% + popularity                      ...
Popularity                                        ...
CF                                                ...
```

## 5. Stage 2 Ranking

Stage 2 reranks the Stage 1 candidates into top 10.

Current formula:

```text
final_score =
  cf_weight     * cf_score_norm
+ artist_weight * same_artist_score_norm
+ pop_weight    * popularity_norm
```

Default weights:

```text
cf_weight     = 0.35
artist_weight = 0.10
pop_weight    = 0.10
```

Required Stage 2 output table:

```text
Model / full description                                                   Recall@10 NDCG@10 Proxy@10
Random catalog ordering, no personalization
Popularity ranking from observed training playlists
CF + popularity on same-artist quota candidates
CF + artist25% + popularity
CF + artist25% + artist score + popularity
CF + artist50% + popularity
CF + artist50% + artist score + popularity
CF + artist75% + popularity
CF + artist75% + artist score + popularity
CF + artist100% + popularity
CF + artist100% + artist score + popularity
```

## 6. Metadata Tables

At the beginning of console output, print aligned metadata-title columns only.

Song metadata fields:

```text
song_id
title
artist
lyrics
popularity
cf_neighbors
```

Playlist metadata fields:

```text
playlist_id
playlist_name
original_track_count
matched_song_count
matched_coverage_percent
pos
track_name
artist_name
```

Do not print language fields.
Do not print metadata source/use descriptions in console output.

## 7. Commands

Small smoke test:

```cmd
F:
cd \ancserProject\ECS172Music
python .\recommandation.py --lyrics-csv .\data\spotify_millsongdata.csv --playlist-csv .\dataFiltered\spotify_playlist_50percent_50item.csv --max-playlists 20 --max-eval-cases 20 --min-playlist-len 20 --holdout-k 10
```

Fuller filtered run:

```cmd
F:
cd \ancserProject\ECS172Music
python .\recommandation.py --lyrics-csv .\data\spotify_millsongdata.csv --playlist-csv .\dataFiltered\spotify_playlist_50percent_50item.csv --max-playlists 0 --max-eval-cases 1000 --min-playlist-len 20 --holdout-k 10
```
