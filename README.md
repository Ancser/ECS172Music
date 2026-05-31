# ECS172 Music Recommendation Project

## Full Run Commands

Strict 50+ songs and 50%+ lyrics coverage:

```cmd
F:
cd \ancserProject\ECS172Music
python .\recommandation.py --lyrics-csv .\data\spotify_millsongdata.csv --playlist-csv .\dataFiltered\spotify_playlist_50percent_50item.csv --max-playlists 0 --max-eval-cases 0 --min-playlist-len 20 --holdout-k 10
```

50+ songs only:

```cmd
F:
cd \ancserProject\ECS172Music
python .\recommandation.py --lyrics-csv .\data\spotify_millsongdata.csv --playlist-csv .\dataFiltered\spotify_playlist_50item.csv --max-playlists 0 --max-eval-cases 0 --min-playlist-len 20 --holdout-k 10
```

50%+ lyrics coverage only:

```cmd
F:
cd \ancserProject\ECS172Music
python .\recommandation.py --lyrics-csv .\data\spotify_millsongdata.csv --playlist-csv .\dataFiltered\spotify_playlist_50percent.csv --max-playlists 0 --max-eval-cases 0 --min-playlist-len 20 --holdout-k 10
```

## Recorded Results

### spotify_playlist_50percent_50item.csv

Stage 1 candidate recall:

```text
Stage 1 same-artist candidate recall ========================================================
Candidate method                                 R@100 R@200 R@300 R@400 R@500
------------------------------------------------------------------------------
CF + artist25% + popularity                      0.743 0.809 0.846 0.876 0.893
CF + artist50% + popularity                      0.757 0.820 0.855 0.879 0.893
CF + artist75% + popularity                      0.761 0.821 0.854 0.875 0.888
CF + artist100% + popularity                     0.754 0.800 0.824 0.848 0.865
Popularity                                       0.166 0.220 0.263 0.305 0.331
CF                                               0.678 0.739 0.777 0.804 0.819
```

Ranking result:

```text
Ranking result ========================================================
Model / full description                                                                Recall@10    NDCG@10   Proxy@10
--------------------------------------------------------------------------------------------------------------------------
Random catalog ordering, no personalization                                               0.00144    0.00155    0.00149
Popularity ranking from observed training playlists                                       0.02240    0.02356    0.02298
CF + artist25% + popularity                                                               0.38933    0.40286    0.39609
CF + artist25% + artist score + popularity                                                0.40952    0.41828    0.41390
CF + artist50% + popularity                                                               0.38933    0.40286    0.39609
CF + artist50% + artist score + popularity                                                0.40952    0.41828    0.41390
CF + artist75% + popularity                                                               0.38942    0.40302    0.39622
CF + artist75% + artist score + popularity                                                0.40962    0.41844    0.41403
CF + artist100% + popularity                                                              0.38750    0.40061    0.39405
CF + artist100% + artist score + popularity                                               0.40769    0.41641    0.41205
```

Current best:

```text
CF + artist75% + artist score + popularity
Recall@10 = 0.40962
NDCG@10   = 0.41844
Proxy@10  = 0.41403
```

### spotify_playlist_50item.csv

```text
Ranking result ========================================================
Model / full description                                                                Recall@10    NDCG@10   Proxy@10
--------------------------------------------------------------------------------------------------------------------------
Random catalog ordering, no personalization                                               0.00053    0.00055    0.00054
Popularity ranking from observed training playlists                                       0.02353    0.02732    0.02543
CF + artist25% + popularity                                                               0.14721    0.15536    0.15128
CF + artist25% + artist score + popularity                                                0.14871    0.15547    0.15209
CF + artist50% + popularity                                                               0.14726    0.15540    0.15133
CF + artist50% + artist score + popularity                                                0.14878    0.15554    0.15216
CF + artist75% + popularity                                                               0.14728    0.15537    0.15133
CF + artist75% + artist score + popularity                                                0.14863    0.15545    0.15204
CF + artist100% + popularity                                                              0.13714    0.14650    0.14182
CF + artist100% + artist score + popularity                                               0.13619    0.14483    0.14051
```

Current best:

```text
CF + artist50% + artist score + popularity
Recall@10 = 0.14878
NDCG@10   = 0.15554
Proxy@10  = 0.15216
```

## Direction

This project is now a non-LLM playlist continuation prototype.

```text
Input:  earlier songs in a playlist
Target: final 10 songs of that playlist
Output: top-10 recommended songs
Metrics: Recall@10 and NDCG@10
```

The current system uses original playlist/song metadata and playlist co-occurrence:

```text
Stage 1: same-artist candidate retrieval + CF + popularity
Stage 2: CF + artist metadata + popularity
```

Generated semantic LLM files are removed from the active algorithm. The recommender does not use `songSemantic.py`, `playlistSemantic.py`, `--emotion-source llm`, generated semantic CSV files, or language features.

## Files

```text
recommandation.py   main prototype and evaluator
download_data.py    one-time Kaggle playlist downloader
playlistMarker.py   marks MPD playlists with matched_song_count and matched_coverage_percent
playlistFilter.py   extracts filtered playlist-track CSV files
dataScan.py         coverage scan against the lyrics catalog
getLLM.py           optional model install helper, not used by the current recommender
skill.md            algorithm workflow notes
data/               local raw datasets
dataMarked/         marked MPD JSON output
dataFiltered/       filtered playlist CSV output
```

## Data

Required lyrics catalog:

```text
data/spotify_millsongdata.csv
```

Spotify Million Playlist Dataset slices should live directly in `data/`:

```text
data/mpd.slice.0-999.json
data/mpd.slice.1000-1999.json
...
```

Download playlist data once:

```cmd
pushd <project-folder>
python .\download_data.py --playlists
```

Mark playlist coverage once:

```cmd
pushd <project-folder>
python .\playlistMarker.py --workers 4
```

Extract filtered playlist CSVs:

```cmd
pushd <project-folder>
python .\playlistFilter.py
```

Current filtered outputs:

```text
dataFiltered/spotify_playlist_50percent_50item.csv
dataFiltered/spotify_playlist_50item.csv
dataFiltered/spotify_playlist_50percent.csv
```

## Metadata Used

Song metadata titles:

```text
song_id, title, artist, lyrics, popularity, cf_neighbors
```

Playlist metadata titles:

```text
playlist_id, playlist_name, original_track_count, matched_song_count,
matched_coverage_percent, pos, track_name, artist_name
```

Language is not modeled or printed because the joined lyric catalog is effectively English-only.

## Validation

Use playlist-order last-10 holdout:

```text
heldout  = playlist[-min(10, len(playlist)-1):]
observed = all songs before heldout
```

Rules:

```text
recommendations exclude observed songs
truth = heldout songs, up to 10
top K = 10
candidate pools are fixed at 100, 200, 300, 400, and 500
default min playlist length = 20
```

The ranking step uses the largest listed Stage 1 pool, 500 candidate songs, before choosing the final top 10.

Metrics:

```text
Stage 1: Recall@100, Recall@200, ..., Recall@500
Stage 2: Recall@10, NDCG@10, Proxy@10
Proxy@10 = (Recall@10 + NDCG@10) / 2
```

## Algorithm

Stage 1 retrieves broad candidates and is optimized for recall:

```text
1. Score CF candidates from song-song co-occurrence.
2. Fill with popular songs.
3. Test same-artist quotas at 25%, 50%, 75%, and 100%.
4. Recent observed artists receive higher priority than older observed artists.
```

Stage 2 ranks Stage 1 candidates:

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

## Commands

Run the real current prototype:

```cmd
pushd <project-folder>
python .\recommandation.py --lyrics-csv .\data\spotify_millsongdata.csv --playlist-csv .\dataFiltered\spotify_playlist_50percent_50item.csv --max-playlists 0 --max-eval-cases 1000 --min-playlist-len 20 --holdout-k 10
```

Run a smaller smoke test:

```cmd
pushd <project-folder>
python .\recommandation.py --lyrics-csv .\data\spotify_millsongdata.csv --playlist-csv .\dataFiltered\spotify_playlist_50percent_50item.csv --max-playlists 20 --max-eval-cases 20 --min-playlist-len 20 --holdout-k 10
```

Run demo data:

```cmd
pushd <project-folder>
python .\recommandation.py --demo
```
