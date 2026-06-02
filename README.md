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

### Experiment Descriptions

| Experiment | Data | Purpose | Semantic source | Notes |
|---|---|---|---|---|
| Full non-semantic baseline | `spotify_playlist_50percent_50item.csv`, full run | Main CF + metadata benchmark | None | Best historical result on strict 50+ songs and 50%+ lyrics coverage. |
| Full non-semantic broader data | `spotify_playlist_50item.csv`, full run | Check performance when only playlist length is constrained | None | Lower quality because many playlists have weaker lyric coverage. |
| Fine playlist semantic, 179 playlists | `spotify_playlist_50percent_50item.csv`, first 180 loaded, 179 eval | Test fine controlled playlist semantic schema | Qwen playlist profiles only | Diagnostic run; playlist semantic alone was weak. |
| Playlist-song semantic, 179 playlists | Same 179 eval playlists | Compare playlist tags to song-side tags | Qwen playlist profiles + heuristic song profiles | Small improvement at `semantic_weight=0.01`, but not the final long-run result. |
| Long playlist-song semantic run | `spotify_playlist_50percent_50item.csv`, first 800 loaded, 799 eval | Main two-hour semantic comparison | Qwen playlist profiles + heuristic song profiles | Actual long run took about 104 minutes. |
| Pre-song-profile semantic comparison | Same 799 eval playlists | Compare old direct playlist semantic scoring to new playlist-song semantic scoring | Qwen playlist profiles | Old method matched playlist tags directly against raw title/artist/lyrics text. |
| Semantic as Stage 1 retrieval | Same 799 eval playlists | Test whether semantic can replace CF candidate retrieval | Qwen playlist profiles + heuristic song profiles | Result: semantic should not replace CF in Stage 1. |

The 799 eval playlist ids are saved in:

```text
dataFiltered/eval_playlist_ids_50_50_800_799.csv
```

### Stage 1

Stage 1 measures candidate recall before final top-10 ranking. NDCG and Proxy do
not apply to this candidate-retrieval table.

| Data | Algorithm | Recall@100 | Recall@200 | Recall@300 | Recall@400 | Recall@500 | NDCG@x | Proxy@x |
|---|---|---:|---:|---:|---:|---:|---|---|
| 50/50 CSV, 799 eval playlists | CF + artist25% + popularity | 0.738 | 0.803 | 0.845 | 0.878 | 0.896 | N/A | N/A |
| 50/50 CSV, 799 eval playlists | CF + artist50% + popularity | 0.750 | 0.817 | 0.856 | 0.881 | 0.897 | N/A | N/A |
| 50/50 CSV, 799 eval playlists | CF + artist75% + popularity | 0.756 | 0.820 | 0.856 | 0.875 | 0.891 | N/A | N/A |
| 50/50 CSV, 799 eval playlists | CF + artist100% + popularity | 0.753 | 0.798 | 0.825 | 0.849 | 0.867 | N/A | N/A |
| 50/50 CSV, 799 eval playlists | Popularity | 0.155 | 0.216 | 0.259 | 0.300 | 0.327 | N/A | N/A |
| 50/50 CSV, 799 eval playlists | CF only | 0.656 | 0.722 | 0.762 | 0.788 | 0.808 | N/A | N/A |
| 50/50 CSV, full historical run | CF + artist25% + popularity | 0.743 | 0.809 | 0.846 | 0.876 | 0.893 | N/A | N/A |
| 50/50 CSV, full historical run | CF + artist50% + popularity | 0.757 | 0.820 | 0.855 | 0.879 | 0.893 | N/A | N/A |
| 50/50 CSV, full historical run | CF + artist75% + popularity | 0.761 | 0.821 | 0.854 | 0.875 | 0.888 | N/A | N/A |
| 50/50 CSV, full historical run | CF + artist100% + popularity | 0.754 | 0.800 | 0.824 | 0.848 | 0.865 | N/A | N/A |
| 50/50 CSV, full historical run | Popularity | 0.166 | 0.220 | 0.263 | 0.305 | 0.331 | N/A | N/A |
| 50/50 CSV, full historical run | CF only | 0.678 | 0.739 | 0.777 | 0.804 | 0.819 | N/A | N/A |

### Stage 2

Stage 2 ranks the Stage 1 candidate pool and evaluates the final top-10
recommendations.

| Data | Algorithm | Recall@10 | NDCG@10 | Proxy@10 |
|---|---|---:|---:|---:|
| 50/50 CSV, 799 eval playlists | Random catalog ordering | 0.00113 | 0.00116 | 0.00114 |
| 50/50 CSV, 799 eval playlists | Popularity ranking | 0.01827 | 0.01802 | 0.01815 |
| 50/50 CSV, 799 eval playlists | CF + artist25% + popularity | 0.38235 | 0.39699 | 0.38967 |
| 50/50 CSV, 799 eval playlists | CF + artist25% + artist score + popularity | 0.40150 | 0.41267 | 0.40708 |
| 50/50 CSV, 799 eval playlists | Old direct playlist semantic only | 0.01765 | 0.01854 | 0.01809 |
| 50/50 CSV, 799 eval playlists | Old direct playlist semantic + popularity | 0.01902 | 0.01954 | 0.01928 |
| 50/50 CSV, 799 eval playlists | Old direct CF + artist + popularity + playlist semantic | 0.39212 | 0.40472 | 0.39842 |
| 50/50 CSV, 799 eval playlists | New playlist-song semantic only | 0.01414 | 0.01451 | 0.01433 |
| 50/50 CSV, 799 eval playlists | New playlist-song semantic + popularity | 0.01427 | 0.01439 | 0.01433 |
| 50/50 CSV, 799 eval playlists | New CF + artist + popularity + playlist-song semantic | 0.39750 | 0.40963 | 0.40356 |
| 50/50 CSV, 179 eval playlists | Fine playlist semantic only | 0.01397 | 0.01556 | 0.01476 |
| 50/50 CSV, 179 eval playlists | Fine playlist semantic + popularity | 0.01117 | 0.01302 | 0.01209 |
| 50/50 CSV, 179 eval playlists | CF + artist + popularity baseline | 0.29721 | 0.31077 | 0.30399 |
| 50/50 CSV, 179 eval playlists | CF + artist + popularity + playlist-song semantic, weight 0.01 | 0.29888 | 0.31398 | 0.30643 |
| 50/50 CSV, 199 eval playlists | Deprecated favorite-artist semantic + popularity | 0.31508 | 0.30880 | 0.31194 |
| 50/50 CSV, 199 eval playlists | Deprecated CF + artist + popularity baseline | 0.29397 | 0.30322 | 0.29860 |
| 50/50 CSV, full historical run | Random catalog ordering | 0.00144 | 0.00155 | 0.00149 |
| 50/50 CSV, full historical run | Popularity ranking | 0.02240 | 0.02356 | 0.02298 |
| 50/50 CSV, full historical run | CF + artist25% + popularity | 0.38933 | 0.40286 | 0.39609 |
| 50/50 CSV, full historical run | CF + artist75% + artist score + popularity | 0.40962 | 0.41844 | 0.41403 |
| 50item CSV, full historical run | Random catalog ordering | 0.00053 | 0.00055 | 0.00054 |
| 50item CSV, full historical run | Popularity ranking | 0.02353 | 0.02732 | 0.02543 |
| 50item CSV, full historical run | CF + artist50% + popularity | 0.14726 | 0.15540 | 0.15133 |
| 50item CSV, full historical run | CF + artist50% + artist score + popularity | 0.14878 | 0.15554 | 0.15216 |

### Stage 1 Only

This experiment treats semantic retrieval itself as Stage 1 candidate generation
and disables forced artist quotas. It tests whether semantic retrieval can
replace original co-occurrence CF before metadata reranking.

| Data | Algorithm | Recall@500 | NDCG@x | Proxy@x |
|---|---|---:|---|---|
| 50/50 CSV, same 799 eval playlists | Original CF Stage 1 | 0.80889 | N/A | N/A |
| 50/50 CSV, same 799 eval playlists | Direct playlist semantic Stage 1 | 0.09136 | N/A | N/A |
| 50/50 CSV, same 799 eval playlists | Playlist-song semantic Stage 1 | 0.07472 | N/A | N/A |

The same Stage 1 candidate sources after metadata reranking:

| Data | Algorithm | Recall@10 | NDCG@10 | Proxy@10 |
|---|---|---:|---:|---:|
| 50/50 CSV, same 799 eval playlists | Original CF Stage 1 + metadata rerank | 0.38849 | 0.40172 | 0.39510 |
| 50/50 CSV, same 799 eval playlists | Direct playlist semantic Stage 1 + metadata rerank | 0.01264 | 0.01304 | 0.01284 |
| 50/50 CSV, same 799 eval playlists | Playlist-song semantic Stage 1 + metadata rerank | 0.00738 | 0.00766 | 0.00752 |

### Main Conclusions

| Finding | Evidence |
|---|---|
| Best overall result remains the non-semantic historical 50/50 full run. | `CF + artist75% + artist score + popularity`: Recall@10 `0.40962`, NDCG@10 `0.41844`, Proxy@10 `0.41403`. |
| In the 799-playlist long run, non-semantic metadata reranking is still best. | `CF + artist25% + artist score + popularity`: Proxy@10 `0.40708`. |
| New playlist-song semantic reranking reduces damage compared with old direct playlist-to-lyrics semantic matching. | Old direct semantic combined Proxy@10 `0.39842`; new playlist-song combined Proxy@10 `0.40356`; non-semantic baseline `0.40708`. |
| Semantic-only retrieval is weak. | Stage 1 Recall@500 is `0.09136` for direct playlist semantic and `0.07472` for playlist-song semantic, versus `0.80889` for original CF. |
| Semantic should stay in Stage 2 as a low-weight reranking signal, not replace Stage 1 CF. | Semantic Stage 1 retrieval collapses both candidate recall and final top-10 quality. |
| Deprecated favorite-artist semantic improved scores but was not a clean semantic result. | It mainly behaved like artist metadata / artist-CF, so it was removed from the active semantic schema. |

## Direction

The default project path remains a non-LLM playlist continuation prototype.
Playlist-side LLM semantics are now available as an optional experiment.

```text
Input:  earlier songs in a playlist
Target: final 10 songs of that playlist
Output: top-10 recommended songs
Metrics: Recall@10 and NDCG@10
```

The default system uses original playlist/song metadata and playlist co-occurrence:

```text
Stage 1: same-artist candidate retrieval + CF + popularity
Stage 2: CF + artist metadata + popularity
```

The optional semantic path generates one controlled semantic profile per
evaluation playlist from the observed songs only, caches the JSONL output, builds
song-side semantic profiles for candidate songs, and adds a playlist-song
semantic-match feature to Stage 2 reranking. It does not use heldout songs when
producing playlist semantics.

## File Structure

```text
recommandation.py              main recommender, evaluator, optional playlist LLM semantics
getLLM.py                      installs/caches Qwen locally in models/llm_cache
semantic_stage1_experiment.py  compares CF Stage 1 with semantic Stage 1 retrieval
playlistFilter.py              extracts filtered playlist-track CSV files
playlistMarker.py              marks MPD playlists with lyrics coverage fields
dataScan.py                    scans MPD coverage against the lyrics catalog
download_data.py               one-time Kaggle dataset downloader
skill.md                       project workflow notes
data/                          raw local datasets, including lyrics CSV and MPD slices
dataMarked/                    MPD JSON files with coverage metadata
dataFiltered/                  filtered CSVs and cached playlist semantic JSONL files
models/llm_cache/              local Hugging Face cache for Qwen model files
logs/                          long-run logs and prompt-experiment logs
paper/                         paper/proposal material
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

Song metadata used by the recommender:

```text
song_id, title, artist, lyrics, popularity, cf_neighbors
```

Playlist metadata available in filtered CSVs:

```text
playlist_id, playlist_name, original_track_count, matched_song_count,
matched_coverage_percent, pos, track_name, artist_name
```

Optional playlist semantic metadata:

```text
top_keywords, affect, energy, valence, genre_style, narrative_theme,
listening_context, cohesion, next_song_role
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
1. Build song-song collaborative filtering neighbors from playlist co-occurrence.
2. Score candidates from observed songs using the CF neighbor graph.
3. Add same-artist forced candidates using recent observed artists first.
4. Fill remaining candidate slots with popularity.
5. Evaluate same-artist quotas at 25%, 50%, 75%, and 100%.
6. Report candidate recall at R@100, R@200, R@300, R@400, and R@500.
```

Stage 2 ranks Stage 1 candidates:

```text
final_score =
  cf_weight     * cf_score_norm
+ artist_weight * same_artist_score_norm
+ pop_weight    * popularity_norm
+ semantic_weight * playlist_song_semantic_score_norm   # only when --playlist-semantics is enabled
```

Default weights:

```text
cf_weight     = 0.35
artist_weight = 0.10
pop_weight    = 0.10
semantic_weight = 0.01
```

The semantic feature is optional. When enabled, the recommender generates or
loads one playlist profile from observed songs only. It also builds song-side
semantic profiles for the evaluation catalog, then scores candidates by comparing
playlist tags against candidate-song tags in the same schema.

## Paper Requirement Coverage

The current system covers these project/paper requirements:

```text
Playlist continuation task:
  Input observed prefix, hold out the final songs, recommend top-K continuations.

Collaborative filtering:
  Uses song-song co-occurrence from playlists.

Metadata-aware recommendation:
  Uses artist metadata, popularity, and lyrics-derived optional semantics.

Two-stage recommendation:
  Stage 1 candidate generation, Stage 2 reranking.

Evaluation:
  Reports Recall@10, NDCG@10, Proxy@10, and Stage 1 recall at multiple candidate sizes.

Ablation/comparison:
  Compares random, popularity, CF, CF+artist, and CF+artist+playlist semantic.

Controlled LLM semantic generation:
  Uses fixed schema, allowed labels, JSONL cache, and no heldout-song leakage.
```

Current limitation:

```text
The current fine-keyword semantic run improves the best proxy score slightly
when used with a low weight. The signal is still weak because song-side profiles
are heuristic rather than LLM-generated or embedding-based.
```

## LLM Prompts

### Active playlist-semantic prompt in `recommandation.py`

This is the prompt structure used by the current recommender:

```text
Classify this playlist for music recommendation.
Use only the observed songs below. Do not infer from hidden future songs.
Return valid JSON only. No markdown. No explanation.
Use only labels from the allowed lists. Do not invent new labels.
top_keywords must be exactly 3 labels describing the playlist's musical intent, not artist identity.
affect, genre_style, and narrative_theme must be arrays with 1 to 3 labels.
All other fields must be one label string.

Every key is required. Never return empty strings or empty arrays.
If uncertain, use mixed. For energy use mid. For next_song_role use same_vibe.

Allowed labels:
- top_keywords: high_arousal, low_arousal, dancefloor, theatrical, rebellious, melancholic_story, nostalgic, cinematic, angsty, romantic_tension, party, workout, roadtrip, chill, singalong, dark, uplifting, confident, dreamy, aggressive, playful, anime, club, acoustic, heartbreak, empowerment, mixed
- affect: confident, dramatic, melancholic, angsty, euphoric, dark, playful, dreamy, aggressive, nostalgic, romantic, chill, uplifting, mixed
- genre_style: rock, alt_rock, classic_rock, pop, dance_pop, hiphop, rnb, country, soul, folk, metal, electronic, electropop, punk, indie, soundtrack, anime, mixed
- narrative_theme: love, heartbreak, desire, self_expression, youth, escape, loneliness, resilience, celebration, conflict, fantasy, coming_of_age, depression, empowerment, mixed
- listening_context: party, wedding, halloween, workout, roadtrip, chill, slow_dance, nostalgia, singalong, background, mixed
- energy: low, mid, high
- valence: negative, mixed, positive
- cohesion: genre, affect, activity, theme, story, mixed
- next_song_role: same_vibe, energy_lift, cooldown, genre_bridge, singalong, romantic, spooky

Required JSON schema:
{"top_keywords":[],"affect":[],"energy":"","valence":"","genre_style":[],"narrative_theme":[],"listening_context":"","cohesion":"","next_song_role":""}

playlist_id: {playlist_id}
observed_song_count: {len(observed)}
observed songs, in playlist order, with short lyric excerpts:
- title: {title}; artist: {artist}; lyrics_excerpt: {lyrics_excerpt}

JSON:
```

The current fine-keyword semantic runs used:

```text
semantic_recent_songs = 8
semantic_lyrics_chars = 100
semantic_max_new_tokens = 130
model = Qwen/Qwen2.5-1.5B-Instruct
```

### Song-level LLM prompt

The current recommender does not use a song-level LLM semantic file. Song-side
semantic profiles are currently heuristic profiles built from title, artist, and
lyrics. The LLM only creates playlist-side JSON profiles from the observed songs.

The old/smoke-test song-style prompt idea was:

```text
Convert this song into a controlled semantic profile for a music recommendation experiment.
Return only one JSON object. Do not output numeric scores.
Schema keys: dominant_emotion, secondary_emotion, valence, arousal, theme_tags, emotion_arc, playlist_role.
Allowed emotions: love, sadness, energy, calm, hope, anger, nostalgia, neutral.
Allowed valence: negative, mixed, positive. Allowed arousal: low, medium, high.
Lyrics: {lyrics}
JSON:
```

That song-level prompt is not part of the active algorithm because previous
emotion-only semantics were not reliable enough and the active experiment moved
to playlist-side semantics.

## Commands

Install/cache the local Qwen model inside this project folder:

```cmd
F:
cd \ancserProject\ECS172Music
python .\getLLM.py --preset qwen1b --cache-dir .\models\llm_cache --download-only
```

Run the real current prototype:

```cmd
pushd <project-folder>
python .\recommandation.py --lyrics-csv .\data\spotify_millsongdata.csv --playlist-csv .\dataFiltered\spotify_playlist_50percent_50item.csv --max-playlists 0 --max-eval-cases 1000 --min-playlist-len 20 --holdout-k 10
```

Run the current cached playlist-semantic comparison:

```cmd
F:
cd \ancserProject\ECS172Music
python .\recommandation.py --lyrics-csv .\data\spotify_millsongdata.csv --playlist-csv .\dataFiltered\spotify_playlist_50percent_50item.csv --max-playlists 800 --max-eval-cases 800 --min-playlist-len 20 --holdout-k 10 --playlist-semantics llm --semantic-cache .\dataFiltered\playlist_semantics_fine_keywords_800_qwen.jsonl --song-semantics heuristic --song-semantic-cache .\dataFiltered\song_semantics_fine_keywords_800_heuristic.jsonl --semantic-model Qwen/Qwen2.5-1.5B-Instruct --semantic-model-cache-dir .\models\llm_cache --semantic-device cuda --semantic-weight 0.01 --semantic-recent-songs 8 --semantic-lyrics-chars 100 --semantic-max-new-tokens 130
```

Run the semantic Stage 1 retrieval experiment:

```cmd
F:
cd \ancserProject\ECS172Music
python .\semantic_stage1_experiment.py --lyrics-csv .\data\spotify_millsongdata.csv --playlist-csv .\dataFiltered\spotify_playlist_50percent_50item.csv --playlist-semantic-cache .\dataFiltered\playlist_semantics_fine_keywords_800_qwen.jsonl --song-semantic-cache .\dataFiltered\song_semantics_fine_keywords_800_heuristic.jsonl --playlist-id-output .\dataFiltered\eval_playlist_ids_50_50_800_799.csv --max-playlists 800 --max-eval-cases 800 --min-playlist-len 20 --holdout-k 10
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
