#!/usr/bin/env python3
"""LEMON-inspired smoke prototype.

This is not the full industrial LEMON model from the paper. It mirrors the
paper structure in a lightweight offline experiment:

1. song-level emotion extraction
2. emotion-enhanced song vectors, either hand-weighted or learned with SVD
3. short-term and long-term user emotion representations
4. adaptive fusion
5. Stage 2 reranking over the existing CF candidate pool
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import Counter
from pathlib import Path

import recommandation as rec


ROOT = Path(__file__).resolve().parent
DEFAULT_PLAYLIST_CSV = ROOT / "dataFiltered" / "playlist_50%_50c_799.csv"
DEFAULT_SONG_SEMANTIC_CSV = ROOT / "dataFiltered" / "song_semantics_fine_keywords_800_heuristic.csv"
DEFAULT_SONG_LLM_CACHE = ROOT / "dataFiltered" / "song_semantics_fine_keywords_qwen.jsonl"

LIST_WEIGHTS = {
    "top_keywords": 1.6,
    "affect": 2.2,
    "genre_style": 1.2,
    "narrative_theme": 1.7,
}
SCALAR_WEIGHTS = {
    "energy": 1.0,
    "valence": 0.8,
    "listening_context": 0.8,
}


def parse_pipe_list(value: str) -> list[str]:
    value = str(value or "").strip()
    if not value:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def load_song_semantic_csv(path: Path, songs: dict[str, rec.Song]) -> dict[str, dict[str, object]]:
    profiles: dict[str, dict[str, object]] = {}
    if not path.exists():
        return profiles
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return profiles
        for row in reader:
            song_id = str(row.get("song_id", ""))
            if song_id not in songs:
                continue
            profile: dict[str, object] = {}
            for key in rec.SEMANTIC_JSON_KEYS:
                if key in rec.SEMANTIC_LIST_KEYS:
                    profile[key] = parse_pipe_list(row.get(key, ""))
                else:
                    profile[key] = row.get(key, "")
            profiles[song_id] = rec.normalize_semantic_profile(profile)
    return profiles


def build_song_profiles(
    songs: dict[str, rec.Song],
    semantic_csv: Path,
) -> dict[str, dict[str, object]]:
    profiles = load_song_semantic_csv(semantic_csv, songs)
    generated = 0
    for song_id, song in songs.items():
        if song_id not in profiles:
            profiles[song_id] = rec.heuristic_song_semantic(song)
            generated += 1
    print(f"  loaded song profiles:    {len(profiles) - generated:,}")
    print(f"  generated fallback:      {generated:,}")
    return profiles


def song_semantic_prompt(song: rec.Song, lyrics_chars: int) -> str:
    allowed_lines = "\n".join(
        f"- {key}: {', '.join(values)}"
        for key, values in rec.SEMANTIC_ALLOWED.items()
    )
    lyrics_excerpt = " ".join(song.lyrics.split())[:lyrics_chars]
    return (
        "Classify this song for music recommendation.\n"
        "Return valid JSON only. No markdown. No explanation.\n"
        "Use only labels from the allowed lists. Do not invent new labels.\n"
        "top_keywords must be exactly 3 labels describing musical intent, not artist identity.\n"
        "affect, genre_style, and narrative_theme must be arrays with 1 to 3 labels.\n"
        "All other fields must be one label string.\n\n"
        "Every key is required. Never return empty strings or empty arrays.\n"
        "If uncertain, use mixed. For energy use mid. For next_song_role use same_vibe.\n\n"
        "Allowed labels:\n"
        f"{allowed_lines}\n\n"
        "Required JSON schema:\n"
        '{"top_keywords":[],"affect":[],"energy":"","valence":"","genre_style":[],"narrative_theme":[],"listening_context":"","cohesion":"","next_song_role":""}\n\n'
        f"title: {song.title}\n"
        f"artist: {song.artist}\n"
        f"lyrics_excerpt: {lyrics_excerpt}\n\n"
        "JSON:"
    )


def llm_song_semantic(pipe, song: rec.Song, lyrics_chars: int, max_new_tokens: int) -> dict[str, object]:
    output = pipe(
        [{"role": "user", "content": song_semantic_prompt(song, lyrics_chars)}],
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    parsed = rec.extract_json_object(rec.generated_text_from_pipeline_output(output))
    if not parsed:
        parsed = rec.heuristic_song_semantic(song)
    return rec.normalize_semantic_profile(parsed)


def llm_song_semantic_batch(
    pipe,
    batch: list[tuple[str, rec.Song]],
    lyrics_chars: int,
    max_new_tokens: int,
    batch_size: int,
) -> list[tuple[str, dict[str, object]]]:
    prompts = [
        [{"role": "user", "content": song_semantic_prompt(song, lyrics_chars)}]
        for _, song in batch
    ]
    try:
        outputs = pipe(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            batch_size=max(1, batch_size),
        )
    except Exception as exc:
        print(f"  batch generation failed, falling back to single-song generation: {exc}", flush=True)
        return [
            (song_id, llm_song_semantic(pipe, song, lyrics_chars, max_new_tokens))
            for song_id, song in batch
        ]

    results: list[tuple[str, dict[str, object]]] = []
    for (song_id, song), output in zip(batch, outputs):
        parsed = rec.extract_json_object(rec.generated_text_from_pipeline_output(output))
        if not parsed:
            parsed = rec.heuristic_song_semantic(song)
        results.append((song_id, rec.normalize_semantic_profile(parsed)))
    return results


def build_qwen_song_profiles(
    songs: dict[str, rec.Song],
    cache_path: Path,
    model_name: str,
    model_cache_dir: Path,
    device: str,
    lyrics_chars: int,
    max_new_tokens: int,
    max_generate: int,
    batch_size: int = 1,
) -> dict[str, dict[str, object]]:
    cache = rec.load_semantic_cache(cache_path)
    profiles: dict[str, dict[str, object]] = {}
    new_records: list[dict[str, object]] = []
    pipe = None
    reused = 0
    generated = 0
    fallback = 0
    no_data = 0
    started = time.time()

    pending: list[tuple[int, str, rec.Song]] = []
    items = sorted(songs.items())

    def flush_pending() -> None:
        nonlocal generated, pipe, new_records
        if not pending:
            return
        if pipe is None:
            pipe = rec.load_text_generation_pipeline(model_name, device, model_cache_dir)
        batch = [(song_id, song) for _, song_id, song in pending]
        for song_id, profile in llm_song_semantic_batch(pipe, batch, lyrics_chars, max_new_tokens, batch_size):
            song = songs[song_id]
            profiles[song_id] = profile
            generated += 1
            new_records.append(
                {
                    "cache_key": rec.song_semantic_cache_key(song_id),
                    "song_id": song_id,
                    "title": song.title,
                    "artist": song.artist,
                    "semantic": profile,
                }
            )
        pending.clear()
        if len(new_records) >= 50:
            rec.append_semantic_cache(cache_path, new_records)
            new_records = []

    for idx, (song_id, song) in enumerate(items, start=1):
        key = rec.song_semantic_cache_key(song_id)
        if key in cache:
            profiles[song_id] = rec.normalize_semantic_profile(cache[key])
            reused += 1
        elif max_generate and generated >= max_generate:
            profiles[song_id] = rec.heuristic_song_semantic(song)
            fallback += 1
        elif not song.title.strip() or not song.artist.strip() or not song.lyrics.strip():
            profiles[song_id] = rec.heuristic_song_semantic(song)
            no_data += 1
            fallback += 1
        else:
            pending.append((idx, song_id, song))
            if len(pending) >= max(1, batch_size):
                flush_pending()

        if idx == 1 or idx == len(songs) or idx % rec.PROGRESS_INTERVAL == 0:
            flush_pending()
            print(
                f"  qwen song semantics {idx:,}/{len(songs):,} | "
                f"reused={reused:,} generated={generated:,} fallback={fallback:,} "
                f"no_data={no_data:,} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    flush_pending()
    rec.append_semantic_cache(cache_path, new_records)
    return profiles


def add_value(vector: dict[str, float], key: str, value: float) -> None:
    if value:
        vector[key] = vector.get(key, 0.0) + value


def profile_to_vector(profile: dict[str, object]) -> dict[str, float]:
    vector: dict[str, float] = {}
    for key, weight in LIST_WEIGHTS.items():
        values = rec.semantic_profile_values(profile, key)
        for value in values:
            if value and value != "mixed":
                add_value(vector, f"{key}:{value}", weight)
    for key, weight in SCALAR_WEIGHTS.items():
        values = rec.semantic_profile_values(profile, key)
        for value in values:
            if value and value != "mixed":
                add_value(vector, f"{key}:{value}", weight)
    return l2_normalize(vector)


def build_feature_vocabulary(profiles: dict[str, dict[str, object]]) -> dict[str, int]:
    vocabulary: dict[str, int] = {}
    for profile in profiles.values():
        for feature in profile_to_vector(profile):
            if feature not in vocabulary:
                vocabulary[feature] = len(vocabulary)
    return vocabulary


def build_svd_song_vectors(
    profiles: dict[str, dict[str, object]],
    dimensions: int,
    random_state: int,
) -> dict[str, dict[str, float]]:
    try:
        import numpy as np
        from scipy import sparse
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
    except Exception as exc:
        print(f"  SVD unavailable, falling back to heuristic vectors: {exc}")
        return {song_id: profile_to_vector(profile) for song_id, profile in profiles.items()}

    song_ids = sorted(profiles)
    vocabulary = build_feature_vocabulary(profiles)
    if not song_ids or not vocabulary:
        return {}

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row_idx, song_id in enumerate(song_ids):
        for feature, value in profile_to_vector(profiles[song_id]).items():
            col_idx = vocabulary.get(feature)
            if col_idx is not None and value:
                rows.append(row_idx)
                cols.append(col_idx)
                values.append(value)

    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(song_ids), len(vocabulary)), dtype=float)
    components = max(1, min(dimensions, matrix.shape[0] - 1, matrix.shape[1] - 1))
    if components <= 0:
        return {song_id: profile_to_vector(profile) for song_id, profile in profiles.items()}

    svd = TruncatedSVD(n_components=components, random_state=random_state)
    dense = svd.fit_transform(matrix)
    dense = normalize(dense)
    explained = float(np.sum(svd.explained_variance_ratio_))
    print(f"  learned SVD dim:      {components}")
    print(f"  feature count:        {len(vocabulary):,}")
    print(f"  explained variance:   {explained:.3f}")

    vectors: dict[str, dict[str, float]] = {}
    for song_id, row in zip(song_ids, dense):
        vectors[song_id] = {
            f"svd:{idx}": float(value)
            for idx, value in enumerate(row)
            if abs(float(value)) > 1e-12
        }
    return vectors


def l2_normalize(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm <= 0:
        return {}
    return {key: value / norm for key, value in vector.items()}


def combine_vectors(parts: list[tuple[dict[str, float], float]]) -> dict[str, float]:
    combined: dict[str, float] = {}
    for vector, weight in parts:
        for key, value in vector.items():
            combined[key] = combined.get(key, 0.0) + weight * value
    return l2_normalize(combined)


def average_vectors(song_ids: list[str], song_vectors: dict[str, dict[str, float]]) -> dict[str, float]:
    vectors = [song_vectors[song_id] for song_id in song_ids if song_id in song_vectors and song_vectors[song_id]]
    if not vectors:
        return {}
    scale = 1.0 / len(vectors)
    return combine_vectors([(vector, scale) for vector in vectors])


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def adaptive_fusion_alpha(long_vector: dict[str, float], short_vector: dict[str, float]) -> float:
    agreement = max(0.0, min(1.0, cosine(long_vector, short_vector)))
    drift = 1.0 - agreement
    return max(0.25, min(0.75, 0.35 + 0.35 * drift))


def build_user_vectors(
    cases: list[rec.EvalCase],
    song_vectors: dict[str, dict[str, float]],
    recent_window: int,
) -> tuple[dict[str, dict[str, float]], list[float]]:
    user_vectors: dict[str, dict[str, float]] = {}
    alphas: list[float] = []
    for case in cases:
        long_vector = average_vectors(case.observed, song_vectors)
        short_vector = average_vectors(case.observed[-recent_window:], song_vectors)
        alpha = adaptive_fusion_alpha(long_vector, short_vector)
        user_vectors[case.playlist_id] = combine_vectors(
            [
                (long_vector, 1.0 - alpha),
                (short_vector, alpha),
            ]
        )
        alphas.append(alpha)
    return user_vectors, alphas


def lemon_scores_for_case(
    prepared: rec.PreparedCase,
    user_vectors: dict[str, dict[str, float]],
    song_vectors: dict[str, dict[str, float]],
) -> dict[str, float]:
    user_vector = user_vectors.get(prepared.playlist_id, {})
    return {
        song_id: cosine(user_vector, song_vectors.get(song_id, {}))
        for song_id in prepared.candidates
    }


def build_lemon_rankings(
    prepared_cases: list[rec.PreparedCase],
    user_vectors: dict[str, dict[str, float]],
    song_vectors: dict[str, dict[str, float]],
    cf_weight: float,
    artist_weight: float,
    pop_weight: float,
    lemon_weight: float,
) -> list[list[str]]:
    rankings: list[list[str]] = []
    for prepared in prepared_cases:
        lemon_norm = rec.normalize_scores(lemon_scores_for_case(prepared, user_vectors, song_vectors), prepared.candidates)
        scored = []
        for song_id in prepared.candidates:
            score = (
                cf_weight * prepared.cf_norm.get(song_id, 0.0)
                + artist_weight * prepared.artist_norm.get(song_id, 0.0)
                + pop_weight * prepared.pop_norm.get(song_id, 0.0)
                + lemon_weight * lemon_norm.get(song_id, 0.0)
            )
            scored.append((score, song_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        rankings.append([song_id for _, song_id in scored])
    return rankings


def random_rankings(cases: list[rec.EvalCase], catalog: set[str], seed: int) -> list[list[str]]:
    rankings = []
    for case in cases:
        available = [song_id for song_id in sorted(catalog) if song_id not in set(case.observed)]
        rng = random.Random(f"{seed}:{case.playlist_id}")
        rng.shuffle(available)
        rankings.append(available)
    return rankings


def popularity_rankings(cases: list[rec.EvalCase], popularity: Counter[str]) -> list[list[str]]:
    top_popular_all = [song_id for song_id, _ in popularity.most_common()]
    return [
        [song_id for song_id in top_popular_all if song_id not in set(case.observed)]
        for case in cases
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LEMON-inspired playlist continuation smoke prototype")
    parser.add_argument("--lyrics-csv", type=Path, default=rec.DEFAULT_LYRICS_CSV)
    parser.add_argument("--playlist-csv", type=Path, default=DEFAULT_PLAYLIST_CSV)
    parser.add_argument("--song-semantic-csv", type=Path, default=DEFAULT_SONG_SEMANTIC_CSV)
    parser.add_argument("--song-semantic-source", choices=["csv", "qwen"], default="csv")
    parser.add_argument("--song-llm-cache", type=Path, default=DEFAULT_SONG_LLM_CACHE)
    parser.add_argument("--song-llm-max-generate", type=int, default=0, help="0 means generate every missing eval song")
    parser.add_argument("--song-llm-lyrics-chars", type=int, default=220)
    parser.add_argument("--song-llm-max-new-tokens", type=int, default=130)
    parser.add_argument("--song-llm-batch-size", type=int, default=1)
    parser.add_argument("--semantic-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--semantic-model-cache-dir", type=Path, default=rec.DEFAULT_LLM_CACHE_DIR)
    parser.add_argument("--semantic-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-playlists", type=int, default=50, help="0 means no cap")
    parser.add_argument("--max-eval-cases", type=int, default=50, help="0 means no cap")
    parser.add_argument("--min-playlist-len", type=int, default=20)
    parser.add_argument("--holdout-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--stage1-recent-window", type=int, default=10)
    parser.add_argument("--lemon-recent-window", type=int, default=8)
    parser.add_argument("--cold-start-threshold", type=int, default=1)
    parser.add_argument("--cf-neighbors", type=int, default=100)
    parser.add_argument("--cf-weight", type=float, default=0.35)
    parser.add_argument("--artist-weight", type=float, default=0.10)
    parser.add_argument("--pop-weight", type=float, default=0.10)
    parser.add_argument("--lemon-weight", type=float, default=0.05)
    parser.add_argument("--embedding-mode", choices=["heuristic", "svd"], default="heuristic")
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=172)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    rec.print_section("LEMON smoke inputs")
    print(f"  lyrics_csv:          {args.lyrics_csv}")
    print(f"  playlist_csv:        {args.playlist_csv}")
    print(f"  song_semantic_csv:   {args.song_semantic_csv}")
    print(f"  song_source:         {args.song_semantic_source}")
    if args.song_semantic_source == "qwen":
        print(f"  song_llm_cache:      {args.song_llm_cache}")
        print(f"  song_llm_limit:      {args.song_llm_max_generate}")
    print(f"  embedding_mode:      {args.embedding_mode}")
    print(f"  max_playlists:       {args.max_playlists}")
    print(f"  max_eval_cases:      {args.max_eval_cases}")

    songs = rec.load_lyrics_csv(args.lyrics_csv)
    playlists = rec.load_playlist_csv(args.playlist_csv, songs, args.max_playlists)
    cases = rec.split_playlists(playlists, args.min_playlist_len, args.holdout_k)
    if args.max_eval_cases and len(cases) > args.max_eval_cases:
        cases = cases[: args.max_eval_cases]
    if not cases:
        raise SystemExit("No evaluation playlists after filtering.")

    eval_song_ids = {song_id for case in cases for song_id in case.observed + case.heldout}
    songs = {song_id: song for song_id, song in songs.items() if song_id in eval_song_ids}
    catalog = set(songs)
    popularity = rec.popularity_counts(cases)

    rec.print_section("LEMON paper-structure stages")
    print("  1. Song-level emotion extraction from title/lyrics labels")
    if args.song_semantic_source == "qwen":
        song_profiles = build_qwen_song_profiles(
            songs=songs,
            cache_path=args.song_llm_cache,
            model_name=args.semantic_model,
            model_cache_dir=args.semantic_model_cache_dir,
            device=args.semantic_device,
            lyrics_chars=args.song_llm_lyrics_chars,
            max_new_tokens=args.song_llm_max_new_tokens,
            max_generate=args.song_llm_max_generate,
            batch_size=args.song_llm_batch_size,
        )
    else:
        song_profiles = build_song_profiles(songs, args.song_semantic_csv)
    print("  2. Emotion-enhanced song vector lookup")
    if args.embedding_mode == "svd":
        song_vectors = build_svd_song_vectors(song_profiles, args.embedding_dim, args.seed)
    else:
        song_vectors = {song_id: profile_to_vector(profile) for song_id, profile in song_profiles.items()}
    nonempty_vectors = sum(1 for vector in song_vectors.values() if vector)
    print(f"  song vectors:        {nonempty_vectors:,}/{len(song_vectors):,}")
    print("  3. Long-term and short-term user emotion encoders")
    user_vectors, alphas = build_user_vectors(cases, song_vectors, args.lemon_recent_window)
    print(f"  user vectors:        {len(user_vectors):,}")
    print("  4. Adaptive fusion coefficient")
    print(f"  alpha avg/min/max:   {sum(alphas) / len(alphas):.3f} / {min(alphas):.3f} / {max(alphas):.3f}")

    rec.print_section("Algorithm input")
    print(f"  eval playlists:      {len(cases):,}")
    print(f"  eval songs:          {len(songs):,}")
    print(f"  heldout per list:    {args.holdout_k}")
    print(f"  candidate pools:     {', '.join(str(pool) for pool in rec.STAGE1_POOL_SIZES)}")
    print(f"  ranking pool:        {rec.STAGE2_POOL_SIZE}")

    rec.print_section("Building co-occurrence CF neighbors")
    cf_neighbors = rec.build_cf_neighbors(cases, args.cf_neighbors)
    print(f"  songs with neighbors:  {len(cf_neighbors):,}")
    print(f"  stored neighbor edges: {sum(len(items) for items in cf_neighbors.values()):,}")
    rec.print_stage1_same_artist_grid(cases, songs, cf_neighbors, popularity, args.stage1_recent_window)

    rows: list[tuple[str, float, float, float]] = []
    rows.append(("Random catalog ordering, no personalization", *rec.evaluate_rankings(random_rankings(cases, catalog, args.seed), cases, args.top_k)))
    rows.append(("Popularity ranking from observed training playlists", *rec.evaluate_rankings(popularity_rankings(cases, popularity), cases, args.top_k)))

    for ratio in rec.STAGE1_RATIOS:
        prepared_cases = rec.prepare_cases(
            cases=cases,
            songs=songs,
            cf_neighbors=cf_neighbors,
            popularity=popularity,
            playlist_semantics={},
            song_semantics={},
            pool_size=rec.STAGE2_POOL_SIZE,
            force_ratio=ratio,
            recent_window=args.stage1_recent_window,
            cold_start_threshold=args.cold_start_threshold,
            progress_interval=rec.PROGRESS_INTERVAL,
        )
        quota = int(ratio * 100)
        cf_pop = rec.build_rankings(prepared_cases, args.cf_weight, 0.0, args.pop_weight)
        rows.append((f"CF + artist{quota}% + popularity", *rec.evaluate_rankings(cf_pop, cases, args.top_k)))

        if ratio == rec.STAGE1_RATIOS[0]:
            lemon_only = build_lemon_rankings(prepared_cases, user_vectors, song_vectors, 0.0, 0.0, 0.0, 1.0)
            lemon_pop = build_lemon_rankings(prepared_cases, user_vectors, song_vectors, 0.0, 0.0, args.pop_weight, 1.0)
            rows.append((f"LEMON emotion-vector only (artist{quota}% candidate pool)", *rec.evaluate_rankings(lemon_only, cases, args.top_k)))
            rows.append((f"LEMON emotion-vector + popularity (artist{quota}% candidate pool)", *rec.evaluate_rankings(lemon_pop, cases, args.top_k)))

        full = rec.build_rankings(prepared_cases, args.cf_weight, args.artist_weight, args.pop_weight)
        rows.append((f"CF + artist{quota}% + artist score + popularity", *rec.evaluate_rankings(full, cases, args.top_k)))
        lemon_full = build_lemon_rankings(
            prepared_cases,
            user_vectors,
            song_vectors,
            args.cf_weight,
            args.artist_weight,
            args.pop_weight,
            args.lemon_weight,
        )
        rows.append((f"CF + artist{quota}% + artist score + popularity + LEMON", *rec.evaluate_rankings(lemon_full, cases, args.top_k)))

    rec.print_stage2_results(rows)
    print(f"\nElapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
