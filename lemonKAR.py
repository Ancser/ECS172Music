#!/usr/bin/env python3
"""LEMON + KAR-style smoke prototype.

KAR in the paper uses LLM-generated factual/preference knowledge, encodes it,
adapts it, and feeds it into a recommender. This file builds a lightweight local
version:

1. LEMON emotion vectors from song semantic profiles.
2. KAR factual item text for each candidate song.
3. KAR preference-reasoning text for each playlist.
4. TF-IDF + TruncatedSVD encoder to put item/user knowledge into one vector space.
5. Compare baseline vs LEMON-only vs LEMON+KAR.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import lemon
import recommandation as rec


ROOT = Path(__file__).resolve().parent
DEFAULT_PLAYLIST_CSV = ROOT / "dataFiltered" / "playlist_50%_50c_799.csv"
DEFAULT_SONG_SEMANTIC_CSV = ROOT / "dataFiltered" / "song_semantics_fine_keywords_800_heuristic.csv"


def profile_terms(profile: dict[str, object]) -> list[str]:
    terms: list[str] = []
    for key in rec.SEMANTIC_JSON_KEYS:
        values = profile.get(key)
        values = values if isinstance(values, list) else [values]
        for value in values:
            text = str(value or "").replace("_", " ").strip()
            if text and text != "mixed":
                terms.append(f"{key} {text}")
                terms.append(text)
    return terms


def item_knowledge_text(
    song_id: str,
    songs: dict[str, rec.Song],
    song_profiles: dict[str, dict[str, object]],
    lyrics_chars: int,
) -> str:
    song = songs[song_id]
    profile = song_profiles.get(song_id, {})
    lyrics_excerpt = " ".join(song.lyrics.split())[:lyrics_chars]
    terms = " ".join(profile_terms(profile))
    return (
        f"item factual knowledge. title {song.title}. artist {song.artist}. "
        f"semantic tags {terms}. lyric evidence {lyrics_excerpt}"
    )


def playlist_reasoning_text(
    case: rec.EvalCase,
    songs: dict[str, rec.Song],
    song_profiles: dict[str, dict[str, object]],
    recent_window: int,
) -> str:
    observed = [song_id for song_id in case.observed if song_id in songs]
    recent = observed[-recent_window:] if recent_window > 0 else observed
    artist_counts = Counter(songs[song_id].artist for song_id in observed)
    term_counts: Counter[str] = Counter()
    recent_term_counts: Counter[str] = Counter()
    for song_id in observed:
        term_counts.update(profile_terms(song_profiles.get(song_id, {})))
    for song_id in recent:
        recent_term_counts.update(profile_terms(song_profiles.get(song_id, {})))
    observed_names = "; ".join(f"{songs[song_id].title} by {songs[song_id].artist}" for song_id in recent[:12])
    top_artists = " ".join(artist for artist, _ in artist_counts.most_common(5))
    top_terms = " ".join(term for term, _ in term_counts.most_common(18))
    recent_terms = " ".join(term for term, _ in recent_term_counts.most_common(12))
    return (
        "preference reasoning knowledge. "
        f"recent observed songs {observed_names}. "
        f"frequent artists {top_artists}. "
        f"long term semantic preferences {top_terms}. "
        f"short term semantic intent {recent_terms}."
    )


def build_kar_vectors(
    cases: list[rec.EvalCase],
    songs: dict[str, rec.Song],
    song_profiles: dict[str, dict[str, object]],
    dimensions: int,
    lyrics_chars: int,
    recent_window: int,
    max_features: int,
    random_state: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float, int]:
    song_ids = sorted(songs)
    playlist_ids = [case.playlist_id for case in cases]
    item_texts = [
        item_knowledge_text(song_id, songs, song_profiles, lyrics_chars)
        for song_id in song_ids
    ]
    user_texts = [
        playlist_reasoning_text(case, songs, song_profiles, recent_window)
        for case in cases
    ]
    texts = item_texts + user_texts
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]+\b",
    )
    tfidf = vectorizer.fit_transform(texts)
    components = max(1, min(dimensions, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=components, random_state=random_state)
    dense = normalize(svd.fit_transform(tfidf))
    explained = float(np.sum(svd.explained_variance_ratio_))
    item_dense = dense[: len(song_ids)]
    user_dense = dense[len(song_ids) :]
    item_vectors = {song_id: item_dense[idx] for idx, song_id in enumerate(song_ids)}
    user_vectors = {pid: user_dense[idx] for idx, pid in enumerate(playlist_ids)}
    return item_vectors, user_vectors, explained, len(vectorizer.vocabulary_)


def vector_score(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    return float(np.dot(left, right))


def kar_scores_for_case(
    prepared: rec.PreparedCase,
    kar_user_vectors: dict[str, np.ndarray],
    kar_item_vectors: dict[str, np.ndarray],
) -> dict[str, float]:
    user_vector = kar_user_vectors.get(prepared.playlist_id)
    return {
        song_id: vector_score(user_vector, kar_item_vectors.get(song_id))
        for song_id in prepared.candidates
    }


def combined_scores_for_case(
    prepared: rec.PreparedCase,
    lemon_user_vectors: dict[str, dict[str, float]],
    lemon_song_vectors: dict[str, dict[str, float]],
    kar_user_vectors: dict[str, np.ndarray],
    kar_item_vectors: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float]]:
    lemon_norm = rec.normalize_scores(
        lemon.lemon_scores_for_case(prepared, lemon_user_vectors, lemon_song_vectors),
        prepared.candidates,
    )
    kar_norm = rec.normalize_scores(
        kar_scores_for_case(prepared, kar_user_vectors, kar_item_vectors),
        prepared.candidates,
    )
    return lemon_norm, kar_norm


def rank_with_features(
    prepared_cases: list[rec.PreparedCase],
    lemon_user_vectors: dict[str, dict[str, float]],
    lemon_song_vectors: dict[str, dict[str, float]],
    kar_user_vectors: dict[str, np.ndarray],
    kar_item_vectors: dict[str, np.ndarray],
    cf_weight: float,
    artist_weight: float,
    pop_weight: float,
    lemon_weight: float,
    kar_weight: float,
) -> list[list[str]]:
    rankings: list[list[str]] = []
    for prepared in prepared_cases:
        lemon_norm, kar_norm = combined_scores_for_case(
            prepared,
            lemon_user_vectors,
            lemon_song_vectors,
            kar_user_vectors,
            kar_item_vectors,
        )
        scored: list[tuple[float, str]] = []
        for song_id in prepared.candidates:
            score = (
                cf_weight * prepared.cf_norm.get(song_id, 0.0)
                + artist_weight * prepared.artist_norm.get(song_id, 0.0)
                + pop_weight * prepared.pop_norm.get(song_id, 0.0)
                + lemon_weight * lemon_norm.get(song_id, 0.0)
                + kar_weight * kar_norm.get(song_id, 0.0)
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
    parser = argparse.ArgumentParser(description="LEMON + KAR-style smoke prototype")
    parser.add_argument("--lyrics-csv", type=Path, default=rec.DEFAULT_LYRICS_CSV)
    parser.add_argument("--playlist-csv", type=Path, default=DEFAULT_PLAYLIST_CSV)
    parser.add_argument("--song-semantic-csv", type=Path, default=DEFAULT_SONG_SEMANTIC_CSV)
    parser.add_argument("--max-playlists", type=int, default=50, help="0 means no cap")
    parser.add_argument("--max-eval-cases", type=int, default=50, help="0 means no cap")
    parser.add_argument("--min-playlist-len", type=int, default=20)
    parser.add_argument("--holdout-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--stage1-force-ratio", type=float, default=0.25)
    parser.add_argument("--stage1-recent-window", type=int, default=10)
    parser.add_argument("--lemon-recent-window", type=int, default=8)
    parser.add_argument("--kar-recent-window", type=int, default=12)
    parser.add_argument("--cold-start-threshold", type=int, default=1)
    parser.add_argument("--cf-neighbors", type=int, default=100)
    parser.add_argument("--cf-weight", type=float, default=0.35)
    parser.add_argument("--artist-weight", type=float, default=0.10)
    parser.add_argument("--pop-weight", type=float, default=0.10)
    parser.add_argument("--lemon-weight", type=float, default=0.01)
    parser.add_argument("--kar-weight", type=float, default=0.01)
    parser.add_argument("--kar-dim", type=int, default=32)
    parser.add_argument("--kar-max-features", type=int, default=4000)
    parser.add_argument("--kar-lyrics-chars", type=int, default=260)
    parser.add_argument("--seed", type=int, default=172)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    rec.print_section("LEMON + KAR smoke inputs")
    print(f"  lyrics_csv:        {args.lyrics_csv}")
    print(f"  playlist_csv:      {args.playlist_csv}")
    print(f"  song_semantic_csv: {args.song_semantic_csv}")
    print(f"  max_playlists:     {args.max_playlists}")
    print(f"  max_eval_cases:    {args.max_eval_cases}")

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

    rec.print_section("Semantic feature preparation")
    song_profiles = lemon.build_song_profiles(songs, args.song_semantic_csv)
    lemon_song_vectors = {song_id: lemon.profile_to_vector(profile) for song_id, profile in song_profiles.items()}
    lemon_user_vectors, alphas = lemon.build_user_vectors(cases, lemon_song_vectors, args.lemon_recent_window)
    print(f"  LEMON vectors:      {sum(1 for v in lemon_song_vectors.values() if v):,}/{len(lemon_song_vectors):,}")
    print(f"  LEMON alpha avg:    {sum(alphas) / len(alphas):.3f}")

    rec.print_section("KAR-style knowledge encoding")
    kar_item_vectors, kar_user_vectors, explained, vocab_size = build_kar_vectors(
        cases=cases,
        songs=songs,
        song_profiles=song_profiles,
        dimensions=args.kar_dim,
        lyrics_chars=args.kar_lyrics_chars,
        recent_window=args.kar_recent_window,
        max_features=args.kar_max_features,
        random_state=args.seed,
    )
    print(f"  KAR encoder:        TF-IDF + TruncatedSVD")
    print(f"  KAR dim:            {min(args.kar_dim, len(next(iter(kar_item_vectors.values()), []))):,}")
    print(f"  KAR vocabulary:     {vocab_size:,}")
    print(f"  KAR explained var:  {explained:.3f}")

    rec.print_section("Algorithm input")
    print(f"  eval playlists:     {len(cases):,}")
    print(f"  eval songs:         {len(songs):,}")
    print(f"  heldout per list:   {args.holdout_k}")
    print(f"  force ratio:        {args.stage1_force_ratio:.2f}")

    rec.print_section("Building co-occurrence CF neighbors")
    cf_neighbors = rec.build_cf_neighbors(cases, args.cf_neighbors)
    print(f"  songs with neighbors:  {len(cf_neighbors):,}")
    print(f"  stored neighbor edges: {sum(len(items) for items in cf_neighbors.values()):,}")
    rec.print_stage1_same_artist_grid(cases, songs, cf_neighbors, popularity, args.stage1_recent_window)

    prepared_cases = rec.prepare_cases(
        cases=cases,
        songs=songs,
        cf_neighbors=cf_neighbors,
        popularity=popularity,
        playlist_semantics={},
        song_semantics={},
        pool_size=rec.STAGE2_POOL_SIZE,
        force_ratio=args.stage1_force_ratio,
        recent_window=args.stage1_recent_window,
        cold_start_threshold=args.cold_start_threshold,
        progress_interval=rec.PROGRESS_INTERVAL,
    )

    rows: list[tuple[str, float, float, float]] = []
    rows.append(("Random catalog ordering, no personalization", *rec.evaluate_rankings(random_rankings(cases, catalog, args.seed), cases, args.top_k)))
    rows.append(("Popularity ranking from observed training playlists", *rec.evaluate_rankings(popularity_rankings(cases, popularity), cases, args.top_k)))

    baseline = rec.build_rankings(prepared_cases, args.cf_weight, args.artist_weight, args.pop_weight)
    lemon_only = rank_with_features(prepared_cases, lemon_user_vectors, lemon_song_vectors, kar_user_vectors, kar_item_vectors, 0.0, 0.0, 0.0, 1.0, 0.0)
    kar_only = rank_with_features(prepared_cases, lemon_user_vectors, lemon_song_vectors, kar_user_vectors, kar_item_vectors, 0.0, 0.0, 0.0, 0.0, 1.0)
    lemon_kar = rank_with_features(prepared_cases, lemon_user_vectors, lemon_song_vectors, kar_user_vectors, kar_item_vectors, 0.0, 0.0, 0.0, 1.0, 1.0)
    baseline_lemon = rank_with_features(
        prepared_cases,
        lemon_user_vectors,
        lemon_song_vectors,
        kar_user_vectors,
        kar_item_vectors,
        args.cf_weight,
        args.artist_weight,
        args.pop_weight,
        args.lemon_weight,
        0.0,
    )
    baseline_kar = rank_with_features(
        prepared_cases,
        lemon_user_vectors,
        lemon_song_vectors,
        kar_user_vectors,
        kar_item_vectors,
        args.cf_weight,
        args.artist_weight,
        args.pop_weight,
        0.0,
        args.kar_weight,
    )
    baseline_lemon_kar = rank_with_features(
        prepared_cases,
        lemon_user_vectors,
        lemon_song_vectors,
        kar_user_vectors,
        kar_item_vectors,
        args.cf_weight,
        args.artist_weight,
        args.pop_weight,
        args.lemon_weight,
        args.kar_weight,
    )

    rows.append(("Baseline CF + artist score + popularity", *rec.evaluate_rankings(baseline, cases, args.top_k)))
    rows.append(("LEMON only", *rec.evaluate_rankings(lemon_only, cases, args.top_k)))
    rows.append(("KAR only", *rec.evaluate_rankings(kar_only, cases, args.top_k)))
    rows.append(("LEMON + KAR only", *rec.evaluate_rankings(lemon_kar, cases, args.top_k)))
    rows.append(("Baseline + LEMON", *rec.evaluate_rankings(baseline_lemon, cases, args.top_k)))
    rows.append(("Baseline + KAR", *rec.evaluate_rankings(baseline_kar, cases, args.top_k)))
    rows.append(("Baseline + LEMON + KAR", *rec.evaluate_rankings(baseline_lemon_kar, cases, args.top_k)))

    rec.print_stage2_results(rows)
    print(f"\nElapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
