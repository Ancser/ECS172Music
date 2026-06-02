#!/usr/bin/env python3
"""Compare CF Stage 1 against playlist/song semantic Stage 1 retrieval."""

from __future__ import annotations

import argparse
import csv
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import recommandation as rec


ROOT = Path(__file__).resolve().parent
DEFAULT_PLAYLIST_CACHE = ROOT / "dataFiltered" / "playlist_semantics_fine_keywords_800_qwen.jsonl"
DEFAULT_SONG_CACHE = ROOT / "dataFiltered" / "song_semantics_fine_keywords_800_heuristic.jsonl"
DEFAULT_ID_OUT = ROOT / "dataFiltered" / "eval_playlist_ids_50_50_800_799.csv"
POOL_SIZE = rec.STAGE2_POOL_SIZE


def load_eval_cases(args: argparse.Namespace) -> tuple[dict[str, rec.Song], list[rec.EvalCase]]:
    songs = rec.load_lyrics_csv(args.lyrics_csv)
    playlists = rec.load_playlist_csv(args.playlist_csv, songs, args.max_playlists)
    cases = rec.split_playlists(playlists, args.min_playlist_len, args.holdout_k)
    if args.max_eval_cases and len(cases) > args.max_eval_cases:
        cases = cases[: args.max_eval_cases]
    eval_song_ids = {song_id for case in cases for song_id in case.observed + case.heldout}
    songs = {song_id: song for song_id, song in songs.items() if song_id in eval_song_ids}
    return songs, cases


def write_playlist_ids(path: Path, cases: list[rec.EvalCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_index", "playlist_id", "observed_count", "heldout_count"])
        for idx, case in enumerate(cases, start=1):
            writer.writerow([idx, case.playlist_id, len(case.observed), len(case.heldout)])


def load_playlist_semantics(path: Path, cases: list[rec.EvalCase]) -> dict[str, dict[str, object]]:
    cache = rec.load_semantic_cache(path)
    semantics = {}
    missing = []
    for case in cases:
        key = rec.semantic_cache_key(case.playlist_id, case.observed)
        profile = cache.get(key)
        if profile is None:
            missing.append(case.playlist_id)
        else:
            semantics[case.playlist_id] = rec.normalize_semantic_profile(profile)
    if missing:
        raise SystemExit(f"Missing playlist semantics for {len(missing)} cases. First missing pid: {missing[0]}")
    return semantics


def load_song_semantics(path: Path, songs: dict[str, rec.Song]) -> dict[str, dict[str, object]]:
    cache = rec.load_semantic_cache(path)
    semantics = {}
    missing = []
    for song_id in songs:
        key = rec.song_semantic_cache_key(song_id)
        profile = cache.get(key)
        if profile is None:
            missing.append(song_id)
        else:
            semantics[song_id] = rec.normalize_semantic_profile(profile)
    if missing:
        raise SystemExit(f"Missing song semantics for {len(missing)} songs. First missing song_id: {missing[0]}")
    return semantics


def order_from_scores(
    scores: dict[str, float],
    top_popular: list[str],
    popularity: dict[str, int],
    exclude: set[str],
    pool_size: int,
) -> list[str]:
    ordered = []
    seen = set()
    for song_id, score in sorted(scores.items(), key=lambda item: (-item[1], -popularity.get(item[0], 0), item[0])):
        if score <= 0:
            break
        if song_id not in exclude and song_id not in seen:
            ordered.append(song_id)
            seen.add(song_id)
        if len(ordered) >= pool_size:
            return ordered
    for song_id in top_popular:
        if song_id not in exclude and song_id not in seen:
            ordered.append(song_id)
            seen.add(song_id)
        if len(ordered) >= pool_size:
            break
    return ordered


def build_token_index(normalized_song_text: dict[str, str]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for song_id, text in normalized_song_text.items():
        for token in set(re.findall(r"\b[a-z0-9]{3,}\b", text)):
            index[token].add(song_id)
    return dict(index)


def direct_playlist_text_scores_indexed(
    playlist_profile: dict[str, object],
    candidates: set[str],
    token_index: dict[str, set[str]],
) -> dict[str, float]:
    terms = rec.semantic_terms(playlist_profile)
    scores: Counter[str] = Counter()
    for term in terms:
        normalized = rec.normalize_text(term)
        if not normalized:
            continue
        tokens = [token for token in re.findall(r"\b[a-z0-9]{3,}\b", normalized) if len(token) >= 3]
        if not tokens:
            continue
        weight = 2.0 if len(tokens) > 1 else 1.0
        for token in tokens:
            for song_id in token_index.get(token, set()) & candidates:
                scores[song_id] += weight / len(tokens)
    return dict(scores)


def semantic_profile_scores(
    playlist_profile: dict[str, object],
    candidates: set[str],
    song_semantics: dict[str, dict[str, object]],
) -> dict[str, float]:
    return {
        song_id: rec.semantic_profile_similarity(playlist_profile, song_semantics.get(song_id))
        for song_id in candidates
    }


def rank_with_metadata_cf(
    candidates: set[str],
    cf_scores: dict[str, float],
    case: rec.EvalCase,
    songs: dict[str, rec.Song],
    popularity: dict[str, int],
    args: argparse.Namespace,
) -> list[str]:
    artist_scores = rec.score_artist_metadata(case.observed, candidates, songs)
    pop_scores = {song_id: float(popularity.get(song_id, 0)) for song_id in candidates}
    cf_norm = rec.normalize_scores(cf_scores, candidates)
    artist_norm = rec.normalize_scores(artist_scores, candidates)
    pop_norm = rec.normalize_scores(pop_scores, candidates)
    scored = []
    for song_id in candidates:
        score = (
            args.cf_weight * cf_norm.get(song_id, 0.0)
            + args.artist_weight * artist_norm.get(song_id, 0.0)
            + args.pop_weight * pop_norm.get(song_id, 0.0)
        )
        scored.append((score, song_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [song_id for _, song_id in scored]


def rank_semantic_as_cf(
    candidates: set[str],
    semantic_scores: dict[str, float],
    case: rec.EvalCase,
    songs: dict[str, rec.Song],
    popularity: dict[str, int],
    args: argparse.Namespace,
) -> list[str]:
    artist_scores = rec.score_artist_metadata(case.observed, candidates, songs)
    pop_scores = {song_id: float(popularity.get(song_id, 0)) for song_id in candidates}
    semantic_norm = rec.normalize_scores(semantic_scores, candidates)
    artist_norm = rec.normalize_scores(artist_scores, candidates)
    pop_norm = rec.normalize_scores(pop_scores, candidates)
    scored = []
    for song_id in candidates:
        score = (
            args.cf_weight * semantic_norm.get(song_id, 0.0)
            + args.artist_weight * artist_norm.get(song_id, 0.0)
            + args.pop_weight * pop_norm.get(song_id, 0.0)
        )
        scored.append((score, song_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [song_id for _, song_id in scored]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare semantic Stage 1 candidate retrieval")
    parser.add_argument("--lyrics-csv", type=Path, default=rec.DEFAULT_LYRICS_CSV)
    parser.add_argument("--playlist-csv", type=Path, default=rec.DEFAULT_FILTERED_PLAYLIST_CSV)
    parser.add_argument("--playlist-semantic-cache", type=Path, default=DEFAULT_PLAYLIST_CACHE)
    parser.add_argument("--song-semantic-cache", type=Path, default=DEFAULT_SONG_CACHE)
    parser.add_argument("--playlist-id-output", type=Path, default=DEFAULT_ID_OUT)
    parser.add_argument("--max-playlists", type=int, default=800)
    parser.add_argument("--max-eval-cases", type=int, default=800)
    parser.add_argument("--min-playlist-len", type=int, default=20)
    parser.add_argument("--holdout-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cf-neighbors", type=int, default=100)
    parser.add_argument("--cf-weight", type=float, default=0.35)
    parser.add_argument("--artist-weight", type=float, default=0.10)
    parser.add_argument("--pop-weight", type=float, default=0.10)
    args = parser.parse_args()

    started = time.time()
    songs, cases = load_eval_cases(args)
    write_playlist_ids(args.playlist_id_output, cases)
    playlist_semantics = load_playlist_semantics(args.playlist_semantic_cache, cases)
    song_semantics = load_song_semantics(args.song_semantic_cache, songs)
    popularity = rec.popularity_counts(cases)
    cf_neighbors = rec.build_cf_neighbors(cases, args.cf_neighbors)
    top_popular = [song_id for song_id, _ in popularity.most_common(POOL_SIZE)]
    catalog = set(songs)
    normalized_song_text = {
        song_id: rec.normalize_text(f"{song.title} {song.artist} {song.lyrics}")
        for song_id, song in songs.items()
    }
    token_index = build_token_index(normalized_song_text)

    stage1_recalls = {
        "Original CF Stage1": [],
        "Direct playlist semantic Stage1": [],
        "Playlist-song semantic Stage1": [],
    }
    rankings = {
        "Original CF Stage1 + metadata rerank": [],
        "Direct playlist semantic Stage1 + metadata rerank": [],
        "Playlist-song semantic Stage1 + metadata rerank": [],
    }

    for idx, case in enumerate(cases, start=1):
        exclude = set(case.observed)
        available = catalog - exclude
        truth = set(case.heldout)
        playlist_profile = playlist_semantics[case.playlist_id]

        cf_scores = rec.score_cf(case.observed, cf_neighbors, exclude, POOL_SIZE)
        cf_ordered = rec.ordered_cf_pop_candidates(cf_scores, top_popular, popularity, POOL_SIZE, exclude)
        cf_candidates = set(cf_ordered)
        stage1_recalls["Original CF Stage1"].append(rec.recall_at_k(cf_ordered, truth, POOL_SIZE))
        rankings["Original CF Stage1 + metadata rerank"].append(
            rank_with_metadata_cf(cf_candidates, cf_scores, case, songs, popularity, args)
        )

        direct_scores = direct_playlist_text_scores_indexed(playlist_profile, available, token_index)
        direct_ordered = order_from_scores(direct_scores, top_popular, popularity, exclude, POOL_SIZE)
        direct_candidates = set(direct_ordered)
        stage1_recalls["Direct playlist semantic Stage1"].append(rec.recall_at_k(direct_ordered, truth, POOL_SIZE))
        rankings["Direct playlist semantic Stage1 + metadata rerank"].append(
            rank_semantic_as_cf(direct_candidates, direct_scores, case, songs, popularity, args)
        )

        profile_scores = semantic_profile_scores(playlist_profile, available, song_semantics)
        profile_ordered = order_from_scores(profile_scores, top_popular, popularity, exclude, POOL_SIZE)
        profile_candidates = set(profile_ordered)
        stage1_recalls["Playlist-song semantic Stage1"].append(rec.recall_at_k(profile_ordered, truth, POOL_SIZE))
        rankings["Playlist-song semantic Stage1 + metadata rerank"].append(
            rank_semantic_as_cf(profile_candidates, profile_scores, case, songs, popularity, args)
        )

        if idx == 1 or idx == len(cases) or idx % rec.PROGRESS_INTERVAL == 0:
            print(f"processed {idx:,}/{len(cases):,} cases | elapsed={time.time() - started:.1f}s", flush=True)

    print()
    print("Semantic Stage 1 experiment")
    print(f"playlist_csv:       {args.playlist_csv}")
    print(f"eval playlists:     {len(cases):,}")
    print(f"eval songs:         {len(songs):,}")
    print(f"playlist ids csv:   {args.playlist_id_output}")
    print()
    print("Stage 1 candidate recall @500")
    for label, values in stage1_recalls.items():
        print(f"{label:<42} {sum(values) / len(values):.5f}")
    print()
    print(f"{'Model':<58} {'Recall@10':>10} {'NDCG@10':>10} {'Proxy@10':>10}")
    print("-" * 92)
    for label, ranked in rankings.items():
        recall, ndcg, proxy = rec.evaluate_rankings(ranked, cases, args.top_k)
        print(f"{label:<58} {recall:>10.5f} {ndcg:>10.5f} {proxy:>10.5f}")


if __name__ == "__main__":
    main()
