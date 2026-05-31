#!/usr/bin/env python3
"""Playlist continuation prototype without LLM song semantics.

Pipeline:
- load Spotify lyrics catalog and filtered playlist rows
- split each playlist into observed songs and last-K heldout songs
- Stage 1: retrieve candidates with same-artist priority plus CF/popularity
- Stage 2: rank Stage 1 candidates with CF, artist metadata, and popularity

This file intentionally does not call an LLM and does not use generated song
semantic files. Language features are omitted because the current joined catalog
is effectively English-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_FILTERED_DIR = ROOT / "dataFiltered"
DEFAULT_LYRICS_CSV = DEFAULT_DATA_DIR / "spotify_millsongdata.csv"
DEFAULT_FILTERED_PLAYLIST_CSV = DEFAULT_FILTERED_DIR / "spotify_playlist_50percent_50item.csv"
STAGE1_RATIOS = (0.25, 0.50, 0.75, 1.00)
STAGE1_POOL_SIZES = (100, 200, 300, 400, 500)
STAGE2_POOL_SIZE = max(STAGE1_POOL_SIZES)
PROGRESS_INTERVAL = 100

@dataclass(frozen=True)
class Song:
    song_id: str
    title: str
    artist: str
    lyrics: str

@dataclass(frozen=True)
class EvalCase:
    playlist_id: str
    observed: list[str]
    heldout: list[str]


@dataclass(frozen=True)
class PreparedCase:
    playlist_id: str
    heldout: list[str]
    candidates: set[str]
    cf_norm: dict[str, float]
    artist_norm: dict[str, float]
    pop_norm: dict[str, float]
    retrieval: float
    cold_start: bool


def print_section(title: str) -> None:
    print()
    print(f"{title} {'=' * 56}")


def normalize_text(value: str) -> str:
    value = (value or "").lower()
    value = re.sub(r"\([^)]*\)|\[[^]]*]", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def song_key(title: str, artist: str) -> str:
    return f"{normalize_text(artist)}::{normalize_text(title)}"


def find_column(fieldnames: list[str], candidates: list[str]) -> str:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(f"Missing one of columns: {', '.join(candidates)}")


def load_lyrics_csv(path: Path) -> dict[str, Song]:
    songs: dict[str, Song] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        title_col = find_column(reader.fieldnames, ["song", "track_name", "track", "title", "name"])
        artist_col = find_column(reader.fieldnames, ["artist", "artist_name", "artists"])
        lyrics_col = find_column(reader.fieldnames, ["text", "lyrics", "lyric"])

        for row in reader:
            title = (row.get(title_col) or "").strip()
            artist = (row.get(artist_col) or "").strip()
            lyrics = (row.get(lyrics_col) or "").strip()
            if not title or not artist or not lyrics:
                continue
            key = song_key(title, artist)
            if key not in songs:
                songs[key] = Song(key, title, artist, lyrics)
    return songs


def iter_mpd_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for pattern in ("*.json", "*.jsonl"):
        yield from sorted(path.glob(pattern))


def load_mpd_playlists(path: Path, lyrics: dict[str, Song], max_playlists: int | None) -> list[tuple[str, list[str]]]:
    playlists: list[tuple[str, list[str]]] = []
    for json_path in iter_mpd_files(path):
        with json_path.open("r", encoding="utf-8", errors="replace") as f:
            if json_path.suffix.lower() == ".jsonl":
                raw_playlists = (json.loads(line) for line in f if line.strip())
            else:
                data = json.load(f)
                raw_playlists = data.get("playlists", data if isinstance(data, list) else [])

            for playlist in raw_playlists:
                pid = str(playlist.get("pid", playlist.get("name", len(playlists))))
                matched: list[str] = []
                for track in playlist.get("tracks", []):
                    key = song_key(
                        track.get("track_name") or track.get("name") or track.get("title") or "",
                        track.get("artist_name") or track.get("artist") or "",
                    )
                    if key in lyrics:
                        matched.append(key)
                if matched:
                    playlists.append((pid, matched))
                if max_playlists and len(playlists) >= max_playlists:
                    return playlists
    return playlists


def load_playlist_csv(path: Path, lyrics: dict[str, Song], max_playlists: int | None) -> list[tuple[str, list[str]]]:
    buckets: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        pid_col = find_column(reader.fieldnames, ["playlist_id", "pid", "playlist", "user_id"])
        title_col = find_column(reader.fieldnames, ["track_name", "song", "track", "title", "name"])
        artist_col = find_column(reader.fieldnames, ["artist_name", "artist", "artists"])
        pos_col = None
        for option in ["pos", "position", "track_pos", "order", "index"]:
            if option in reader.fieldnames:
                pos_col = option
                break

        for row_idx, row in enumerate(reader):
            key = song_key(row.get(title_col, ""), row.get(artist_col, ""))
            if key not in lyrics:
                continue
            pid = str(row.get(pid_col, ""))
            try:
                pos = int(row.get(pos_col, row_idx)) if pos_col else row_idx
            except ValueError:
                pos = row_idx
            buckets[pid].append((pos, key))

    playlists: list[tuple[str, list[str]]] = []
    for pid, rows in buckets.items():
        rows.sort()
        playlists.append((pid, [key for _, key in rows]))
        if max_playlists and len(playlists) >= max_playlists:
            break
    return playlists


def make_demo_data() -> tuple[dict[str, Song], list[tuple[str, list[str]]]]:
    raw = [
        ("Dancing Queen", "ABBA", "dance night young sweet music party"),
        ("Billie Jean", "Michael Jackson", "dance floor night beat"),
        ("Someone Like You", "Adele", "sad goodbye remember love"),
        ("Hello", "Adele", "hello from outside sorry heart"),
        ("Bad Romance", "Lady Gaga", "love romance dance party"),
        ("Poker Face", "Lady Gaga", "dance club love"),
        ("Let It Be", "The Beatles", "mother wisdom peace"),
        ("Hey Jude", "The Beatles", "sad song better heart"),
    ]
    songs = {song_key(title, artist): Song(song_key(title, artist), title, artist, lyrics) for title, artist, lyrics in raw}
    playlists = [
        ("demo_pop", [song_key("Dancing Queen", "ABBA"), song_key("Billie Jean", "Michael Jackson"), song_key("Bad Romance", "Lady Gaga"), song_key("Poker Face", "Lady Gaga")]),
        ("demo_adele", [song_key("Someone Like You", "Adele"), song_key("Hello", "Adele"), song_key("Hey Jude", "The Beatles")]),
    ]
    return songs, playlists


def split_playlists(playlists: list[tuple[str, list[str]]], min_len: int, holdout_k: int) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for pid, tracks in playlists:
        deduped = list(dict.fromkeys(tracks))
        if len(deduped) < min_len or len(deduped) < 2:
            continue
        case_holdout_k = min(holdout_k, len(deduped) - 1)
        observed, heldout = deduped[:-case_holdout_k], deduped[-case_holdout_k:]
        if observed and heldout:
            cases.append(EvalCase(pid, observed, heldout))
    return cases


def popularity_counts(cases: list[EvalCase]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for case in cases:
        counts.update(case.observed)
    return counts


def build_cf_neighbors(cases: list[EvalCase], max_neighbors: int) -> dict[str, list[tuple[str, float]]]:
    popularity = popularity_counts(cases)
    co_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        observed = list(dict.fromkeys(case.observed))
        for i, left in enumerate(observed):
            for right in observed[i + 1 :]:
                co_counts[left][right] += 1
                co_counts[right][left] += 1

    neighbors: dict[str, list[tuple[str, float]]] = {}
    for song_id, counts in co_counts.items():
        scored = []
        for other_id, count in counts.items():
            denom = math.sqrt(popularity[song_id] * popularity[other_id]) or 1.0
            scored.append((other_id, count / denom))
        scored.sort(key=lambda item: (-item[1], item[0]))
        neighbors[song_id] = scored[:max_neighbors]
    return neighbors


def score_cf(
    observed: list[str],
    cf_neighbors: dict[str, list[tuple[str, float]]],
    exclude: set[str],
    top_k: int,
) -> dict[str, float]:
    scores: Counter[str] = Counter()
    for song_id in observed:
        for candidate, weight in cf_neighbors.get(song_id, []):
            if candidate not in exclude:
                scores[candidate] += weight
    return dict(scores.most_common(top_k))


def ordered_cf_pop_candidates(
    cf_scores: dict[str, float],
    top_popular: list[str],
    popularity: Counter[str],
    pool_size: int,
    exclude: set[str],
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for song_id, _ in sorted(cf_scores.items(), key=lambda item: (-item[1], -popularity.get(item[0], 0), item[0])):
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


def same_artist_candidates(
    observed: list[str],
    songs: dict[str, Song],
    popularity: Counter[str],
    max_candidates: int,
    recent_window: int,
    exclude: set[str],
) -> list[str]:
    if max_candidates <= 0:
        return []
    all_artists = Counter(songs[song_id].artist for song_id in observed if song_id in songs)
    recent_artists = Counter(songs[song_id].artist for song_id in observed[-recent_window:] if song_id in songs)
    if not all_artists:
        return []

    scored: list[tuple[float, str]] = []
    for song_id, song in songs.items():
        if song_id in exclude:
            continue
        score = 0.0
        if song.artist in recent_artists:
            score += 1000.0 * recent_artists[song.artist]
        if song.artist in all_artists:
            score += 200.0 * all_artists[song.artist]
        if score > 0:
            score += 0.001 * popularity.get(song_id, 0)
            scored.append((score, song_id))
    scored.sort(key=lambda item: (-item[0], -popularity.get(item[1], 0), item[1]))
    return [song_id for _, song_id in scored[:max_candidates]]


def merge_forced_stage1_candidates(forced: list[str], baseline_ordered: list[str], pool_size: int) -> set[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for song_id in forced + baseline_ordered:
        if song_id not in seen:
            ordered.append(song_id)
            seen.add(song_id)
        if len(ordered) >= pool_size:
            break
    return set(ordered)


def normalize_scores(scores: dict[str, float], candidates: set[str]) -> dict[str, float]:
    if not scores or not candidates:
        return {}
    values = [scores.get(song_id, 0.0) for song_id in candidates]
    high = max(values)
    low = min(values)
    if high <= low:
        return {song_id: 1.0 if scores.get(song_id, 0.0) > 0 else 0.0 for song_id in candidates}
    return {song_id: (scores.get(song_id, 0.0) - low) / (high - low) for song_id in candidates}


def score_artist_metadata(
    observed: list[str],
    candidates: set[str],
    songs: dict[str, Song],
) -> dict[str, float]:
    observed_artists = Counter(songs[song_id].artist for song_id in observed if song_id in songs)
    artist_scores: dict[str, float] = {}
    for candidate in candidates:
        song = songs.get(candidate)
        artist_scores[candidate] = float(observed_artists.get(song.artist, 0)) if song else 0.0
    return artist_scores


def recall_at_k(ranked: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    return len(set(ranked[:k]) & truth) / len(truth)


def ndcg_at_k(ranked: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    dcg = 0.0
    for idx, song_id in enumerate(ranked[:k], start=1):
        if song_id in truth:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_rankings(rankings: list[list[str]], cases: list[EvalCase], top_k: int) -> tuple[float, float, float]:
    recalls = []
    ndcgs = []
    for ranked, case in zip(rankings, cases):
        truth = set(case.heldout)
        recalls.append(recall_at_k(ranked, truth, top_k))
        ndcgs.append(ndcg_at_k(ranked, truth, top_k))
    recall = sum(recalls) / len(recalls) if recalls else 0.0
    ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    return recall, ndcg, (recall + ndcg) / 2.0


def print_metadata_tables() -> None:
    print_section("Song metadata titles")
    song_fields = [
        "song_id",
        "title",
        "artist",
        "lyrics",
        "popularity",
        "cf_neighbors",
    ]
    print_columns(song_fields, width=28, columns=3)

    print_section("Playlist metadata titles")
    playlist_fields = [
        "playlist_id",
        "playlist_name",
        "original_track_count",
        "matched_song_count",
        "matched_coverage_percent",
        "pos",
        "track_name",
        "artist_name",
    ]
    print_columns(playlist_fields, width=28, columns=3)


def print_columns(values: list[str], width: int, columns: int) -> None:
    for start in range(0, len(values), columns):
        row = values[start : start + columns]
        print("".join(f"{value:<{width}}" for value in row).rstrip())


def print_stage1_same_artist_grid(
    cases: list[EvalCase],
    songs: dict[str, Song],
    cf_neighbors: dict[str, list[tuple[str, float]]],
    popularity: Counter[str],
    recent_window: int,
) -> None:
    print_section("Stage 1 same-artist candidate recall")
    pool_sizes = list(STAGE1_POOL_SIZES)
    max_pool = max(pool_sizes)
    top_popular = [song_id for song_id, _ in popularity.most_common(max_pool)]
    ratio_methods = [f"CF + artist{int(ratio * 100)}% + popularity" for ratio in STAGE1_RATIOS]
    method_names = ratio_methods + ["Popularity", "CF"]
    recalls_by_method: dict[str, dict[int, list[float]]] = {
        method: {pool_size: [] for pool_size in pool_sizes}
        for method in method_names
    }

    for case in cases:
        exclude = set(case.observed)
        cf_scores = score_cf(case.observed, cf_neighbors, exclude, max_pool)
        baseline_ordered = ordered_cf_pop_candidates(cf_scores, top_popular, popularity, max_pool, exclude)
        popularity_ordered = [song_id for song_id in top_popular if song_id not in exclude]
        cf_ordered = [
            song_id
            for song_id, _ in sorted(cf_scores.items(), key=lambda item: (-item[1], -popularity.get(item[0], 0), item[0]))
            if song_id not in exclude
        ]
        for pool_size in pool_sizes:
            truth = set(case.heldout)
            for ratio, method_name in zip(STAGE1_RATIOS, ratio_methods):
                forced_limit = max(0, min(pool_size, int(round(pool_size * ratio))))
                forced = same_artist_candidates(case.observed, songs, popularity, forced_limit, recent_window, exclude)
                candidates = merge_forced_stage1_candidates(forced, baseline_ordered[:pool_size], pool_size)
                recalls_by_method[method_name][pool_size].append(
                    recall_at_k(list(candidates), truth, len(candidates))
                )
            recalls_by_method["Popularity"][pool_size].append(
                recall_at_k(popularity_ordered[:pool_size], truth, pool_size)
            )
            recalls_by_method["CF"][pool_size].append(
                recall_at_k(cf_ordered[:pool_size], truth, pool_size)
            )

    header = f"{'Candidate method':<48} " + " ".join(f"R@{pool:<3}" for pool in pool_sizes)
    print(header)
    print("-" * len(header))
    for method in method_names:
        values = [
            sum(recalls_by_method[method][pool]) / len(recalls_by_method[method][pool])
            if recalls_by_method[method][pool]
            else 0.0
            for pool in pool_sizes
        ]
        print(f"{method:<48} " + " ".join(f"{value:>5.3f}" for value in values))


def prepare_cases(
    cases: list[EvalCase],
    songs: dict[str, Song],
    cf_neighbors: dict[str, list[tuple[str, float]]],
    popularity: Counter[str],
    pool_size: int,
    force_ratio: float,
    recent_window: int,
    cold_start_threshold: int,
    progress_interval: int,
) -> list[PreparedCase]:
    print_section(f"Preparing candidate scores with same-artist quota {int(force_ratio * 100)}%")
    print("  candidate source: same artist + CF + popularity")
    top_popular = [song_id for song_id, _ in popularity.most_common(pool_size)]
    prepared: list[PreparedCase] = []
    cold = 0
    candidate_total = 0
    start = time.time()

    for idx, case in enumerate(cases, start=1):
        exclude = set(case.observed)
        cf_scores = score_cf(case.observed, cf_neighbors, exclude, pool_size)
        baseline_ordered = ordered_cf_pop_candidates(cf_scores, top_popular, popularity, pool_size, exclude)
        forced_limit = max(0, min(pool_size, int(round(pool_size * force_ratio))))
        forced = same_artist_candidates(case.observed, songs, popularity, forced_limit, recent_window, exclude)
        candidates = merge_forced_stage1_candidates(forced, baseline_ordered, pool_size)
        is_cold = len(case.observed) <= cold_start_threshold or not cf_scores
        if is_cold:
            cold += 1
        artist_scores = score_artist_metadata(case.observed, candidates, songs)
        pop_scores = {song_id: float(popularity.get(song_id, 0)) for song_id in candidates}
        candidate_total += len(candidates)
        prepared.append(
            PreparedCase(
                playlist_id=case.playlist_id,
                heldout=case.heldout,
                candidates=candidates,
                cf_norm=normalize_scores(cf_scores, candidates),
                artist_norm=normalize_scores(artist_scores, candidates),
                pop_norm=normalize_scores(pop_scores, candidates),
                retrieval=recall_at_k(list(candidates), set(case.heldout), len(candidates)),
                cold_start=is_cold,
            )
        )
        should_print = idx == 1 or idx == len(cases) or (progress_interval > 0 and idx % progress_interval == 0)
        if should_print:
            avg_candidates = candidate_total / idx
            elapsed = time.time() - start
            print(f"  prepared {idx:,}/{len(cases):,} cases | cold={cold:,} | avg_candidates={avg_candidates:.1f} | elapsed={elapsed:.1f}s")

    print(f"  cold-start popularity cases: {cold:,}")
    return prepared


def build_rankings(
    prepared_cases: list[PreparedCase],
    cf_weight: float,
    artist_weight: float,
    pop_weight: float,
) -> list[list[str]]:
    rankings: list[list[str]] = []
    for case in prepared_cases:
        scored: list[tuple[float, str]] = []
        for song_id in case.candidates:
            score = (
                cf_weight * case.cf_norm.get(song_id, 0.0)
                + artist_weight * case.artist_norm.get(song_id, 0.0)
                + pop_weight * case.pop_norm.get(song_id, 0.0)
            )
            scored.append((score, song_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        rankings.append([song_id for _, song_id in scored])
    return rankings


def print_stage2_results(rows: list[tuple[str, float, float, float]]) -> None:
    print_section("Ranking result")
    print(f"{'Model / full description':<86} {'Recall@10':>10} {'NDCG@10':>10} {'Proxy@10':>10}")
    print("-" * 122)
    for label, recall, ndcg, proxy in rows:
        print(f"{label:<86} {recall:>10.5f} {ndcg:>10.5f} {proxy:>10.5f}")


def format_song(song_id: str, songs: dict[str, Song]) -> str:
    song = songs.get(song_id)
    if not song:
        return song_id
    return f"{song.title} / {song.artist}"


def print_example(prepared_cases: list[PreparedCase], cases: list[EvalCase], songs: dict[str, Song], rankings: list[list[str]]) -> None:
    if not prepared_cases or not cases or not rankings:
        return
    case = cases[0]
    print_section("Example recommendation")
    print(f"  playlist_id: {case.playlist_id}")
    print("  input songs:")
    for song_id in case.observed[-5:]:
        print(f"    - {format_song(song_id, songs)}")
    print("  hidden heldout songs:")
    for song_id in case.heldout[:5]:
        print(f"    - {format_song(song_id, songs)}")
    print("  recommended songs:")
    for song_id in rankings[0][:5]:
        print(f"    - {format_song(song_id, songs)}")


def default_mpd_path() -> Path | None:
    if list(DEFAULT_DATA_DIR.glob("mpd.slice.*.json")):
        return DEFAULT_DATA_DIR
    playlist_dir = DEFAULT_DATA_DIR / "playlists"
    if playlist_dir.exists() and list(playlist_dir.glob("mpd.slice.*.json")):
        return playlist_dir
    return None


def default_playlist_csv() -> Path | None:
    if DEFAULT_FILTERED_PLAYLIST_CSV.exists():
        return DEFAULT_FILTERED_PLAYLIST_CSV
    for name in ("playlists.csv", "playlist.csv"):
        path = DEFAULT_DATA_DIR / name
        if path.exists():
            return path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CF + metadata playlist continuation prototype")
    parser.add_argument("--lyrics-csv", type=Path, help="Spotify Million Song lyrics CSV")
    parser.add_argument("--mpd-path", type=Path, help="MPD JSON file/directory with playlists")
    parser.add_argument("--playlist-csv", type=Path, help="Filtered playlist CSV")
    parser.add_argument("--demo", action="store_true", help="Force the tiny built-in demo")
    parser.add_argument("--max-playlists", type=int, default=0, help="Cap loaded playlists; 0 means no cap")
    parser.add_argument("--max-eval-cases", type=int, default=1000, help="Cap eval playlists after splitting; 0 means no cap")
    parser.add_argument("--min-playlist-len", type=int, default=20)
    parser.add_argument("--holdout-k", type=int, default=10, help="Use up to the last K songs as heldout truth")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--stage1-recent-window", type=int, default=10, help="Recent observed songs used to prioritize same-artist candidates")
    parser.add_argument("--cold-start-threshold", type=int, default=1)
    parser.add_argument("--cf-neighbors", type=int, default=100)
    parser.add_argument("--cf-weight", type=float, default=0.35)
    parser.add_argument("--artist-weight", type=float, default=0.10)
    parser.add_argument("--pop-weight", type=float, default=0.10)
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--show-example", action="store_true", help="Print one playlist example after metric tables")
    parser.add_argument("--seed", type=int, default=172)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    print_metadata_tables()

    if args.demo:
        print("Running built-in demo data.")
        songs, playlists = make_demo_data()
        args.holdout_k = min(args.holdout_k, 1)
        args.min_playlist_len = min(args.min_playlist_len, 3)
    else:
        if not args.lyrics_csv and DEFAULT_LYRICS_CSV.exists():
            args.lyrics_csv = DEFAULT_LYRICS_CSV
        if not args.mpd_path and not args.playlist_csv:
            args.playlist_csv = default_playlist_csv()
            if not args.playlist_csv:
                args.mpd_path = default_mpd_path()
        if not args.lyrics_csv:
            print("No lyrics CSV supplied; running built-in demo data.")
            songs, playlists = make_demo_data()
        else:
            songs = load_lyrics_csv(args.lyrics_csv)
            if args.playlist_csv:
                playlists = load_playlist_csv(args.playlist_csv, songs, args.max_playlists)
            elif args.mpd_path:
                playlists = load_mpd_playlists(args.mpd_path, songs, args.max_playlists)
            else:
                playlists = []

    if not playlists:
        print()
        print("No playlist data found. Run playlistFilter.py or pass --playlist-csv/--mpd-path.")
        return

    cases = split_playlists(playlists, args.min_playlist_len, args.holdout_k)
    if not cases:
        raise SystemExit("No evaluation playlists after filtering. Lower --min-playlist-len/--holdout-k or check joined data.")
    if args.max_eval_cases and len(cases) > args.max_eval_cases:
        print(f"Capping eval playlists from {len(cases):,} to {args.max_eval_cases:,}.")
        cases = cases[: args.max_eval_cases]

    eval_song_ids = {song_id for case in cases for song_id in case.observed + case.heldout}
    songs = {song_id: song for song_id, song in songs.items() if song_id in eval_song_ids}
    catalog = set(songs)
    popularity = popularity_counts(cases)

    print_section("Algorithm input")
    if args.playlist_csv:
        print(f"  playlist_csv:       {args.playlist_csv}")
    elif args.mpd_path:
        print(f"  mpd_path:           {args.mpd_path}")
    print(f"  eval playlists:     {len(cases):,}")
    print(f"  eval songs:         {len(songs):,}")
    print(f"  heldout per list:   {args.holdout_k}")
    print(f"  candidate pools:    {', '.join(str(pool) for pool in STAGE1_POOL_SIZES)}")
    print(f"  ranking pool:       {STAGE2_POOL_SIZE}")

    print_section("Building co-occurrence CF neighbors")
    cf_neighbors = build_cf_neighbors(cases, args.cf_neighbors)
    neighbor_edges = sum(len(items) for items in cf_neighbors.values())
    print(f"  songs with neighbors:  {len(cf_neighbors):,}")
    print(f"  stored neighbor edges: {neighbor_edges:,}")

    print_stage1_same_artist_grid(
        cases=cases,
        songs=songs,
        cf_neighbors=cf_neighbors,
        popularity=popularity,
        recent_window=args.stage1_recent_window,
    )

    random_rankings: list[list[str]] = []
    for case in cases:
        available = [song_id for song_id in sorted(catalog) if song_id not in set(case.observed)]
        rng = random.Random(f"{args.seed}:{case.playlist_id}")
        rng.shuffle(available)
        random_rankings.append(available)

    top_popular_all = [song_id for song_id, _ in popularity.most_common()]
    popularity_rankings = [
        [song_id for song_id in top_popular_all if song_id not in set(case.observed)]
        for case in cases
    ]

    rows: list[tuple[str, float, float, float]] = []
    rows.append(("Random catalog ordering, no personalization", *evaluate_rankings(random_rankings, cases, args.top_k)))
    rows.append(("Popularity ranking from observed training playlists", *evaluate_rankings(popularity_rankings, cases, args.top_k)))

    best_proxy = -1.0
    best_prepared_cases: list[PreparedCase] = []
    best_full_rankings: list[list[str]] = []

    for ratio in STAGE1_RATIOS:
        prepared_cases = prepare_cases(
            cases=cases,
            songs=songs,
            cf_neighbors=cf_neighbors,
            popularity=popularity,
            pool_size=STAGE2_POOL_SIZE,
            force_ratio=ratio,
            recent_window=args.stage1_recent_window,
            cold_start_threshold=args.cold_start_threshold,
            progress_interval=PROGRESS_INTERVAL,
        )
        quota = int(ratio * 100)
        cf_pop = build_rankings(prepared_cases, args.cf_weight, 0.0, args.pop_weight)
        rows.append((f"CF + artist{quota}% + popularity", *evaluate_rankings(cf_pop, cases, args.top_k)))

        full_rankings = build_rankings(
            prepared_cases,
            args.cf_weight,
            args.artist_weight,
            args.pop_weight,
        )
        full_metrics = evaluate_rankings(full_rankings, cases, args.top_k)
        rows.append((f"CF + artist{quota}% + artist score + popularity", *full_metrics))
        if full_metrics[2] > best_proxy:
            best_proxy = full_metrics[2]
            best_prepared_cases = prepared_cases
            best_full_rankings = full_rankings

    print_stage2_results(rows)
    if args.show_example:
        print_example(best_prepared_cases, cases, songs, best_full_rankings)


if __name__ == "__main__":
    main()
