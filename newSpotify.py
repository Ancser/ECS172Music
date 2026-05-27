#!/usr/bin/env python3
"""Lyrics TF-IDF playlist-continuation prototype.

This is the first implementation slice for the ECS172 music project:
- no LLM calls
- lyrics are represented with TF-IDF/content-IDF vectors
- each playlist uses earlier songs as input and the last 10 songs as heldout truth
- recommendations are evaluated with Recall@10 and NDCG@10

The script runs a tiny built-in demo when no dataset paths are supplied.
For real data, pass an MPD JSON directory and a Spotify lyrics CSV.
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


TOKEN_RE = re.compile(r"[a-z][a-z']+")
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_LYRICS_CSV = DEFAULT_DATA_DIR / "spotify_millsongdata.csv"


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
    long_norm: dict[str, float]
    short_norm: dict[str, float]
    pop_norm: dict[str, float]
    retrieval: float
    cold_start: bool


def normalize_text(value: str) -> str:
    value = (value or "").lower()
    value = re.sub(r"\([^)]*\)|\[[^]]*]", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def song_key(title: str, artist: str) -> str:
    return f"{normalize_text(artist)}::{normalize_text(title)}"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def find_column(fieldnames: list[str], options: list[str]) -> str:
    normalized = {normalize_text(name): name for name in fieldnames}
    for option in options:
        key = normalize_text(option)
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not find any of columns {options}; available={fieldnames}")


def load_lyrics_csv(path: Path) -> dict[str, Song]:
    """Load Kaggle-style lyrics CSV and return song_key -> Song."""
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
    """Load Spotify MPD JSON slices and keep only tracks matched to lyrics."""
    playlists: list[tuple[str, list[str]]] = []
    for json_path in iter_mpd_files(path):
        with json_path.open("r", encoding="utf-8", errors="replace") as f:
            if json_path.suffix.lower() == ".jsonl":
                payloads = (json.loads(line) for line in f if line.strip())
                raw_playlists = payloads
            else:
                data = json.load(f)
                raw_playlists = data.get("playlists", data if isinstance(data, list) else [])

            for playlist in raw_playlists:
                pid = str(playlist.get("pid", playlist.get("name", len(playlists))))
                matched: list[str] = []
                for track in playlist.get("tracks", []):
                    title = track.get("track_name") or track.get("name") or track.get("title") or ""
                    artist = track.get("artist_name") or track.get("artist") or ""
                    key = song_key(title, artist)
                    if key in lyrics:
                        matched.append(key)
                if matched:
                    playlists.append((pid, matched))
                if max_playlists and len(playlists) >= max_playlists:
                    return playlists
    return playlists


def load_playlist_csv(path: Path, lyrics: dict[str, Song], max_playlists: int | None) -> list[tuple[str, list[str]]]:
    """Load a simple playlist CSV with one track per row."""
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
            try:
                pos_col = find_column(reader.fieldnames, [option])
                break
            except ValueError:
                pass

        for row_idx, row in enumerate(reader):
            key = song_key(row.get(title_col, ""), row.get(artist_col, ""))
            if key not in lyrics:
                continue
            pid = str(row.get(pid_col, ""))
            pos = row_idx
            if pos_col:
                try:
                    pos = int(row.get(pos_col, row_idx))
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
    rows = [
        ("sunrise drive", "Nova", "bright morning golden road dancing light hope higher sky"),
        ("city sparks", "Nova", "neon city fast heart electric fire dancing night"),
        ("afterglow", "Nova", "warm light hope home heartbeat glowing sky"),
        ("quiet rain", "Mira", "soft rain window lonely midnight calm blue memory"),
        ("old letters", "Mira", "paper memory regret lonely soft voice rain"),
        ("empty station", "Mira", "midnight train goodbye grief lonely cold platform"),
        ("wild pulse", "Kite", "drums thunder running fire energy jump crowd"),
        ("break the room", "Kite", "loud guitar fire rebel scream energy night"),
        ("open field", "Kite", "running sunlight wind freedom drums bright"),
        ("slow harbor", "Lune", "calm ocean breathing moon soft sleep harbor"),
        ("moon water", "Lune", "moon river calm reflection sleep gentle night"),
        ("blue lantern", "Lune", "gentle blue lantern quiet memory moon"),
    ]
    songs = {song_key(t, a): Song(song_key(t, a), t, a, lyrics) for t, a, lyrics in rows}
    k = lambda title, artist: song_key(title, artist)
    playlists = [
        ("p1", [k("sunrise drive", "Nova"), k("afterglow", "Nova"), k("open field", "Kite"), k("city sparks", "Nova"), k("wild pulse", "Kite")]),
        ("p2", [k("quiet rain", "Mira"), k("old letters", "Mira"), k("blue lantern", "Lune"), k("moon water", "Lune"), k("empty station", "Mira")]),
        ("p3", [k("wild pulse", "Kite"), k("break the room", "Kite"), k("city sparks", "Nova"), k("open field", "Kite"), k("sunrise drive", "Nova")]),
        ("p4", [k("slow harbor", "Lune"), k("moon water", "Lune"), k("blue lantern", "Lune"), k("quiet rain", "Mira"), k("old letters", "Mira")]),
        ("p5", [k("afterglow", "Nova"), k("sunrise drive", "Nova"), k("slow harbor", "Lune"), k("moon water", "Lune"), k("blue lantern", "Lune")]),
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


class TfidfIndex:
    def __init__(self, songs: dict[str, Song], min_df: int, max_features: int):
        self.songs = songs
        self.min_df = min_df
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: dict[int, float] = {}
        self.item_vectors: dict[str, dict[int, float]] = {}
        self.inverted: dict[int, list[tuple[str, float]]] = defaultdict(list)
        self._fit()

    def _fit(self) -> None:
        tokenized: dict[str, list[str]] = {}
        df: Counter[str] = Counter()
        for song_id, song in self.songs.items():
            tokens = tokenize(song.lyrics)
            tokenized[song_id] = tokens
            df.update(set(tokens))

        terms = [(term, count) for term, count in df.items() if count >= self.min_df]
        terms.sort(key=lambda item: (-item[1], item[0]))
        if self.max_features > 0:
            terms = terms[: self.max_features]
        self.vocab = {term: idx for idx, (term, _) in enumerate(terms)}
        n_docs = max(1, len(tokenized))
        self.idf = {
            idx: math.log((1.0 + n_docs) / (1.0 + df[term])) + 1.0
            for term, idx in self.vocab.items()
        }

        for song_id, tokens in tokenized.items():
            counts = Counter(self.vocab[t] for t in tokens if t in self.vocab)
            if not counts:
                continue
            vec = {idx: (1.0 + math.log(tf)) * self.idf[idx] for idx, tf in counts.items()}
            norm = math.sqrt(sum(weight * weight for weight in vec.values())) or 1.0
            vec = {idx: weight / norm for idx, weight in vec.items()}
            self.item_vectors[song_id] = vec
            for idx, weight in vec.items():
                self.inverted[idx].append((song_id, weight))

    def profile(self, song_ids: list[str], decay: float = 1.0) -> dict[int, float]:
        weights: Counter[int] = Counter()
        total_weight = 0.0
        n_songs = len(song_ids)
        for pos, song_id in enumerate(song_ids):
            vec = self.item_vectors.get(song_id)
            if not vec:
                continue
            sequence_weight = decay ** (n_songs - pos - 1) if decay < 1.0 else 1.0
            total_weight += sequence_weight
            for idx, weight in vec.items():
                weights[idx] += weight * sequence_weight
        if not total_weight:
            return {}
        raw = {idx: weight / total_weight for idx, weight in weights.items()}
        norm = math.sqrt(sum(weight * weight for weight in raw.values())) or 1.0
        return {idx: weight / norm for idx, weight in raw.items()}

    def score_profile(self, profile: dict[int, float], exclude: set[str], top_k: int) -> dict[str, float]:
        scores: Counter[str] = Counter()
        for idx, profile_weight in profile.items():
            for song_id, item_weight in self.inverted.get(idx, []):
                if song_id not in exclude:
                    scores[song_id] += profile_weight * item_weight
        if top_k <= 0:
            return dict(scores)
        return dict(scores.most_common(top_k))


def normalize_scores(scores: dict[str, float], candidates: set[str]) -> dict[str, float]:
    if not candidates:
        return {}
    values = [scores.get(candidate, 0.0) for candidate in candidates]
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return {candidate: 0.0 for candidate in candidates}
    return {candidate: (scores.get(candidate, 0.0) - low) / (high - low) for candidate in candidates}


def recall_at_k(ranked: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    hits = len(set(ranked[:k]) & truth)
    return hits / len(truth)


def ndcg_at_k(ranked: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    dcg = 0.0
    for rank, song_id in enumerate(ranked[:k], start=1):
        if song_id in truth:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(truth), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_rankings(rankings: dict[str, list[str]], cases: list[EvalCase], k: int) -> tuple[float, float]:
    recalls, ndcgs = [], []
    for case in cases:
        ranked = rankings.get(case.playlist_id, [])
        truth = set(case.heldout)
        recalls.append(recall_at_k(ranked, truth, k))
        ndcgs.append(ndcg_at_k(ranked, truth, k))
    return sum(recalls) / len(recalls), sum(ndcgs) / len(ndcgs)


def print_results_table(rows: list[dict[str, float | str]]) -> None:
    print()
    print("Results table")
    print("| Model | Retrieval | Recall | NDCG | Proxy |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        retrieval = row["retrieval"]
        retrieval_text = "-" if retrieval == "" else f"{float(retrieval):.5f}"
        recall = float(row["recall"])
        ndcg = float(row["ndcg"])
        proxy = float(row["proxy"])
        print(f"| {row['model']} | {retrieval_text} | {recall:.5f} | {ndcg:.5f} | {proxy:.5f} |")


def popularity_counts(cases: list[EvalCase]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for case in cases:
        counts.update(case.observed)
    return counts


def rank_popularity(case: EvalCase, popularity: Counter[str], catalog: set[str], k: int) -> list[str]:
    exclude = set(case.observed)
    candidates = [song_id for song_id in catalog if song_id not in exclude]
    candidates.sort(key=lambda song_id: (-popularity.get(song_id, 0), song_id))
    return candidates[:k]


def rank_random(case: EvalCase, catalog: set[str], k: int, rng: random.Random) -> list[str]:
    candidates = sorted(song_id for song_id in catalog if song_id not in set(case.observed))
    rng.shuffle(candidates)
    return candidates[:k]


def prepare_content_cases(
    cases: list[EvalCase],
    index: TfidfIndex,
    popularity: Counter[str],
    pool_size: int,
    short_window: int,
    time_decay: float,
    cold_start_threshold: int,
    progress_interval: int,
) -> list[PreparedCase]:
    prepared: list[PreparedCase] = []
    top_popular = [song_id for song_id, _ in popularity.most_common(pool_size)]
    started = time.time()
    cold_count = 0

    print("Preparing candidate scores once before alpha sweep...")
    for idx, case in enumerate(cases, start=1):
        exclude = set(case.observed)
        cold_start = len(case.observed) <= cold_start_threshold

        if cold_start:
            cold_count += 1
            candidates = {song_id for song_id in top_popular if song_id not in exclude}
            long_norm: dict[str, float] = {}
            short_norm: dict[str, float] = {}
        else:
            long_profile = index.profile(case.observed, decay=time_decay)
            short_profile = index.profile(case.observed[-short_window:], decay=time_decay)
            long_scores = index.score_profile(long_profile, exclude, pool_size)
            short_scores = index.score_profile(short_profile, exclude, pool_size)
            candidates = set(long_scores) | set(short_scores)
            candidates.update(song_id for song_id in top_popular if song_id not in exclude)
            long_norm = normalize_scores(long_scores, candidates)
            short_norm = normalize_scores(short_scores, candidates)

        pop_raw = {song_id: float(popularity.get(song_id, 0)) for song_id in candidates}
        pop_norm = normalize_scores(pop_raw, candidates)
        retrieval = recall_at_k(list(candidates), set(case.heldout), len(candidates))
        prepared.append(
            PreparedCase(
                playlist_id=case.playlist_id,
                heldout=case.heldout,
                candidates=candidates,
                long_norm=long_norm,
                short_norm=short_norm,
                pop_norm=pop_norm,
                retrieval=retrieval,
                cold_start=cold_start,
            )
        )

        if progress_interval and (idx == 1 or idx % progress_interval == 0 or idx == len(cases)):
            elapsed = time.time() - started
            avg_candidates = sum(len(item.candidates) for item in prepared) / len(prepared)
            print(
                f"  prepared {idx:,}/{len(cases):,} cases | "
                f"cold={cold_count:,} | avg_candidates={avg_candidates:.1f} | elapsed={elapsed:.1f}s",
                flush=True,
            )

    return prepared


def build_content_rankings(
    prepared_cases: list[PreparedCase],
    k: int,
    alpha: float,
    pop_weight: float,
) -> tuple[dict[str, list[str]], float]:
    rankings: dict[str, list[str]] = {}
    retrieval_recalls: list[float] = []

    for case in prepared_cases:
        retrieval_recalls.append(case.retrieval)
        final_scores = {
            song_id: alpha * case.long_norm.get(song_id, 0.0)
            + (1.0 - alpha) * case.short_norm.get(song_id, 0.0)
            + pop_weight * case.pop_norm.get(song_id, 0.0)
            for song_id in case.candidates
        }
        ranked = sorted(final_scores, key=lambda song_id: (-final_scores[song_id], song_id))
        rankings[case.playlist_id] = ranked[:k]

    retrieval = sum(retrieval_recalls) / len(retrieval_recalls) if retrieval_recalls else 0.0
    return rankings, retrieval


def print_dataset_study(songs: dict[str, Song], playlists: list[tuple[str, list[str]]], cases: list[EvalCase]) -> None:
    lengths = [len(tracks) for _, tracks in playlists]
    matched_tracks = sum(lengths)
    users = len(playlists)
    catalog = len(songs)
    density = matched_tracks / (users * catalog) if users and catalog else 0.0
    observed_lengths = [len(case.observed) for case in cases]
    heldout_lengths = [len(case.heldout) for case in cases]
    print("Dataset study")
    print(f"  songs with lyrics:       {len(songs):,}")
    print(f"  matched playlists:       {len(playlists):,}")
    print(f"  eval playlists:          {len(cases):,}")
    print(f"  matched interactions:    {matched_tracks:,}")
    print(f"  matrix density:          {density:.6f}")
    if observed_lengths:
        sorted_lengths = sorted(observed_lengths)
        median = sorted_lengths[len(sorted_lengths) // 2]
        cold = sum(1 for n in observed_lengths if n <= 3)
        print(f"  avg observed length:     {sum(observed_lengths) / len(observed_lengths):.2f}")
        print(f"  median observed length:  {median}")
        sorted_heldout = sorted(heldout_lengths)
        print(f"  avg heldout length:      {sum(heldout_lengths) / len(heldout_lengths):.2f}")
        print(f"  median heldout length:   {sorted_heldout[len(sorted_heldout) // 2]}")
        print(f"  min/max heldout length:  {min(heldout_lengths)} / {max(heldout_lengths)}")
        print(f"  cold playlists <= 3:     {cold:,}")


def preview_text(value: str, limit: int = 90) -> str:
    compact = re.sub(r"\s+", " ", value or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def print_song_head(songs: dict[str, Song], limit: int = 10) -> None:
    print("Head 10 songs")
    for idx, song in enumerate(list(songs.values())[:limit], start=1):
        print(
            f"  {idx:>2}. artist={song.artist} | song={song.title} | "
            f"lyrics={preview_text(song.lyrics)}"
        )


def print_playlist_head(
    playlists: list[tuple[str, list[str]]],
    songs: dict[str, Song],
    limit: int = 10,
    track_limit: int = 10,
) -> None:
    print("Head 10 playlists")
    if not playlists:
        print("  No playlist data loaded yet.")
        return
    for idx, (playlist_id, track_ids) in enumerate(playlists[:limit], start=1):
        names = []
        for song_id in track_ids[:track_limit]:
            song = songs.get(song_id)
            names.append(f"{song.title} / {song.artist}" if song else song_id)
        suffix = " ..." if len(track_ids) > track_limit else ""
        print(f"  {idx:>2}. playlist={playlist_id} | tracks={len(track_ids)} | {', '.join(names)}{suffix}")


def print_all_data_stats(songs: dict[str, Song], playlists: list[tuple[str, list[str]]]) -> None:
    playlist_lengths = [len(track_ids) for _, track_ids in playlists]
    unique_playlist_tracks = {song_id for _, track_ids in playlists for song_id in track_ids}
    lyric_token_counts = [len(tokenize(song.lyrics)) for song in songs.values()]
    print("All loaded data stats")
    print(f"  song lyric rows:         {len(songs):,}")
    print(f"  playlists loaded:        {len(playlists):,}")
    print(f"  matched playlist tracks: {sum(playlist_lengths):,}")
    print(f"  unique matched tracks:   {len(unique_playlist_tracks):,}")
    if playlist_lengths:
        sorted_lengths = sorted(playlist_lengths)
        print(f"  avg playlist length:     {sum(playlist_lengths) / len(playlist_lengths):.2f}")
        print(f"  median playlist length:  {sorted_lengths[len(sorted_lengths) // 2]}")
        print(f"  min playlist length:     {min(playlist_lengths)}")
        print(f"  max playlist length:     {max(playlist_lengths)}")
    if lyric_token_counts:
        sorted_tokens = sorted(lyric_token_counts)
        print(f"  avg lyric tokens:        {sum(lyric_token_counts) / len(lyric_token_counts):.2f}")
        print(f"  median lyric tokens:     {sorted_tokens[len(sorted_tokens) // 2]}")
        print(f"  min lyric tokens:        {min(lyric_token_counts)}")
        print(f"  max lyric tokens:        {max(lyric_token_counts)}")


def print_example_recs(cases: list[EvalCase], rankings: dict[str, list[str]], songs: dict[str, Song], limit: int) -> None:
    if not cases:
        return
    case = cases[0]
    print()
    print(f"Example recommendations for playlist {case.playlist_id}")
    print("  input:")
    for song_id in case.observed[:limit]:
        song = songs[song_id]
        print(f"    - {song.title} / {song.artist}")
    print("  heldout:")
    for song_id in case.heldout[:limit]:
        song = songs[song_id]
        print(f"    - {song.title} / {song.artist}")
    print("  recommended:")
    for song_id in rankings.get(case.playlist_id, [])[:limit]:
        song = songs[song_id]
        print(f"    - {song.title} / {song.artist}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lyrics TF-IDF playlist continuation prototype")
    parser.add_argument("--lyrics-csv", type=Path, help="Spotify Million Song lyrics CSV")
    parser.add_argument("--mpd-path", type=Path, help="MPD JSON file/directory with playlists")
    parser.add_argument("--playlist-csv", type=Path, help="Alternative simple playlist CSV")
    parser.add_argument("--demo", action="store_true", help="Force the tiny built-in demo instead of repo data")
    parser.add_argument("--max-playlists", type=int, default=1000)
    parser.add_argument("--max-eval-cases", type=int, default=1000, help="Cap validation playlists after splitting; 0 means no cap")
    parser.add_argument("--min-playlist-len", type=int, default=2)
    parser.add_argument("--holdout-k", type=int, default=10, help="Use up to the last K songs as heldout truth")
    parser.add_argument("--pool-size", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--short-window", type=int, default=5)
    parser.add_argument("--time-decay", type=float, default=0.9, help="Older observed songs get decay^(distance from newest)")
    parser.add_argument("--cold-start-threshold", type=int, default=1, help="Use popularity-only ranking when observed history has this many songs or fewer")
    parser.add_argument("--progress-interval", type=int, default=1000, help="Print candidate-prep progress every N eval playlists")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--pop-weight", type=float, default=0.05)
    parser.add_argument("--alpha-grid-step", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=172)
    return parser.parse_args()


def default_mpd_path() -> Path | None:
    if list(DEFAULT_DATA_DIR.glob("mpd.slice.*.json")):
        return DEFAULT_DATA_DIR
    playlist_dir = DEFAULT_DATA_DIR / "playlists"
    if playlist_dir.exists() and list(playlist_dir.glob("mpd.slice.*.json")):
        return playlist_dir
    return None


def default_playlist_csv() -> Path | None:
    for name in ("playlists.csv", "playlist.csv"):
        path = DEFAULT_DATA_DIR / name
        if path.exists():
            return path
    return None


def main() -> None:
    args = parse_args()
    if args.demo:
        print("Running built-in demo data.")
        songs, playlists = make_demo_data()
        args.holdout_k = min(args.holdout_k, 1)
        args.min_playlist_len = min(args.min_playlist_len, 5)
        args.min_df = 1
        args.pool_size = min(args.pool_size, 20)
    else:
        if not args.lyrics_csv and DEFAULT_LYRICS_CSV.exists():
            args.lyrics_csv = DEFAULT_LYRICS_CSV
        if not args.mpd_path and not args.playlist_csv:
            args.mpd_path = default_mpd_path()
            if not args.mpd_path:
                args.playlist_csv = default_playlist_csv()

    if not args.demo and args.lyrics_csv:
        songs = load_lyrics_csv(args.lyrics_csv)
        if args.mpd_path:
            playlists = load_mpd_playlists(args.mpd_path, songs, args.max_playlists)
        elif args.playlist_csv:
            playlists = load_playlist_csv(args.playlist_csv, songs, args.max_playlists)
        else:
            playlists = []
    elif not args.demo:
        print("No dataset paths supplied; running built-in demo data.")
        songs, playlists = make_demo_data()
        args.min_playlist_len = min(args.min_playlist_len, 5)
        args.min_df = 1
        args.pool_size = min(args.pool_size, 20)

    print()
    print_song_head(songs, limit=10)
    print()
    print_playlist_head(playlists, songs, limit=10)
    print()
    print_all_data_stats(songs, playlists)

    if not playlists:
        print()
        print("No playlist data found under data/. Put MPD mpd.slice.*.json files in data/ or pass --mpd-path.")
        return

    cases = split_playlists(playlists, args.min_playlist_len, args.holdout_k)
    if not cases:
        raise SystemExit("No evaluation playlists after lyrics join/filtering. Lower --min-playlist-len/--holdout-k or check join columns.")
    if args.max_eval_cases and len(cases) > args.max_eval_cases:
        print(f"Capping eval playlists from {len(cases):,} to {args.max_eval_cases:,}.")
        cases = cases[: args.max_eval_cases]

    eval_song_ids = {song_id for case in cases for song_id in case.observed + case.heldout}
    songs = {song_id: song for song_id, song in songs.items() if song_id in eval_song_ids}
    catalog = set(songs)
    popularity = popularity_counts(cases)

    print_dataset_study(songs, playlists, cases)
    print()
    print("Building lyrics TF-IDF/content-IDF index...")
    index = TfidfIndex(songs, args.min_df, args.max_features)
    print(f"  vocab size:              {len(index.vocab):,}")
    print(f"  indexed songs:           {len(index.item_vectors):,}")

    prepared_cases = prepare_content_cases(
        cases=cases,
        index=index,
        popularity=popularity,
        pool_size=args.pool_size,
        short_window=args.short_window,
        time_decay=args.time_decay,
        cold_start_threshold=args.cold_start_threshold,
        progress_interval=args.progress_interval,
    )
    cold_prepared = sum(1 for case in prepared_cases if case.cold_start)
    print(f"  cold-start popularity cases: {cold_prepared:,}")

    rng = random.Random(args.seed)
    random_rankings = {case.playlist_id: rank_random(case, catalog, args.top_k, rng) for case in cases}
    pop_rankings = {case.playlist_id: rank_popularity(case, popularity, catalog, args.top_k) for case in cases}
    random_recall, random_ndcg = evaluate_rankings(random_rankings, cases, args.top_k)
    pop_recall, pop_ndcg = evaluate_rankings(pop_rankings, cases, args.top_k)

    print()
    print("Baselines")
    print(f"  random       Recall@{args.top_k}: {random_recall:.5f}  NDCG@{args.top_k}: {random_ndcg:.5f}")
    print(f"  popularity   Recall@{args.top_k}: {pop_recall:.5f}  NDCG@{args.top_k}: {pop_ndcg:.5f}")
    result_rows: list[dict[str, float | str]] = [
        {
            "model": "Random",
            "retrieval": "",
            "recall": random_recall,
            "ndcg": random_ndcg,
            "proxy": (random_recall + random_ndcg) / 2.0,
        },
        {
            "model": "Popularity",
            "retrieval": "",
            "recall": pop_recall,
            "ndcg": pop_ndcg,
            "proxy": (pop_recall + pop_ndcg) / 2.0,
        },
    ]

    alphas: list[float] = []
    value = 0.0
    while value <= 1.000001:
        alphas.append(round(value, 10))
        value += args.alpha_grid_step
    if 1.0 not in alphas:
        alphas.append(1.0)

    best = None
    print()
    print("Lyrics TF-IDF fusion sweep")
    for alpha in alphas:
        rankings, retrieval = build_content_rankings(
            k=args.top_k,
            prepared_cases=prepared_cases,
            alpha=alpha,
            pop_weight=args.pop_weight,
        )
        recall, ndcg = evaluate_rankings(rankings, cases, args.top_k)
        proxy = (recall + ndcg) / 2.0
        print(
            f"  alpha={alpha:>4.2f}  Retrieval@{args.pool_size}: {retrieval:.5f}  "
            f"Recall@{args.top_k}: {recall:.5f}  NDCG@{args.top_k}: {ndcg:.5f}  Proxy: {proxy:.5f}"
        )
        result_rows.append(
            {
                "model": f"TF-IDF alpha={alpha:.2f}",
                "retrieval": retrieval,
                "recall": recall,
                "ndcg": ndcg,
                "proxy": proxy,
            }
        )
        candidate = (proxy, recall, ndcg, retrieval, alpha, rankings)
        if best is None or candidate[:4] > best[:4]:
            best = candidate

    assert best is not None
    proxy, recall, ndcg, retrieval, alpha, rankings = best
    print()
    print("Best prototype result")
    print(f"  method:                 lyrics_tfidf_long_short_fusion")
    print(f"  alpha long-term:        {alpha:.2f}")
    print(f"  alpha short-term:       {1.0 - alpha:.2f}")
    print(f"  time decay:             {args.time_decay:.2f}")
    print(f"  popularity tie weight:  {args.pop_weight:.2f}")
    print(f"  Retrieval@{args.pool_size}:          {retrieval:.5f}")
    print(f"  Recall@{args.top_k}:             {recall:.5f}")
    print(f"  NDCG@{args.top_k}:               {ndcg:.5f}")
    print(f"  Proxy:                 {proxy:.5f}")
    result_rows.append(
        {
            "model": f"Best TF-IDF alpha={alpha:.2f}",
            "retrieval": retrieval,
            "recall": recall,
            "ndcg": ndcg,
            "proxy": proxy,
        }
    )
    print_results_table(result_rows)
    print_example_recs(cases, rankings, songs, limit=5)


if __name__ == "__main__":
    main()
