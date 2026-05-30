#!/usr/bin/env python3
"""Scan MPD playlist coverage against the local lyrics catalog."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_LYRICS_CSV = DEFAULT_DATA_DIR / "spotify_millsongdata.csv"

MATCHED_COUNT_THRESHOLD_LABELS = (
    ">= 1 song",
    ">= 5 songs",
    ">= 10 songs",
    ">= 20 songs",
    ">= 30 songs",
    ">= 50 songs",
    ">= 100 songs",
)

COVERAGE_THRESHOLD_LABELS = (
    ">= 10% coverage",
    ">= 20% coverage",
    ">= 30% coverage",
    ">= 50% coverage",
    ">= 70% coverage",
    ">= 90% coverage",
    "100% coverage",
)


def section(title: str) -> None:
    print()
    print(title)
    print("=" * 20)


def normalize_text(value: str) -> str:
    value = (value or "").lower()
    value = re.sub(r"\([^)]*\)|\[[^]]*]", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def song_key(title: str, artist: str) -> str:
    return f"{normalize_text(artist)}::{normalize_text(title)}"


def find_column(fieldnames: list[str], options: list[str]) -> str:
    normalized = {normalize_text(name): name for name in fieldnames}
    for option in options:
        key = normalize_text(option)
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not find any of columns {options}; available={fieldnames}")


def percent_text(numerator: int, denominator: int) -> str:
    pct = (100.0 * numerator / denominator) if denominator else 0.0
    return f"({numerator:,}/{denominator:,}) {pct:.2f}%"


def iter_mpd_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    seen: set[Path] = set()
    for pattern in ("mpd.slice.*.json", "*.json", "*.jsonl"):
        for item in sorted(path.glob(pattern), key=lambda p: natural_key(p.name)):
            if item not in seen:
                seen.add(item)
                yield item


def natural_key(value: str) -> list[int | str]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts]


def load_lyrics_keys(path: Path) -> tuple[set[str], dict[str, int]]:
    keys: set[str] = set()
    stats = {
        "rows": 0,
        "has_title": 0,
        "has_artist": 0,
        "has_lyrics": 0,
        "has_title_artist": 0,
        "usable_rows": 0,
        "duplicate_usable_keys": 0,
    }
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        title_col = find_column(reader.fieldnames, ["song", "track_name", "track", "title", "name"])
        artist_col = find_column(reader.fieldnames, ["artist", "artist_name", "artists"])
        lyrics_col = find_column(reader.fieldnames, ["text", "lyrics", "lyric"])
        for row in reader:
            stats["rows"] += 1
            title = (row.get(title_col) or "").strip()
            artist = (row.get(artist_col) or "").strip()
            lyrics = (row.get(lyrics_col) or "").strip()
            if title:
                stats["has_title"] += 1
            if artist:
                stats["has_artist"] += 1
            if lyrics:
                stats["has_lyrics"] += 1
            if title and artist:
                stats["has_title_artist"] += 1
            if title and artist and lyrics:
                stats["usable_rows"] += 1
                key = song_key(title, artist)
                if key in keys:
                    stats["duplicate_usable_keys"] += 1
                keys.add(key)
    return keys, stats


def playlist_length_bin(length: int) -> str:
    if length <= 0:
        return "0"
    if length <= 10:
        return "1-10"
    if length <= 20:
        return "11-20"
    if length <= 30:
        return "21-30"
    if length <= 40:
        return "31-40"
    if length <= 50:
        return "41-50"
    if length <= 60:
        return "51-60"
    if length <= 70:
        return "61-70"
    if length <= 80:
        return "71-80"
    if length <= 90:
        return "81-90"
    if length <= 100:
        return "91-100"
    return "100+"


def print_bar_chart(counts: Counter[str], total: int, width: int = 48) -> None:
    labels = ["0", "1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100", "100+"]
    max_count = max(counts.values(), default=0)
    for label in labels:
        count = counts.get(label, 0)
        bar_len = round((count / max_count) * width) if max_count else 0
        pct = (100.0 * count / total) if total else 0.0
        print(f"  {label:>6} | {'#' * bar_len:<{width}} {count:>8,}  {pct:>6.2f}%")


def print_threshold_table(title: str, counts: dict[str, int], total: int) -> None:
    section(title)
    for label, count in counts.items():
        print(f"  {label:<18} {percent_text(count, total)}")


def empty_mpd_stats(started: float | None = None) -> dict[str, object]:
    return {
        "started": time.time() if started is None else started,
        "total_playlists": 0,
        "total_tracks": 0,
        "matched_tracks": 0,
        "empty_playlists": 0,
        "full_playlists": 0,
        "partial_playlists": 0,
        "zero_match_playlists": 0,
        "length_bins": Counter(),
        "matched_length_bins": Counter(),
        "unique_tracks": set(),
        "unique_matched_tracks": set(),
        "unique_missing_tracks": set(),
        "matched_count_thresholds": {label: 0 for label in MATCHED_COUNT_THRESHOLD_LABELS},
        "coverage_thresholds": {label: 0 for label in COVERAGE_THRESHOLD_LABELS},
    }


def update_stats_for_playlist(stats: dict[str, object], playlist: dict[str, object], lyrics_keys: set[str]) -> None:
    tracks = playlist.get("tracks", [])
    if not isinstance(tracks, list):
        tracks = []
    length = len(tracks)
    matched = 0
    stats["total_playlists"] = int(stats["total_playlists"]) + 1
    stats["total_tracks"] = int(stats["total_tracks"]) + length
    stats["length_bins"][playlist_length_bin(length)] += 1  # type: ignore[index]
    if length == 0:
        stats["empty_playlists"] = int(stats["empty_playlists"]) + 1

    unique_tracks: set[str] = stats["unique_tracks"]  # type: ignore[assignment]
    unique_matched_tracks: set[str] = stats["unique_matched_tracks"]  # type: ignore[assignment]
    unique_missing_tracks: set[str] = stats["unique_missing_tracks"]  # type: ignore[assignment]
    for track in tracks:
        title = track.get("track_name") or track.get("name") or track.get("title") or ""
        artist = track.get("artist_name") or track.get("artist") or ""
        key = song_key(str(title), str(artist))
        if not key or key == "::":
            continue
        unique_tracks.add(key)
        if key in lyrics_keys:
            matched += 1
            unique_matched_tracks.add(key)
        else:
            unique_missing_tracks.add(key)

    stats["matched_tracks"] = int(stats["matched_tracks"]) + matched
    stats["matched_length_bins"][playlist_length_bin(matched)] += 1  # type: ignore[index]
    coverage = matched / length if length else 0.0
    matched_thresholds: dict[str, int] = stats["matched_count_thresholds"]  # type: ignore[assignment]
    coverage_thresholds: dict[str, int] = stats["coverage_thresholds"]  # type: ignore[assignment]
    if matched >= 1:
        matched_thresholds[">= 1 song"] += 1
    if matched >= 5:
        matched_thresholds[">= 5 songs"] += 1
    if matched >= 10:
        matched_thresholds[">= 10 songs"] += 1
    if matched >= 20:
        matched_thresholds[">= 20 songs"] += 1
    if matched >= 30:
        matched_thresholds[">= 30 songs"] += 1
    if matched >= 50:
        matched_thresholds[">= 50 songs"] += 1
    if matched >= 100:
        matched_thresholds[">= 100 songs"] += 1
    if coverage >= 0.10:
        coverage_thresholds[">= 10% coverage"] += 1
    if coverage >= 0.20:
        coverage_thresholds[">= 20% coverage"] += 1
    if coverage >= 0.30:
        coverage_thresholds[">= 30% coverage"] += 1
    if coverage >= 0.50:
        coverage_thresholds[">= 50% coverage"] += 1
    if coverage >= 0.70:
        coverage_thresholds[">= 70% coverage"] += 1
    if coverage >= 0.90:
        coverage_thresholds[">= 90% coverage"] += 1
    if length and matched == length:
        coverage_thresholds["100% coverage"] += 1
        stats["full_playlists"] = int(stats["full_playlists"]) + 1
    elif matched == 0:
        stats["zero_match_playlists"] = int(stats["zero_match_playlists"]) + 1
    else:
        stats["partial_playlists"] = int(stats["partial_playlists"]) + 1


def merge_mpd_stats(target: dict[str, object], source: dict[str, object]) -> None:
    for key in (
        "total_playlists",
        "total_tracks",
        "matched_tracks",
        "empty_playlists",
        "full_playlists",
        "partial_playlists",
        "zero_match_playlists",
    ):
        target[key] = int(target[key]) + int(source[key])
    target["length_bins"].update(source["length_bins"])  # type: ignore[union-attr]
    target["matched_length_bins"].update(source["matched_length_bins"])  # type: ignore[union-attr]
    target["unique_tracks"].update(source["unique_tracks"])  # type: ignore[union-attr]
    target["unique_matched_tracks"].update(source["unique_matched_tracks"])  # type: ignore[union-attr]
    target["unique_missing_tracks"].update(source["unique_missing_tracks"])  # type: ignore[union-attr]
    for label in MATCHED_COUNT_THRESHOLD_LABELS:
        target["matched_count_thresholds"][label] += source["matched_count_thresholds"][label]  # type: ignore[index]
    for label in COVERAGE_THRESHOLD_LABELS:
        target["coverage_thresholds"][label] += source["coverage_thresholds"][label]  # type: ignore[index]


def scan_mpd_file(json_path: Path, lyrics_keys: set[str]) -> dict[str, object]:
    stats = empty_mpd_stats()
    with json_path.open("r", encoding="utf-8", errors="replace") as f:
        if json_path.suffix.lower() == ".jsonl":
            raw_playlists = (json.loads(line) for line in f if line.strip())
        else:
            data = json.load(f)
            raw_playlists = data.get("playlists", data if isinstance(data, list) else [])
        for playlist in raw_playlists:
            update_stats_for_playlist(stats, playlist, lyrics_keys)
    return stats


def scan_mpd(path: Path, lyrics_keys: set[str], max_playlists: int, workers: int) -> dict[str, object]:
    started = time.time()
    stats = empty_mpd_stats(started)
    files = list(iter_mpd_files(path))

    if workers > 1 and not max_playlists and len(files) > 1:
        print(f"  workers={workers:,} files={len(files):,}")
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(scan_mpd_file, json_path, lyrics_keys): json_path for json_path in files}
            for future in as_completed(futures):
                merge_mpd_stats(stats, future.result())
                completed += 1
                if completed == 1 or completed % 25 == 0 or completed == len(files):
                    elapsed = time.time() - started
                    print(
                        f"  scanned files={completed:,}/{len(files):,} "
                        f"playlists={int(stats['total_playlists']):,} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
        return stats

    if workers > 1 and max_playlists:
        print("  workers disabled for --max-playlists so the sample size stays exact.")

    for file_idx, json_path in enumerate(files, start=1):
        with json_path.open("r", encoding="utf-8", errors="replace") as f:
            if json_path.suffix.lower() == ".jsonl":
                raw_playlists = (json.loads(line) for line in f if line.strip())
            else:
                data = json.load(f)
                raw_playlists = data.get("playlists", data if isinstance(data, list) else [])

            for playlist in raw_playlists:
                update_stats_for_playlist(stats, playlist, lyrics_keys)

                if max_playlists and int(stats["total_playlists"]) >= max_playlists:
                    return stats

        if file_idx == 1 or file_idx % 25 == 0:
            elapsed = time.time() - started
            print(f"  scanned files={file_idx:,} playlists={int(stats['total_playlists']):,} elapsed={elapsed:.1f}s", flush=True)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan MPD playlist coverage against lyrics data")
    parser.add_argument("--lyrics-csv", type=Path, default=DEFAULT_LYRICS_CSV)
    parser.add_argument("--mpd-path", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-playlists", type=int, default=0, help="0 means scan all playlists")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2))), help="Thread workers for full MPD scan; exact --max-playlists samples run single-threaded")
    parser.add_argument("--songs-only", action="store_true", help="Only scan the lyrics CSV song fields")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    section("Data scan inputs")
    print(f"  lyrics_csv:     {args.lyrics_csv}")
    print(f"  mpd_path:       {args.mpd_path}")
    print(f"  max_playlists:  {args.max_playlists or 'all'}")
    print(f"  workers:        {args.workers}")

    section("Loading lyrics catalog")
    lyrics_keys, lyrics_stats = load_lyrics_keys(args.lyrics_csv)
    total_rows = lyrics_stats["rows"]
    print(f"  total CSV rows:                 {total_rows:,}")
    print(f"  rows with title:                {percent_text(lyrics_stats['has_title'], total_rows)}")
    print(f"  rows with artist:               {percent_text(lyrics_stats['has_artist'], total_rows)}")
    print(f"  rows with lyrics:               {percent_text(lyrics_stats['has_lyrics'], total_rows)}")
    print(f"  rows with title + artist:       {percent_text(lyrics_stats['has_title_artist'], total_rows)}")
    print(f"  rows usable for MPD matching:   {percent_text(lyrics_stats['usable_rows'], total_rows)}")
    print(f"  duplicate usable title/artists: {percent_text(lyrics_stats['duplicate_usable_keys'], max(1, lyrics_stats['usable_rows']))}")
    print(f"  unique usable title/artists:    {percent_text(len(lyrics_keys), max(1, lyrics_stats['usable_rows']))}")
    if args.songs_only:
        return

    section("Scanning MPD playlists")
    stats = scan_mpd(args.mpd_path, lyrics_keys, args.max_playlists, max(1, args.workers))
    elapsed = time.time() - float(stats["started"])

    total_playlists = int(stats["total_playlists"])
    total_tracks = int(stats["total_tracks"])
    matched_tracks = int(stats["matched_tracks"])
    unique_tracks = stats["unique_tracks"]
    unique_matched_tracks = stats["unique_matched_tracks"]
    unique_missing_tracks = stats["unique_missing_tracks"]

    section("Coverage summary")
    print(f"  playlist count:              {total_playlists:,}")
    print(f"  total MPD track entries:     {total_tracks:,}")
    print(f"  matched track entries:       {percent_text(matched_tracks, total_tracks)}")
    print(f"  missing track entries:       {percent_text(total_tracks - matched_tracks, total_tracks)}")
    print(f"  playlists fully available:   {percent_text(int(stats['full_playlists']), total_playlists)}")
    print(f"  playlists partially matched: {percent_text(int(stats['partial_playlists']), total_playlists)}")
    print(f"  playlists with zero matches: {percent_text(int(stats['zero_match_playlists']), total_playlists)}")
    print(f"  empty playlists:             {percent_text(int(stats['empty_playlists']), total_playlists)}")
    print(f"  unique MPD songs:            {len(unique_tracks):,}")
    print(f"  unique MPD songs matched:    {percent_text(len(unique_matched_tracks), len(unique_tracks))}")
    print(f"  unique MPD songs missing:    {percent_text(len(unique_missing_tracks), len(unique_tracks))}")
    print(f"  elapsed:                     {elapsed:.1f}s")

    print_threshold_table(
        "Playlist filter by matched song count",
        stats["matched_count_thresholds"],
        total_playlists,
    )

    print_threshold_table(
        "Playlist filter by matched coverage percent",
        stats["coverage_thresholds"],
        total_playlists,
    )

    section("Playlist length bar chart")
    print("  Original MPD playlist length")
    print_bar_chart(stats["length_bins"], total_playlists)

    section("Matched song count bar chart")
    print("  Number of songs per playlist that exist in lyrics catalog")
    print_bar_chart(stats["matched_length_bins"], total_playlists)


if __name__ == "__main__":
    main()
