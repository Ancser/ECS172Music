#!/usr/bin/env python3
"""Scan MPD playlist coverage against the local lyrics catalog."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_LYRICS_CSV = DEFAULT_DATA_DIR / "spotify_millsongdata.csv"


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


def scan_mpd(path: Path, lyrics_keys: set[str], max_playlists: int) -> dict[str, object]:
    started = time.time()
    total_playlists = 0
    total_tracks = 0
    matched_tracks = 0
    empty_playlists = 0
    full_playlists = 0
    partial_playlists = 0
    zero_match_playlists = 0
    length_bins: Counter[str] = Counter()
    matched_length_bins: Counter[str] = Counter()
    unique_tracks: set[str] = set()
    unique_matched_tracks: set[str] = set()
    unique_missing_tracks: set[str] = set()

    for file_idx, json_path in enumerate(iter_mpd_files(path), start=1):
        with json_path.open("r", encoding="utf-8", errors="replace") as f:
            if json_path.suffix.lower() == ".jsonl":
                raw_playlists = (json.loads(line) for line in f if line.strip())
            else:
                data = json.load(f)
                raw_playlists = data.get("playlists", data if isinstance(data, list) else [])

            for playlist in raw_playlists:
                tracks = playlist.get("tracks", [])
                length = len(tracks)
                matched = 0
                total_playlists += 1
                total_tracks += length
                length_bins[playlist_length_bin(length)] += 1
                if length == 0:
                    empty_playlists += 1

                for track in tracks:
                    title = track.get("track_name") or track.get("name") or track.get("title") or ""
                    artist = track.get("artist_name") or track.get("artist") or ""
                    key = song_key(title, artist)
                    if not key or key == "::":
                        continue
                    unique_tracks.add(key)
                    if key in lyrics_keys:
                        matched += 1
                        unique_matched_tracks.add(key)
                    else:
                        unique_missing_tracks.add(key)

                matched_tracks += matched
                matched_length_bins[playlist_length_bin(matched)] += 1
                if length and matched == length:
                    full_playlists += 1
                elif matched == 0:
                    zero_match_playlists += 1
                else:
                    partial_playlists += 1

                if max_playlists and total_playlists >= max_playlists:
                    return {
                        "started": started,
                        "total_playlists": total_playlists,
                        "total_tracks": total_tracks,
                        "matched_tracks": matched_tracks,
                        "empty_playlists": empty_playlists,
                        "full_playlists": full_playlists,
                        "partial_playlists": partial_playlists,
                        "zero_match_playlists": zero_match_playlists,
                        "length_bins": length_bins,
                        "matched_length_bins": matched_length_bins,
                        "unique_tracks": unique_tracks,
                        "unique_matched_tracks": unique_matched_tracks,
                        "unique_missing_tracks": unique_missing_tracks,
                        "last_file_idx": file_idx,
                    }

        if file_idx == 1 or file_idx % 25 == 0:
            elapsed = time.time() - started
            print(f"  scanned files={file_idx:,} playlists={total_playlists:,} elapsed={elapsed:.1f}s", flush=True)

    return {
        "started": started,
        "total_playlists": total_playlists,
        "total_tracks": total_tracks,
        "matched_tracks": matched_tracks,
        "empty_playlists": empty_playlists,
        "full_playlists": full_playlists,
        "partial_playlists": partial_playlists,
        "zero_match_playlists": zero_match_playlists,
        "length_bins": length_bins,
        "matched_length_bins": matched_length_bins,
        "unique_tracks": unique_tracks,
        "unique_matched_tracks": unique_matched_tracks,
        "unique_missing_tracks": unique_missing_tracks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan MPD playlist coverage against lyrics data")
    parser.add_argument("--lyrics-csv", type=Path, default=DEFAULT_LYRICS_CSV)
    parser.add_argument("--mpd-path", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-playlists", type=int, default=0, help="0 means scan all playlists")
    parser.add_argument("--songs-only", action="store_true", help="Only scan the lyrics CSV song fields")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    section("Data scan inputs")
    print(f"  lyrics_csv:     {args.lyrics_csv}")
    print(f"  mpd_path:       {args.mpd_path}")
    print(f"  max_playlists:  {args.max_playlists or 'all'}")

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
    stats = scan_mpd(args.mpd_path, lyrics_keys, args.max_playlists)
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

    section("Playlist length bar chart")
    print("  Original MPD playlist length")
    print_bar_chart(stats["length_bins"], total_playlists)

    section("Matched song count bar chart")
    print("  Number of songs per playlist that exist in lyrics catalog")
    print_bar_chart(stats["matched_length_bins"], total_playlists)


if __name__ == "__main__":
    main()
