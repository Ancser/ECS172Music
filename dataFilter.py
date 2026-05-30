#!/usr/bin/env python3
"""Filter marked MPD playlists into playlist-track CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from dataMarker import COVERAGE_FIELD, DEFAULT_MARKED_DIR, MATCH_COUNT_FIELD, mark_playlist
from dataScan import DEFAULT_LYRICS_CSV, ROOT, iter_mpd_files, load_lyrics_keys, percent_text, section, song_key


DEFAULT_OUTPUT_DIR = ROOT / "dataFiltered"


FILTERS = {
    "playlists_50songs_50coverage.csv": lambda matched, coverage: matched >= 50 and coverage >= 50,
    "playlists_50songs.csv": lambda matched, coverage: matched >= 50,
    "playlists_50coverage.csv": lambda matched, coverage: coverage >= 50,
}


CSV_FIELDS = [
    "playlist_id",
    "playlist_name",
    "original_track_count",
    "matched_song_count",
    "matched_coverage_percent",
    "pos",
    "track_name",
    "artist_name",
    "song_id",
]


def matched_track_rows(playlist: dict[str, object], lyrics_keys: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    tracks = playlist.get("tracks", [])
    if not isinstance(tracks, list):
        return rows
    playlist_id = playlist.get("pid", "")
    playlist_name = playlist.get("name", "")
    matched_song_count = int(playlist.get(MATCH_COUNT_FIELD, 0))
    matched_coverage = int(playlist.get(COVERAGE_FIELD, 0))
    for idx, track in enumerate(tracks):
        if not isinstance(track, dict):
            continue
        title = str(track.get("track_name") or track.get("name") or track.get("title") or "")
        artist = str(track.get("artist_name") or track.get("artist") or "")
        key = song_key(title, artist)
        if key not in lyrics_keys:
            continue
        rows.append(
            {
                "playlist_id": playlist_id,
                "playlist_name": playlist_name,
                "original_track_count": len(tracks),
                "matched_song_count": matched_song_count,
                "matched_coverage_percent": matched_coverage,
                "pos": track.get("pos", idx),
                "track_name": title,
                "artist_name": artist,
                "song_id": key,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract filtered playlist CSVs from MPD JSON")
    parser.add_argument("--lyrics-csv", type=Path, default=DEFAULT_LYRICS_CSV)
    parser.add_argument("--mpd-path", type=Path, default=DEFAULT_MARKED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all MPD JSON files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    section("Data extract inputs")
    print(f"  lyrics_csv: {args.lyrics_csv}")
    print(f"  mpd_path:   {args.mpd_path}")
    print(f"  out_dir:    {args.out_dir}")
    print("  filters:    50+ songs & 50%+ coverage; 50+ songs; 50%+ coverage")

    section("Loading lyrics catalog")
    lyrics_keys, _ = load_lyrics_keys(args.lyrics_csv)
    print(f"  unique usable title/artists: {len(lyrics_keys):,}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {name: args.out_dir / name for name in FILTERS}
    files = {name: path.open("w", encoding="utf-8", newline="") for name, path in output_paths.items()}
    writers = {name: csv.DictWriter(handle, fieldnames=CSV_FIELDS) for name, handle in files.items()}
    for writer in writers.values():
        writer.writeheader()

    playlist_counts = {name: 0 for name in FILTERS}
    row_counts = {name: 0 for name in FILTERS}
    total_playlists = 0
    total_matched_tracks = 0
    total_tracks = 0

    section("Extracting filtered CSVs")
    try:
        mpd_files = list(iter_mpd_files(args.mpd_path))
        if args.max_files:
            mpd_files = mpd_files[: args.max_files]
        for file_idx, json_path in enumerate(mpd_files, start=1):
            with json_path.open("r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            playlists = data.get("playlists", data if isinstance(data, list) else [])
            if not isinstance(playlists, list):
                playlists = []
            for playlist in playlists:
                if not isinstance(playlist, dict):
                    continue
                if MATCH_COUNT_FIELD not in playlist or COVERAGE_FIELD not in playlist:
                    mark_playlist(playlist, lyrics_keys)
                matched = int(playlist.get(MATCH_COUNT_FIELD, 0))
                coverage = int(playlist.get(COVERAGE_FIELD, 0))
                tracks = playlist.get("tracks", [])
                total_playlists += 1
                total_tracks += len(tracks) if isinstance(tracks, list) else 0
                total_matched_tracks += matched
                rows = None
                for name, predicate in FILTERS.items():
                    if not predicate(matched, coverage):
                        continue
                    if rows is None:
                        rows = matched_track_rows(playlist, lyrics_keys)
                    writers[name].writerows(rows)
                    playlist_counts[name] += 1
                    row_counts[name] += len(rows)
            if file_idx == 1 or file_idx % 25 == 0 or file_idx == len(mpd_files):
                elapsed = time.time() - started
                print(f"  files={file_idx:,}/{len(mpd_files):,} playlists={total_playlists:,} elapsed={elapsed:.1f}s", flush=True)
    finally:
        for handle in files.values():
            handle.close()

    section("Extract summary")
    print(f"  playlists scanned:       {total_playlists:,}")
    print(f"  matched track entries:   {percent_text(total_matched_tracks, total_tracks)}")
    for name, path in output_paths.items():
        print(f"  {path.name}")
        print(f"    playlists: {playlist_counts[name]:,}")
        print(f"    rows:      {row_counts[name]:,}")
        print(f"    path:      {path}")
    print(f"  elapsed:                 {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
