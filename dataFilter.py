#!/usr/bin/env python3
"""Mark MPD playlist JSON files with lyrics-match coverage fields."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dataScan import DEFAULT_DATA_DIR, DEFAULT_LYRICS_CSV, ROOT, iter_mpd_files, load_lyrics_keys, percent_text, section, song_key


MATCH_COUNT_FIELD = "matched_song_count"
COVERAGE_FIELD = "matched_coverage_percent"
DEFAULT_MARKED_DIR = ROOT / "dataMarked"


def mark_playlist(playlist: dict[str, object], lyrics_keys: set[str]) -> tuple[int, int]:
    tracks = playlist.get("tracks", [])
    if not isinstance(tracks, list):
        tracks = []
    matched = 0
    for track in tracks:
        if not isinstance(track, dict):
            continue
        title = track.get("track_name") or track.get("name") or track.get("title") or ""
        artist = track.get("artist_name") or track.get("artist") or ""
        if song_key(str(title), str(artist)) in lyrics_keys:
            matched += 1
    coverage = (matched * 100 // len(tracks)) if tracks else 0
    playlist[MATCH_COUNT_FIELD] = int(matched)
    playlist[COVERAGE_FIELD] = int(coverage)
    return matched, coverage


def mark_file(
    json_path: Path,
    output_path: Path,
    lyrics_keys: set[str],
    dry_run: bool,
    force: bool,
) -> dict[str, int | str]:
    if output_path.exists() and not force:
        with output_path.open("r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        playlists = data.get("playlists", data if isinstance(data, list) else [])
        if not isinstance(playlists, list):
            playlists = []
        already_marked = bool(playlists) and all(
            isinstance(playlist, dict) and MATCH_COUNT_FIELD in playlist and COVERAGE_FIELD in playlist
            for playlist in playlists
        )
        if already_marked:
            return {
                "file": str(json_path),
                "output": str(output_path),
                "playlists": len(playlists),
                "tracks": sum(len(playlist.get("tracks", [])) for playlist in playlists if isinstance(playlist, dict)),
                "matched": sum(int(playlist.get(MATCH_COUNT_FIELD, 0)) for playlist in playlists if isinstance(playlist, dict)),
                "skipped": 1,
                "written": 0,
            }

    with json_path.open("r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    playlists = data.get("playlists", data if isinstance(data, list) else [])
    if not isinstance(playlists, list):
        playlists = []

    total_tracks = 0
    matched_tracks = 0
    for playlist in playlists:
        if not isinstance(playlist, dict):
            continue
        tracks = playlist.get("tracks", [])
        total_tracks += len(tracks) if isinstance(tracks, list) else 0
        matched, _ = mark_playlist(playlist, lyrics_keys)
        matched_tracks += matched

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.write("\n")
        tmp_path.replace(output_path)

    return {
        "file": str(json_path),
        "output": str(output_path),
        "playlists": len(playlists),
        "tracks": total_tracks,
        "matched": matched_tracks,
        "skipped": 0,
        "written": 0 if dry_run else 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create marked MPD JSON copies with matched_song_count and matched_coverage_percent")
    parser.add_argument("--lyrics-csv", type=Path, default=DEFAULT_LYRICS_CSV)
    parser.add_argument("--mpd-path", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MARKED_DIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all MPD JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Compute marks without writing marked JSON copies")
    parser.add_argument("--force", action="store_true", help="Recompute and rewrite marked output files even if fields already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    section("Data filter inputs")
    print(f"  lyrics_csv: {args.lyrics_csv}")
    print(f"  mpd_path:   {args.mpd_path}")
    print(f"  out_dir:    {args.out_dir}")
    print(f"  workers:    {args.workers}")
    print(f"  max_files:  {args.max_files or 'all'}")
    print(f"  dry_run:    {args.dry_run}")
    print(f"  fields:     {MATCH_COUNT_FIELD}, {COVERAGE_FIELD}")

    section("Loading lyrics catalog")
    lyrics_keys, lyrics_stats = load_lyrics_keys(args.lyrics_csv)
    print(f"  unique usable title/artists: {len(lyrics_keys):,}")
    print(f"  usable CSV rows:             {percent_text(lyrics_stats['usable_rows'], lyrics_stats['rows'])}")

    files = list(iter_mpd_files(args.mpd_path))
    if args.max_files:
        files = files[: args.max_files]
    output_paths = {
        json_path: args.out_dir / json_path.relative_to(args.mpd_path) if args.mpd_path.is_dir() else args.out_dir / json_path.name
        for json_path in files
    }

    section("Marking MPD JSON files")
    total_playlists = 0
    total_tracks = 0
    total_matched = 0
    skipped_files = 0
    written_files = 0
    workers = max(1, args.workers)

    if workers == 1:
        for idx, json_path in enumerate(files, start=1):
            result = mark_file(json_path, output_paths[json_path], lyrics_keys, args.dry_run, args.force)
            total_playlists += int(result["playlists"])
            total_tracks += int(result["tracks"])
            total_matched += int(result["matched"])
            skipped_files += int(result["skipped"])
            written_files += int(result["written"])
            if idx == 1 or idx % 25 == 0 or idx == len(files):
                elapsed = time.time() - started
                print(f"  files={idx:,}/{len(files):,} playlists={total_playlists:,} elapsed={elapsed:.1f}s", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(mark_file, json_path, output_paths[json_path], lyrics_keys, args.dry_run, args.force)
                for json_path in files
            ]
            for idx, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                total_playlists += int(result["playlists"])
                total_tracks += int(result["tracks"])
                total_matched += int(result["matched"])
                skipped_files += int(result["skipped"])
                written_files += int(result["written"])
                if idx == 1 or idx % 25 == 0 or idx == len(files):
                    elapsed = time.time() - started
                    print(f"  files={idx:,}/{len(files):,} playlists={total_playlists:,} elapsed={elapsed:.1f}s", flush=True)

    section("Mark summary")
    print(f"  files seen:       {len(files):,}")
    print(f"  files skipped:    {skipped_files:,}")
    print(f"  files written:    {written_files:,}")
    print(f"  output folder:    {args.out_dir}")
    print(f"  playlists marked: {total_playlists:,}")
    print(f"  matched tracks:   {percent_text(total_matched, total_tracks)}")
    print(f"  elapsed:          {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
