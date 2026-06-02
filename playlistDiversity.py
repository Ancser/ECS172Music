#!/usr/bin/env python3
"""Split artist-diverse playlists into standalone CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from dataScan import ROOT, iter_mpd_files, section, song_key


DEFAULT_INPUT_CSV = ROOT / "dataFiltered" / "spotify_playlist_50percent_50item.csv"
DEFAULT_OUTPUT_CSV = ROOT / "dataFiltered" / "spotify_playlist_50percent_50item_artist_diverse.csv"
DEFAULT_SUMMARY_CSV = ROOT / "dataFiltered" / "spotify_playlist_50percent_50item_artist_diversity_summary.csv"

DIVERSITY_FIELDS = [
    "top_artist",
    "top_artist_count",
    "top_artist_share",
    "unique_artist_count",
    "deduped_track_count",
    "is_artist_dominated",
    "is_diverse",
]

MPD_TRACK_FIELDS = [
    "playlist_id",
    "playlist_name",
    "original_track_count",
    "pos",
    "track_name",
    "artist_name",
    "album_name",
    "track_uri",
] + DIVERSITY_FIELDS


def normalized_artist(value: object) -> str:
    return str(value or "").strip().lower()


def playlist_stats(
    artists: Iterable[str],
    artist_dominated_threshold: float,
    diverse_threshold: float,
    min_unique_artists: int,
    min_playlist_len: int,
) -> dict[str, object]:
    artist_list = [artist for artist in artists if artist]
    if not artist_list:
        top_artist = ""
        top_artist_count = 0
        top_artist_share = 0.0
        unique_artist_count = 0
    else:
        counts = Counter(artist_list)
        top_artist, top_artist_count = counts.most_common(1)[0]
        top_artist_share = top_artist_count / len(artist_list)
        unique_artist_count = len(counts)

    is_artist_dominated = top_artist_share >= artist_dominated_threshold
    is_diverse = (
        len(artist_list) >= min_playlist_len
        and top_artist_share <= diverse_threshold
        and unique_artist_count >= min_unique_artists
    )
    return {
        "top_artist": top_artist,
        "top_artist_count": top_artist_count,
        "top_artist_share": f"{top_artist_share:.6f}",
        "unique_artist_count": unique_artist_count,
        "deduped_track_count": len(artist_list),
        "is_artist_dominated": int(is_artist_dominated),
        "is_diverse": int(is_diverse),
    }


def find_column(fieldnames: list[str], candidates: list[str]) -> str:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(f"Missing one of columns: {', '.join(candidates)}")


def split_playlist_csv(args: argparse.Namespace) -> tuple[int, int, int]:
    stats_by_pid: dict[str, dict[str, object]] = {}
    rows_by_pid: dict[str, int] = defaultdict(int)
    artists_by_pid: dict[str, dict[str, str]] = defaultdict(dict)

    with args.playlist_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{args.playlist_csv} has no CSV header")
        fieldnames = reader.fieldnames
        pid_col = find_column(fieldnames, ["playlist_id", "pid", "playlist", "user_id"])
        title_col = find_column(fieldnames, ["track_name", "song", "track", "title", "name"])
        artist_col = find_column(fieldnames, ["artist_name", "artist", "artists"])
        song_id_col = None
        for option in ["song_id", "track_id", "track_uri"]:
            if option in fieldnames:
                song_id_col = option
                break

        for row in reader:
            pid = str(row.get(pid_col, ""))
            artist = str(row.get(artist_col, "")).strip()
            if not pid or not artist:
                continue
            track_id = str(row.get(song_id_col, "")).strip() if song_id_col else ""
            if not track_id:
                track_id = song_key(str(row.get(title_col, "")), artist)
            rows_by_pid[pid] += 1
            artists_by_pid[pid].setdefault(track_id, artist)

    for pid, track_artists in artists_by_pid.items():
        stats_by_pid[pid] = playlist_stats(
            (normalized_artist(artist) for artist in track_artists.values()),
            args.artist_dominated_threshold,
            args.diverse_threshold,
            args.min_unique_artists,
            args.min_playlist_len,
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.summary_csv.open("w", encoding="utf-8", newline="") as f:
        summary_fields = ["playlist_id", "input_row_count"] + DIVERSITY_FIELDS
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for pid, stats in stats_by_pid.items():
            writer.writerow({"playlist_id": pid, "input_row_count": rows_by_pid[pid], **stats})

    diverse_ids = {pid for pid, stats in stats_by_pid.items() if int(stats["is_diverse"]) == 1}
    diverse_rows = 0

    with args.playlist_csv.open("r", encoding="utf-8", errors="replace", newline="") as src:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise ValueError(f"{args.playlist_csv} has no CSV header")
        pid_col = find_column(reader.fieldnames, ["playlist_id", "pid", "playlist", "user_id"])
        output_fields = list(reader.fieldnames)
        for field in DIVERSITY_FIELDS:
            if field not in output_fields:
                output_fields.append(field)

        with args.output_csv.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=output_fields)
            writer.writeheader()
            for row in reader:
                pid = str(row.get(pid_col, ""))
                if pid not in diverse_ids:
                    continue
                writer.writerow({**row, **stats_by_pid[pid]})
                diverse_rows += 1

    return len(stats_by_pid), len(diverse_ids), diverse_rows


def iter_playlists_from_mpd(path: Path, max_files: int) -> Iterable[dict[str, object]]:
    files = list(iter_mpd_files(path))
    if max_files:
        files = files[:max_files]
    for json_path in files:
        with json_path.open("r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        playlists = data.get("playlists", data if isinstance(data, list) else [])
        if not isinstance(playlists, list):
            continue
        for playlist in playlists:
            if isinstance(playlist, dict):
                yield playlist


def split_mpd(args: argparse.Namespace) -> tuple[int, int, int]:
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    total_playlists = 0
    diverse_playlists = 0
    diverse_rows = 0

    with args.summary_csv.open("w", encoding="utf-8", newline="") as summary_f, args.output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as out_f:
        summary_writer = csv.DictWriter(
            summary_f,
            fieldnames=["playlist_id", "playlist_name", "input_row_count"] + DIVERSITY_FIELDS,
        )
        output_writer = csv.DictWriter(out_f, fieldnames=MPD_TRACK_FIELDS)
        summary_writer.writeheader()
        output_writer.writeheader()

        for playlist in iter_playlists_from_mpd(args.mpd_path, args.max_files):
            total_playlists += 1
            tracks = playlist.get("tracks", [])
            if not isinstance(tracks, list):
                tracks = []

            deduped_artists: dict[str, str] = {}
            for idx, track in enumerate(tracks):
                if not isinstance(track, dict):
                    continue
                title = str(track.get("track_name") or track.get("name") or track.get("title") or "")
                artist = str(track.get("artist_name") or track.get("artist") or "")
                track_id = str(track.get("track_uri") or "").strip() or song_key(title, artist) or str(idx)
                deduped_artists.setdefault(track_id, artist)

            stats = playlist_stats(
                (normalized_artist(artist) for artist in deduped_artists.values()),
                args.artist_dominated_threshold,
                args.diverse_threshold,
                args.min_unique_artists,
                args.min_playlist_len,
            )
            playlist_id = str(playlist.get("pid", ""))
            playlist_name = str(playlist.get("name", ""))
            summary_writer.writerow(
                {
                    "playlist_id": playlist_id,
                    "playlist_name": playlist_name,
                    "input_row_count": len(tracks),
                    **stats,
                }
            )
            if int(stats["is_diverse"]) != 1:
                continue

            diverse_playlists += 1
            for idx, track in enumerate(tracks):
                if not isinstance(track, dict):
                    continue
                output_writer.writerow(
                    {
                        "playlist_id": playlist_id,
                        "playlist_name": playlist_name,
                        "original_track_count": len(tracks),
                        "pos": track.get("pos", idx),
                        "track_name": track.get("track_name") or track.get("name") or track.get("title") or "",
                        "artist_name": track.get("artist_name") or track.get("artist") or "",
                        "album_name": track.get("album_name") or track.get("album") or "",
                        "track_uri": track.get("track_uri", ""),
                        **stats,
                    }
                )
                diverse_rows += 1

    return total_playlists, diverse_playlists, diverse_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split artist-diverse playlist CSVs")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--playlist-csv", type=Path, default=DEFAULT_INPUT_CSV)
    source.add_argument("--mpd-path", type=Path)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--artist-dominated-threshold", type=float, default=0.50)
    parser.add_argument("--diverse-threshold", type=float, default=0.40)
    parser.add_argument("--min-unique-artists", type=int, default=5)
    parser.add_argument("--min-playlist-len", type=int, default=20)
    parser.add_argument("--max-files", type=int, default=0, help="Only applies to --mpd-path; 0 means all files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    section("Playlist artist diversity split")
    print(f"  artist-dominated rule: top artist >= {args.artist_dominated_threshold:.2f}")
    print(
        "  diverse rule:          "
        f"top artist <= {args.diverse_threshold:.2f}, "
        f"unique artists >= {args.min_unique_artists}, "
        f"deduped tracks >= {args.min_playlist_len}"
    )

    if args.mpd_path:
        print(f"  mpd_path:              {args.mpd_path}")
        total_playlists, diverse_playlists, diverse_rows = split_mpd(args)
    else:
        print(f"  playlist_csv:          {args.playlist_csv}")
        total_playlists, diverse_playlists, diverse_rows = split_playlist_csv(args)

    elapsed = time.time() - started
    print(f"  output_csv:            {args.output_csv}")
    print(f"  summary_csv:           {args.summary_csv}")
    print(f"  total playlists:       {total_playlists:,}")
    print(f"  diverse playlists:     {diverse_playlists:,}")
    print(f"  diverse rows:          {diverse_rows:,}")
    print(f"  elapsed:               {elapsed:.1f}s")


if __name__ == "__main__":
    main()
