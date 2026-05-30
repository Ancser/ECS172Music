#!/usr/bin/env python3
"""Build a song-level LLM semantic CSV for filtered playlist songs."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from newSpotify import (
    DEFAULT_FILTERED_DIR,
    DEFAULT_LLM_CACHE_DIR,
    DEFAULT_LLM_MODEL,
    DEFAULT_LYRICS_CSV,
    DEFAULT_SEMANTIC_CACHE_DIR,
    DEFAULT_SEMANTIC_SONG_CSV,
    SemanticProfile,
    Song,
    build_llm_song_features,
    load_lyrics_csv,
    print_section,
    read_semantic_profile_cache,
    song_key,
)


DEFAULT_PLAYLIST_CSVS = (
    DEFAULT_FILTERED_DIR / "spotify_playlist_50item.csv",
    DEFAULT_FILTERED_DIR / "spotify_playlist_50percent.csv",
)

OUTPUT_FIELDS = [
    "song_id",
    "title",
    "artist",
    "lyrics",
    "language",
    "semantic_summary",
    "themes",
    "lyrical_narrative",
    "listening_context",
    "playlist_function",
    "transition_note",
    "keywords",
    "training_text",
    "source",
]


def read_song_ids_from_playlist_csvs(paths: list[Path]) -> set[str]:
    song_ids: set[str] = set()
    for path in paths:
        if not path.exists():
            print(f"  missing playlist CSV, skipped: {path}")
            continue
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            for row in reader:
                song_id = (row.get("song_id") or "").strip()
                if not song_id:
                    song_id = song_key(row.get("track_name", ""), row.get("artist_name", ""))
                if song_id:
                    song_ids.add(song_id)
    return song_ids


def write_song_semantic_csv(path: Path, songs: dict[str, Song], profiles: dict[str, SemanticProfile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for song_id in sorted(songs):
            profile = profiles.get(song_id)
            if not profile:
                continue
            song = songs[song_id]
            writer.writerow(
                {
                    "song_id": song_id,
                    "title": song.title,
                    "artist": song.artist,
                    "lyrics": song.lyrics,
                    "language": profile.language,
                    "semantic_summary": profile.semantic_summary,
                    "themes": "|".join(profile.themes),
                    "lyrical_narrative": profile.lyrical_narrative,
                    "listening_context": profile.listening_context,
                    "playlist_function": profile.playlist_function,
                    "transition_note": profile.transition_note,
                    "keywords": "|".join(profile.keywords),
                    "training_text": profile.training_text,
                    "source": profile.source,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate song-level LLM semantic profiles for filtered playlist songs")
    parser.add_argument("--lyrics-csv", type=Path, default=DEFAULT_LYRICS_CSV)
    parser.add_argument("--playlist-csv", type=Path, action="append", help="Filtered playlist CSV; can be repeated")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_SEMANTIC_SONG_CSV)
    parser.add_argument("--max-songs", type=int, default=0, help="0 means all selected songs")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-cache-dir", type=Path, default=DEFAULT_LLM_CACHE_DIR)
    parser.add_argument("--semantic-cache-dir", type=Path, default=DEFAULT_SEMANTIC_CACHE_DIR)
    parser.add_argument("--llm-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--llm-batch-size", type=int, default=20)
    parser.add_argument("--llm-max-new-tokens", type=int, default=320)
    parser.add_argument("--emotion-max-chars", type=int, default=1200)
    parser.add_argument("--emotion-limit", type=int, default=0, help="Only process this many missing songs; 0 means all")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--llm-debug-output", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    playlist_csvs = args.playlist_csv or list(DEFAULT_PLAYLIST_CSVS)

    print_section("Song semantic inputs")
    print(f"  lyrics_csv:        {args.lyrics_csv}")
    print(f"  output_csv:        {args.out_csv}")
    print(f"  playlist_csvs:     {len(playlist_csvs)}")
    for path in playlist_csvs:
        print(f"    - {path}")
    print(f"  llm_model:         {args.llm_model}")
    print(f"  llm_device:        {args.llm_device}")
    print(f"  llm_batch_size:    {args.llm_batch_size}")

    print_section("Selecting songs")
    requested_song_ids = read_song_ids_from_playlist_csvs(playlist_csvs)
    print(f"  unique filtered songs: {len(requested_song_ids):,}")

    lyrics = load_lyrics_csv(args.lyrics_csv)
    songs = {song_id: lyrics[song_id] for song_id in requested_song_ids if song_id in lyrics}
    missing_from_lyrics = requested_song_ids - set(songs)
    if args.max_songs and len(songs) > args.max_songs:
        songs = dict(list(sorted(songs.items()))[: args.max_songs])
        print(f"  max_songs cap:        {args.max_songs:,}")
    print(f"  songs with lyrics:    {len(songs):,}")
    print(f"  missing from lyrics:  {len(missing_from_lyrics):,}")
    if not songs:
        raise SystemExit("No matching songs found. Run playlistFilter.py first or check --playlist-csv paths.")

    seed_profiles = read_semantic_profile_cache(args.out_csv)
    seed_profiles = {song_id: profile for song_id, profile in seed_profiles.items() if song_id in songs}
    print(f"  existing output rows: {len(seed_profiles):,}")

    print_section("Generating LLM semantics")
    _, profiles = build_llm_song_features(
        songs=songs,
        model_name=args.llm_model,
        llm_cache_dir=args.llm_cache_dir,
        semantic_cache_dir=args.semantic_cache_dir,
        seed_profiles=seed_profiles,
        max_chars=args.emotion_max_chars,
        limit=args.emotion_limit,
        progress_interval=args.progress_interval,
        device=args.llm_device,
        max_new_tokens=args.llm_max_new_tokens,
        debug_output=args.llm_debug_output,
        batch_size=args.llm_batch_size,
    )

    print_section("Writing song semantic CSV")
    write_song_semantic_csv(args.out_csv, songs, profiles)
    print(f"  path:              {args.out_csv}")
    print(f"  rows written:      {sum(1 for song_id in songs if song_id in profiles):,}")
    print(f"  elapsed:           {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
