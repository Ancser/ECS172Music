#!/usr/bin/env python3
"""Playlist continuation prototype with optional playlist-side LLM semantics.

Pipeline:
- load Spotify lyrics catalog and filtered playlist rows
- split each playlist into observed songs and last-K heldout songs
- Stage 1: retrieve candidates with same-artist priority plus CF/popularity
- optional pre-CF playlist semantic generation from observed songs
- Stage 2: rank Stage 1 candidates with CF, artist metadata, popularity, and
  optional playlist semantic match
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_FILTERED_DIR = ROOT / "dataFiltered"
DEFAULT_LYRICS_CSV = DEFAULT_DATA_DIR / "spotify_millsongdata.csv"
DEFAULT_FILTERED_PLAYLIST_CSV = DEFAULT_FILTERED_DIR / "spotify_playlist_50percent_50item.csv"
DEFAULT_SEMANTIC_CACHE = DEFAULT_FILTERED_DIR / "playlist_semantics.jsonl"
DEFAULT_SONG_SEMANTIC_CACHE = DEFAULT_FILTERED_DIR / "song_semantics_fine_keywords.jsonl"
DEFAULT_LLM_CACHE_DIR = ROOT / "models" / "llm_cache"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
STAGE1_RATIOS = (0.25, 0.50, 0.75, 1.00)
STAGE1_POOL_SIZES = (100, 200, 300, 400, 500)
STAGE2_POOL_SIZE = max(STAGE1_POOL_SIZES)
PROGRESS_INTERVAL = 100
SEMANTIC_SCHEMA_VERSION = "fine_keywords_v1"
SEMANTIC_JSON_KEYS = [
    "top_keywords",
    "affect",
    "energy",
    "valence",
    "genre_style",
    "narrative_theme",
    "listening_context",
    "cohesion",
    "next_song_role",
]
SEMANTIC_ALLOWED = {
    "top_keywords": [
        "high_arousal",
        "low_arousal",
        "dancefloor",
        "theatrical",
        "rebellious",
        "melancholic_story",
        "nostalgic",
        "cinematic",
        "angsty",
        "romantic_tension",
        "party",
        "workout",
        "roadtrip",
        "chill",
        "singalong",
        "dark",
        "uplifting",
        "confident",
        "dreamy",
        "aggressive",
        "playful",
        "anime",
        "club",
        "acoustic",
        "heartbreak",
        "empowerment",
        "mixed",
    ],
    "affect": [
        "confident",
        "dramatic",
        "melancholic",
        "angsty",
        "euphoric",
        "dark",
        "playful",
        "dreamy",
        "aggressive",
        "nostalgic",
        "romantic",
        "chill",
        "uplifting",
        "mixed",
    ],
    "genre_style": [
        "rock",
        "alt_rock",
        "classic_rock",
        "pop",
        "dance_pop",
        "hiphop",
        "rnb",
        "country",
        "soul",
        "folk",
        "metal",
        "electronic",
        "electropop",
        "punk",
        "indie",
        "soundtrack",
        "anime",
        "mixed",
    ],
    "narrative_theme": [
        "love",
        "heartbreak",
        "desire",
        "self_expression",
        "youth",
        "escape",
        "loneliness",
        "resilience",
        "celebration",
        "conflict",
        "fantasy",
        "coming_of_age",
        "depression",
        "empowerment",
        "mixed",
    ],
    "listening_context": [
        "party",
        "wedding",
        "halloween",
        "workout",
        "roadtrip",
        "chill",
        "slow_dance",
        "nostalgia",
        "singalong",
        "background",
        "mixed",
    ],
    "energy": ["low", "mid", "high"],
    "valence": ["negative", "mixed", "positive"],
    "cohesion": ["genre", "affect", "activity", "theme", "story", "mixed"],
    "next_song_role": ["same_vibe", "energy_lift", "cooldown", "genre_bridge", "singalong", "romantic", "spooky"],
}
SEMANTIC_LIST_KEYS = {"top_keywords", "affect", "genre_style", "narrative_theme"}
SEMANTIC_FREE_TEXT_KEYS: set[str] = set()
SEMANTIC_TERM_EXPANSIONS = {
    "high_arousal": ["dance", "fire", "wild", "tonight", "loud", "run", "fight", "party"],
    "low_arousal": ["slow", "soft", "quiet", "sleep", "dream", "rain", "alone"],
    "dancefloor": ["dance", "club", "floor", "dj", "beat", "body", "tonight"],
    "theatrical": ["drama", "fame", "show", "stage", "monster", "applause", "glory"],
    "rebellious": ["fight", "break", "riot", "rebel", "wild", "control", "rules"],
    "melancholic_story": ["sad", "cry", "tears", "lonely", "goodbye", "miss", "broken", "pain"],
    "nostalgic": ["remember", "yesterday", "old", "again", "home", "memory", "time"],
    "cinematic": ["dream", "sky", "world", "story", "night", "light", "hero"],
    "angsty": ["pain", "hate", "alone", "broken", "scream", "dark", "inside"],
    "romantic_tension": ["love", "heart", "kiss", "touch", "desire", "need", "want"],
    "party": ["party", "tonight", "dance", "drink", "club", "everybody"],
    "workout": ["run", "strong", "fight", "power", "move", "body"],
    "roadtrip": ["road", "drive", "highway", "home", "miles", "ride"],
    "chill": ["slow", "easy", "relax", "quiet", "soft", "dream"],
    "singalong": ["sing", "song", "na", "la", "everybody", "chorus"],
    "dark": ["dark", "black", "night", "shadow", "dead", "fear"],
    "uplifting": ["hope", "rise", "light", "free", "alive", "higher"],
    "confident": ["fame", "money", "power", "boss", "strong", "winner"],
    "dreamy": ["dream", "sleep", "sky", "moon", "stars", "float"],
    "aggressive": ["fight", "kill", "rage", "blood", "scream", "fire"],
    "playful": ["fun", "play", "baby", "smile", "crazy", "sweet"],
    "anime": ["hero", "dream", "world", "story", "fight", "future"],
    "club": ["club", "dance", "dj", "beat", "floor", "bass"],
    "acoustic": ["guitar", "home", "simple", "voice", "song"],
    "heartbreak": ["heart", "broken", "goodbye", "tears", "miss", "alone"],
    "empowerment": ["strong", "free", "power", "rise", "fight", "alive"],
    "depression": ["sad", "alone", "dark", "cry", "pain", "empty", "broken"],
}

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
    semantic_norm: dict[str, float]
    retrieval: float
    cold_start: bool


@dataclass(frozen=True)
class ContentIndex:
    song_ids: list[str]
    row_by_song: dict[str, int]
    matrix: Any


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


def playlist_artist_stats(case: EvalCase, songs: dict[str, Song]) -> tuple[float, int]:
    track_ids = case.observed + case.heldout
    artists = [songs[song_id].artist for song_id in track_ids if song_id in songs]

    if not artists:
        return 0.0, 0

    artist_counts = Counter(artists)
    top_artist_count = artist_counts.most_common(1)[0][1]
    top_artist_share = top_artist_count / len(artists)
    unique_artist_count = len(artist_counts)

    return top_artist_share, unique_artist_count


def split_cases_by_artist_diversity(
    cases: list[EvalCase],
    songs: dict[str, Song],
    artist_dominated_threshold: float = 0.50,
    diverse_threshold: float = 0.40,
    min_unique_artists: int = 5,
) -> tuple[list[EvalCase], list[EvalCase]]:
    artist_dominated = []
    diverse = []

    for case in cases:
        top_artist_share, unique_artist_count = playlist_artist_stats(case, songs)

        if top_artist_share >= artist_dominated_threshold:
            artist_dominated.append(case)

        if top_artist_share <= diverse_threshold and unique_artist_count >= min_unique_artists:
            diverse.append(case)

    return artist_dominated, diverse


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


def semantic_cache_key(playlist_id: str, observed: list[str]) -> str:
    payload = json.dumps([SEMANTIC_SCHEMA_VERSION, playlist_id, observed], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_semantic_cache(path: Path) -> dict[str, dict[str, object]]:
    cache: dict[str, dict[str, object]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(record.get("cache_key", ""))
            semantic = record.get("semantic")
            if key and isinstance(semantic, dict):
                cache[key] = semantic
    return cache


def append_semantic_cache(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def song_prompt_line(song_id: str, songs: dict[str, Song], lyrics_chars: int) -> str:
    song = songs.get(song_id)
    if not song:
        return song_id
    lyrics = re.sub(r"\s+", " ", song.lyrics).strip()
    if len(lyrics) > lyrics_chars:
        lyrics = lyrics[:lyrics_chars].rsplit(" ", 1)[0]
    return f"- title: {song.title}; artist: {song.artist}; lyrics_excerpt: {lyrics}"


def playlist_semantic_prompt(
    case: EvalCase,
    songs: dict[str, Song],
    recent_songs: int,
    lyrics_chars: int,
) -> str:
    observed = case.observed[-recent_songs:] if recent_songs > 0 else case.observed
    song_lines = "\n".join(song_prompt_line(song_id, songs, lyrics_chars) for song_id in observed)
    allowed_lines = "\n".join(
        f"- {key}: {', '.join(values)}"
        for key, values in SEMANTIC_ALLOWED.items()
    )
    return (
        "Classify this playlist for music recommendation.\n"
        "Use only the observed songs below. Do not infer from hidden future songs.\n"
        "Return valid JSON only. No markdown. No explanation.\n"
        "Use only labels from the allowed lists. Do not invent new labels.\n"
        "top_keywords must be exactly 3 labels describing the playlist's musical intent, not artist identity.\n"
        "affect, genre_style, and narrative_theme must be arrays with 1 to 3 labels.\n"
        "All other fields must be one label string.\n\n"
        "Every key is required. Never return empty strings or empty arrays.\n"
        "If uncertain, use mixed. For energy use mid. For next_song_role use same_vibe.\n\n"
        "Allowed labels:\n"
        f"{allowed_lines}\n\n"
        "Required JSON schema:\n"
        '{"top_keywords":[],"affect":[],"energy":"","valence":"","genre_style":[],"narrative_theme":[],"listening_context":"","cohesion":"","next_song_role":""}\n\n'
        f"playlist_id: {case.playlist_id}\n"
        f"observed_song_count: {len(case.observed)}\n"
        "observed songs, in playlist order, with short lyric excerpts:\n"
        f"{song_lines}\n\n"
        "JSON:"
    )


def extract_json_object(text: str) -> dict[str, object]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict) and parsed:
            return parsed

    assignment_profile: dict[str, object] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("`").rstrip(",")
        if not line:
            continue
        for key in SEMANTIC_JSON_KEYS:
            match = re.match(rf"^{re.escape(key)}\s*[:=]\s*(.+)$", line)
            if not match:
                continue
            value_text = match.group(1).strip().rstrip(",")
            try:
                assignment_profile[key] = ast.literal_eval(value_text)
            except (ValueError, SyntaxError):
                assignment_profile[key] = value_text.strip("\"'")
            break
    return assignment_profile


def normalize_semantic_profile(profile: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key in SEMANTIC_JSON_KEYS:
        value = profile.get(key)
        if key in SEMANTIC_FREE_TEXT_KEYS:
            normalized[key] = normalize_free_semantic_text(str(value)) if value is not None else ""
        elif key in SEMANTIC_LIST_KEYS:
            values = value if isinstance(value, list) else [value]
            cleaned = []
            for item in values:
                label = normalize_semantic_label(str(item), key)
                if label and label not in cleaned:
                    cleaned.append(label)
            target_len = 3 if key == "top_keywords" else None
            if not cleaned:
                cleaned = fallback_semantic_list(key)
            if target_len:
                for fallback in fallback_semantic_list(key):
                    if len(cleaned) >= target_len:
                        break
                    if fallback not in cleaned:
                        cleaned.append(fallback)
            normalized[key] = cleaned[:3]
        else:
            label = normalize_semantic_label(str(value), key) if value is not None else ""
            normalized[key] = label or fallback_semantic_label(key)
    return normalized


def normalize_free_semantic_text(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \"'`")[:80]


def fallback_semantic_label(key: str) -> str:
    if key == "energy":
        return "mid"
    if key == "next_song_role":
        return "same_vibe"
    allowed = SEMANTIC_ALLOWED.get(key, [])
    return "mixed" if "mixed" in allowed else (allowed[0] if allowed else "")


def fallback_semantic_list(key: str) -> list[str]:
    if key == "top_keywords":
        return ["mixed", "high_arousal", "nostalgic"]
    label = fallback_semantic_label(key)
    return [label] if label else []


def normalize_semantic_label(value: str, key: str) -> str:
    label = normalize_text(value).replace(" ", "_")
    aliases = {
        "medium": "mid",
        "moderate": "mid",
        "alt": "alt_rock",
        "alternative": "alt_rock",
        "alternative_rock": "alt_rock",
        "classic": "classic_rock",
        "dance": "dance_pop",
        "rap": "hiphop",
        "hip_hop": "hiphop",
        "r_b": "rnb",
        "r_and_b": "rnb",
        "edm": "electronic",
        "00s": "2000s",
        "2000": "2000s",
        "10s": "2010s",
        "2010": "2010s",
        "90": "90s",
        "90s_": "90s",
        "1990s": "90s",
        "80": "80s",
        "1980s": "80s",
        "70": "70s",
        "1970s": "70s",
        "60": "60s",
        "1960s": "60s",
        "positive_upbeat": "positive",
        "upbeat": "happy",
        "energetic": "high",
        "intense": "high_arousal",
        "hype": "high_arousal",
        "excited": "high_arousal",
        "exciting": "high_arousal",
        "pump_up": "high_arousal",
        "calm": "low_arousal",
        "soft": "low_arousal",
        "clubby": "dancefloor",
        "club_music": "dancefloor",
        "dance": "dancefloor",
        "dance_pop": "dance_pop",
        "dramatic": "theatrical",
        "drama": "theatrical",
        "melancholy": "melancholic",
        "sad": "melancholic",
        "sadness": "melancholic",
        "depressed": "depression",
        "depressive": "depression",
        "anime_opening": "anime",
        "anime_ost": "anime",
        "ost": "soundtrack",
        "film_score": "cinematic",
        "movie": "cinematic",
        "empowering": "empowerment",
        "self_expression": "self_expression",
        "heart_break": "heartbreak",
        "heartbroken": "heartbreak",
        "angst": "angsty",
        "same_mood": "same_vibe",
        "bridge": "genre_bridge",
        "thematic": "theme",
        "theme_based": "theme",
        "genre_based": "genre",
        "mood_based": "affect",
        "activity_based": "activity",
        "vibe": "affect",
    }
    label = aliases.get(label, label)
    allowed = set(SEMANTIC_ALLOWED.get(key, []))
    return label if label in allowed else ""


def favorite_observed_artist(case: EvalCase, songs: dict[str, Song]) -> str:
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for idx, song_id in enumerate(case.observed):
        song = songs.get(song_id)
        if not song:
            continue
        artist = song.artist.strip()
        if not artist:
            continue
        counts[artist] += 1
        first_seen.setdefault(artist, idx)
    if not counts:
        return ""
    return min(counts, key=lambda artist: (-counts[artist], first_seen[artist], artist.lower()))


def enrich_playlist_semantic(profile: dict[str, object], case: EvalCase, songs: dict[str, Song]) -> dict[str, object]:
    return normalize_semantic_profile(profile)


def heuristic_playlist_semantic(case: EvalCase, songs: dict[str, Song]) -> dict[str, object]:
    positive = {"love", "happy", "dance", "party", "smile", "hope", "free", "tonight", "dream"}
    negative = {"sad", "cry", "miss", "lonely", "alone", "tears", "goodbye", "pain", "broken"}
    high_energy = {"dance", "party", "club", "fire", "run", "jump", "loud", "wild", "fight"}
    low_energy = {"sleep", "quiet", "soft", "slow", "rain", "alone", "dream", "night"}
    stop = {
        "the", "and", "you", "your", "that", "with", "for", "this", "from", "are", "was", "were",
        "have", "has", "had", "not", "but", "all", "out", "get", "got", "just", "like", "will",
    }
    tokens: Counter[str] = Counter()
    artists = []
    for song_id in case.observed:
        song = songs.get(song_id)
        if not song:
            continue
        artists.append(song.artist.lower())
        text = f"{song.title} {song.artist} {song.lyrics}"
        for token in re.findall(r"[a-zA-Z][a-zA-Z']{2,}", text.lower()):
            if token not in stop:
                tokens[token] += 1
    top_terms = [token for token, _ in tokens.most_common(8)]
    pos_hits = sum(tokens.get(token, 0) for token in positive)
    neg_hits = sum(tokens.get(token, 0) for token in negative)
    high_hits = sum(tokens.get(token, 0) for token in high_energy)
    low_hits = sum(tokens.get(token, 0) for token in low_energy)
    artist_counts = Counter(artists)
    max_artist_share = max(artist_counts.values()) / len(artists) if artists else 0.0
    return enrich_playlist_semantic(
        {
            "top_keywords": ["dancefloor", "high_arousal", "party"] if high_hits > low_hits else ["melancholic_story", "low_arousal", "chill"] if low_hits > high_hits else ["mixed", "nostalgic", "singalong"],
            "affect": ["euphoric"] if pos_hits > neg_hits else ["melancholic"] if neg_hits > pos_hits else ["mixed"],
            "genre_style": ["mixed"],
            "narrative_theme": ["celebration"] if pos_hits > neg_hits else ["heartbreak"] if neg_hits > pos_hits else ["mixed"],
            "listening_context": "party" if any(term in tokens for term in ("party", "dance", "club")) else "mixed",
            "energy": "high" if high_hits > low_hits else "low" if low_hits > high_hits else "mid",
            "valence": "positive" if pos_hits > neg_hits else "negative" if neg_hits > pos_hits else "mixed",
            "cohesion": "genre" if max_artist_share >= 0.5 else "mixed",
            "next_song_role": "same_vibe",
        },
        case,
        songs,
    )


def count_text_hits(text: str, words: Iterable[str]) -> int:
    score = 0
    for word in words:
        normalized = normalize_text(word)
        if normalized and re.search(rf"\b{re.escape(normalized)}\b", text):
            score += 1
    return score


def top_labels_from_scores(scores: dict[str, float], allowed: list[str], limit: int, fallback: list[str]) -> list[str]:
    ranked = [
        (score, label)
        for label, score in scores.items()
        if score > 0 and label in allowed
    ]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    labels = [label for _, label in ranked[:limit]]
    for label in fallback:
        if len(labels) >= limit:
            break
        if label in allowed and label not in labels:
            labels.append(label)
    return labels[:limit]


def heuristic_song_semantic(song: Song) -> dict[str, object]:
    text = normalize_text(f"{song.title} {song.lyrics}")
    keyword_scores = {
        label: float(count_text_hits(text, terms))
        for label, terms in SEMANTIC_TERM_EXPANSIONS.items()
        if label in SEMANTIC_ALLOWED["top_keywords"]
    }
    genre_scores = {
        "rock": count_text_hits(text, ["guitar", "rock", "band", "loud"]),
        "dance_pop": count_text_hits(text, ["dance", "party", "club", "beat", "body"]),
        "electropop": count_text_hits(text, ["electric", "neon", "synth", "fame", "monster"]),
        "hiphop": count_text_hits(text, ["rap", "flow", "money", "mic", "street"]),
        "rnb": count_text_hits(text, ["baby", "love", "touch", "slow", "body"]),
        "country": count_text_hits(text, ["road", "truck", "home", "whiskey", "town"]),
        "soundtrack": count_text_hits(text, ["story", "hero", "world", "dream", "sky"]),
        "anime": count_text_hits(text, ["hero", "future", "world", "fight", "dream"]),
        "metal": count_text_hits(text, ["blood", "rage", "scream", "dark", "fire"]),
        "folk": count_text_hits(text, ["home", "river", "simple", "guitar", "old"]),
    }
    affect_scores = {
        "confident": count_text_hits(text, ["strong", "power", "winner", "fame", "money"]),
        "dramatic": count_text_hits(text, ["drama", "stage", "show", "applause", "glory"]),
        "melancholic": count_text_hits(text, ["sad", "tears", "cry", "lonely", "goodbye"]),
        "angsty": count_text_hits(text, ["pain", "hate", "broken", "scream", "inside"]),
        "euphoric": count_text_hits(text, ["tonight", "party", "alive", "higher", "free"]),
        "dark": count_text_hits(text, ["dark", "black", "shadow", "dead", "fear"]),
        "playful": count_text_hits(text, ["fun", "play", "baby", "smile", "crazy"]),
        "dreamy": count_text_hits(text, ["dream", "moon", "stars", "sleep", "sky"]),
        "nostalgic": count_text_hits(text, ["remember", "yesterday", "again", "home", "time"]),
        "romantic": count_text_hits(text, ["love", "heart", "kiss", "touch", "desire"]),
        "chill": count_text_hits(text, ["slow", "easy", "quiet", "soft", "relax"]),
        "uplifting": count_text_hits(text, ["hope", "rise", "light", "free", "alive"]),
    }
    theme_scores = {
        "love": count_text_hits(text, ["love", "heart", "kiss", "baby", "desire"]),
        "heartbreak": count_text_hits(text, ["broken", "goodbye", "tears", "miss", "alone"]),
        "desire": count_text_hits(text, ["want", "need", "touch", "body", "desire"]),
        "self_expression": count_text_hits(text, ["fame", "show", "applause", "born", "free"]),
        "youth": count_text_hits(text, ["young", "school", "summer", "teen", "kid"]),
        "escape": count_text_hits(text, ["run", "away", "escape", "leave", "road"]),
        "loneliness": count_text_hits(text, ["alone", "lonely", "empty", "night", "miss"]),
        "resilience": count_text_hits(text, ["rise", "strong", "fight", "survive", "alive"]),
        "celebration": count_text_hits(text, ["party", "tonight", "dance", "drink", "everybody"]),
        "conflict": count_text_hits(text, ["fight", "war", "hate", "break", "hurt"]),
        "fantasy": count_text_hits(text, ["dream", "magic", "world", "monster", "fairy"]),
        "coming_of_age": count_text_hits(text, ["young", "grow", "learn", "home", "time"]),
        "depression": count_text_hits(text, ["sad", "alone", "dark", "pain", "empty"]),
        "empowerment": count_text_hits(text, ["power", "strong", "free", "rise", "fight"]),
    }
    high_hits = count_text_hits(text, SEMANTIC_TERM_EXPANSIONS["high_arousal"])
    low_hits = count_text_hits(text, SEMANTIC_TERM_EXPANSIONS["low_arousal"])
    positive_hits = count_text_hits(text, ["love", "party", "hope", "free", "smile", "alive", "dance"])
    negative_hits = count_text_hits(text, ["sad", "cry", "tears", "lonely", "pain", "broken", "dark"])
    if high_hits > low_hits:
        energy = "high"
    elif low_hits > high_hits:
        energy = "low"
    else:
        energy = "mid"
    if positive_hits > negative_hits:
        valence = "positive"
    elif negative_hits > positive_hits:
        valence = "negative"
    else:
        valence = "mixed"
    context_scores = {
        "party": count_text_hits(text, ["party", "club", "dance", "tonight", "drink"]),
        "workout": count_text_hits(text, ["run", "strong", "fight", "move", "body"]),
        "roadtrip": count_text_hits(text, ["road", "drive", "highway", "ride", "miles"]),
        "chill": count_text_hits(text, ["slow", "easy", "quiet", "soft", "dream"]),
        "singalong": count_text_hits(text, ["sing", "song", "everybody", "chorus"]),
        "nostalgia": count_text_hits(text, ["remember", "old", "again", "home", "time"]),
    }
    listening_context = max(context_scores, key=lambda label: (context_scores[label], label))
    if context_scores[listening_context] <= 0:
        listening_context = "mixed"
    return normalize_semantic_profile(
        {
            "top_keywords": top_labels_from_scores(keyword_scores, SEMANTIC_ALLOWED["top_keywords"], 3, ["mixed", "high_arousal", "nostalgic"]),
            "affect": top_labels_from_scores(affect_scores, SEMANTIC_ALLOWED["affect"], 3, ["mixed"]),
            "energy": energy,
            "valence": valence,
            "genre_style": top_labels_from_scores(genre_scores, SEMANTIC_ALLOWED["genre_style"], 3, ["mixed"]),
            "narrative_theme": top_labels_from_scores(theme_scores, SEMANTIC_ALLOWED["narrative_theme"], 3, ["mixed"]),
            "listening_context": listening_context,
            "cohesion": "mixed",
            "next_song_role": "same_vibe",
        }
    )


def song_semantic_cache_key(song_id: str) -> str:
    payload = json.dumps([SEMANTIC_SCHEMA_VERSION, "song", song_id], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_song_semantics(
    songs: dict[str, Song],
    mode: str,
    cache_path: Path,
) -> dict[str, dict[str, object]]:
    if mode == "off":
        return {}
    print_section("Preparing song semantics")
    print(f"  mode:          {mode}")
    print(f"  cache:         {cache_path}")
    cache = load_semantic_cache(cache_path)
    semantics: dict[str, dict[str, object]] = {}
    new_records: list[dict[str, object]] = []
    reused = 0
    generated = 0
    started = time.time()
    items = sorted(songs.items())
    for idx, (song_id, song) in enumerate(items, start=1):
        key = song_semantic_cache_key(song_id)
        if key in cache:
            semantics[song_id] = normalize_semantic_profile(cache[key])
            reused += 1
        else:
            profile = heuristic_song_semantic(song)
            semantics[song_id] = profile
            generated += 1
            new_records.append(
                {
                    "cache_key": key,
                    "song_id": song_id,
                    "title": song.title,
                    "artist": song.artist,
                    "semantic": profile,
                }
            )
        if len(new_records) >= 500:
            append_semantic_cache(cache_path, new_records)
            new_records = []
        if idx == 1 or idx == len(items) or (PROGRESS_INTERVAL > 0 and idx % PROGRESS_INTERVAL == 0):
            print(f"  song semantics {idx:,}/{len(items):,} | reused={reused:,} | generated={generated:,} | elapsed={time.time() - started:.1f}s", flush=True)
    append_semantic_cache(cache_path, new_records)
    return semantics


def generated_text_from_pipeline_output(output: object) -> str:
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            generated = first.get("generated_text", "")
            if isinstance(generated, list) and generated:
                last = generated[-1]
                if isinstance(last, dict):
                    return str(last.get("content", ""))
            return str(generated)
    return str(output)


def semantic_response_schema() -> dict[str, object]:
    properties: dict[str, object] = {}
    for key in SEMANTIC_JSON_KEYS:
        allowed = SEMANTIC_ALLOWED[key]
        if key in SEMANTIC_LIST_KEYS:
            properties[key] = {
                "type": "array",
                "items": {"type": "string", "enum": allowed},
                "minItems": 1,
                "maxItems": 3,
            }
        else:
            properties[key] = {"type": "string", "enum": allowed}
    return {
        "type": "object",
        "properties": properties,
        "required": SEMANTIC_JSON_KEYS,
        "propertyOrdering": SEMANTIC_JSON_KEYS,
    }


def gemini_playlist_semantic(
    case: EvalCase,
    songs: dict[str, Song],
    model_name: str,
    api_key_env: str,
    recent_songs: int,
    lyrics_chars: int,
    max_new_tokens: int,
) -> dict[str, object]:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing Gemini API key. Set {api_key_env}=your_google_ai_studio_key.")
    prompt = playlist_semantic_prompt(case, songs, recent_songs, lyrics_chars)
    endpoint_model = urllib.parse.quote(model_name, safe="")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{endpoint_model}:generateContent?key={urllib.parse.quote(api_key)}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_new_tokens,
            "responseMimeType": "application/json",
            "responseSchema": semantic_response_schema(),
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        raise SystemExit(f"Gemini API request failed with HTTP {exc.code}:\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Gemini API request failed: {exc}") from exc
    parts = body.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
    parsed = extract_json_object(text)
    if not parsed:
        parsed = heuristic_playlist_semantic(case, songs)
    return enrich_playlist_semantic(parsed, case, songs)


def load_text_generation_pipeline(model_name: str, device: str, model_cache_dir: Path):
    try:
        import torch
        from transformers import pipeline
    except Exception as exc:
        raise SystemExit(
            "LLM playlist semantics require transformers and torch. "
            "Run getLLM.py first, or use --playlist-semantics heuristic."
        ) from exc
    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested for playlist semantics, but torch.cuda.is_available() is false.")
    hub_cache = model_cache_dir / "hub"
    if hub_cache.exists():
        os.environ["HF_HUB_CACHE"] = str(hub_cache)
    return pipeline(
        "text-generation",
        model_name,
        model_kwargs={"dtype": "auto"},
        device=0 if resolved_device == "cuda" else -1,
    )


def llm_playlist_semantic(
    pipe,
    case: EvalCase,
    songs: dict[str, Song],
    recent_songs: int,
    lyrics_chars: int,
    max_new_tokens: int,
) -> dict[str, object]:
    prompt = playlist_semantic_prompt(case, songs, recent_songs, lyrics_chars)
    output = pipe(
        [{"role": "user", "content": prompt}],
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    parsed = extract_json_object(generated_text_from_pipeline_output(output))
    if not parsed:
        parsed = heuristic_playlist_semantic(case, songs)
    return enrich_playlist_semantic(parsed, case, songs)


def build_playlist_semantics(
    cases: list[EvalCase],
    songs: dict[str, Song],
    mode: str,
    provider: str,
    cache_path: Path,
    model_name: str,
    model_cache_dir: Path,
    api_key_env: str,
    device: str,
    recent_songs: int,
    lyrics_chars: int,
    max_new_tokens: int,
    max_generate: int,
    force_regenerate: bool,
) -> dict[str, dict[str, object]]:
    if mode == "off":
        return {}
    print_section("Preparing playlist semantics")
    print(f"  mode:          {mode}")
    print(f"  provider:      {provider}")
    print(f"  cache:         {cache_path}")
    cache = {} if force_regenerate else load_semantic_cache(cache_path)
    pipe = None
    semantics: dict[str, dict[str, object]] = {}
    generated = 0
    reused = 0
    started = time.time()
    for idx, case in enumerate(cases, start=1):
        key = semantic_cache_key(case.playlist_id, case.observed)
        if key in cache:
            semantics[case.playlist_id] = enrich_playlist_semantic(cache[key], case, songs)
            reused += 1
        elif max_generate and generated >= max_generate:
            semantics[case.playlist_id] = heuristic_playlist_semantic(case, songs)
        else:
            if mode == "llm":
                if provider == "gemini":
                    profile = gemini_playlist_semantic(
                        case,
                        songs,
                        model_name,
                        api_key_env,
                        recent_songs,
                        lyrics_chars,
                        max_new_tokens,
                    )
                else:
                    if pipe is None:
                        pipe = load_text_generation_pipeline(model_name, device, model_cache_dir)
                    profile = llm_playlist_semantic(pipe, case, songs, recent_songs, lyrics_chars, max_new_tokens)
            else:
                profile = heuristic_playlist_semantic(case, songs)
            semantics[case.playlist_id] = profile
            generated += 1
            append_semantic_cache(
                cache_path,
                [
                    {
                    "cache_key": key,
                    "playlist_id": case.playlist_id,
                    "observed_count": len(case.observed),
                    "semantic": profile,
                    }
                ],
            )
        if idx == 1 or idx == len(cases) or (PROGRESS_INTERVAL > 0 and idx % PROGRESS_INTERVAL == 0):
            print(f"  semantics {idx:,}/{len(cases):,} | reused={reused:,} | generated={generated:,} | elapsed={time.time() - started:.1f}s", flush=True)
    return semantics


def semantic_terms(profile: dict[str, object]) -> set[str]:
    terms: set[str] = set()
    for value in profile.values():
        values = value if isinstance(value, list) else [value]
        for item in values:
            text = str(item).lower()
            for phrase in re.split(r"[,;/|]+", text):
                raw_label = normalize_text(phrase).replace(" ", "_")
                for expansion in SEMANTIC_TERM_EXPANSIONS.get(raw_label, []):
                    terms.add(expansion)
                phrase = re.sub(r"[^a-z0-9 ]+", " ", phrase)
                phrase = re.sub(r"\s+", " ", phrase).strip()
                if len(phrase) >= 3:
                    terms.add(phrase)
                for token in phrase.split():
                    if len(token) >= 4:
                        terms.add(token)
    return terms


def score_playlist_semantics(
    playlist_profile: dict[str, object] | None,
    candidates: set[str],
    songs: dict[str, Song],
    song_semantics: dict[str, dict[str, object]] | None = None,
) -> dict[str, float]:
    if not playlist_profile:
        return {song_id: 0.0 for song_id in candidates}
    if song_semantics:
        return {
            song_id: semantic_profile_similarity(playlist_profile, song_semantics.get(song_id))
            for song_id in candidates
        }
    terms = semantic_terms(playlist_profile)
    if not terms:
        return {song_id: 0.0 for song_id in candidates}
    scores: dict[str, float] = {}
    for song_id in candidates:
        song = songs.get(song_id)
        if not song:
            scores[song_id] = 0.0
            continue
        text = normalize_text(f"{song.title} {song.artist} {song.lyrics}")
        score = 0.0
        for term in terms:
            normalized = normalize_text(term)
            if not normalized:
                continue
            if " " in normalized:
                if normalized in text:
                    score += 2.0
            elif re.search(rf"\b{re.escape(normalized)}\b", text):
                score += 1.0
        scores[song_id] = score
    return scores


def song_content_text(song: Song, profile: dict[str, object] | None, lyrics_chars: int) -> str:
    lyrics = re.sub(r"\s+", " ", song.lyrics).strip()
    if len(lyrics) > lyrics_chars:
        lyrics = lyrics[:lyrics_chars].rsplit(" ", 1)[0]
    semantic_bits: list[str] = []
    if profile:
        for key in SEMANTIC_JSON_KEYS:
            values = profile.get(key)
            values = values if isinstance(values, list) else [values]
            for value in values:
                text = str(value or "").replace("_", " ").strip()
                if text and text != "mixed":
                    semantic_bits.append(text)
                    semantic_bits.append(f"{key} {text}")
    return " ".join([song.title, song.artist, lyrics, " ".join(semantic_bits)])


def build_content_index(
    songs: dict[str, Song],
    song_semantics: dict[str, dict[str, object]],
    max_features: int,
    lyrics_chars: int,
) -> ContentIndex | None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
    except Exception as exc:
        print(f"  content route unavailable; install scikit-learn to enable TF-IDF retrieval: {exc}")
        return None
    song_ids = sorted(songs)
    if not song_ids:
        return None
    texts = [song_content_text(songs[song_id], song_semantics.get(song_id), lyrics_chars) for song_id in song_ids]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]+\b",
    )
    matrix = normalize(vectorizer.fit_transform(texts))
    return ContentIndex(
        song_ids=song_ids,
        row_by_song={song_id: idx for idx, song_id in enumerate(song_ids)},
        matrix=matrix,
    )


def score_content_route(
    observed: list[str],
    content_index: ContentIndex | None,
    exclude: set[str],
    top_k: int,
    recent_window: int,
) -> dict[str, float]:
    if content_index is None:
        return {}
    row_weights: list[tuple[int, float]] = []
    observed_window = observed[-recent_window:] if recent_window > 0 else observed
    recent_set = set(observed_window)
    for song_id in observed:
        row_idx = content_index.row_by_song.get(song_id)
        if row_idx is None:
            continue
        row_weights.append((row_idx, 2.0 if song_id in recent_set else 1.0))
    if not row_weights:
        return {}
    profile = None
    total_weight = 0.0
    for row_idx, weight in row_weights:
        row = content_index.matrix[row_idx] * weight
        profile = row if profile is None else profile + row
        total_weight += weight
    if profile is None or total_weight <= 0:
        return {}
    profile = profile / total_weight
    raw_scores = content_index.matrix @ profile.T
    scores_array = raw_scores.toarray().ravel() if hasattr(raw_scores, "toarray") else raw_scores.ravel()
    scored = [
        (float(score), song_id)
        for song_id, score in zip(content_index.song_ids, scores_array)
        if song_id not in exclude and float(score) > 0.0
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return {song_id: score for score, song_id in scored[:top_k]}


def ordered_scores(
    scores: dict[str, float],
    popularity: Counter[str],
    exclude: set[str],
) -> list[str]:
    return [
        song_id
        for song_id, _ in sorted(scores.items(), key=lambda item: (-item[1], -popularity.get(item[0], 0), item[0]))
        if song_id not in exclude
    ]


def merge_stage1_routes(
    routes: list[list[str]],
    top_popular: list[str],
    pool_size: int,
    exclude: set[str],
) -> set[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    max_route_len = max((len(route) for route in routes), default=0)
    for idx in range(max_route_len):
        for route in routes:
            if idx >= len(route):
                continue
            song_id = route[idx]
            if song_id in exclude or song_id in seen:
                continue
            ordered.append(song_id)
            seen.add(song_id)
            if len(ordered) >= pool_size:
                return set(ordered)
    for song_id in top_popular:
        if song_id not in exclude and song_id not in seen:
            ordered.append(song_id)
            seen.add(song_id)
        if len(ordered) >= pool_size:
            break
    return set(ordered)


def semantic_profile_values(profile: dict[str, object] | None, key: str) -> set[str]:
    if not profile:
        return set()
    value = profile.get(key)
    values = value if isinstance(value, list) else [value]
    return {str(item) for item in values if str(item)}


def semantic_profile_similarity(playlist_profile: dict[str, object], song_profile: dict[str, object] | None) -> float:
    if not song_profile:
        return 0.0
    score = 0.0
    weighted_overlap = {
        "top_keywords": 4.0,
        "affect": 2.0,
        "genre_style": 2.0,
        "narrative_theme": 2.0,
    }
    for key, weight in weighted_overlap.items():
        overlap = semantic_profile_values(playlist_profile, key) & semantic_profile_values(song_profile, key)
        score += weight * len(overlap)

    exact_weights = {
        "energy": 1.0,
        "valence": 0.75,
        "listening_context": 1.0,
    }
    for key, weight in exact_weights.items():
        playlist_values = semantic_profile_values(playlist_profile, key)
        song_values = semantic_profile_values(song_profile, key)
        if playlist_values and song_values and playlist_values != {"mixed"} and playlist_values & song_values:
            score += weight

    playlist_terms = semantic_profile_values(playlist_profile, "top_keywords")
    song_affect = semantic_profile_values(song_profile, "affect")
    song_theme = semantic_profile_values(song_profile, "narrative_theme")
    cross_map = {
        "romantic_tension": {"romantic", "love", "desire"},
        "heartbreak": {"heartbreak", "melancholic", "loneliness", "depression"},
        "melancholic_story": {"melancholic", "heartbreak", "loneliness", "depression"},
        "theatrical": {"dramatic", "self_expression"},
        "confident": {"confident", "empowerment"},
        "dancefloor": {"euphoric", "celebration"},
        "party": {"euphoric", "celebration"},
        "dark": {"dark", "conflict", "depression"},
        "angsty": {"angsty", "conflict"},
        "uplifting": {"uplifting", "resilience", "empowerment"},
    }
    song_cross_values = song_affect | song_theme
    for term in playlist_terms:
        if cross_map.get(term, set()) & song_cross_values:
            score += 1.0
    return score


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
        "playlist_semantic_profile",
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
    content_index: ContentIndex | None = None,
    mode: str = "legacy",
) -> None:
    print_section("Stage 1 candidate recall")
    pool_sizes = list(STAGE1_POOL_SIZES)
    max_pool = max(pool_sizes)
    top_popular = [song_id for song_id, _ in popularity.most_common(max_pool)]
    ratio_methods = [f"CF + artist{int(ratio * 100)}% + popularity" for ratio in STAGE1_RATIOS]
    hybrid_methods = ["Route A: Item-CF", "Route B: Content TF-IDF", "Hybrid CF + content + popularity"]
    method_names = (hybrid_methods if mode == "hybrid" else ratio_methods) + ["Popularity", "CF"]
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
        content_scores = score_content_route(case.observed, content_index, exclude, max_pool, recent_window)
        content_ordered = ordered_scores(content_scores, popularity, exclude)
        for pool_size in pool_sizes:
            truth = set(case.heldout)
            if mode == "hybrid":
                hybrid_candidates = merge_stage1_routes(
                    [cf_ordered[:pool_size], content_ordered[:pool_size]],
                    top_popular,
                    pool_size,
                    exclude,
                )
                recalls_by_method["Route A: Item-CF"][pool_size].append(recall_at_k(cf_ordered[:pool_size], truth, pool_size))
                recalls_by_method["Route B: Content TF-IDF"][pool_size].append(recall_at_k(content_ordered[:pool_size], truth, pool_size))
                recalls_by_method["Hybrid CF + content + popularity"][pool_size].append(
                    recall_at_k(list(hybrid_candidates), truth, len(hybrid_candidates))
                )
            else:
                for ratio, method_name in zip(STAGE1_RATIOS, ratio_methods):
                    forced_limit = max(0, min(pool_size, int(round(pool_size * ratio))))
                    forced = same_artist_candidates(case.observed, songs, popularity, forced_limit, recent_window, exclude)
                    candidates = merge_forced_stage1_candidates(forced, baseline_ordered[:pool_size], pool_size)
                    recalls_by_method[method_name][pool_size].append(
                        recall_at_k(list(candidates), truth, len(candidates))
                    )
            recalls_by_method["Popularity"][pool_size].append(recall_at_k(popularity_ordered[:pool_size], truth, pool_size))
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
    playlist_semantics: dict[str, dict[str, object]],
    song_semantics: dict[str, dict[str, object]],
    pool_size: int,
    force_ratio: float,
    recent_window: int,
    cold_start_threshold: int,
    progress_interval: int,
    content_index: ContentIndex | None = None,
    mode: str = "legacy",
) -> list[PreparedCase]:
    if mode == "hybrid":
        print_section("Preparing candidate scores with two-route Stage 1")
        print("  candidate source: item-CF route + content TF-IDF route + popularity backfill")
    else:
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
        if mode == "hybrid":
            cf_ordered = ordered_scores(cf_scores, popularity, exclude)
            content_scores = score_content_route(case.observed, content_index, exclude, pool_size, recent_window)
            content_ordered = ordered_scores(content_scores, popularity, exclude)
            candidates = merge_stage1_routes([cf_ordered, content_ordered], top_popular, pool_size, exclude)
        else:
            baseline_ordered = ordered_cf_pop_candidates(cf_scores, top_popular, popularity, pool_size, exclude)
            forced_limit = max(0, min(pool_size, int(round(pool_size * force_ratio))))
            forced = same_artist_candidates(case.observed, songs, popularity, forced_limit, recent_window, exclude)
            candidates = merge_forced_stage1_candidates(forced, baseline_ordered, pool_size)
        is_cold = len(case.observed) <= cold_start_threshold or not cf_scores
        if is_cold:
            cold += 1
        artist_scores = score_artist_metadata(case.observed, candidates, songs)
        pop_scores = {song_id: float(popularity.get(song_id, 0)) for song_id in candidates}
        semantic_scores = score_playlist_semantics(playlist_semantics.get(case.playlist_id), candidates, songs, song_semantics)
        candidate_total += len(candidates)
        prepared.append(
            PreparedCase(
                playlist_id=case.playlist_id,
                heldout=case.heldout,
                candidates=candidates,
                cf_norm=normalize_scores(cf_scores, candidates),
                artist_norm=normalize_scores(artist_scores, candidates),
                pop_norm=normalize_scores(pop_scores, candidates),
                semantic_norm=normalize_scores(semantic_scores, candidates),
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
    semantic_weight: float = 0.0,
) -> list[list[str]]:
    rankings: list[list[str]] = []
    for case in prepared_cases:
        scored: list[tuple[float, str]] = []
        for song_id in case.candidates:
            score = (
                cf_weight * case.cf_norm.get(song_id, 0.0)
                + artist_weight * case.artist_norm.get(song_id, 0.0)
                + pop_weight * case.pop_norm.get(song_id, 0.0)
                + semantic_weight * case.semantic_norm.get(song_id, 0.0)
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
    parser.add_argument("--semantic-weight", type=float, default=0.01)
    parser.add_argument("--stage1-mode", choices=["legacy", "hybrid"], default="legacy", help="legacy uses same-artist + CF; hybrid merges CF and content TF-IDF routes")
    parser.add_argument("--content-max-features", type=int, default=6000)
    parser.add_argument("--content-lyrics-chars", type=int, default=500)
    parser.add_argument("--playlist-semantics", choices=["off", "heuristic", "llm"], default="off")
    parser.add_argument("--semantic-provider", choices=["local", "gemini"], default="local", help="Provider used when --playlist-semantics llm")
    parser.add_argument("--semantic-cache", type=Path, default=DEFAULT_SEMANTIC_CACHE)
    parser.add_argument("--song-semantics", choices=["off", "heuristic"], default="heuristic")
    parser.add_argument("--song-semantic-cache", type=Path, default=DEFAULT_SONG_SEMANTIC_CACHE)
    parser.add_argument("--semantic-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--semantic-model-cache-dir", type=Path, default=DEFAULT_LLM_CACHE_DIR)
    parser.add_argument("--gemini-api-key-env", default="GEMINI_API_KEY", help="Environment variable containing your Google AI Studio API key")
    parser.add_argument("--semantic-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--semantic-recent-songs", type=int, default=20)
    parser.add_argument("--semantic-lyrics-chars", type=int, default=240)
    parser.add_argument("--semantic-max-new-tokens", type=int, default=256)
    parser.add_argument("--semantic-max-generate", type=int, default=0, help="0 means generate/cache semantics for all eval playlists")
    parser.add_argument("--force-semantic-regenerate", action="store_true")
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

    artist_dominated_cases, diverse_cases = split_cases_by_artist_diversity(cases, songs)

    print_section("Playlist diversity analysis")
    print(f"  all playlists:              {len(cases):,}")
    print(f"  artist-dominated playlists: {len(artist_dominated_cases):,}")
    print(f"  diverse playlists:          {len(diverse_cases):,}")
    print("  artist-dominated rule: top artist >= 50% of playlist")
    print("  diverse rule: top artist <= 40% and at least 5 unique artists")

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
    print(f"  stage1_mode:        {args.stage1_mode}")
    print(f"  semantics:          {args.playlist_semantics}")
    print(f"  song semantics:     {args.song_semantics if args.playlist_semantics != 'off' else 'off'}")
    if args.playlist_semantics == "llm":
        print(f"  semantic provider:  {args.semantic_provider}")
        if args.semantic_provider == "gemini" and args.semantic_model.startswith("Qwen/"):
            args.semantic_model = DEFAULT_GEMINI_MODEL
        print(f"  semantic model:     {args.semantic_model}")

    playlist_semantics = build_playlist_semantics(
        cases=cases,
        songs=songs,
        mode=args.playlist_semantics,
        provider=args.semantic_provider,
        cache_path=args.semantic_cache,
        model_name=args.semantic_model,
        model_cache_dir=args.semantic_model_cache_dir,
        api_key_env=args.gemini_api_key_env,
        device=args.semantic_device,
        recent_songs=args.semantic_recent_songs,
        lyrics_chars=args.semantic_lyrics_chars,
        max_new_tokens=args.semantic_max_new_tokens,
        max_generate=args.semantic_max_generate,
        force_regenerate=args.force_semantic_regenerate,
    )
    song_semantics = build_song_semantics(
        songs=songs,
        mode=args.song_semantics if (playlist_semantics or args.stage1_mode == "hybrid") else "off",
        cache_path=args.song_semantic_cache,
    )
    content_index = None
    if args.stage1_mode == "hybrid":
        print_section("Building content TF-IDF Stage 1 route")
        content_index = build_content_index(
            songs=songs,
            song_semantics=song_semantics,
            max_features=args.content_max_features,
            lyrics_chars=args.content_lyrics_chars,
        )
        if content_index is not None:
            print(f"  route B songs:       {len(content_index.song_ids):,}")
            print(f"  route B features:    {content_index.matrix.shape[1]:,}")

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
        content_index=content_index,
        mode=args.stage1_mode,
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

    stage1_ratios = (STAGE1_RATIOS[0],) if args.stage1_mode == "hybrid" else STAGE1_RATIOS
    for ratio in stage1_ratios:
        prepared_cases = prepare_cases(
            cases=cases,
            songs=songs,
            cf_neighbors=cf_neighbors,
            popularity=popularity,
            playlist_semantics=playlist_semantics,
            song_semantics=song_semantics,
            pool_size=STAGE2_POOL_SIZE,
            force_ratio=ratio,
            recent_window=args.stage1_recent_window,
            cold_start_threshold=args.cold_start_threshold,
            progress_interval=PROGRESS_INTERVAL,
            content_index=content_index,
            mode=args.stage1_mode,
        )
        quota = int(ratio * 100)
        stage1_label = "Hybrid Stage1" if args.stage1_mode == "hybrid" else f"artist{quota}%"
        cf_pop = build_rankings(prepared_cases, args.cf_weight, 0.0, args.pop_weight)
        rows.append((f"CF + {stage1_label} + popularity", *evaluate_rankings(cf_pop, cases, args.top_k)))
        if playlist_semantics and ratio == STAGE1_RATIOS[0]:
            semantic_only = build_rankings(prepared_cases, 0.0, 0.0, 0.0, 1.0)
            semantic_pop = build_rankings(prepared_cases, 0.0, 0.0, args.pop_weight, 1.0)
            semantic_label = "Playlist-song semantic" if song_semantics else "Playlist semantic"
            rows.append((f"{semantic_label} only ({stage1_label} candidate pool)", *evaluate_rankings(semantic_only, cases, args.top_k)))
            rows.append((f"{semantic_label} + popularity ({stage1_label} candidate pool)", *evaluate_rankings(semantic_pop, cases, args.top_k)))

        full_rankings = build_rankings(
            prepared_cases,
            args.cf_weight,
            args.artist_weight,
            args.pop_weight,
        )
        full_metrics = evaluate_rankings(full_rankings, cases, args.top_k)
        rows.append((f"CF + {stage1_label} + artist score + popularity", *full_metrics))
        if playlist_semantics:
            semantic_rankings = build_rankings(
                prepared_cases,
                args.cf_weight,
                args.artist_weight,
                args.pop_weight,
                args.semantic_weight,
            )
            semantic_metrics = evaluate_rankings(semantic_rankings, cases, args.top_k)
            semantic_suffix = "playlist-song semantic" if song_semantics else "playlist semantic"
            rows.append((f"CF + {stage1_label} + artist score + popularity + {semantic_suffix}", *semantic_metrics))
        else:
            semantic_rankings = full_rankings
            semantic_metrics = full_metrics
        if full_metrics[2] > best_proxy:
            best_proxy = full_metrics[2]
            best_prepared_cases = prepared_cases
            best_full_rankings = full_rankings
        if playlist_semantics and semantic_metrics[2] > best_proxy:
            best_proxy = semantic_metrics[2]
            best_prepared_cases = prepared_cases
            best_full_rankings = semantic_rankings

    print_stage2_results(rows)
    if args.show_example:
        print_example(best_prepared_cases, cases, songs, best_full_rankings)


if __name__ == "__main__":
    main()
