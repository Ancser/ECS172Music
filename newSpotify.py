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
DEFAULT_LLM_CACHE_DIR = ROOT / "models" / "llm_cache"
DEFAULT_LLM_EMOTION_CACHE_DIR = ROOT / "models" / "llm_emotion_cache"
DEFAULT_LLM_MODEL = "google/gemma-3-270m-it"

EMOTION_LEXICON = {
    "love": {
        "love", "baby", "heart", "kiss", "hold", "touch", "darling", "sweet", "together", "forever",
        "lover", "romance", "beautiful",
    },
    "sadness": {
        "cry", "tears", "lonely", "alone", "sad", "pain", "hurt", "broken", "goodbye", "miss",
        "lost", "blue", "empty", "sorrow",
    },
    "energy": {
        "dance", "party", "night", "fire", "rock", "jump", "move", "beat", "club", "run",
        "wild", "loud", "alive",
    },
    "anger": {
        "hate", "fight", "kill", "rage", "mad", "war", "enemy", "burn", "scream", "angry",
    },
    "calm": {
        "sleep", "dream", "quiet", "soft", "slow", "peace", "breathe", "gentle", "moon", "rain",
        "river", "home",
    },
    "hope": {
        "hope", "rise", "sun", "light", "free", "fly", "believe", "better", "tomorrow", "shine",
    },
    "nostalgia": {
        "remember", "memory", "yesterday", "old", "again", "back", "days", "child", "home",
    },
}

EMOTION_VECTOR = {
    "love": (0.70, 0.45),
    "sadness": (-0.70, 0.25),
    "energy": (0.45, 0.90),
    "anger": (-0.55, 0.85),
    "calm": (0.20, 0.15),
    "hope": (0.65, 0.55),
    "nostalgia": (0.15, 0.30),
    "neutral": (0.0, 0.35),
    "other": (0.0, 0.5),
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
    long_norm: dict[str, float]
    short_norm: dict[str, float]
    cf_norm: dict[str, float]
    artist_norm: dict[str, float]
    type_norm: dict[str, float]
    language_norm: dict[str, float]
    mood_norm: dict[str, float]
    pop_norm: dict[str, float]
    retrieval: float
    cold_start: bool


@dataclass(frozen=True)
class SongFeature:
    language: str
    primary_type: str
    valence: float
    arousal: float


def normalize_text(value: str) -> str:
    value = (value or "").lower()
    value = re.sub(r"\([^)]*\)|\[[^]]*]", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def song_key(title: str, artist: str) -> str:
    return f"{normalize_text(artist)}::{normalize_text(title)}"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def cosine2(a: tuple[float, float], b: tuple[float, float]) -> float:
    dot = a[0] * b[0] + a[1] * b[1]
    na = math.sqrt(a[0] * a[0] + a[1] * a[1])
    nb = math.sqrt(b[0] * b[0] + b[1] * b[1])
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def detect_language(tokens: list[str], raw_text: str) -> str:
    if not raw_text:
        return "unknown"
    ascii_chars = sum(1 for char in raw_text if ord(char) < 128)
    ascii_ratio = ascii_chars / max(1, len(raw_text))
    common_english = {"the", "and", "you", "me", "i", "to", "my", "a", "in", "it", "of"}
    english_hits = sum(1 for token in tokens[:300] if token in common_english)
    if ascii_ratio > 0.95 and english_hits >= 3:
        return "english"
    if ascii_ratio > 0.90:
        return "mostly_english"
    return "non_english"


def build_song_features(songs: dict[str, Song]) -> dict[str, SongFeature]:
    features: dict[str, SongFeature] = {}
    for song_id, song in songs.items():
        tokens = tokenize(song.lyrics)
        token_counts = Counter(tokens)
        scores: dict[str, float] = {}
        for emotion, terms in EMOTION_LEXICON.items():
            scores[emotion] = sum(token_counts.get(term, 0) for term in terms)
        primary = max(scores, key=lambda key: (scores[key], key)) if scores else "other"
        if scores.get(primary, 0.0) <= 0:
            primary = "other"

        total = sum(scores.values())
        if total:
            valence = sum(scores[key] * EMOTION_VECTOR[key][0] for key in scores) / total
            arousal = sum(scores[key] * EMOTION_VECTOR[key][1] for key in scores) / total
        else:
            valence, arousal = EMOTION_VECTOR["other"]

        features[song_id] = SongFeature(
            language=detect_language(tokens, song.lyrics),
            primary_type=primary,
            valence=valence,
            arousal=arousal,
        )
    return features


def emotion_cache_path(cache_dir: Path, model_name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name)
    return cache_dir / f"{safe_name}.csv"


def read_emotion_feature_cache(path: Path) -> dict[str, SongFeature]:
    if not path.exists():
        return {}
    cached: dict[str, SongFeature] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song_id = row.get("song_id", "")
            if not song_id:
                continue
            try:
                cached[song_id] = SongFeature(
                    language=row.get("language", "unknown"),
                    primary_type=row.get("primary_type", "other"),
                    valence=float(row.get("valence", "0")),
                    arousal=float(row.get("arousal", "0.5")),
                )
            except ValueError:
                continue
    return cached


def write_emotion_feature_cache(path: Path, features: dict[str, SongFeature]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["song_id", "language", "primary_type", "valence", "arousal"])
        writer.writeheader()
        for song_id, feature in sorted(features.items()):
            writer.writerow(
                {
                    "song_id": song_id,
                    "language": feature.language,
                    "primary_type": feature.primary_type,
                    "valence": f"{feature.valence:.6f}",
                    "arousal": f"{feature.arousal:.6f}",
                }
            )

def parse_llm_feature(text: str, language: str) -> SongFeature:
    allowed = set(EMOTION_VECTOR)
    label = "neutral"
    valence = EMOTION_VECTOR[label][0]
    arousal = EMOTION_VECTOR[label][1]
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            raw_label = normalize_text(str(payload.get("primary_type", payload.get("emotion", label))))
            if raw_label in allowed:
                label = raw_label
            valence = max(-1.0, min(1.0, float(payload.get("valence", EMOTION_VECTOR[label][0]))))
            arousal = max(0.0, min(1.0, float(payload.get("arousal", EMOTION_VECTOR[label][1]))))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return SongFeature(language=language, primary_type=label, valence=valence, arousal=arousal)


def llm_prompt(song: Song, max_chars: int) -> str:
    lyrics = song.lyrics[:max_chars].replace("\r", " ").replace("\n", " ")
    return (
        "Classify the song lyrics for a music recommendation experiment.\n"
        "Return only compact JSON with keys primary_type, valence, arousal.\n"
        "primary_type must be one of: love, sadness, energy, anger, calm, hope, nostalgia, neutral.\n"
        "valence is from -1.0 negative to 1.0 positive. arousal is from 0.0 calm to 1.0 energetic.\n"
        f"Artist: {song.artist}\n"
        f"Title: {song.title}\n"
        f"Lyrics: {lyrics}\n"
        "JSON:"
    )


def build_llm_song_features(
    songs: dict[str, Song],
    model_name: str,
    llm_cache_dir: Path,
    emotion_cache_dir: Path,
    max_chars: int,
    limit: int,
    progress_interval: int,
    device: str,
) -> dict[str, SongFeature]:
    cache_path = emotion_cache_path(emotion_cache_dir, model_name)
    features = read_emotion_feature_cache(cache_path)
    missing_ids = [song_id for song_id in songs if song_id not in features]
    if limit > 0:
        missing_ids = missing_ids[:limit]
    print(f"  LLM emotion cache:      {cache_path}")
    print(f"  cached features:        {len(features):,}")
    print(f"  missing this run:       {len(missing_ids):,}")
    if not missing_ids:
        return {song_id: features[song_id] for song_id in songs if song_id in features}

    import os

    llm_hub_cache = llm_cache_dir / "hub"
    llm_hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(llm_hub_cache)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if resolved_device == "cuda" else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        message = str(exc)
        print()
        print("Could not load the LLM emotion model.")
        print("Most common reason for Gemma: the model is gated on Hugging Face.")
        print("Fix:")
        print("  1. Open https://huggingface.co/google/gemma-3-270m-it")
        print("  2. Accept the Google usage license")
        print("  3. Run: python -m huggingface_hub.cli.hf auth login")
        print("  4. Re-run install_llm.py, then re-run this command")
        print()
        print(message[:1200])
        raise SystemExit(2) from exc
    model.to(resolved_device)
    model.eval()

    started = time.time()
    for idx, song_id in enumerate(missing_ids, start=1):
        song = songs[song_id]
        tokens = tokenize(song.lyrics)
        language = detect_language(tokens, song.lyrics)
        prompt = llm_prompt(song, max_chars)
        encoded = tokenizer(prompt, return_tensors="pt").to(resolved_device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][encoded["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        features[song_id] = parse_llm_feature(text, language)
        if progress_interval and (idx == 1 or idx % progress_interval == 0 or idx == len(missing_ids)):
            elapsed = time.time() - started
            print(f"  llm-tagged {idx:,}/{len(missing_ids):,} missing songs | total_cached={len(features):,} | elapsed={elapsed:.1f}s", flush=True)
            write_emotion_feature_cache(cache_path, features)

    write_emotion_feature_cache(cache_path, features)
    return {song_id: features[song_id] for song_id in songs if song_id in features}


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


def print_results_table(
    rows: list[dict[str, float | str]],
    comparison_model: str = "CF + popularity baseline",
) -> None:
    comparison_row = next((row for row in rows if row["model"] == comparison_model), None)
    if comparison_row is None:
        comparison_row = next((row for row in rows if row["model"] == "Popularity"), None)
        comparison_model = "Popularity"

    base_recall = float(comparison_row["recall"]) if comparison_row else 0.0
    base_ndcg = float(comparison_row["ndcg"]) if comparison_row else 0.0
    base_proxy = float(comparison_row["proxy"]) if comparison_row else 0.0

    print()
    print("Results table")
    print(f"Comparison baseline: {comparison_model}")
    header = (
        f"{'Model':<28} {'Retrieval':>10} {'Recall':>10} {'dRecall':>10} "
        f"{'NDCG':>10} {'dNDCG':>10} {'Proxy':>10} {'dProxy':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        retrieval = row["retrieval"]
        retrieval_text = "-" if retrieval == "" else f"{float(retrieval):.5f}"
        recall = float(row["recall"])
        ndcg = float(row["ndcg"])
        proxy = float(row["proxy"])
        print(
            f"{str(row['model']):<28} {retrieval_text:>10} "
            f"{recall:>10.5f} {recall - base_recall:>+10.5f} "
            f"{ndcg:>10.5f} {ndcg - base_ndcg:>+10.5f} "
            f"{proxy:>10.5f} {proxy - base_proxy:>+10.5f}"
        )


def result_row(
    model: str,
    retrieval: float | str,
    recall: float,
    ndcg: float,
) -> dict[str, float | str]:
    return {
        "model": model,
        "retrieval": retrieval,
        "recall": recall,
        "ndcg": ndcg,
        "proxy": (recall + ndcg) / 2.0,
    }


def evaluate_prepared_model(
    model: str,
    prepared_cases: list[PreparedCase],
    cases: list[EvalCase],
    top_k: int,
    alpha: float,
    lyrics_weight: float,
    cf_weight: float,
    mood_weight: float,
    artist_weight: float,
    type_weight: float,
    language_weight: float,
    pop_weight: float,
) -> tuple[dict[str, float | str], dict[str, list[str]], float]:
    rankings, retrieval = build_content_rankings(
        k=top_k,
        prepared_cases=prepared_cases,
        alpha=alpha,
        lyrics_weight=lyrics_weight,
        cf_weight=cf_weight,
        mood_weight=mood_weight,
        artist_weight=artist_weight,
        type_weight=type_weight,
        language_weight=language_weight,
        pop_weight=pop_weight,
    )
    recall, ndcg = evaluate_rankings(rankings, cases, top_k)
    return result_row(model, retrieval, recall, ndcg), rankings, retrieval


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


def score_cf(observed: list[str], cf_neighbors: dict[str, list[tuple[str, float]]], exclude: set[str], top_k: int) -> dict[str, float]:
    scores: Counter[str] = Counter()
    for song_id in observed:
        for candidate, weight in cf_neighbors.get(song_id, []):
            if candidate not in exclude:
                scores[candidate] += weight
    return dict(scores.most_common(top_k))


def average_mood(song_ids: list[str], features: dict[str, SongFeature], decay: float = 1.0) -> tuple[float, float]:
    total_weight = 0.0
    valence = 0.0
    arousal = 0.0
    n_songs = len(song_ids)
    for pos, song_id in enumerate(song_ids):
        feature = features.get(song_id)
        if not feature:
            continue
        weight = decay ** (n_songs - pos - 1) if decay < 1.0 else 1.0
        total_weight += weight
        valence += feature.valence * weight
        arousal += feature.arousal * weight
    if not total_weight:
        return EMOTION_VECTOR["other"]
    return valence / total_weight, arousal / total_weight


def score_metadata_and_mood(
    observed: list[str],
    candidates: set[str],
    songs: dict[str, Song],
    features: dict[str, SongFeature],
    short_window: int,
    time_decay: float,
    mood_drift_beta: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    observed_artists = Counter(songs[song_id].artist for song_id in observed if song_id in songs)
    observed_types = Counter(features[song_id].primary_type for song_id in observed if song_id in features)
    observed_languages = Counter(features[song_id].language for song_id in observed if song_id in features)

    first_half = observed[: max(1, len(observed) // 2)]
    second_half = observed[max(1, len(observed) // 2) :]
    recent = average_mood(observed[-short_window:], features, decay=time_decay)
    first_mood = average_mood(first_half, features, decay=time_decay)
    second_mood = average_mood(second_half, features, decay=time_decay)
    projected_mood = (
        recent[0] + mood_drift_beta * (second_mood[0] - first_mood[0]),
        recent[1] + mood_drift_beta * (second_mood[1] - first_mood[1]),
    )

    artist_scores: dict[str, float] = {}
    type_scores: dict[str, float] = {}
    language_scores: dict[str, float] = {}
    mood_scores: dict[str, float] = {}
    for candidate in candidates:
        song = songs.get(candidate)
        feature = features.get(candidate)
        if not song or not feature:
            continue
        artist_scores[candidate] = float(observed_artists.get(song.artist, 0))
        type_scores[candidate] = float(observed_types.get(feature.primary_type, 0))
        language_scores[candidate] = float(observed_languages.get(feature.language, 0))
        mood_scores[candidate] = cosine2((feature.valence, feature.arousal), projected_mood)

    return artist_scores, type_scores, language_scores, mood_scores


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
    songs: dict[str, Song],
    features: dict[str, SongFeature],
    index: TfidfIndex,
    cf_neighbors: dict[str, list[tuple[str, float]]],
    popularity: Counter[str],
    pool_size: int,
    short_window: int,
    time_decay: float,
    mood_drift_beta: float,
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
            cf_norm: dict[str, float] = {}
        else:
            long_profile = index.profile(case.observed, decay=time_decay)
            short_profile = index.profile(case.observed[-short_window:], decay=time_decay)
            long_scores = index.score_profile(long_profile, exclude, pool_size)
            short_scores = index.score_profile(short_profile, exclude, pool_size)
            cf_scores = score_cf(case.observed, cf_neighbors, exclude, pool_size)
            candidates = set(long_scores) | set(short_scores)
            candidates.update(cf_scores)
            candidates.update(song_id for song_id in top_popular if song_id not in exclude)
            long_norm = normalize_scores(long_scores, candidates)
            short_norm = normalize_scores(short_scores, candidates)
            cf_norm = normalize_scores(cf_scores, candidates)

        artist_scores, type_scores, language_scores, mood_scores = score_metadata_and_mood(
            observed=case.observed,
            candidates=candidates,
            songs=songs,
            features=features,
            short_window=short_window,
            time_decay=time_decay,
            mood_drift_beta=mood_drift_beta,
        )
        pop_raw = {song_id: float(popularity.get(song_id, 0)) for song_id in candidates}
        pop_norm = normalize_scores(pop_raw, candidates)
        artist_norm = normalize_scores(artist_scores, candidates)
        type_norm = normalize_scores(type_scores, candidates)
        language_norm = normalize_scores(language_scores, candidates)
        mood_norm = normalize_scores(mood_scores, candidates)
        retrieval = recall_at_k(list(candidates), set(case.heldout), len(candidates))
        prepared.append(
            PreparedCase(
                playlist_id=case.playlist_id,
                heldout=case.heldout,
                candidates=candidates,
                long_norm=long_norm,
                short_norm=short_norm,
                cf_norm=cf_norm,
                artist_norm=artist_norm,
                type_norm=type_norm,
                language_norm=language_norm,
                mood_norm=mood_norm,
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
    lyrics_weight: float,
    cf_weight: float,
    mood_weight: float,
    artist_weight: float,
    type_weight: float,
    language_weight: float,
    pop_weight: float,
) -> tuple[dict[str, list[str]], float]:
    rankings: dict[str, list[str]] = {}
    retrieval_recalls: list[float] = []

    for case in prepared_cases:
        retrieval_recalls.append(case.retrieval)
        final_scores = {
            song_id: lyrics_weight
            * (alpha * case.long_norm.get(song_id, 0.0) + (1.0 - alpha) * case.short_norm.get(song_id, 0.0))
            + cf_weight * case.cf_norm.get(song_id, 0.0)
            + mood_weight * case.mood_norm.get(song_id, 0.0)
            + artist_weight * case.artist_norm.get(song_id, 0.0)
            + type_weight * case.type_norm.get(song_id, 0.0)
            + language_weight * case.language_norm.get(song_id, 0.0)
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


def describe_song(song_id: str, songs: dict[str, Song], features: dict[str, SongFeature]) -> str:
    song = songs[song_id]
    feature = features.get(song_id)
    if not feature:
        return f"{song.title} / {song.artist}"
    return (
        f"{song.title} / {song.artist} "
        f"[type={feature.primary_type}, lang={feature.language}, mood=({feature.valence:.2f},{feature.arousal:.2f})]"
    )


def print_example_recs(
    cases: list[EvalCase],
    rankings: dict[str, list[str]],
    songs: dict[str, Song],
    features: dict[str, SongFeature],
    limit: int,
    short_window: int,
    time_decay: float,
) -> None:
    if not cases:
        return
    case = cases[0]
    first_mood = average_mood(case.observed[: max(1, len(case.observed) // 2)], features, decay=time_decay)
    recent_mood = average_mood(case.observed[-short_window:], features, decay=time_decay)
    heldout_mood = average_mood(case.heldout, features, decay=time_decay)
    print()
    print(f"Example recommendations for playlist {case.playlist_id}")
    print(
        "  mood path: "
        f"early=({first_mood[0]:.2f},{first_mood[1]:.2f}) -> "
        f"recent=({recent_mood[0]:.2f},{recent_mood[1]:.2f}) -> "
        f"heldout=({heldout_mood[0]:.2f},{heldout_mood[1]:.2f})"
    )
    print("  input:")
    for song_id in case.observed[:limit]:
        print(f"    - {describe_song(song_id, songs, features)}")
    print("  heldout:")
    for song_id in case.heldout[:limit]:
        print(f"    - {describe_song(song_id, songs, features)}")
    print("  recommended:")
    for song_id in rankings.get(case.playlist_id, [])[:limit]:
        print(f"    - {describe_song(song_id, songs, features)}")


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
    parser.add_argument("--mood-drift-beta", type=float, default=0.5, help="Extrapolate recent mood by beta * mood drift")
    parser.add_argument("--cold-start-threshold", type=int, default=1, help="Use popularity-only ranking when observed history has this many songs or fewer")
    parser.add_argument("--progress-interval", type=int, default=1000, help="Print candidate-prep progress every N eval playlists")
    parser.add_argument("--cf-neighbors", type=int, default=100, help="Keep top N co-occurrence neighbors per song")
    parser.add_argument("--emotion-source", choices=["weak", "llm"], default="weak")
    parser.add_argument("--emotion-max-chars", type=int, default=1200)
    parser.add_argument("--emotion-limit", type=int, default=0, help="Only tag this many missing songs; 0 means all missing songs")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-cache-dir", type=Path, default=DEFAULT_LLM_CACHE_DIR)
    parser.add_argument("--llm-emotion-cache-dir", type=Path, default=DEFAULT_LLM_EMOTION_CACHE_DIR)
    parser.add_argument("--llm-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--lyrics-weight", type=float, default=0.25)
    parser.add_argument("--cf-weight", type=float, default=0.35)
    parser.add_argument("--mood-weight", type=float, default=0.15)
    parser.add_argument("--artist-weight", type=float, default=0.10)
    parser.add_argument("--type-weight", type=float, default=0.05)
    parser.add_argument("--language-weight", type=float, default=0.02)
    parser.add_argument("--no-ablations", action="store_true", help="Skip cheap post-sweep ablation comparisons")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--pop-weight", type=float, default=0.10)
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
    print("Building song feature table...")
    if args.emotion_source == "llm":
        features = build_llm_song_features(
            songs=songs,
            model_name=args.llm_model,
            llm_cache_dir=args.llm_cache_dir,
            emotion_cache_dir=args.llm_emotion_cache_dir,
            max_chars=args.emotion_max_chars,
            limit=args.emotion_limit,
            progress_interval=args.progress_interval,
            device=args.llm_device,
        )
        missing_features = set(songs) - set(features)
        if missing_features:
            print(f"  LLM features incomplete; using weak fallback for {len(missing_features):,} songs.")
            weak_features = build_song_features({song_id: songs[song_id] for song_id in missing_features})
            features.update(weak_features)
    else:
        features = build_song_features(songs)
    print(f"  emotion source:         {args.emotion_source}")
    type_counts = Counter(feature.primary_type for feature in features.values())
    language_counts = Counter(feature.language for feature in features.values())
    print("  top lyric types:         " + ", ".join(f"{k}={v:,}" for k, v in type_counts.most_common(5)))
    print("  languages:               " + ", ".join(f"{k}={v:,}" for k, v in language_counts.most_common(5)))
    print()
    print("Building co-occurrence CF neighbors...")
    cf_neighbors = build_cf_neighbors(cases, args.cf_neighbors)
    neighbor_edges = sum(len(items) for items in cf_neighbors.values())
    print(f"  songs with neighbors:    {len(cf_neighbors):,}")
    print(f"  stored neighbor edges:   {neighbor_edges:,}")
    print()
    print("Building lyrics TF-IDF/content-IDF index...")
    index = TfidfIndex(songs, args.min_df, args.max_features)
    print(f"  vocab size:              {len(index.vocab):,}")
    print(f"  indexed songs:           {len(index.item_vectors):,}")

    prepared_cases = prepare_content_cases(
        cases=cases,
        songs=songs,
        features=features,
        index=index,
        cf_neighbors=cf_neighbors,
        popularity=popularity,
        pool_size=args.pool_size,
        short_window=args.short_window,
        time_decay=args.time_decay,
        mood_drift_beta=args.mood_drift_beta,
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
        result_row("Random", "", random_recall, random_ndcg),
        result_row("Popularity", "", pop_recall, pop_ndcg),
    ]

    cf_pop_row, _, _ = evaluate_prepared_model(
        model="CF + popularity baseline",
        prepared_cases=prepared_cases,
        cases=cases,
        top_k=args.top_k,
        alpha=0.0,
        lyrics_weight=0.0,
        cf_weight=args.cf_weight,
        mood_weight=0.0,
        artist_weight=0.0,
        type_weight=0.0,
        language_weight=0.0,
        pop_weight=args.pop_weight,
    )
    result_rows.append(cf_pop_row)
    print(
        f"  CF+popularity Recall@{args.top_k}: {float(cf_pop_row['recall']):.5f}  "
        f"NDCG@{args.top_k}: {float(cf_pop_row['ndcg']):.5f}"
    )

    alphas: list[float] = []
    value = 0.0
    while value <= 1.000001:
        alphas.append(round(value, 10))
        value += args.alpha_grid_step
    if 1.0 not in alphas:
        alphas.append(1.0)

    hybrid_label = f"Hybrid {args.emotion_source}"
    best = None
    print()
    print("Hybrid fusion sweep")
    for alpha in alphas:
        rankings, retrieval = build_content_rankings(
            k=args.top_k,
            prepared_cases=prepared_cases,
            alpha=alpha,
            lyrics_weight=args.lyrics_weight,
            cf_weight=args.cf_weight,
            mood_weight=args.mood_weight,
            artist_weight=args.artist_weight,
            type_weight=args.type_weight,
            language_weight=args.language_weight,
            pop_weight=args.pop_weight,
        )
        recall, ndcg = evaluate_rankings(rankings, cases, args.top_k)
        proxy = (recall + ndcg) / 2.0
        print(
            f"  alpha={alpha:>4.2f}  Retrieval@{args.pool_size}: {retrieval:.5f}  "
            f"Recall@{args.top_k}: {recall:.5f}  NDCG@{args.top_k}: {ndcg:.5f}  Proxy: {proxy:.5f}"
        )
        result_rows.append(result_row(f"{hybrid_label} alpha={alpha:.2f}", retrieval, recall, ndcg))
        candidate = (proxy, recall, ndcg, retrieval, alpha, rankings)
        if best is None or candidate[:4] > best[:4]:
            best = candidate

    assert best is not None
    proxy, recall, ndcg, retrieval, alpha, rankings = best
    print()
    print("Best prototype result")
    print(f"  method:                 hybrid_cf_metadata_mood_tfidf_{args.emotion_source}")
    print(f"  alpha long-term:        {alpha:.2f}")
    print(f"  alpha short-term:       {1.0 - alpha:.2f}")
    print(f"  time decay:             {args.time_decay:.2f}")
    print(f"  weights:                lyrics={args.lyrics_weight:.2f}, cf={args.cf_weight:.2f}, mood={args.mood_weight:.2f}, artist={args.artist_weight:.2f}, type={args.type_weight:.2f}, language={args.language_weight:.2f}, pop={args.pop_weight:.2f}")
    print(f"  Retrieval@{args.pool_size}:          {retrieval:.5f}")
    print(f"  Recall@{args.top_k}:             {recall:.5f}")
    print(f"  NDCG@{args.top_k}:               {ndcg:.5f}")
    print(f"  Proxy:                 {proxy:.5f}")
    result_rows.append(result_row(f"Best {hybrid_label} alpha={alpha:.2f}", retrieval, recall, ndcg))
    if not args.no_ablations:
        print()
        print("Ablation comparisons at best alpha")
        ablations = [
            ("No CF", args.lyrics_weight, 0.0, args.mood_weight, args.artist_weight, args.type_weight, args.language_weight, args.pop_weight),
            ("No mood/drift", args.lyrics_weight, args.cf_weight, 0.0, args.artist_weight, args.type_weight, args.language_weight, args.pop_weight),
            ("No metadata", args.lyrics_weight, args.cf_weight, args.mood_weight, 0.0, 0.0, 0.0, args.pop_weight),
            ("No lyrics", 0.0, args.cf_weight, args.mood_weight, args.artist_weight, args.type_weight, args.language_weight, args.pop_weight),
            ("CF + popularity", 0.0, args.cf_weight, 0.0, 0.0, 0.0, 0.0, args.pop_weight),
        ]
        for name, lyrics_w, cf_w, mood_w, artist_w, type_w, language_w, pop_w in ablations:
            row, _, _ = evaluate_prepared_model(
                model=name,
                prepared_cases=prepared_cases,
                cases=cases,
                top_k=args.top_k,
                alpha=alpha,
                lyrics_weight=lyrics_w,
                cf_weight=cf_w,
                mood_weight=mood_w,
                artist_weight=artist_w,
                type_weight=type_w,
                language_weight=language_w,
                pop_weight=pop_w,
            )
            result_rows.append(row)
    print_results_table(result_rows)
    print_example_recs(
        cases=cases,
        rankings=rankings,
        songs=songs,
        features=features,
        limit=5,
        short_window=args.short_window,
        time_decay=args.time_decay,
    )


if __name__ == "__main__":
    main()
