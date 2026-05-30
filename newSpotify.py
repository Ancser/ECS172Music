#!/usr/bin/env python3
"""CF + structured semantic text playlist-continuation prototype.

This is the first implementation slice for the ECS172 music project:
- optional Gemma LLM calls only create cached structured song profiles
- raw lyrics are not directly compared in the hybrid LLM ranking path
- structured semantic text is represented with TF-IDF/content-IDF vectors
- each playlist uses earlier songs as input and the last 10 songs as heldout truth
- recommendations are evaluated with Recall@10 and NDCG@10

The script runs a tiny built-in demo when no dataset paths are supplied.
For real data, pass an MPD JSON directory and a Spotify lyrics CSV.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import re
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z][a-z']+")
STOPWORDS = {
    "the", "and", "you", "your", "for", "with", "that", "this", "are", "was", "were", "from",
    "have", "has", "had", "but", "not", "all", "can", "just", "like", "into", "out", "our",
    "his", "her", "she", "him", "they", "them", "their", "what", "when", "where", "why",
    "how", "who", "will", "would", "could", "should", "there", "here", "been", "being",
    "about", "after", "before", "over", "under", "again", "then", "than", "too", "very",
    "get", "got", "let", "make", "made", "say", "said", "see", "know", "come", "go",
    "one", "two", "yes", "yeah", "oh", "hey", "la", "na", "woo", "ooh", "ah",
}
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_LYRICS_CSV = DEFAULT_DATA_DIR / "spotify_millsongdata.csv"
DEFAULT_LLM_CACHE_DIR = ROOT / "models" / "llm_cache"
DEFAULT_SEMANTIC_CACHE_DIR = ROOT / "models" / "semantic_cache"
DEFAULT_LLM_MODEL = "google/gemma-3-270m-it"

ALLOWED_EMOTIONS = ("love", "sadness", "energy", "calm", "hope", "anger", "nostalgia", "neutral")
ALLOWED_VALENCE = ("negative", "mixed", "positive")
ALLOWED_AROUSAL = ("low", "medium", "high")
ALLOWED_PLAYLIST_ROLES = (
    "continue_mood",
    "intensify_mood",
    "resolve_mood",
    "contrast_mood",
    "transition_mood",
)
ALLOWED_THEME_TAGS = (
    "breakup",
    "romance",
    "loneliness",
    "memory",
    "confidence",
    "healing",
    "party",
    "struggle",
    "regret",
    "freedom",
    "friendship",
    "desire",
    "loss",
    "self_growth",
    "rebellion",
    "comfort",
    "celebration",
    "anxiety",
)

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
    cf_norm: dict[str, float]
    artist_norm: dict[str, float]
    type_norm: dict[str, float]
    language_norm: dict[str, float]
    semantic_norm: dict[str, float]
    pop_norm: dict[str, float]
    retrieval: float
    cold_start: bool


@dataclass(frozen=True)
class SongFeature:
    language: str
    primary_type: str
    valence: float
    arousal: float


@dataclass(frozen=True)
class SemanticProfile:
    language: str
    semantic_summary: str
    themes: tuple[str, str, str]
    lyrical_narrative: str
    listening_context: str
    playlist_function: str
    transition_note: str
    keywords: tuple[str, str, str, str, str]
    training_text: str
    source: str = "llm"


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


def semantic_cache_path(cache_dir: Path, model_name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name)
    return cache_dir / f"{safe_name}__structured_semantic_v4.csv"


def fixed_training_text(profile: SemanticProfile) -> str:
    return (
        f"Semantic profile: themes={', '.join(profile.themes)}; "
        f"narrative={profile.lyrical_narrative}; "
        f"context={profile.listening_context}; "
        f"playlist_function={profile.playlist_function}; "
        f"transition={profile.transition_note}; "
        f"keywords={', '.join(profile.keywords)}."
    )


def profile_to_feature(profile: SemanticProfile) -> SongFeature:
    # Structured semantic profiles are not raw emotion labels. Keep numerical
    # mood/type features on the weak lexical path and use profiles only for
    # semantic-text matching.
    return SongFeature(
        language=profile.language,
        primary_type="semantic",
        valence=0.0,
        arousal=0.35,
    )


def read_semantic_profile_cache(path: Path) -> dict[str, SemanticProfile]:
    if not path.exists():
        return {}
    cached: dict[str, SemanticProfile] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song_id = row.get("song_id", "")
            if not song_id:
                continue
            themes = tuple((row.get("themes", "") or "").split("|")[:3])
            keywords = tuple((row.get("keywords", "") or "").split("|")[:5])
            if len(themes) != 3:
                continue
            if len(keywords) != 5:
                keywords = ("relationship", "memory", "movement", "reflection", "playlist flow")
            profile = SemanticProfile(
                language=row.get("language", "unknown"),
                semantic_summary=row.get("semantic_summary", ""),
                themes=(themes[0], themes[1], themes[2]),
                lyrical_narrative=row.get("lyrical_narrative", ""),
                listening_context=row.get("listening_context", ""),
                playlist_function=row.get("playlist_function", ""),
                transition_note=row.get("transition_note", ""),
                keywords=(keywords[0], keywords[1], keywords[2], keywords[3], keywords[4]),
                training_text=row.get("training_text", ""),
                source=row.get("source", "llm"),
            )
            cached[song_id] = normalize_semantic_profile(profile, language=profile.language, source=profile.source)
    return cached


def write_semantic_profile_cache(path: Path, profiles: dict[str, SemanticProfile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "song_id",
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
            ],
        )
        writer.writeheader()
        for song_id, profile in sorted(profiles.items()):
            writer.writerow(
                {
                    "song_id": song_id,
                    "language": profile.language,
                    "semantic_summary": profile.semantic_summary,
                    "themes": "|".join(profile.themes),
                    "lyrical_narrative": profile.lyrical_narrative,
                    "listening_context": profile.listening_context,
                    "playlist_function": profile.playlist_function,
                    "transition_note": profile.transition_note,
                    "keywords": "|".join(profile.keywords),
                    "training_text": fixed_training_text(profile),
                    "source": profile.source,
                }
            )

def clean_phrase(value: object, max_words: int = 8) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'_-]*", str(value or "").lower())
    cleaned = [word.strip("'_-") for word in words[:max_words]]
    return " ".join(word for word in cleaned if word).strip()


def clean_phrase_list(value: object, length: int, fallback: list[str]) -> tuple[str, ...]:
    items: list[str] = []
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[,|;/]+", str(value or ""))
    for item in raw_items:
        phrase = clean_phrase(item, max_words=4)
        if phrase and phrase not in items:
            items.append(phrase)
    for phrase in fallback:
        cleaned = clean_phrase(phrase, max_words=4)
        if len(items) >= length:
            break
        if cleaned and cleaned not in items:
            items.append(cleaned)
    return tuple(items[:length])


def normalize_semantic_profile(payload: object, language: str, source: str = "llm") -> SemanticProfile:
    if isinstance(payload, SemanticProfile):
        raw = {
            "semantic_summary": payload.semantic_summary,
            "themes": payload.themes,
            "lyrical_narrative": payload.lyrical_narrative,
            "listening_context": payload.listening_context,
            "playlist_function": payload.playlist_function,
            "transition_note": payload.transition_note,
            "keywords": payload.keywords,
        }
    elif isinstance(payload, dict):
        raw = payload
    else:
        raw = {}
    keywords = clean_phrase_list(raw.get("keywords"), 5, ["relationship", "memory", "movement", "reflection", "playlist flow"])
    themes = clean_phrase_list(raw.get("themes"), 3, list(keywords[:3]) or ["relationship", "memory", "reflection"])
    summary = clean_phrase(raw.get("semantic_summary", raw.get("summary")), max_words=16)
    narrative = clean_phrase(raw.get("lyrical_narrative"), max_words=18)
    context = clean_phrase(raw.get("listening_context"), max_words=12)
    function = clean_phrase(raw.get("playlist_function"), max_words=12)
    transition = clean_phrase(raw.get("transition_note"), max_words=14)
    if not summary:
        summary = f"{themes[0]} and {themes[1]} oriented playlist continuation"
    if not narrative:
        narrative = f"lyrics focus on {', '.join(themes)}"
    if not context:
        context = "general listening sequence"
    if not function:
        function = "continue related playlist context"
    if not transition:
        transition = "connect when recent songs share topic or singer"
    profile = SemanticProfile(
        language=language,
        semantic_summary=summary,
        themes=(themes[0], themes[1], themes[2]),  # type: ignore[index]
        lyrical_narrative=narrative,
        listening_context=context,
        playlist_function=function,
        transition_note=transition,
        keywords=(keywords[0], keywords[1], keywords[2], keywords[3], keywords[4]),  # type: ignore[index]
        training_text="",
        source=str(raw.get("source", source)),
    )
    return SemanticProfile(
        language=profile.language,
        semantic_summary=profile.semantic_summary,
        themes=profile.themes,
        lyrical_narrative=profile.lyrical_narrative,
        listening_context=profile.listening_context,
        playlist_function=profile.playlist_function,
        transition_note=profile.transition_note,
        keywords=profile.keywords,
        training_text=fixed_training_text(profile),
        source=profile.source,
    )


def fallback_semantic_profile(song: Song, language: str) -> SemanticProfile:
    tokens = [
        token
        for token in tokenize(f"{song.title} {song.artist} {song.lyrics}")
        if len(token) > 2 and token not in STOPWORDS
    ]
    counts = Counter(tokens)
    top_words = [word for word, _ in counts.most_common(8)]
    while len(top_words) < 8:
        top_words.append(["relationship", "memory", "movement", "reflection", "playlist", "energy", "story", "voice"][len(top_words)])
    payload = {
        "semantic_summary": f"{top_words[0]} and {top_words[1]} focused song context",
        "themes": top_words[:3],
        "lyrical_narrative": f"lyrics emphasize {top_words[0]} {top_words[1]} {top_words[2]}",
        "listening_context": f"{top_words[3]} oriented listening",
        "playlist_function": f"connect through {top_words[0]} and {top_words[1]}",
        "transition_note": f"works after songs about {top_words[2]}",
        "keywords": top_words[:5],
        "source": "fallback_keyword",
    }
    return normalize_semantic_profile(payload, language=language, source="fallback_keyword")


def extract_json_object(text: str) -> str:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1)
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""


def parse_llm_semantic_profile(text: str, language: str, song: Song) -> SemanticProfile:
    payload: dict[str, object] = {}
    json_text = extract_json_object(text)
    if json_text:
        try:
            loaded = json.loads(json_text)
            if isinstance(loaded, dict):
                payload = loaded
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    if not payload:
        return fallback_semantic_profile(song, language)
    return normalize_semantic_profile(payload, language=language, source="llm")


def llm_prompt(song: Song, max_chars: int) -> str:
    lyrics = song.lyrics[:max_chars].replace("\r", " ").replace("\n", " ")
    return (
        "Create a controlled song semantic profile for a music recommender. "
        "Return exactly one minified JSON object. No markdown. No explanation. No recommendations. "
        "Do not output raw emotion labels or numeric scores. "
        "Do not repeat the song title or artist in any value. Use compact English phrases, not paragraphs.\n"
        "Required keys: semantic_summary, themes, lyrical_narrative, listening_context, playlist_function, transition_note, keywords.\n"
        "themes must be an array of exactly 3 short phrases. keywords must be an array of exactly 5 short phrases. "
        "Every value must be under 8 words and grounded in the title or lyrics. "
        "Do not copy generic examples. Do not use regret/desire/memory unless those ideas are clearly in the song.\n"
        'JSON schema: {"semantic_summary":string,"themes":[string,string,string],"lyrical_narrative":string,"listening_context":string,"playlist_function":string,"transition_note":string,"keywords":[string,string,string,string,string]}\n'
        f"Artist: {song.artist}\n"
        f"Title: {song.title}\n"
        f"Lyrics: {lyrics}\n"
        "JSON:"
    )


def extract_pipeline_text(output: object) -> str:
    try:
        if isinstance(output, list) and output and isinstance(output[0], list):
            return extract_pipeline_text(output[0])
        first = output[0]  # type: ignore[index]
        generated = first.get("generated_text", "")  # type: ignore[union-attr]
        if isinstance(generated, list):
            for item in reversed(generated):
                if isinstance(item, dict) and item.get("role") == "assistant":
                    return str(item.get("content", ""))
            return str(generated[-1].get("content", "")) if generated and isinstance(generated[-1], dict) else ""
        return str(generated)
    except Exception:
        return ""


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


@contextlib.contextmanager
def quiet_transformers_warnings() -> Iterable[None]:
    """Hide noisy Hugging Face generation warnings while keeping our progress output."""
    try:
        from transformers.utils import logging as hf_logging
    except Exception:
        hf_logging = None
    old_verbosity = hf_logging.get_verbosity() if hf_logging else None
    if hf_logging:
        hf_logging.set_verbosity_error()
    with warnings.catch_warnings():
        with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stderr(devnull):
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers\..*")
            warnings.filterwarnings("ignore", message=r".*max_new_tokens.*max_length.*")
            warnings.filterwarnings("ignore", message=r".*pipelines sequentially on GPU.*")
            warnings.filterwarnings("ignore", message=r".*clean_up_tokenization_spaces.*")
            warnings.filterwarnings("ignore", message=r".*generation_config.*deprecated.*")
            try:
                yield
            finally:
                if hf_logging and old_verbosity is not None:
                    hf_logging.set_verbosity(old_verbosity)


def build_llm_song_features(
    songs: dict[str, Song],
    model_name: str,
    llm_cache_dir: Path,
    semantic_cache_dir: Path,
    max_chars: int,
    limit: int,
    progress_interval: int,
    device: str,
    max_new_tokens: int,
    debug_output: int,
    batch_size: int,
) -> tuple[dict[str, SongFeature], dict[str, SemanticProfile]]:
    cache_path = semantic_cache_path(semantic_cache_dir, model_name)
    profiles = read_semantic_profile_cache(cache_path)
    missing_ids = [song_id for song_id in songs if song_id not in profiles]
    if limit > 0:
        missing_ids = missing_ids[:limit]
    print(f"  LLM semantic cache:     {cache_path}")
    print(f"  cached profiles:        {len(profiles):,}")
    print(f"  missing this run:       {len(missing_ids):,}")
    if not missing_ids:
        return (
            {song_id: profile_to_feature(profiles[song_id]) for song_id in songs if song_id in profiles},
            {song_id: profiles[song_id] for song_id in songs if song_id in profiles},
        )

    llm_hub_cache = llm_cache_dir / "hub"
    llm_hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(llm_hub_cache)
    import torch
    from transformers import pipeline

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        print()
        print("CUDA was requested, but this Python environment has CPU-only PyTorch.")
        print("Install CUDA PyTorch, then rerun with --llm-device cuda.")
        print("Suggested command:")
        print("  python -m pip install --user --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128")
        raise SystemExit(2)
    print(f"  LLM device:              {resolved_device}")
    if resolved_device == "cuda":
        props = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"  GPU:                     {props.name} ({total_bytes / (1024 ** 3):.2f} GB total, {free_bytes / (1024 ** 3):.2f} GB free)")
    try:
        with quiet_transformers_warnings():
            pipe = pipeline(
                "text-generation",
                model=model_name,
                device=0 if resolved_device == "cuda" else -1,
            )
    except Exception as exc:
        message = str(exc)
        print()
        print("Could not load the LLM semantic model.")
        print("Most common reason for Gemma: the model is gated on Hugging Face.")
        print("Fix:")
        print("  1. Open https://huggingface.co/google/gemma-3-270m-it")
        print("  2. Accept the Google usage license")
        print("  3. Run: python -m huggingface_hub.cli.hf auth login")
        print("  4. Re-run install_llm.py, then re-run this command")
        print()
        print(message[:1200])
        raise SystemExit(2) from exc

    started = time.time()
    batch_size = max(1, batch_size)
    print(f"  LLM batch size:          {batch_size}")
    if len(missing_ids) >= max(100, progress_interval):
        print("  semantic ETA:            estimating after first batch...")
    done_count = 0
    next_progress = 1
    for start in range(0, len(missing_ids), batch_size):
        batch_ids = missing_ids[start : start + batch_size]
        batch_messages = []
        batch_languages = []
        for song_id in batch_ids:
            song = songs[song_id]
            tokens = tokenize(song.lyrics)
            batch_languages.append(detect_language(tokens, song.lyrics))
            batch_messages.append([{"role": "user", "content": llm_prompt(song, max_chars)}])
        with quiet_transformers_warnings():
            outputs = pipe(
                batch_messages,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                batch_size=batch_size,
            )
        for offset, (song_id, language, output) in enumerate(zip(batch_ids, batch_languages, outputs), start=1):
            idx = start + offset
            text = extract_pipeline_text(output)
            if debug_output and idx <= debug_output:
                print(f"  raw llm output {idx}: {preview_text(text, limit=300)}")
                if not text:
                    print(f"  raw pipeline object {idx}: {preview_text(repr(output), limit=500)}")
            profiles[song_id] = parse_llm_semantic_profile(text, language, songs[song_id])
            done_count = idx
        should_print = done_count >= next_progress or done_count == len(missing_ids)
        if progress_interval and should_print:
            elapsed = time.time() - started
            rate = done_count / elapsed if elapsed > 0 else 0.0
            remaining = len(missing_ids) - done_count
            eta = remaining / rate if rate else 0.0
            percent = (done_count / len(missing_ids)) * 100 if missing_ids else 100.0
            print(
                "  semantic songs "
                f"{done_count:,}/{len(missing_ids):,} ({percent:5.1f}%) | "
                f"cached={len(profiles):,} | batch={batch_size} | "
                f"rate={rate:.2f}/s | elapsed={format_duration(elapsed)} | eta={format_duration(eta)}",
                flush=True,
            )
            write_semantic_profile_cache(cache_path, profiles)
            if progress_interval:
                if next_progress == 1:
                    next_progress = max(progress_interval, done_count + 1)
                else:
                    while next_progress <= done_count:
                        next_progress += progress_interval

    write_semantic_profile_cache(cache_path, profiles)
    return (
        {song_id: profile_to_feature(profiles[song_id]) for song_id in songs if song_id in profiles},
        {song_id: profiles[song_id] for song_id in songs if song_id in profiles},
    )


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


class SemanticTextIndex:
    def __init__(self, profiles: dict[str, SemanticProfile], min_df: int, max_features: int):
        self.profiles = profiles
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
        for song_id, profile in self.profiles.items():
            tokens = tokenize(profile.training_text)
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

    def score_candidates(
        self,
        profile: dict[int, float],
        candidates: set[str],
        exclude: set[str],
    ) -> dict[str, float]:
        if not profile or not candidates:
            return {}
        scores: dict[str, float] = {}
        for song_id in candidates:
            if song_id in exclude:
                continue
            vec = self.item_vectors.get(song_id)
            if not vec:
                continue
            scores[song_id] = sum(profile.get(idx, 0.0) * weight for idx, weight in vec.items())
        return scores


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

    print_section("Results table")
    print(f"Comparison baseline: {comparison_model}")
    model_width = max(28, *(len(str(row["model"])) for row in rows))
    header = (
        f"{'Model':<{model_width}} {'Retrieval':>10} {'Recall':>10} {'dRecall':>10} "
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
            f"{str(row['model']):<{model_width}} {retrieval_text:>10} "
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
    cf_weight: float,
    artist_weight: float,
    type_weight: float,
    language_weight: float,
    semantic_weight: float,
    pop_weight: float,
) -> tuple[dict[str, float | str], dict[str, list[str]], float]:
    rankings, retrieval = build_content_rankings(
        k=top_k,
        prepared_cases=prepared_cases,
        cf_weight=cf_weight,
        artist_weight=artist_weight,
        type_weight=type_weight,
        language_weight=language_weight,
        semantic_weight=semantic_weight,
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


def score_metadata(
    observed: list[str],
    candidates: set[str],
    songs: dict[str, Song],
    features: dict[str, SongFeature],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    observed_artists = Counter(songs[song_id].artist for song_id in observed if song_id in songs)
    observed_types = Counter(features[song_id].primary_type for song_id in observed if song_id in features)
    observed_languages = Counter(features[song_id].language for song_id in observed if song_id in features)

    artist_scores: dict[str, float] = {}
    type_scores: dict[str, float] = {}
    language_scores: dict[str, float] = {}
    for candidate in candidates:
        song = songs.get(candidate)
        feature = features.get(candidate)
        if not song or not feature:
            continue
        artist_scores[candidate] = float(observed_artists.get(song.artist, 0))
        type_scores[candidate] = float(observed_types.get(feature.primary_type, 0))
        language_scores[candidate] = float(observed_languages.get(feature.language, 0))

    return artist_scores, type_scores, language_scores


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
    semantic_index: SemanticTextIndex | None,
    cf_neighbors: dict[str, list[tuple[str, float]]],
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

    print_section("Preparing candidate scores")
    print("  candidate source: CF + popularity")
    for idx, case in enumerate(cases, start=1):
        exclude = set(case.observed)
        cold_start = len(case.observed) <= cold_start_threshold

        if cold_start:
            cold_count += 1
            candidates = {song_id for song_id in top_popular if song_id not in exclude}
            cf_norm: dict[str, float] = {}
        else:
            cf_scores = score_cf(case.observed, cf_neighbors, exclude, pool_size)
            candidates = set(cf_scores)
            candidates.update(song_id for song_id in top_popular if song_id not in exclude)
            cf_norm = normalize_scores(cf_scores, candidates)

        artist_scores, type_scores, language_scores = score_metadata(
            observed=case.observed,
            candidates=candidates,
            songs=songs,
            features=features,
        )
        semantic_scores: dict[str, float] = {}
        if semantic_index:
            semantic_profile = semantic_index.profile(case.observed[-short_window:], decay=time_decay)
            semantic_scores = semantic_index.score_candidates(semantic_profile, candidates, exclude)
        pop_raw = {song_id: float(popularity.get(song_id, 0)) for song_id in candidates}
        pop_norm = normalize_scores(pop_raw, candidates)
        artist_norm = normalize_scores(artist_scores, candidates)
        type_norm = normalize_scores(type_scores, candidates)
        language_norm = normalize_scores(language_scores, candidates)
        semantic_norm = normalize_scores(semantic_scores, candidates)
        retrieval = recall_at_k(list(candidates), set(case.heldout), len(candidates))
        prepared.append(
            PreparedCase(
                playlist_id=case.playlist_id,
                heldout=case.heldout,
                candidates=candidates,
                cf_norm=cf_norm,
                artist_norm=artist_norm,
                type_norm=type_norm,
                language_norm=language_norm,
                semantic_norm=semantic_norm,
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
    cf_weight: float,
    artist_weight: float,
    type_weight: float,
    language_weight: float,
    semantic_weight: float,
    pop_weight: float,
) -> tuple[dict[str, list[str]], float]:
    rankings: dict[str, list[str]] = {}
    retrieval_recalls: list[float] = []

    for case in prepared_cases:
        retrieval_recalls.append(case.retrieval)
        final_scores = {
            song_id: cf_weight * case.cf_norm.get(song_id, 0.0)
            + artist_weight * case.artist_norm.get(song_id, 0.0)
            + type_weight * case.type_norm.get(song_id, 0.0)
            + language_weight * case.language_norm.get(song_id, 0.0)
            + semantic_weight * case.semantic_norm.get(song_id, 0.0)
            + pop_weight * case.pop_norm.get(song_id, 0.0)
            for song_id in case.candidates
        }
        ranked = sorted(final_scores, key=lambda song_id: (-final_scores[song_id], song_id))
        rankings[case.playlist_id] = ranked[:k]

    retrieval = sum(retrieval_recalls) / len(retrieval_recalls) if retrieval_recalls else 0.0
    return rankings, retrieval


def print_section(title: str) -> None:
    print()
    print(title)
    print("=" * 20)


def print_dataset_study(songs: dict[str, Song], playlists: list[tuple[str, list[str]]], cases: list[EvalCase]) -> None:
    lengths = [len(tracks) for _, tracks in playlists]
    matched_tracks = sum(lengths)
    users = len(playlists)
    catalog = len(songs)
    density = matched_tracks / (users * catalog) if users and catalog else 0.0
    observed_lengths = [len(case.observed) for case in cases]
    heldout_lengths = [len(case.heldout) for case in cases]
    print_section("Dataset study")
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
    print_section("Song example")
    song = next(iter(songs.values()), None)
    if not song:
        print("  No songs loaded.")
        return
    print(f"  song_id: {song.song_id}")
    print(f"  title:   {song.title}")
    print(f"  artist:  {song.artist}")
    print("  lyrics:")
    print(f"    {preview_text(song.lyrics, limit=240)}")


def print_playlist_head(
    playlists: list[tuple[str, list[str]]],
    songs: dict[str, Song],
    limit: int = 10,
    track_limit: int = 10,
) -> None:
    print_section("Playlist example")
    if not playlists:
        print("  No playlist data loaded yet.")
        return
    playlist_id, track_ids = playlists[0]
    print(f"  playlist_id: {playlist_id}")
    print(f"  tracks:      {len(track_ids)}")
    print("  song list:")
    for idx, song_id in enumerate(track_ids[:track_limit], start=1):
        song = songs.get(song_id)
        label = f"{song.title} / {song.artist}" if song else song_id
        print(f"    {idx:>2}. {label}")
    if len(track_ids) > track_limit:
        print(f"    ... {len(track_ids) - track_limit:,} more tracks not shown")


def print_all_data_stats(songs: dict[str, Song], playlists: list[tuple[str, list[str]]]) -> None:
    playlist_lengths = [len(track_ids) for _, track_ids in playlists]
    unique_playlist_tracks = {song_id for _, track_ids in playlists for song_id in track_ids}
    lyric_token_counts = [len(tokenize(song.lyrics)) for song in songs.values()]
    print_section("All loaded data stats")
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
    return f"{song.title} / {song.artist} [type={feature.primary_type}, lang={feature.language}]"


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
    print_section("Example recommendation")
    print(f"  playlist_id: {case.playlist_id}")
    print("  input songs:")
    for song_id in case.observed[:limit]:
        print(f"    - {describe_song(song_id, songs, features)}")
    print("  hidden heldout songs:")
    for song_id in case.heldout[:limit]:
        print(f"    - {describe_song(song_id, songs, features)}")
    print("  recommended songs:")
    for song_id in rankings.get(case.playlist_id, [])[:limit]:
        print(f"    - {describe_song(song_id, songs, features)}")


def print_semantic_examples(
    songs: dict[str, Song],
    profiles: dict[str, SemanticProfile],
    limit: int = 5,
) -> None:
    if not profiles:
        return
    print_section("Semantic profile example")
    shown = 0
    for song_id in songs:
        profile = profiles.get(song_id)
        song = songs.get(song_id)
        if not profile or not song:
            continue
        shown += 1
        print(f"  song_id:            {song_id}")
        print(f"  title:              {song.title}")
        print(f"  artist:             {song.artist}")
        print(f"  language:           {profile.language}")
        print(f"  source:             {profile.source}")
        print(f"  semantic_summary:   {profile.semantic_summary}")
        print(f"  themes:             {', '.join(profile.themes)}")
        print(f"  lyrical_narrative:  {profile.lyrical_narrative}")
        print(f"  listening_context:  {profile.listening_context}")
        print(f"  playlist_function:  {profile.playlist_function}")
        print(f"  transition_note:    {profile.transition_note}")
        print(f"  keywords:           {', '.join(profile.keywords)}")
        print("  training_text:")
        print(f"    {profile.training_text}")
        if shown >= limit:
            break
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CF + structured semantic text playlist continuation prototype")
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
    parser.add_argument("--mood-drift-beta", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--cold-start-threshold", type=int, default=1, help="Use popularity-only ranking when observed history has this many songs or fewer")
    parser.add_argument("--progress-interval", type=int, default=1000, help="Print candidate-prep progress every N eval playlists")
    parser.add_argument("--cf-neighbors", type=int, default=100, help="Keep top N co-occurrence neighbors per song")
    parser.add_argument("--emotion-source", choices=["weak", "llm"], default="weak")
    parser.add_argument("--emotion-max-chars", type=int, default=1200)
    parser.add_argument("--emotion-limit", type=int, default=0, help="Only tag this many missing songs; 0 means all missing songs")
    parser.add_argument("--llm-max-new-tokens", type=int, default=320)
    parser.add_argument("--llm-batch-size", type=int, default=1, help="Batch LLM semantic profiling prompts; try 2 on 4GB GPUs")
    parser.add_argument("--llm-debug-output", type=int, default=0, help="Print raw LLM output for the first N newly profiled songs")
    parser.add_argument("--require-semantic-coverage", action="store_true", help="Fail instead of using weak fallback if any selected eval song lacks LLM semantic profile")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-cache-dir", type=Path, default=DEFAULT_LLM_CACHE_DIR)
    parser.add_argument("--semantic-cache-dir", type=Path, default=DEFAULT_SEMANTIC_CACHE_DIR)
    parser.add_argument("--llm-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--lyrics-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--cf-weight", type=float, default=0.35)
    parser.add_argument("--mood-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--artist-weight", type=float, default=0.10)
    parser.add_argument("--type-weight", type=float, default=0.05)
    parser.add_argument("--language-weight", type=float, default=0.02)
    parser.add_argument("--semantic-weight", type=float, default=0.15)
    parser.add_argument("--no-ablations", action="store_true", help="Skip cheap post-sweep ablation comparisons")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--pop-weight", type=float, default=0.10)
    parser.add_argument("--alpha-grid-step", type=float, default=0.1, help=argparse.SUPPRESS)
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

    print_song_head(songs, limit=10)
    print_playlist_head(playlists, songs, limit=10)
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
    print_section("Building song feature table")
    features = build_song_features(songs)
    semantic_profiles: dict[str, SemanticProfile] = {}
    if args.emotion_source == "llm":
        _, semantic_profiles = build_llm_song_features(
            songs=songs,
            model_name=args.llm_model,
            llm_cache_dir=args.llm_cache_dir,
            semantic_cache_dir=args.semantic_cache_dir,
            max_chars=args.emotion_max_chars,
            limit=args.emotion_limit,
            progress_interval=args.progress_interval,
            device=args.llm_device,
            max_new_tokens=args.llm_max_new_tokens,
            debug_output=args.llm_debug_output,
            batch_size=args.llm_batch_size,
        )
        missing_semantic = set(songs) - set(semantic_profiles)
        if missing_semantic:
            if args.require_semantic_coverage:
                raise SystemExit(
                    "LLM semantic coverage is incomplete. "
                    f"Missing {len(missing_semantic):,} selected eval songs. "
                    "Rerun with --emotion-limit 0 or a larger limit, or omit --require-semantic-coverage for weak fallback."
                )
            print(f"  LLM semantic profiles incomplete; semantic-text score disabled for {len(missing_semantic):,} songs.")
    print(f"  emotion source:         {args.emotion_source}")
    type_counts = Counter(feature.primary_type for feature in features.values())
    language_counts = Counter(feature.language for feature in features.values())
    print("  top lyric types:         " + ", ".join(f"{k}={v:,}" for k, v in type_counts.most_common(5)))
    print("  languages:               " + ", ".join(f"{k}={v:,}" for k, v in language_counts.most_common(5)))
    if semantic_profiles:
        source_counts = Counter(profile.source for profile in semantic_profiles.values())
        theme_counts: Counter[str] = Counter()
        role_counts: Counter[str] = Counter()
        for profile in semantic_profiles.values():
            theme_counts.update(profile.themes)
            role_counts[profile.playlist_function] += 1
        print("  semantic sources:       " + ", ".join(f"{k}={v:,}" for k, v in source_counts.most_common()))
        print("  top semantic themes:     " + ", ".join(f"{k}={v:,}" for k, v in theme_counts.most_common(5)))
        print("  playlist functions:      " + ", ".join(f"{k}={v:,}" for k, v in role_counts.most_common(5)))
        print_semantic_examples(songs, semantic_profiles, limit=5)
    print_section("Building co-occurrence CF neighbors")
    cf_neighbors = build_cf_neighbors(cases, args.cf_neighbors)
    neighbor_edges = sum(len(items) for items in cf_neighbors.values())
    print(f"  songs with neighbors:    {len(cf_neighbors):,}")
    print(f"  stored neighbor edges:   {neighbor_edges:,}")
    semantic_index: SemanticTextIndex | None = None
    if semantic_profiles:
        print_section("Building structured semantic text index")
        semantic_index = SemanticTextIndex(semantic_profiles, args.min_df, args.max_features)
        print(f"  semantic vocab size:     {len(semantic_index.vocab):,}")
        print(f"  indexed semantic songs:  {len(semantic_index.item_vectors):,}")
    else:
        print_section("Building structured semantic text index")
        print("No structured semantic profiles found; semantic-text score is disabled.")

    prepared_cases = prepare_content_cases(
        cases=cases,
        songs=songs,
        features=features,
        semantic_index=semantic_index,
        cf_neighbors=cf_neighbors,
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

    print_section("Baselines")
    print("  random")
    print(f"    Recall@{args.top_k}: {random_recall:.5f}")
    print(f"    NDCG@{args.top_k}:   {random_ndcg:.5f}")
    print("  popularity")
    print(f"    Recall@{args.top_k}: {pop_recall:.5f}")
    print(f"    NDCG@{args.top_k}:   {pop_ndcg:.5f}")
    result_rows: list[dict[str, float | str]] = [
        result_row("Random", "", random_recall, random_ndcg),
        result_row("Popularity", "", pop_recall, pop_ndcg),
    ]

    cf_pop_row, _, _ = evaluate_prepared_model(
        model="CF + popularity baseline",
        prepared_cases=prepared_cases,
        cases=cases,
        top_k=args.top_k,
        cf_weight=args.cf_weight,
        artist_weight=0.0,
        type_weight=0.0,
        language_weight=0.0,
        semantic_weight=0.0,
        pop_weight=args.pop_weight,
    )
    result_rows.append(cf_pop_row)
    print("  CF + popularity")
    print(f"    Recall@{args.top_k}: {float(cf_pop_row['recall']):.5f}")
    print(f"    NDCG@{args.top_k}:   {float(cf_pop_row['ndcg']):.5f}")

    hybrid_label = "Hybrid structured semantic"
    print_section("Structured semantic rerank")
    rankings, retrieval = build_content_rankings(
        k=args.top_k,
        prepared_cases=prepared_cases,
        cf_weight=args.cf_weight,
        artist_weight=args.artist_weight,
        type_weight=args.type_weight,
        language_weight=args.language_weight,
        semantic_weight=args.semantic_weight,
        pop_weight=args.pop_weight,
    )
    recall, ndcg = evaluate_rankings(rankings, cases, args.top_k)
    proxy = (recall + ndcg) / 2.0
    print(f"  Retrieval@{args.pool_size}: {retrieval:.5f}")
    print(f"  Recall@{args.top_k}:    {recall:.5f}")
    print(f"  NDCG@{args.top_k}:      {ndcg:.5f}")
    print(f"  Proxy:         {proxy:.5f}")
    result_rows.append(result_row(hybrid_label, retrieval, recall, ndcg))

    print_section("Best prototype result")
    print("  method:                 hybrid_cf_metadata_structured_semantic")
    print(f"  time decay:             {args.time_decay:.2f}")
    print(f"  weights:                cf={args.cf_weight:.2f}, artist={args.artist_weight:.2f}, type={args.type_weight:.2f}, language={args.language_weight:.2f}, semantic_text={args.semantic_weight:.2f}, pop={args.pop_weight:.2f}")
    print(f"  Retrieval@{args.pool_size}:          {retrieval:.5f}")
    print(f"  Recall@{args.top_k}:             {recall:.5f}")
    print(f"  NDCG@{args.top_k}:               {ndcg:.5f}")
    print(f"  Proxy:                 {proxy:.5f}")
    result_rows.append(result_row(f"Best {hybrid_label}", retrieval, recall, ndcg))
    if not args.no_ablations:
        print_section("Ablation comparisons")
        ablations = [
            ("No CF", 0.0, args.artist_weight, args.type_weight, args.language_weight, args.semantic_weight, args.pop_weight),
            ("No metadata", args.cf_weight, 0.0, 0.0, 0.0, args.semantic_weight, args.pop_weight),
            ("No structured semantic", args.cf_weight, args.artist_weight, args.type_weight, args.language_weight, 0.0, args.pop_weight),
            ("CF + popularity", args.cf_weight, 0.0, 0.0, 0.0, 0.0, args.pop_weight),
        ]
        for name, cf_w, artist_w, type_w, language_w, semantic_w, pop_w in ablations:
            row, _, _ = evaluate_prepared_model(
                model=name,
                prepared_cases=prepared_cases,
                cases=cases,
                top_k=args.top_k,
                cf_weight=cf_w,
                artist_weight=artist_w,
                type_weight=type_w,
                language_weight=language_w,
                semantic_weight=semantic_w,
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
