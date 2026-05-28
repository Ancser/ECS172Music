#!/usr/bin/env python3
"""Install/cache the smallest Gemma 3 LLM for optional lyric emotion tests."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = HERE / "models" / "llm_cache"
DEFAULT_MODEL = "google/gemma-3-270m-it"


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def ensure_dependencies(cpu_torch: bool, cuda_torch: bool) -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--user",
            "--upgrade",
            "transformers",
            "safetensors",
            "accelerate",
            "huggingface_hub",
        ]
    )
    if cuda_torch:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--user",
                "--upgrade",
                "--force-reinstall",
                "torch",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ]
        )
    elif not module_available("torch"):
        if cpu_torch:
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--user",
                    "--upgrade",
                    "torch",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                ]
            )
        else:
            run([sys.executable, "-m", "pip", "install", "--user", "--upgrade", "torch"])


def prompt_text() -> str:
    return (
        "Convert this song into a controlled semantic profile for a music recommendation experiment.\n"
        "Return only one JSON object. Do not output numeric scores.\n"
        "Schema keys: dominant_emotion, secondary_emotion, valence, arousal, theme_tags, emotion_arc, playlist_role.\n"
        "Allowed emotions: love, sadness, energy, calm, hope, anger, nostalgia, neutral.\n"
        "Allowed valence: negative, mixed, positive. Allowed arousal: low, medium, high.\n"
        "Lyrics: I miss you every night, but I still hope the morning brings your smile back.\n"
        "JSON:"
    )


def download_and_test(model_name: str, cache_dir: Path, device: str) -> None:
    hub_cache = cache_dir / "hub"
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    hub_cache.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import pipeline

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested, but this Python environment has CPU-only PyTorch.")
        print("Run: python .\\install_llm.py --cuda-torch --device cuda")
        raise SystemExit(2)
    print(f"HF_HUB_CACHE: {hub_cache}")
    print(f"Model:   {model_name}")
    print(f"Device:  {resolved_device}")
    if resolved_device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:     {props.name} ({props.total_memory / (1024 ** 3):.2f} GB)")
        print("Note: this project defaults to the 270M Gemma 3 model for small GPUs.")

    try:
        pipe = pipeline(
            "text-generation",
            model_name,
            device=0 if resolved_device == "cuda" else -1,
        )
    except Exception as exc:
        message = str(exc)
        print()
        print("Could not load the Gemma model.")
        print("Most common reason: Gemma is gated on Hugging Face.")
        print("Fix:")
        print("  1. Open https://huggingface.co/google/gemma-3-270m-it")
        print("  2. Accept the Google usage license")
        print("  3. Run: python -m huggingface_hub.cli.hf auth login")
        print("  4. Re-run this script")
        print()
        print(message[:1200])
        raise SystemExit(2) from exc

    output = pipe(
        [{"role": "user", "content": prompt_text()}],
        max_new_tokens=96,
        do_sample=False,
    )
    print()
    print("Smoke test output:")
    print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install/cache optional Gemma 3 LLM")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-install", action="store_true", help="Only load/test the model; do not pip install dependencies")
    parser.add_argument("--cpu-torch", action="store_true", help="Install torch from the official CPU wheel index if torch is missing")
    parser.add_argument("--cuda-torch", action="store_true", help="Force reinstall torch from the official CUDA 12.8 wheel index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_install:
        ensure_dependencies(cpu_torch=args.cpu_torch, cuda_torch=args.cuda_torch)
    download_and_test(args.model, args.cache_dir.resolve(), args.device)
    print()
    print("Gemma LLM ready.")


if __name__ == "__main__":
    main()
