#!/usr/bin/env python3
"""Install/cache a small local LLM for playlist semantic experiments.

The model cache intentionally lives inside this project by default:

    models/llm_cache/

This keeps large Hugging Face model files off the C: drive on Windows machines.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = HERE / "models" / "llm_cache"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

MODEL_PRESETS = {
    "qwen1b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "gemma270m": "google/gemma-3-270m-it",
}


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
        "Classify the playlist. Return only JSON.\n"
        "Allowed era: 60s, 70s, 80s, 90s, 2000s, 2010s, mixed.\n"
        "Allowed style: rock, alt_rock, classic_rock, pop, dance_pop, hiphop, rnb, country, soul, folk, metal, mixed.\n"
        "Allowed context: party, wedding, halloween, workout, roadtrip, chill, slow_dance, nostalgia, singalong, background, mixed.\n"
        "Allowed energy: low, mid, high.\n"
        "Schema keys: era, style, context, energy.\n"
        "Playlist name: 90s\n"
        "Tracks: Creep / Radiohead; Even Flow / Pearl Jam; Basket Case / Green Day; Black Hole Sun / Soundgarden.\n"
        "JSON:"
    )


def configure_hf_cache(cache_dir: Path) -> Path:
    hub_cache = cache_dir / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    return hub_cache


def resolve_model(model: str, preset: str | None) -> str:
    if preset:
        return MODEL_PRESETS[preset]
    return MODEL_PRESETS.get(model, model)


def download_model(model_name: str, cache_dir: Path) -> None:
    hub_cache = configure_hf_cache(cache_dir)
    from huggingface_hub import snapshot_download

    print(f"HF_HOME:      {cache_dir}")
    print(f"HF_HUB_CACHE: {hub_cache}")
    print(f"Model:        {model_name}")
    print("Downloading model files into the project cache...")
    snapshot_path = snapshot_download(repo_id=model_name, cache_dir=hub_cache)
    print(f"Snapshot:     {snapshot_path}")


def download_and_test(model_name: str, cache_dir: Path, device: str, download_only: bool) -> None:
    download_model(model_name, cache_dir)
    if download_only:
        print("Download-only mode; skipping model load smoke test.")
        return

    import torch
    from transformers import pipeline

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested, but this Python environment has CPU-only PyTorch.")
        print("Run: python .\\getLLM.py --cuda-torch --device cuda")
        raise SystemExit(2)
    print(f"Device:       {resolved_device}")
    if resolved_device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:          {props.name} ({props.total_memory / (1024 ** 3):.2f} GB)")

    try:
        pipe = pipeline(
            "text-generation",
            model_name,
            model_kwargs={"dtype": "auto"},
            device=0 if resolved_device == "cuda" else -1,
        )
    except Exception as exc:
        message = str(exc)
        print()
        print("Could not load the model for the smoke test.")
        print("The files may still have downloaded successfully.")
        print("If CUDA runs out of memory, retry with --device cpu or --download-only.")
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
    parser = argparse.ArgumentParser(description="Install/cache optional local LLM")
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), help="Convenient model preset")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--download-only", action="store_true", help="Download/cache files without loading the model")
    parser.add_argument("--skip-install", action="store_true", help="Only load/test the model; do not pip install dependencies")
    parser.add_argument("--cpu-torch", action="store_true", help="Install torch from the official CPU wheel index if torch is missing")
    parser.add_argument("--cuda-torch", action="store_true", help="Force reinstall torch from the official CUDA 12.8 wheel index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_install:
        ensure_dependencies(cpu_torch=args.cpu_torch, cuda_torch=args.cuda_torch)
    model_name = resolve_model(args.model, args.preset)
    download_and_test(model_name, args.cache_dir.resolve(), args.device, args.download_only)
    print()
    print("Local LLM ready.")


if __name__ == "__main__":
    main()
