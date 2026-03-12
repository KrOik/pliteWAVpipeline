"""
Audio caching module for pliteWAVpipeline.

This module handles audio format conversion, resampling, and caching to standard PCM format.
"""

import glob
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio

from .utils import (
    get_stable_name,
    load_audio_robust,
    scan_files,
    resample_if_needed,
    is_fake_stereo,
    ensure_channels,
    DEFAULT_SAMPLE_RATE,
)


__all__ = [
    "cache_audio_files",
    "get_stable_name",
    "load_audio_robust",
    "scan_files",
]


def cache_audio_files(
    input_files: List[str],
    output_dir: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    check_quality: bool = False,
    force_stereo: bool = True,
) -> int:
    """
    Process and cache audio files to standard format.
    
    Converts various audio formats to standardized PCM 16-bit WAV files.
    
    Args:
        input_files: List of file paths to process
        output_dir: Directory to save cached WAVs and manifest
        sample_rate: Target sample rate (default: 48000)
        check_quality: If True, log but process low quality files anyway
        force_stereo: If True, convert mono to stereo
        
    Returns:
        Number of files successfully processed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "wav_cache"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    
    fp = manifest_path.open("w", encoding="utf-8")
    
    target_sr = int(sample_rate)
    processed_count = 0
    
    for src in input_files:
        try:
            # Quality Check (informational only)
            if check_quality:
                try:
                    info = torchaudio.info(src)
                    if info.sample_rate < 44100:
                        print(f"Processing low quality file (SR={info.sample_rate}): {src}")
                except Exception:
                    pass

            wav, sr = load_audio_robust(src, target_sr)
            
            # Smart Channel Processing
            if is_fake_stereo(wav):
                # Convert Fake Stereo to Mono to avoid sample contamination
                wav = wav.mean(dim=0, keepdim=True)
            
            if force_stereo:
                wav = ensure_channels(wav, target_channels=2)
            else:
                # Keep original channels (mono stays mono, stereo stays stereo)
                pass
            
            wav, _ = resample_if_needed(wav, int(sr), target_sr)
            wav = wav.detach().cpu().clamp(-1.0, 1.0)

            name = get_stable_name(src)
            out_path = wav_dir / name
            torchaudio.save(str(out_path), wav, target_sr, encoding="PCM_S", bits_per_sample=16)
            
            entry = {
                "path": f"wav_cache/{name}",
                "src": str(src),
                "sample_rate": int(target_sr),
                "channels": int(wav.shape[0]),
                "num_frames": int(wav.shape[-1]),
            }
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
            processed_count += 1
            print(f"Processed: {src} (Ch: {wav.shape[0]})")
            
        except Exception as e:
            print(f"Failed to process {src}: {e}")

    fp.close()
    return processed_count


def cache_audio_directory(
    data_dirs: str,
    output_dir: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    extensions: Optional[List[str]] = None,
    **kwargs
) -> int:
    """
    Convenience function to cache all audio files in directories.
    
    Args:
        data_dirs: Comma-separated directory paths
        output_dir: Directory to save cached WAVs and manifest
        sample_rate: Target sample rate
        extensions: List of file extensions to include
        
    Returns:
        Number of files successfully processed
    """
    if extensions is None:
        extensions = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".wma", ".xm"]
    
    files = []
    for d in data_dirs.split(","):
        d = d.strip()
        if not d:
            continue
        for ext in extensions:
            files.extend(glob.glob(os.path.join(d, f"**/*{ext}"), recursive=True))
    
    files = sorted(list(set(files)))
    return cache_audio_files(files, output_dir, sample_rate, **kwargs)


# CLI support
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Cache audio as stereo PCM16 WAVs for fast training IO")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data_dirs", type=str, help="Comma separated audio directories")
    group.add_argument("--input_files", type=str, help="Comma separated audio file paths")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory root")
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Target sample rate")
    parser.add_argument("--check_quality", action="store_true", help="Log low quality files")
    parser.add_argument("--mono", action="store_true", help="Keep mono instead of forcing stereo")
    args = parser.parse_args()

    if args.input_files is not None:
        files = [p.strip() for p in str(args.input_files).split(",") if p.strip()]
    else:
        files = scan_files(args.data_dirs)

    count = cache_audio_files(
        files, 
        args.output_dir, 
        args.sample_rate, 
        check_quality=args.check_quality,
        force_stereo=not args.mono,
    )
    print(f"Successfully processed {count} files.")


if __name__ == "__main__":
    main()