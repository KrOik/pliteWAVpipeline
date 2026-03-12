"""
Shared utility functions for pliteWAVpipeline.

This module provides common utilities used across the audio processing pipeline:
- Audio channel handling (fake stereo detection, channel normalization)
- Audio resampling
- Atomic file operations
"""

import os
import glob
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torchaudio


def is_fake_stereo(wav: torch.Tensor, threshold: float = 0.99) -> bool:
    """
    Check if stereo audio is actually mono (channels are identical).
    
    Args:
        wav: Audio tensor of shape (C, T)
        threshold: Correlation threshold for fake stereo detection
        
    Returns:
        True if audio is fake stereo (mono duplicated to 2 channels)
    """
    if wav.shape[0] < 2:
        return False
    
    # Check first two channels
    c1 = wav[0].float()
    c2 = wav[1].float()
    
    # Optimization: Check L1 distance first
    diff = (c1 - c2).abs().mean()
    if diff < 1e-6:
        return True
        
    # Check Correlation
    c1_ctr = c1 - c1.mean()
    c2_ctr = c2 - c2.mean()
    norm = torch.sqrt((c1_ctr**2).sum() * (c2_ctr**2).sum())
    if norm < 1e-6:
        return True # Silence
        
    corr = (c1_ctr * c2_ctr).sum() / norm
    return corr.item() > threshold


def ensure_channels(wav: torch.Tensor, target_channels: int = 2) -> torch.Tensor:
    """
    Ensure audio tensor has target_channels.
    
    - Mono -> Duplicate
    - Multi -> Crop (or TODO: Downmix)
    
    Args:
        wav: Audio tensor of shape (C, T)
        target_channels: Desired number of channels
        
    Returns:
        Audio tensor with target_channels
    """
    if wav.ndim != 2:
        raise ValueError(f"Expected (C, T), got {tuple(wav.shape)}")
        
    C, T = wav.shape
    if C == target_channels:
        return wav
        
    if C < target_channels:
        # Duplicate channels to fill
        # e.g. 1 -> 2: [0, 0]
        # e.g. 1 -> 4: [0, 0, 0, 0]
        return wav.repeat(target_channels // C + 1, 1)[:target_channels, :]
        
    if C > target_channels:
        # Crop
        return wav[:target_channels, :]
        
    return wav


def force_stereo(wav: torch.Tensor) -> torch.Tensor:
    """Deprecated: Use ensure_channels instead. Kept for backward compatibility."""
    return ensure_channels(wav, 2)


def resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> Tuple[torch.Tensor, int]:
    """
    Resample audio if sample rate differs from target.
    
    Args:
        wav: Audio tensor of shape (C, T)
        sr: Current sample rate
        target_sr: Target sample rate
        
    Returns:
        Tuple of (resampled_audio, actual_sample_rate)
    """
    if int(sr) == int(target_sr):
        return wav, int(sr)
    wav = torchaudio.functional.resample(wav, int(sr), int(target_sr))
    return wav, int(target_sr)


def atomic_write_text(path: Path, text: str):
    """
    Write text to file atomically using rename.
    
    Args:
        path: Destination file path
        text: Content to write
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def atomic_torch_save(obj, path: Path):
    """
    Save torch object to file atomically using temporary file.
    
    Args:
        obj: Object to save (torch tensor, dict, etc.)
        path: Destination file path
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(str(tmp), str(path))


def get_stable_name(src_path: str | Path) -> str:
    """
    Generate a stable filename based on source path hash.
    
    Args:
        src_path: Source file path
        
    Returns:
        Stable filename with hash prefix
    """
    p = Path(src_path)
    stem = p.stem
    h = hashlib.sha1(str(src_path).encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{h}.wav"


def scan_files(data_dirs: str, exts: List[str] = None) -> List[str]:
    """
    Scan directories for audio files with given extensions.
    
    Args:
        data_dirs: Comma-separated directory paths
        exts: List of file extensions to include
        
    Returns:
        Sorted list of absolute file paths
    """
    if exts is None:
        exts = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".wma", ".xm"]
    
    files = []
    for d in data_dirs.split(","):
        d = d.strip()
        if not d:
            continue
        for ext in exts:
            files.extend(glob.glob(os.path.join(d, f"**/*{ext}"), recursive=True))
    return sorted(list(set(files)))


def load_audio_robust(path: str | Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    """
    Load audio from various formats, using ffmpeg as fallback.
    
    Supports: WAV, FLAC, MP3, M4A, OGG, WMA, XM, etc.
    
    Args:
        path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    path = str(path)
    try:
        wav, sr = torchaudio.load(path)
    except Exception:
        # Fallback to ffmpeg for problematic formats
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", path,
                "-ar", str(target_sr),
                "-f", "wav",
                tmp_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wav, sr = torchaudio.load(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    return wav, sr


# Default constants
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_MIN_SEGMENT_S = 5.0
DEFAULT_MAX_SEGMENT_S = 10.0
DEFAULT_SILENCE_THRESHOLD_DB = -30.0
DEFAULT_ANALYSIS_FRAME_MS = 30.0
DEFAULT_ANALYSIS_HOP_MS = 10.0
DEFAULT_MIN_SILENCE_MS = 500.0
PT_PCM_SCALE = 32768.0