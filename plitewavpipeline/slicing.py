"""
Audio slicing module for pliteWAVpipeline.

This module provides voice activity detection (VAD) and intelligent audio segmentation.
Supports both fixed-length and variable-length segment extraction based on silence detection.
"""

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio

from .utils import (
    scan_files,
    force_stereo,
    resample_if_needed,
    atomic_write_text,
    atomic_torch_save,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_MIN_SEGMENT_S,
    DEFAULT_MAX_SEGMENT_S,
    DEFAULT_SILENCE_THRESHOLD_DB,
    DEFAULT_ANALYSIS_FRAME_MS,
    DEFAULT_ANALYSIS_HOP_MS,
    DEFAULT_MIN_SILENCE_MS,
    PT_PCM_SCALE,
)


__all__ = [
    "EnergyVAD",
    "FenwickTree", 
    "cut_segments",
    "scan_files",
]


class EnergyVAD:
    """
    Voice Activity Detection based on energy threshold.
    
    Uses RMS energy analysis to detect voiced regions in audio.
    """
    
    def __init__(
        self, 
        sample_rate: int = DEFAULT_SAMPLE_RATE, 
        frame_ms: float = DEFAULT_ANALYSIS_FRAME_MS, 
        hop_ms: float = DEFAULT_ANALYSIS_HOP_MS, 
        threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB
    ):
        self.sample_rate = int(sample_rate)
        self.frame_len = max(1, int(round(float(frame_ms) * self.sample_rate / 1000.0)))
        self.hop_len = max(1, int(round(float(hop_ms) * self.sample_rate / 1000.0)))
        self.threshold_db = float(threshold_db)

    def detect(self, wav: torch.Tensor) -> Tuple[np.ndarray, int, int]:
        """
        Detect voice activity using RMS energy.
        
        Args:
            wav: Audio tensor of shape (C, T) or (T,)
            
        Returns:
            Tuple of (voiced_mask, frame_len, hop_len)
            - voiced_mask: Boolean array indicating voiced frames
            - frame_len: Analysis frame length in samples
            - hop_len: Analysis hop length in samples
        """
        if wav.ndim > 1:
            x = wav.mean(dim=0).numpy()
        else:
            x = wav.numpy()
        
        n = x.size
        if n < self.frame_len:
            return np.ones((0,), dtype=bool), self.frame_len, self.hop_len
            
        # Optimized RMS calculation using cumulative sum of squares
        sq = x.astype(np.float64) ** 2
        cum_sq = np.concatenate(([0.0], np.cumsum(sq)))
        
        n_frames = 1 + (n - self.frame_len) // self.hop_len
        starts = np.arange(n_frames) * self.hop_len
        ends = starts + self.frame_len
        
        # Mean Square = Sum Square / N
        energies = (cum_sq[ends] - cum_sq[starts]) / self.frame_len
        
        # dB = 10 * log10(Mean Square) = 20 * log10(RMS)
        rms_db = 10.0 * np.log10(np.maximum(energies, 1e-12))
        
        mask = (rms_db > self.threshold_db)
        return mask, self.frame_len, self.hop_len


class FenwickTree:
    """
    Binary Indexed Tree for efficient prefix sum queries and kth element finding.
    
    Used in the dynamic programming segment search algorithm.
    """
    
    def __init__(self, n: int):
        self.n = int(n)
        self.bit = [0] * (self.n + 1)

    def add(self, idx_1based: int, delta: int):
        i = int(idx_1based)
        while i <= self.n:
            self.bit[i] += int(delta)
            i += i & -i

    def sum(self, idx_1based: int) -> int:
        s = 0
        i = int(idx_1based)
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s

    def find_by_prefix(self, prefix_sum: int) -> int:
        """Find smallest index i such that sum(1..i) >= prefix_sum."""
        target = int(prefix_sum)
        if target <= 0:
            return 0
        idx = 0
        bit_mask = 1 << (self.n.bit_length() - 1)
        while bit_mask:
            t = idx + bit_mask
            if t <= self.n and self.bit[t] < target:
                idx = t
                target -= self.bit[t]
            bit_mask >>= 1
        return idx + 1


def _silence_cut_positions(
    voiced_mask: np.ndarray,
    hop_len: int,
    n_samples: int,
    min_silence_frames: int,
) -> List[int]:
    """
    Find cut positions at silence boundaries.
    
    Args:
        voiced_mask: Boolean array indicating voiced frames
        hop_len: Analysis hop length in samples
        n_samples: Total audio samples
        min_silence_frames: Minimum silence frames to trigger a cut
        
    Returns:
        List of cut positions in samples
    """
    silence = ~voiced_mask
    if silence.size == 0 or not bool(silence.any()):
        return []
    min_silence_frames = int(min_silence_frames)
    if min_silence_frames <= 0:
        min_silence_frames = 1

    n = int(silence.size)
    out: List[int] = []
    i = 0
    while i < n:
        if not bool(silence[i]):
            i += 1
            continue
        j = i + 1
        while j < n and bool(silence[j]):
            j += 1
        if (j - i) >= min_silence_frames:
            mid = (i + j) // 2
            p = int(mid * int(hop_len) + int(hop_len) // 2)
            if 0 < p < int(n_samples):
                out.append(p)
        i = j
    return sorted(set(out))


def _segment_by_cutpoints(
    n_samples: int,
    cut_positions: List[int],
    min_len: int,
    max_len: int,
) -> Optional[List[Tuple[int, int]]]:
    """
    Find optimal segment boundaries using dynamic programming.
    
    Uses Fenwick tree for O(n log n) segment search.
    
    Args:
        n_samples: Total audio samples
        cut_positions: List of valid cut positions
        min_len: Minimum segment length in samples
        max_len: Maximum segment length in samples
        
    Returns:
        List of (start, end) tuples, or None if impossible
    """
    n_samples = int(n_samples)
    min_len = int(min_len)
    max_len = int(max_len)
    if n_samples <= 0:
        return []
    if min_len <= 0 or max_len <= 0 or min_len > max_len:
        raise ValueError("Invalid min/max segment length.")

    positions = sorted(set([0] + [int(p) for p in cut_positions if 0 < int(p) < n_samples] + [n_samples]))
    m = len(positions)
    reachable = [False] * m
    next_idx = [-1] * m

    bit = FenwickTree(m)
    reachable[m - 1] = True
    bit.add(m, 1)

    for i in range(m - 2, -1, -1):
        start = positions[i]
        lo = start + min_len
        hi = start + max_len
        j_min = bisect_left(positions, lo, i + 1, m)
        j_max = bisect_right(positions, hi, i + 1, m) - 1
        if j_min > j_max:
            continue

        cnt = bit.sum(j_max + 1) - bit.sum(j_min)
        if cnt <= 0:
            continue

        total = bit.sum(j_max + 1)
        j = bit.find_by_prefix(total) - 1
        if j < j_min:
            continue

        reachable[i] = True
        next_idx[i] = j
        bit.add(i + 1, 1)

    if not reachable[0]:
        return None

    out: List[Tuple[int, int]] = []
    i = 0
    while i != m - 1:
        j = next_idx[i]
        if j <= i:
            return None
        out.append((positions[i], positions[j]))
        i = j
    return out


def _tile_to_len(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Tile audio to reach target length."""
    target_len = int(target_len)
    if int(x.shape[-1]) == target_len:
        return x
    if int(x.shape[-1]) <= 0:
        return torch.zeros((int(x.shape[0]), target_len), dtype=x.dtype, device=x.device)
    while int(x.shape[-1]) < target_len:
        x = torch.cat([x, x], dim=-1)
    return x[:, :target_len].contiguous()


def cut_segments(
    data_dirs: str,
    output_dir: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    min_segment_s: float = DEFAULT_MIN_SEGMENT_S,
    max_segment_s: float = DEFAULT_MAX_SEGMENT_S,
    output_format: str = "pt",
    overwrite: bool = False,
    resume: bool = False,
    silence_threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
    analysis_frame_ms: float = DEFAULT_ANALYSIS_FRAME_MS,
    analysis_hop_ms: float = DEFAULT_ANALYSIS_HOP_MS,
    min_silence_ms: float = DEFAULT_MIN_SILENCE_MS,
    segments_per_shard: int = 256,
) -> dict:
    """
    Cut audio files into segments using VAD.
    
    Args:
        data_dirs: Comma-separated audio directories
        output_dir: Output directory for shards and index
        sample_rate: Target sample rate
        min_segment_s: Minimum segment duration in seconds
        max_segment_s: Maximum segment duration in seconds
        output_format: "pt", "wav", or "both"
        overwrite: Overwrite existing output
        resume: Resume from existing output
        silence_threshold_db: VAD energy threshold in dB
        analysis_frame_ms: VAD analysis frame length
        analysis_hop_ms: VAD analysis hop length
        min_silence_ms: Minimum silence to trigger cut
        segments_per_shard: Number of segments per saved shard file
        
    Returns:
        Dictionary with processing statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wav_dir = output_dir / "wav"
    pt_dir = output_dir / "pt"
    
    if output_format in ("wav", "both"):
        wav_dir.mkdir(parents=True, exist_ok=True)
    if output_format in ("pt", "both"):
        pt_dir.mkdir(parents=True, exist_ok=True)
        
    state_path = output_dir / "state.json"
    processed_path = output_dir / "processed.jsonl"
    manifest_path = output_dir / "manifest.jsonl"

    # Check for existing output
    has_existing = False
    if pt_dir.exists() and any(pt_dir.glob("segments_*.pt")):
        has_existing = True
    if wav_dir.exists() and any(wav_dir.glob("*.wav")):
        has_existing = True
    if (output_dir / "index.pt").exists():
        has_existing = True
    if (output_dir / "manifest.jsonl").exists():
        has_existing = True
        
    if has_existing and not resume and not overwrite:
        raise RuntimeError("output_dir is not empty; use --resume or --overwrite to proceed.")

    if overwrite and has_existing:
        if pt_dir.exists():
            for p in pt_dir.glob("segments_*.pt"):
                p.unlink(missing_ok=True)
            for p in pt_dir.glob("segments_*.pt.tmp"):
                p.unlink(missing_ok=True)
        if wav_dir.exists():
            for p in wav_dir.glob("*.wav"):
                p.unlink(missing_ok=True)
        (output_dir / "index.pt").unlink(missing_ok=True)
        (output_dir / "manifest.jsonl").unlink(missing_ok=True)
        processed_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    # Initialize manifest
    if not resume and not overwrite:
        if not manifest_path.exists():
            manifest_path.write_text("", encoding="utf-8")
    elif not resume and overwrite:
        manifest_path.write_text("", encoding="utf-8")
    if resume and not manifest_path.exists():
        manifest_path.write_text("", encoding="utf-8")

    # Scan files
    files = scan_files(data_dirs)
    target_sr = int(sample_rate)
    
    min_segment_samples = int(round(min_segment_s * float(target_sr)))
    max_segment_samples = int(round(max_segment_s * float(target_sr)))
    
    if min_segment_samples <= 0 or max_segment_samples <= 0:
        raise ValueError("min/max segment must be positive.")
    if min_segment_samples > max_segment_samples:
        raise ValueError("min segment must be <= max segment.")
    if output_format in ("pt", "both") and min_segment_samples != max_segment_samples:
        raise ValueError("pt output requires fixed-length segments (min == max).")

    # Initialize VAD
    vad = EnergyVAD(
        sample_rate=target_sr,
        frame_ms=analysis_frame_ms,
        hop_ms=analysis_hop_ms,
        threshold_db=silence_threshold_db
    )

    frame_len = max(1, int(round(float(analysis_frame_ms) * target_sr / 1000.0)))
    hop_len = max(1, int(round(float(analysis_hop_ms) * target_sr / 1000.0)))
    min_silence_frames = max(1, int(round(float(min_silence_ms) / float(analysis_hop_ms))))

    # Processing state
    shards = []
    buffer = []
    buffer_scores = []
    buffer_meta = []
    shard_idx = 0
    wav_idx = 0
    processed_sig: dict = {}
    last_file_i = 0
    
    segment_samples = min_segment_samples if min_segment_samples == max_segment_samples else None
    fixed_segment_len = int(min_segment_samples) if min_segment_samples == max_segment_samples else None

    # Resume support
    if resume and state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            shard_idx = int(state.get("shard_idx", 0))
            wav_idx = int(state.get("wav_idx", 0))
            shards = list(state.get("shards", []))
            last_file_i = int(state.get("last_file_i", 0))
        except Exception:
            state = None

    def flush():
        nonlocal shard_idx, buffer, buffer_scores, buffer_meta, shards
        if not buffer:
            return
        print(f"Flushing shard {shard_idx} ({len(buffer)} segments)...")
        
        audio_f32 = torch.stack(buffer, dim=0).detach().cpu().clamp(-1.0, 1.0)
        audio = (audio_f32 * float(PT_PCM_SCALE)).round().to(torch.int16)
        scores = torch.tensor(buffer_scores, dtype=torch.float32)

        name = f"segments_{shard_idx:05d}.pt"
        atomic_torch_save(
            {
                "audio": audio,
                "scores": scores,
                "sample_rate": target_sr,
                "pcm_scale": float(PT_PCM_SCALE),
            },
            pt_dir / name,
        )
        
        # Write metadata
        with manifest_path.open("a", encoding="utf-8") as fp:
            for idx, meta in enumerate(buffer_meta):
                meta["path"] = f"pt/{name}"
                meta["shard_index"] = int(idx)
                fp.write(json.dumps(meta, ensure_ascii=False) + "\n")

        shards.append({"path": f"pt/{name}", "count": int(audio.shape[0])})
        shard_idx += 1
        buffer = []
        buffer_scores = []
        buffer_meta = []
        
        atomic_write_text(
            state_path,
            json.dumps(
                {"shard_idx": int(shard_idx), "wav_idx": int(wav_idx), "shards": shards, "last_file_i": int(last_file_i)},
                ensure_ascii=False,
            ),
        )

    def save_wav(seg: torch.Tensor, out_path: Path):
        x = seg.detach().cpu().clamp(-1.0, 1.0)
        torchaudio.save(str(out_path), x, target_sr, encoding="PCM_S", bits_per_sample=16)

    t0 = time.perf_counter()
    total_files = len(files)
    start_from = int(last_file_i) if resume and int(last_file_i) > 0 else 0
    
    for file_i, path in enumerate(files[start_from:], start=start_from + 1):
        try:
            wav, sr = torchaudio.load(path)
            wav = force_stereo(wav)
            wav, sr = resample_if_needed(wav, int(sr), target_sr)

            # Tile short files
            if wav.shape[-1] < min_segment_samples:
                while wav.shape[-1] < min_segment_samples:
                    wav = torch.cat([wav, wav], dim=-1)

            voiced, frame_len, hop_len = vad.detect(wav)
            
            n_samples = int(wav.shape[-1])
            
            if min_segment_samples == max_segment_samples and output_format in ("pt", "both"):
                # Fixed-length slicing
                seg_len = int(fixed_segment_len)
                spans = [(int(s), int(min(n_samples, s + seg_len))) for s in range(0, n_samples, seg_len)]
            else:
                # Variable-length slicing with VAD
                cut_positions = _silence_cut_positions(
                    voiced,
                    hop_len=hop_len,
                    n_samples=n_samples,
                    min_silence_frames=min_silence_frames,
                )
                spans = _segment_by_cutpoints(
                    n_samples=n_samples,
                    cut_positions=cut_positions,
                    min_len=min_segment_samples,
                    max_len=max_segment_samples,
                )
                if spans is None and len(cut_positions) == 0:
                    # Fallback to uniform slicing
                    fallback = list(range(int(max_segment_samples), n_samples, int(max_segment_samples)))
                    spans = _segment_by_cutpoints(
                        n_samples=n_samples,
                        cut_positions=fallback,
                        min_len=min_segment_samples,
                        max_len=max_segment_samples,
                    )
                if spans is None:
                    continue

            for start, end in spans:
                seg = wav[:, start:end].contiguous()
                
                # Calculate quality score (RMS dB)
                seg_f32 = seg
                if seg_f32.numel() > 0:
                    rms = torch.sqrt(torch.mean(seg_f32**2))
                    score = 20.0 * torch.log10(torch.maximum(rms, torch.tensor(1e-6)))
                    score_val = float(score.item())
                else:
                    score_val = -100.0

                if output_format in ("pt", "both"):
                    if fixed_segment_len is not None:
                        seg = _tile_to_len(seg, fixed_segment_len)
                    buffer.append(seg)
                    buffer_scores.append(score_val)
                    buffer_meta.append({
                        "src": str(path),
                        "start": int(start),
                        "end": int(end),
                        "sample_rate": int(target_sr),
                        "channels": int(seg.shape[0]),
                        "score": float(score_val),
                    })
                    if len(buffer) >= segments_per_shard:
                        flush()
                        
                if output_format in ("wav", "both"):
                    src_stem = Path(path).stem
                    out_name = f"{src_stem}_seg_{wav_idx:06d}_s{start}_e{end}.wav"
                    out_path = wav_dir / out_name
                    save_wav(seg, out_path)
                    with manifest_path.open("a", encoding="utf-8") as fp:
                        fp.write(
                            json.dumps(
                                {
                                    "path": f"wav/{out_name}",
                                    "src": str(path),
                                    "start": int(start),
                                    "end": int(end),
                                    "sample_rate": int(target_sr),
                                    "channels": int(seg.shape[0]),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    wav_idx += 1

        except Exception as e:
            print(f"Error processing {path}: {e}")

        last_file_i = int(file_i)
        if len(buffer) == 0:
            atomic_write_text(
                state_path,
                json.dumps(
                    {"shard_idx": int(shard_idx), "wav_idx": int(wav_idx), "shards": shards, "last_file_i": int(last_file_i)},
                    ensure_ascii=False,
                ),
            )

        if (file_i % 5 == 0 or file_i == total_files):
            elapsed = time.perf_counter() - t0
            avg = elapsed / max(1, file_i)
            remaining = max(0, total_files - file_i) * avg
            print(f"progress {file_i}/{total_files}  elapsed={elapsed:.1f}s  eta={remaining/60:.1f}m")

    # Flush remaining
    if output_format in ("pt", "both"):
        flush()

    # Save index
    if output_format in ("pt", "both"):
        index = {
            "shards": shards,
            "sample_rate": target_sr,
            "segment_samples": int(segment_samples) if segment_samples else 0,
            "channels": 2,
            "dtype": "int16",
            "pcm_scale": float(PT_PCM_SCALE),
        }
        atomic_torch_save(index, output_dir / "index.pt")

    return {
        "shards": len(shards),
        "total_segments": sum(s["count"] for s in shards),
        "output_dir": str(output_dir),
    }


# CLI support
def main():
    parser = argparse.ArgumentParser(description="Cut voiced audio segments and pack into shard tensors")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data_dirs", type=str, help="Comma separated audio directories")
    group.add_argument("--input_files", type=str, help="Comma separated audio file paths")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for shards and index.pt")
    parser.add_argument("--output_format", type=str, default="pt", choices=["pt", "wav", "both"], help="Output format")
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Target sample rate")
    parser.add_argument("--min_segment_s", type=float, default=DEFAULT_MIN_SEGMENT_S, help="Minimum segment seconds")
    parser.add_argument("--max_segment_s", type=float, default=DEFAULT_MAX_SEGMENT_S, help="Maximum segment seconds")
    parser.add_argument("--segments_per_shard", type=int, default=256, help="Segments per saved shard file")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output_dir")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_dir")
    
    # VAD parameters
    parser.add_argument("--silence_threshold_db", type=float, default=DEFAULT_SILENCE_THRESHOLD_DB, help="RMS threshold for silence")
    parser.add_argument("--analysis_frame_ms", type=float, default=DEFAULT_ANALYSIS_FRAME_MS, help="Analysis frame length in ms")
    parser.add_argument("--analysis_hop_ms", type=float, default=DEFAULT_ANALYSIS_HOP_MS, help="Analysis hop length in ms")
    parser.add_argument("--min_silence_ms", type=float, default=DEFAULT_MIN_SILENCE_MS, help="Minimum silence duration in ms")
    
    args = parser.parse_args()
    
    result = cut_segments(
        data_dirs=args.data_dirs or "",
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        min_segment_s=args.min_segment_s,
        max_segment_s=args.max_segment_s,
        output_format=args.output_format,
        overwrite=args.overwrite,
        resume=args.resume,
        silence_threshold_db=args.silence_threshold_db,
        analysis_frame_ms=args.analysis_frame_ms,
        analysis_hop_ms=args.analysis_hop_ms,
        min_silence_ms=args.min_silence_ms,
        segments_per_shard=args.segments_per_shard,
    )
    
    print(f"\nDone! Created {result['shards']} shards with {result['total_segments']} total segments.")


if __name__ == "__main__":
    main()