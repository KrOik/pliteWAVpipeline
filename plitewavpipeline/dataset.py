"""
Dataset module for pliteWAVpipeline.

This module provides various PyTorch Dataset implementations for loading
audio data in different formats (shards, memmap, wav).
"""

import glob
import json
import os
import random
import time
from bisect import bisect_right
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Sampler


__all__ = [
    "AudioDataset",
    "SegmentShardDataset",
    "MemmapDataset",
    "WavSegmentDataset",
    "MemmapSegmentDataset",
    "ShardGroupedBatchSampler",
    "collate_audio_batch",
    "scan_files",
    "scan_wav_segments",
]


# Default constants
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_PCM_SCALE = 32768.0


def scan_files(data_dirs: str, exts: List[str] = None) -> List[str]:
    """Scan directories for audio files."""
    if exts is None:
        exts = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".wma"]
    
    files = []
    for d in data_dirs.split(","):
        for ext in exts:
            files.extend(glob.glob(os.path.join(d, f"**/*{ext}"), recursive=True))
    return sorted(list(set(files)))


def scan_wav_segments(segments_root: str) -> List[str]:
    """Scan for WAV segment files."""
    root = Path(segments_root)
    if (root / "wav").is_dir():
        root = root / "wav"
    files = sorted([str(p) for p in root.glob("*.wav")])
    return files


def collate_audio_batch(batch):
    """
    Collate function to filter out None samples (failed loads).
    
    Args:
        batch: List of samples from Dataset.__getitem__
        
    Returns:
        Batched tensor or None if all samples failed
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.stack(batch, dim=0)


def _extract_audio_tensor(obj) -> torch.Tensor:
    """Extract audio tensor from various shard formats."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        if "audio" in obj:
            return obj["audio"]
        if "segments" in obj:
            return obj["segments"]
    raise ValueError("Unsupported shard format. Expected Tensor or dict with 'audio'/'segments'.")


def _ensure_channels(wav: torch.Tensor, target_channels: int = 2) -> torch.Tensor:
    """Ensure audio has target number of channels."""
    if wav.ndim != 2:
        raise ValueError(f"Expected (C, T), got {tuple(wav.shape)}")
    
    C, T = wav.shape
    if C == target_channels:
        return wav
    if C < target_channels:
        return wav.repeat(target_channels // C + 1, 1)[:target_channels, :]
    if C > target_channels:
        return wav[:target_channels, :]
    return wav


class AudioDataset(Dataset):
    """
    Basic audio dataset for raw audio files.
    
    Handles resampling, channel normalization, and random cropping.
    """
    
    def __init__(
        self,
        files: List[str],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        segment_duration_s: float = 5.0,
        target_channels: int = 2,
    ):
        self.files = files
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_duration_s)
        self.target_channels = target_channels
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Optional[torch.Tensor]:
        path = self.files[idx]
        try:
            wav, sr = torchaudio.load(path)
            
            # Resample
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

            # Ensure channels
            wav = _ensure_channels(wav, self.target_channels)
            
            # Tile if too short
            while wav.shape[-1] < self.segment_samples:
                wav = torch.cat([wav, wav], dim=-1)
            
            # Random crop
            if wav.shape[-1] > self.segment_samples:
                start = random.randint(0, wav.shape[-1] - self.segment_samples)
                wav = wav[:, start:start + self.segment_samples]
                
            return wav
        except Exception as e:
            return None


class SegmentShardDataset(Dataset):
    """
    Dataset for loading audio from PyTorch shard files.
    
    Supports lazy loading, caching, and efficient random access via index.
    """
    
    def __init__(
        self,
        segments_dir: str,
        segment_samples: Optional[int] = None,
        cache_in_memory: bool = False,
        index_name: str = "index.pt",
        max_shards: Optional[int] = None,
        pcm_scale: Optional[float] = DEFAULT_PCM_SCALE,
    ):
        self.segments_dir = Path(segments_dir)
        self.segment_samples = segment_samples
        self.cache_in_memory = cache_in_memory
        self.index_path = self.segments_dir / index_name
        self.pcm_scale = pcm_scale

        shard_paths, shard_counts, meta = self._load_index_or_scan()
        
        if max_shards is not None and max_shards > 0:
            shard_paths = shard_paths[:max_shards]
            shard_counts = shard_counts[:max_shards]
            
        self.shard_paths = shard_paths
        self.shard_counts = shard_counts
        self.meta = meta

        # Build cumulative counts for efficient lookup
        self.cum_counts = []
        total = 0
        for c in self.shard_counts:
            total += int(c)
            self.cum_counts.append(total)
        self.total = total

        # Set segment_samples from metadata if not provided
        if self.segment_samples is None and isinstance(self.meta, dict):
            if "segment_samples" in self.meta:
                self.segment_samples = int(self.meta["segment_samples"])
        if isinstance(self.meta, dict) and "pcm_scale" in self.meta:
            self.pcm_scale = float(self.meta["pcm_scale"])

        # In-memory cache
        self._all_audio = None
        self._last_shard_i = None
        self._last_shard_audio = None
        self._last_shard_pcm_scale = None
        
        if self.cache_in_memory:
            chunks = []
            for p in self.shard_paths:
                obj = torch.load(p, map_location="cpu", weights_only=False)
                audio = _extract_audio_tensor(obj)
                chunks.append(audio)
            self._all_audio = torch.cat(chunks, dim=0)
            if self._all_audio.dtype == torch.int16:
                scale = self.pcm_scale or DEFAULT_PCM_SCALE
                self._all_audio = self._all_audio.to(torch.float32) / scale

    def _load_index_or_scan(self) -> Tuple[List[Path], List[int], dict]:
        """Load shard index or scan directory for shard files."""
        if self.index_path.exists():
            obj = torch.load(self.index_path, map_location="cpu", weights_only=False)
            shards = obj.get("shards", [])
            shard_paths = [self.segments_dir / s["path"] for s in shards]
            shard_counts = [int(s["count"]) for s in shards]
            
            missing = [str(p) for p in shard_paths if not Path(p).exists()]
            if len(missing) == 0:
                meta = {
                    k: obj.get(k)
                    for k in ["sample_rate", "segment_samples", "channels", "dtype", "pcm_scale"]
                    if k in obj
                }
                return shard_paths, shard_counts, meta

        # Scan for loose shard files
        shard_paths = sorted([p for p in self.segments_dir.glob("*.pt") if p.name != self.index_path.name])
        if len(shard_paths) == 0 and (self.segments_dir / "pt").is_dir():
            shard_paths = sorted([
                p for p in (self.segments_dir / "pt").glob("*.pt") 
                if p.name != self.index_path.name
            ])

        shard_counts = []
        for p in shard_paths:
            obj = torch.load(p, map_location="cpu", weights_only=False)
            audio = _extract_audio_tensor(obj)
            shard_counts.append(int(audio.shape[0]))
            
        return shard_paths, shard_counts, {}

    def _locate(self, idx: int) -> Tuple[int, int]:
        """Locate shard index and local index within shard."""
        shard_i = bisect_right(self.cum_counts, idx)
        prev = 0 if shard_i == 0 else self.cum_counts[shard_i - 1]
        local_i = idx - prev
        return shard_i, local_i

    def _load_shard_audio(self, shard_i: int) -> torch.Tensor:
        """Load audio from a specific shard with caching."""
        if self._last_shard_i == shard_i and self._last_shard_audio is not None:
            return self._last_shard_audio

        p = self.shard_paths[int(shard_i)]
        
        # Retry logic
        last_err = None
        for attempt in range(3):
            try:
                obj = torch.load(p, map_location="cpu", weights_only=False)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.2 * (attempt + 1))
        
        if last_err is not None:
            # Return silence on failure
            channels = 2
            if isinstance(self.meta, dict) and "channels" in self.meta:
                channels = int(self.meta["channels"])
            audio = torch.zeros(
                (int(self.shard_counts[int(shard_i)]), channels, int(self.segment_samples or 1)),
                dtype=torch.float32,
            )
            self._last_shard_i = int(shard_i)
            self._last_shard_audio = audio
            return audio
            
        audio = _extract_audio_tensor(obj)
        
        # Get PCM scale
        pcm_scale = self.pcm_scale or DEFAULT_PCM_SCALE
        if isinstance(obj, dict) and "pcm_scale" in obj and obj["pcm_scale"] is not None:
            pcm_scale = float(obj["pcm_scale"])

        # Convert and normalize
        if audio.dtype == torch.int16:
            audio = audio.to(torch.float32) / pcm_scale
        else:
            audio = audio.to(torch.float32)
        audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)

        self._last_shard_i = int(shard_i)
        self._last_shard_audio = audio
        return audio

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            idx = self.total + idx
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)

        # Return cached data if available
        if self._all_audio is not None:
            return self._all_audio[idx]

        shard_i, local_i = self._locate(idx)
        audio = self._load_shard_audio(shard_i)
        
        # Safe access
        if local_i >= audio.shape[0]:
            return torch.zeros((2, int(self.segment_samples or 16000*5)), dtype=torch.float32)

        seg = audio[local_i]

        # Handle length mismatch
        if self.segment_samples is not None:
            curr_len = seg.shape[-1]
            target_len = int(self.segment_samples)
            if curr_len > target_len:
                start = random.randint(0, curr_len - target_len)
                seg = seg[..., start:start+target_len]
            elif curr_len < target_len:
                while seg.shape[-1] < target_len:
                    seg = torch.cat([seg, seg], dim=-1)
                seg = seg[..., :target_len]

        return seg


class MemmapDataset(Dataset):
    """
    Dataset for loading from memory-mapped files.
    
    Provides efficient random access without loading entire dataset into memory.
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.index_path = self.data_dir / "mmap_index.json"
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"mmap_index.json not found in {data_dir}")
            
        with open(self.index_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
            
        self.mmap_path = self.data_dir / self.meta["path"]
        self.count = int(self.meta["count"])
        self.channels = int(self.meta["channels"])
        self.segment_samples = int(self.meta["segment_samples"])
        self.pcm_scale = float(self.meta.get("pcm_scale", DEFAULT_PCM_SCALE))
        
        # Open memmap in read-only mode
        self.mm = np.memmap(
            str(self.mmap_path),
            dtype=np.int16,
            mode="r",
            shape=(self.count, self.channels, self.segment_samples)
        )
        
    def __len__(self) -> int:
        return self.count
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            idx = self.count + idx
        if idx >= self.count or idx < 0:
            raise IndexError(f"Index {idx} out of range [0, {self.count})")
            
        audio_np = self.mm[idx].copy()
        audio = torch.from_numpy(audio_np).float()
        audio = audio / self.pcm_scale
        
        return audio


class WavSegmentDataset(Dataset):
    """
    Dataset for loading WAV segment files.
    
    Simple dataset for loose WAV files.
    """
    
    def __init__(
        self,
        segments_root: str,
        expected_sample_rate: Optional[int] = None,
        cache_in_memory: bool = False,
    ):
        self.files = scan_wav_segments(segments_root)
        if len(self.files) == 0:
            raise ValueError(f"No .wav segments found under {segments_root}")
        self.expected_sample_rate = expected_sample_rate
        self.cache_in_memory = cache_in_memory

        self._audio = None
        if self.cache_in_memory:
            audio = []
            for p in self.files:
                wav, sr = torchaudio.load(p)
                if expected_sample_rate is not None and int(sr) != int(expected_sample_rate):
                    wav = torchaudio.functional.resample(wav, sr, expected_sample_rate)
                wav = _ensure_channels(wav, 2)
                audio.append(wav.contiguous())
            self._audio = audio

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._audio is not None:
            return self._audio[idx]

        p = self.files[idx]
        wav, sr = torchaudio.load(p)
        if self.expected_sample_rate is not None and int(sr) != int(self.expected_sample_rate):
            wav = torchaudio.functional.resample(wav, sr, self.expected_sample_rate)
        wav = _ensure_channels(wav, 2)
        return wav.contiguous()


class MemmapSegmentDataset(Dataset):
    """
    Alternative memmap dataset implementation.
    """
    
    def __init__(
        self,
        memmap_root: str,
        cache_in_memory: bool = False,
        index_name: str = "mmap_index.json",
    ):
        self.memmap_root = Path(memmap_root)
        meta = json.loads((self.memmap_root / index_name).read_text(encoding="utf-8"))
        self.meta = meta

        self.count = int(meta["count"])
        self.channels = int(meta["channels"])
        self.segment_samples = int(meta["segment_samples"])
        self.sample_rate = int(meta["sample_rate"])
        self.pcm_scale = float(meta.get("pcm_scale", DEFAULT_PCM_SCALE))

        mmap_path = self.memmap_root / meta["path"]
        self._mm = np.memmap(
            str(mmap_path),
            dtype=np.int16,
            mode="r",
            shape=(self.count, self.channels, self.segment_samples),
        )
        
        self._all = None
        if cache_in_memory:
            self._all = torch.from_numpy(np.asarray(self._mm)).to(torch.float32) / float(self.pcm_scale)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            idx = self.count + idx
        if idx >= self.count or idx < 0:
            raise IndexError(idx)

        if self._all is not None:
            return self._all[idx]

        x = self._mm[int(idx)]
        return torch.from_numpy(np.array(x, copy=True)).to(torch.float32) / float(self.pcm_scale)


class ShardGroupedBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that groups samples by shard for efficient loading.
    
    Ensures each batch contains samples from the same shard to minimize
    random access overhead.
    """
    
    def __init__(
        self,
        ds: SegmentShardDataset,
        batch_size: int,
        shuffle_shards: bool = True,
        shuffle_batches: bool = True,
        drop_last: bool = False,
    ):
        self.ds = ds
        self.batch_size = int(batch_size)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_batches = bool(shuffle_batches)
        self.drop_last = bool(drop_last)

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

    def __iter__(self):
        shard_ids = list(range(len(self.ds.shard_counts)))
        if self.shuffle_shards:
            random.shuffle(shard_ids)

        for shard_i in shard_ids:
            count = int(self.ds.shard_counts[int(shard_i)])
            start_global = 0 if int(shard_i) == 0 else int(self.ds.cum_counts[int(shard_i) - 1])
            n_batches = (count + self.batch_size - 1) // self.batch_size
            batch_ids = list(range(n_batches))
            if self.shuffle_batches:
                random.shuffle(batch_ids)
            for b in batch_ids:
                s = int(b) * self.batch_size
                e = min(count, (int(b) + 1) * self.batch_size)
                if self.drop_last and (e - s) < self.batch_size:
                    continue
                yield [start_global + k for k in range(s, e)]

    def __len__(self) -> int:
        total_batches = 0
        for c in self.ds.shard_counts:
            count = int(c)
            if self.drop_last:
                total_batches += count // self.batch_size
            else:
                total_batches += (count + self.batch_size - 1) // self.batch_size
        return int(total_batches)


# Utility function for creating dataloaders
def create_dataloader(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    dataset_type: str = "auto",
    **kwargs
) -> DataLoader:
    """
    Convenience function to create a dataloader from various data formats.
    
    Args:
        data_path: Path to data (mmap dir, segments dir, or wav dir)
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        dataset_type: "auto", "memmap", "shard", or "wav"
        
    Returns:
        Configured DataLoader
    """
    data_path = Path(data_path)
    
    # Auto-detect dataset type
    if dataset_type == "auto":
        if (data_path / "mmap_index.json").exists():
            dataset_type = "memmap"
        elif (data_path / "index.pt").exists():
            dataset_type = "shard"
        else:
            dataset_type = "wav"
    
    # Create appropriate dataset
    if dataset_type == "memmap":
        ds = MemmapDataset(data_path)
    elif dataset_type == "shard":
        ds = SegmentShardDataset(data_path, **kwargs)
    else:
        ds = WavSegmentDataset(data_path, **kwargs)
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_audio_batch if isinstance(ds, AudioDataset) else None,
    )