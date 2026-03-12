"""
Memmap packing module for pliteWAVpipeline.

This module handles packing audio segment shards into a single memory-mapped file
for efficient random access during training.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


__all__ = [
    "pack_memmap",
    "parse_size",
]


# Default constants
DEFAULT_PCM_SCALE = 32768.0


def parse_size(size_str: str) -> Optional[int]:
    """
    Parse size string like '1.5G', '500M', '1K' to bytes.
    
    Args:
        size_str: Size string (e.g., "1.5G", "500M", "1K")
        
    Returns:
        Size in bytes, or None if invalid
    """
    if not size_str:
        return None
    size_str = str(size_str).strip().upper()
    multiplier = 1
    
    if size_str.endswith("G") or size_str.endswith("GB"):
        multiplier = 1024**3
        val = size_str.rstrip("GB")
    elif size_str.endswith("M") or size_str.endswith("MB"):
        multiplier = 1024**2
        val = size_str.rstrip("MB")
    elif size_str.endswith("K") or size_str.endswith("KB"):
        multiplier = 1024
        val = size_str.rstrip("KB")
    else:
        val = size_str
    
    try:
        return int(float(val) * multiplier)
    except ValueError:
        return None


def pack_memmap(
    segments_dir: str,
    output_dir: str,
    max_size: Optional[str] = None,
    quality_sort: bool = False,
) -> dict:
    """
    Pack segment shards into a single memmap file.
    
    Args:
        segments_dir: Directory with segment shards (output root from slicing)
        output_dir: Output directory root
        max_size: Maximum size of output mmap (e.g., "1.2G", "500M")
        quality_sort: If True and max_size is set, prioritize high-quality segments
        
    Returns:
        Dictionary with packing statistics
    """
    max_bytes = parse_size(max_size) if max_size else None
    if max_bytes is not None:
        print(f"Max size limit set to: {max_bytes / (1024**3):.2f} GB ({max_bytes} bytes)")

    segments_dir = Path(segments_dir)
    index = torch.load(segments_dir / "index.pt", map_location="cpu")
    shards = index.get("shards", [])
    sample_rate = int(index["sample_rate"])
    segment_samples = int(index["segment_samples"])
    channels = int(index.get("channels", 2))
    pcm_scale = float(index.get("pcm_scale", DEFAULT_PCM_SCALE))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mmap_dir = output_dir / "mmap"
    mmap_dir.mkdir(parents=True, exist_ok=True)
    mmap_path = mmap_dir / "audio_i16.mmap"

    dtype_size = 2  # int16
    bytes_per_sample = channels * segment_samples * dtype_size
    
    # Quality sorting logic
    sorted_samples = []  # List of (shard_path, local_index, score)
    use_quality_sort = quality_sort and max_bytes is not None
    
    if use_quality_sort:
        manifest_path = segments_dir / "manifest.jsonl"
        if manifest_path.exists():
            print("Scanning manifest for quality scores...")
            samples = []
            has_scores = False
            
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        meta = json.loads(line)
                        path = meta.get("path")
                        idx = meta.get("shard_index", 0)
                        score = meta.get("score")
                        
                        if score is not None:
                            has_scores = True
                            samples.append({
                                "path": path,
                                "idx": int(idx),
                                "score": float(score)
                            })
                        else:
                            samples.append({
                                "path": path,
                                "idx": int(idx),
                                "score": -100.0
                            })
                
                if has_scores:
                    print(f"Found {len(samples)} samples with scores. Sorting by quality...")
                    samples.sort(key=lambda x: x["score"], reverse=True)
                    sorted_samples = samples
                else:
                    print("Manifest found but no scores detected. Using sequential packing.")
            except Exception as e:
                print(f"Error reading manifest: {e}. Using sequential packing.")
        else:
            print("No manifest.jsonl found. Using sequential packing.")

    # Quality-sorted packing
    if use_quality_sort and sorted_samples:
        max_count = max_bytes // bytes_per_sample
        if max_count < len(sorted_samples):
            print(f"Truncating to top {max_count} samples (from {len(sorted_samples)}) based on score.")
            sorted_samples = sorted_samples[:max_count]
        
        total_count = len(sorted_samples)
        expected_bytes = total_count * bytes_per_sample
        
        print(f"Packing {total_count} sorted samples into {mmap_path}...")
        
        # Build task mapping
        shard_tasks = {}
        for target_idx, s in enumerate(sorted_samples):
            p = s["path"]
            if p not in shard_tasks:
                shard_tasks[p] = []
            shard_tasks[p].append((s["idx"], target_idx))
        
        # Pre-allocate file
        print(f"Pre-allocating {expected_bytes / 1024**3:.2f} GB...")
        with open(mmap_path, "wb") as f:
            f.seek(expected_bytes - 1)
            f.write(b'\0')
            
        # Open mmap for writing
        out_mmap = np.memmap(
            mmap_path, 
            dtype=np.int16, 
            mode="r+", 
            shape=(total_count, channels, segment_samples)
        )
        
        written_count = 0
        t0 = time.perf_counter()
        sorted_shard_paths = sorted(shard_tasks.keys())
        n_tasks = len(sorted_shard_paths)
        
        for i, p_str in enumerate(sorted_shard_paths):
            p = segments_dir / p_str
            tasks = shard_tasks[p_str]
            
            if not p.exists():
                print(f"Warning: Shard {p} not found, skipping {len(tasks)} samples.")
                continue
                
            try:
                obj = torch.load(p, map_location="cpu")
                audio = obj["audio"] if isinstance(obj, dict) and "audio" in obj else obj
                
                # Convert if needed
                if audio.dtype == torch.float32:
                    audio = (audio * 32768.0).clamp(-32768, 32767).to(torch.int16)
                
                if audio.dim() == 2:
                    audio = audio.unsqueeze(0)
                
                for local_idx, target_idx in tasks:
                    if local_idx < audio.shape[0]:
                        data = audio[local_idx].numpy()
                        out_mmap[target_idx] = data
                        written_count += 1

            except Exception as e:
                print(f"Error processing shard {p}: {e}")
            
            if i % 5 == 0 or i + 1 == n_tasks:
                elapsed = time.perf_counter() - t0
                pct = (i + 1) / n_tasks * 100
                print(f"  Processed {i+1}/{n_tasks} shards ({pct:.1f}%) - Written {written_count} samples", flush=True)

        out_mmap.flush()
        final_count = written_count
        
    else:
        # Sequential packing (legacy / no quality sort)
        total = 0
        for s in shards:
            total += int(s["count"])

        print(f"Packing {len(shards)} shards into {mmap_path}...")
        
        full_expected_bytes = total * bytes_per_sample
        expected_bytes = full_expected_bytes
        
        if max_bytes is not None and expected_bytes > max_bytes:
            print(f"Dataset total size {expected_bytes / 1024**3:.2f} GB > limit {max_bytes / 1024**3:.2f} GB. Will truncate.")
            expected_bytes = max_bytes

        written_bytes = 0
        t0 = time.perf_counter()
        last_print_time = t0
        n_shards = len(shards)
        
        with open(mmap_path, "wb") as f_out:
            for i, s in enumerate(shards):
                if max_bytes is not None and written_bytes >= max_bytes:
                    break

                p = segments_dir / s["path"]
                
                if p.suffix == ".pt":
                    try:
                        obj = torch.load(p, map_location="cpu")
                        audio = obj["audio"] if isinstance(obj, dict) and "audio" in obj else obj
                        if audio.dtype != torch.int16:
                            if audio.dtype == torch.float32:
                                audio = (audio * 32768.0).clamp(-32768, 32767).to(torch.int16)
                            else:
                                raise ValueError(f"Expected int16 or float32 audio in {p}, got {audio.dtype}")
                        
                        if audio.dim() == 2:
                            audio = audio.unsqueeze(0)
                        audio_np = audio.numpy()
                    except Exception as e:
                        print(f"Error reading {p}: {e}")
                        continue
                elif p.suffix == ".bin":
                    try:
                        raw_bytes = p.read_bytes()
                        audio_np = np.frombuffer(raw_bytes, dtype=np.int16)
                        expected_len = channels * segment_samples
                        if audio_np.size != expected_len:
                            if audio_np.size % expected_len == 0:
                                n = audio_np.size // expected_len
                            else:
                                print(f"Warning: {p.name} size {audio_np.size} does not match {channels}x{segment_samples}")
                                continue
                        else:
                            n = 1
                        audio_np = audio_np.reshape(n, channels, segment_samples)
                    except Exception as e:
                        print(f"Error reading {p}: {e}")
                        continue
                else:
                    continue
                
                b = audio_np.tobytes()
                
                stop_writing = False
                if max_bytes is not None:
                    remaining = max_bytes - written_bytes
                    if len(b) > remaining:
                        samples_to_take = remaining // bytes_per_sample
                        if samples_to_take > 0:
                            audio_np = audio_np[:samples_to_take]
                            b = audio_np.tobytes()
                            stop_writing = True
                        else:
                            break
                
                f_out.write(b)
                written_bytes += len(b)
                
                now = time.perf_counter()
                if now - last_print_time > 5.0 or (i + 1) == n_shards or stop_writing:
                    elapsed = now - t0
                    
                    if max_bytes is not None:
                        progress = min(1.0, written_bytes / expected_bytes) if expected_bytes > 0 else 1.0
                    else:
                        progress = (i + 1) / n_shards
                        
                    mb_written = written_bytes / (1024 * 1024)
                    
                    if progress > 0:
                        total_time_est = elapsed / progress
                        remaining = total_time_est - elapsed
                    else:
                        remaining = 0
                    
                    print(f"  Processed {i + 1}/{n_shards} shards - {mb_written:.1f} MB ({progress*100:.1f}%) - ETA: {remaining:.1f}s", flush=True)
                    last_print_time = now
                
                if stop_writing:
                    print(f"Reached max size limit.")
                    break

        final_count = written_bytes // bytes_per_sample

    # Write metadata
    meta = {
        "path": "mmap/audio_i16.mmap",
        "dtype": "int16",
        "count": int(final_count),
        "channels": int(channels),
        "segment_samples": int(segment_samples),
        "sample_rate": int(sample_rate),
        "pcm_scale": float(pcm_scale),
    }
    (output_dir / "mmap_index.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    
    print(f"\nDone! Packed {final_count} segments into {mmap_path}")
    
    return {
        "count": final_count,
        "channels": channels,
        "segment_samples": segment_samples,
        "sample_rate": sample_rate,
        "output_path": str(mmap_path),
    }


# CLI support
def main():
    parser = argparse.ArgumentParser(description="Pack segment shards into a single memmap file")
    parser.add_argument("--segments_dir", type=str, required=True, help="Directory with segment shards")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory root")
    parser.add_argument("--max_size", type=str, default=None, help="Max size of output mmap (e.g. 1.2G, 500M)")
    parser.add_argument("--quality_sort", action="store_true", help="Sort by quality score (requires manifest)")
    args = parser.parse_args()

    result = pack_memmap(
        segments_dir=args.segments_dir,
        output_dir=args.output_dir,
        max_size=args.max_size,
        quality_sort=args.quality_sort,
    )
    
    print(f"\nOutput: {result['count']} segments, {result['channels']} channels, {result['segment_samples']} samples/segment")


if __name__ == "__main__":
    main()