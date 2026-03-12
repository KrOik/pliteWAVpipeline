#!/usr/bin/env python3
"""
Rigorous IO Performance Benchmark v2.0
======================================
Key Improvements over v1:
- Statistical rigor: 30 runs with bootstrap confidence intervals
- Data usage verification: prevent compiler optimization
- Environment capture: full system info for reproducibility  
- Uniform sample counts: fair comparison across formats
- Outlier detection: IQR method
- Random seed: reproducible results
- Warmup: sufficient iterations to reach steady state

Reference: Kalibera & Jones, "Rigorous Benchmarking in Reasonable Time" (ACM SIGPLAN 2013)
"""

import json
import os
import sys
import time
import gc
import subprocess
import random
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42
NUM_RUNS = 30  # Minimum for reliable statistics
WARMUP_RUNS = 5  # Reach steady state
MIN_SAMPLES_PER_TEST = 200  # Uniform sample count
CONFIDENT_LEVEL = 0.95
NUM_BOOTSTRAP = 10000


# =============================================================================
# Statistical Utilities
# =============================================================================

def set_random_seed(seed: int = RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_confidence_interval(
    data: np.ndarray, 
    n_bootstrap: int = NUM_BOOTSTRAP, 
    confidence: float = CONFIDENT_LEVEL
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    Returns: (mean, lower_bound, upper_bound)
    """
    data = np.array(data)
    n = len(data)
    bootstrap_means = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return float(np.mean(data)), float(lower), float(upper)


def remove_outliers_iqr(values: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Remove outliers using IQR method.
    Returns: (cleaned_array, list_of_removed_indices)
    """
    values = np.array(values)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (values >= lower_bound) & (values <= upper_bound)
    removed_indices = np.where(~mask)[0].tolist()
    
    return values[mask], removed_indices


def statistical_significance_test(
    data1: np.ndarray, 
    data2: np.ndarray
) -> Dict:
    """
    Perform Welch's t-test for statistical significance.
    """
    from scipy import stats
    
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    # Welch's t-test (does not assume equal variances)
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant_at_0.05": bool(p_value < 0.05),
        "significant_at_0.01": bool(p_value < 0.01),
    }


# =============================================================================
# Environment Detection
# =============================================================================

def get_system_info() -> Dict:
    """Capture complete system environment for reproducibility."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "cpu": {
            "count_physical": os.cpu_count(),
            "count_logical": os.cpu_count() or 0,
        },
        "memory": {},
        "disk": {},
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
        }
    }
    
    # Memory info
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
            "percent_used": vm.percent,
        }
    except ImportError:
        info["memory"]["note"] = "psutil not available"
    
    # Disk info
    try:
        import shutil
        disk = shutil.disk_usage("/")
        info["disk"] = {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
        }
    except Exception:
        pass
    
    # GPU info
    if torch.cuda.is_available():
        info["gpu"] = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
            "cuda_version": torch.version.cuda,
        }
    else:
        info["gpu"] = {"available": False}
    
    # Check if SSD/NVMe (Windows)
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "diskdrive", "get", "model,mediatype"],
                capture_output=True, text=True, timeout=5
            )
            info["disk"]["details"] = result.stdout.strip()
        except Exception:
            pass
    
    return info


def get_page_cache_size() -> Optional[int]:
    """Get OS page cache size (if available)."""
    try:
        if platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'Cached:' in line:
                        return int(line.split()[1]) * 1024  # Convert to bytes
    except Exception:
        pass
    return None


# =============================================================================
# Cache Management
# =============================================================================

def clear_caches():
    """Attempt to clear OS caches between runs."""
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Note: Full cache clearing requires admin rights on Windows
    # We record cache state instead
    cache_size = get_page_cache_size()
    
    if platform.system() == "Linux":
        try:
            subprocess.run(["sync"], capture_output=True)
            subprocess.run(["echo", "3", ">", "/proc/sys/vm/drop_caches"], shell=True)
        except Exception:
            pass


# =============================================================================
# Dataset Implementations
# =============================================================================

class OriginalAudioDataset(Dataset):
    """Dataset for original audio files (FLAC, MP3, etc.)."""
    
    def __init__(self, audio_root, sample_rate=48000):
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate
        
        supported_exts = ["*.flac", "*.mp3", "*.wav", "*.m4a", "*.ogg", "*.xm"]
        self.files = []
        for ext in supported_exts:
            self.files.extend(sorted(self.audio_root.glob(f"**/{ext}")))
        
        if not self.files:
            raise ValueError(f"No supported audio files found in {audio_root}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            wav, sr = torchaudio.load(self.files[idx])
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            if wav.shape[0] > 2:
                wav = wav[:2, :]
            elif wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            return wav
        except Exception:
            return torch.zeros(2, self.sample_rate * 5)


class CachedWavDataset(Dataset):
    """Dataset for pre-cached WAV files."""
    
    def __init__(self, wav_cache_root, sample_rate=48000):
        self.wav_root = Path(wav_cache_root)
        self.sample_rate = sample_rate
        self.files = sorted(list(self.wav_root.glob("*.wav")))
        
        if not self.files:
            raise ValueError(f"No wav files found in {wav_cache_root}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.files[idx])
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav


class MemmapDataset(Dataset):
    """Dataset for memory-mapped files."""
    
    def __init__(self, mmap_dir):
        self.mmap_dir = Path(mmap_dir)
        with open(self.mmap_dir / "mmap_index.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        
        mmap_path = self.mmap_dir / self.meta["path"]
        self.count = int(self.meta["count"])
        self.channels = int(self.meta["channels"])
        self.segment_samples = int(self.meta["segment_samples"])
        self.pcm_scale = float(self.meta.get("pcm_scale", 32768.0))
        
        self._mm = np.memmap(
            str(mmap_path), dtype=np.int16, mode="r",
            shape=(self.count, self.channels, self.segment_samples)
        )
    
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        x = self._mm[int(idx)]
        return torch.from_numpy(np.array(x, copy=True)).float() / self.pcm_scale


def identity_collate(batch):
    """Identity collate function."""
    return batch


# =============================================================================
# Benchmark Engine
# =============================================================================

@dataclass
class BenchmarkResult:
    """Structured benchmark result with full statistics."""
    test_name: str
    format_type: str
    
    # Raw timing data
    raw_times: List[float] = field(default_factory=list)
    raw_throughputs: List[float] = field(default_factory=list)
    
    # After outlier removal
    clean_times: List[float] = field(default_factory=list)
    clean_throughputs: List[float] = field(default_factory=list)
    outliers_removed: int = 0
    
    # Statistics on clean data
    mean_throughput: float = 0.0
    median_throughput: float = 0.0
    std_throughput: float = 0.0
    min_throughput: float = 0.0
    max_throughput: float = 0.0
    
    # Bootstrap CI
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    
    # Test metadata
    num_samples: int = 0
    batch_size: int = 0
    num_workers: int = 0
    num_runs: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


def run_rigorous_benchmark(
    dataset: Dataset,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    warmup_runs: int = WARMUP_RUNS,
    num_runs: int = NUM_RUNS,
    test_name: str = "test",
    format_type: str = "unknown",
) -> BenchmarkResult:
    """
    Run rigorous benchmark with statistical rigor.
    
    Key features:
    - Multiple warmup runs to reach steady state
    - Many measurement runs (30+) for statistical validity
    - Data usage verification (prevent compiler optimization)
    - Outlier removal (IQR method)
    - Bootstrap confidence intervals
    """
    result = BenchmarkResult(
        test_name=test_name,
        format_type=format_type,
        num_samples=num_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        num_runs=num_runs,
    )
    
    collate_fn = identity_collate if format_type != "mmap" else None
    
    # Warmup runs - reach steady state (JIT compilation, cache population)
    for _ in range(warmup_runs):
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        it = iter(dl)
        for _ in range(min(10, len(dl))):
            try:
                batch = next(it)
                # CRITICAL: Verify data is actually used to prevent optimization
                if isinstance(batch, list):
                    _ = sum(b.sum() for b in batch)
                else:
                    _ = batch.sum()
            except StopIteration:
                break
    
    # Measurement runs
    times = []
    throughputs = []
    
    for run_idx in range(num_runs):
        # Clear caches between runs (best effort)
        clear_caches()
        
        # Create fresh DataLoader for each run
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(RANDOM_SEED + run_idx)
        )
        
        it = iter(dl)
        
        # Warmup within run (1 iteration)
        try:
            batch = next(it)
            if isinstance(batch, list):
                _ = sum(b.sum() for b in batch)
            else:
                _ = batch.sum()
        except StopIteration:
            continue
        
        # Timed measurement
        loaded = 0
        t0 = time.perf_counter()
        
        while loaded < num_samples:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            
            # CRITICAL: Force actual data usage to prevent compiler optimization
            if isinstance(batch, list):
                _ = sum(b.sum() for b in batch)
            else:
                _ = batch.sum()
            
            bsz = len(batch) if isinstance(batch, list) else batch.shape[0]
            loaded += bsz
        
        dt = time.perf_counter() - t0
        times.append(dt)
        throughputs.append(loaded / dt)
    
    result.raw_times = times
    result.raw_throughputs = throughputs
    
    # Outlier removal (IQR method)
    clean_throughputs, removed = remove_outliers_iqr(np.array(throughputs))
    result.clean_throughputs = clean_throughputs.tolist()
    result.outliers_removed = len(removed)
    
    # Statistics on clean data
    if len(clean_throughputs) > 0:
        result.mean_throughput = float(np.mean(clean_throughputs))
        result.median_throughput = float(np.median(clean_throughputs))
        result.std_throughput = float(np.std(clean_throughputs)) if len(clean_throughputs) > 1 else 0.0
        result.min_throughput = float(np.min(clean_throughputs))
        result.max_throughput = float(np.max(clean_throughputs))
        
        # Bootstrap confidence interval
        mean, lower, upper = bootstrap_confidence_interval(clean_throughputs)
        result.ci_95_lower = lower
        result.ci_95_upper = upper
    
    return result


def format_comparison_table(results: Dict[str, BenchmarkResult]) -> str:
    """Generate formatted comparison table."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("RIGOROUS BENCHMARK RESULTS")
    lines.append("=" * 100)
    
    # Header
    lines.append(f"\n{'Format':<20} {'Mean (s/s)':<15} {'Median':<15} {'95% CI':<25} {'Stdev':<12} {'Outliers'}")
    lines.append("-" * 100)
    
    for name, res in results.items():
        ci_str = f"[{res.ci_95_lower:.1f}, {res.ci_95_upper:.1f}]"
        lines.append(
            f"{name:<20} {res.mean_throughput:>14.2f} "
            f"{res.median_throughput:>14.2f} {ci_str:<25} "
            f"{res.std_throughput:>11.2f} {res.outliers_removed}"
        )
    
    return "\n".join(lines)


def generate_summary_report(
    results: Dict[str, BenchmarkResult],
    system_info: Dict
) -> str:
    """Generate comprehensive summary report."""
    lines = []
    
    # Find baseline (original audio)
    baseline = results.get("original_audio", None)
    
    if baseline:
        lines.append("\n" + "=" * 80)
        lines.append("SPEEDUP ANALYSIS (vs Original Audio)")
        lines.append("=" * 80)
        lines.append(f"\n{'Format':<25} {'Speedup':<15} {'95% CI of Speedup'}")
        lines.append("-" * 60)
        
        for name, res in results.items():
            if name != "original_audio" and baseline.mean_throughput > 0:
                speedup = res.mean_throughput / baseline.mean_throughput
                
                # Delta method for CI of ratio
                lower_ci = res.ci_95_lower / baseline.ci_95_upper
                upper_ci = res.ci_95_upper / baseline.ci_95_lower
                
                lines.append(f"{name:<25} {speedup:>10.2f}x    [{lower_ci:.2f}x, {upper_ci:.2f}x]")
    
    # Statistical significance (if we have baseline)
    if baseline and len(baseline.clean_throughputs) > 2:
        lines.append("\n" + "=" * 80)
        lines.append("STATISTICAL SIGNIFICANCE (Welch's t-test vs Original)")
        lines.append("=" * 80)
        
        for name, res in results.items():
            if name != "original_audio" and len(res.clean_throughputs) > 2:
                sig = statistical_significance_test(
                    baseline.clean_throughputs,
                    res.clean_throughputs
                )
                sig_marker = "***" if sig["significant_at_0.01"] else ("**" if sig["significant_at_0.05"] else "")
                lines.append(
                    f"{name:<25} p={sig['p_value']:.2e} {sig_marker:5} "
                    f"Cohen's d={sig['cohens_d']:.2f}"
                )
    
    # System info summary
    lines.append("\n" + "=" * 80)
    lines.append("TEST ENVIRONMENT")
    lines.append("=" * 80)
    
    sys_info = system_info
    lines.append(f"  Platform:     {sys_info['platform']['system']} {sys_info['platform']['release']}")
    lines.append(f"  Python:       {sys_info['versions']['python']}")
    lines.append(f"  PyTorch:      {sys_info['versions']['torch']}")
    lines.append(f"  CPU:          {sys_info['cpu']['count_physical']} cores")
    
    if sys_info.get('gpu', {}).get('available'):
        lines.append(f"  GPU:           {sys_info['gpu']['device_name']}")
    
    if sys_info.get('memory', {}).get('total_gb'):
        lines.append(f"  RAM:           {sys_info['memory']['total_gb']} GB")
    
    return "\n".join(lines)


# =============================================================================
# Main Benchmark Execution
# =============================================================================

def main():
    print("=" * 80)
    print("RIGOROUS IO PERFORMANCE BENCHMARK v2.0")
    print("=" * 80)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Runs per test: {NUM_RUNS}")
    print(f"Warmup runs: {WARMUP_RUNS}")
    print(f"Samples per test: {MIN_SAMPLES_PER_TEST}")
    
    # Set random seed
    set_random_seed(RANDOM_SEED)
    
    # Capture system info
    system_info = get_system_info()
    
    original_dir = Path(os.environ.get("AUDIO_BENCHMARK_ORIGINAL", "./test_data/original"))
    cached_dir = Path(os.environ.get("AUDIO_BENCHMARK_CACHED", "./test_data/cached"))
    mmap_dir = Path(os.environ.get("AUDIO_BENCHMARK_MMAP", "./test_data/mmap"))
    
    print(f"\n[INFO] Original Audio: {len(list(original_dir.glob('**/*.*')))} files")
    print(f"[INFO] Cached WAV: {len(list(cached_dir.glob('*.wav')))} files")
    
    # Check mmap index
    mmap_index = mmap_dir / "mmap_index.json"
    if mmap_index.exists():
        with open(mmap_index) as f:
            mmap_meta = json.load(f)
        print(f"[INFO] Mmap: {mmap_meta.get('count', 'N/A')} segments")
    else:
        print(f"[ERROR] Mmap index not found: {mmap_index}")
        return
    
    # Find minimum dataset size for uniform sampling
    min_len = float('inf')
    datasets = {}
    
    try:
        ds_orig = OriginalAudioDataset(original_dir)
        datasets["original_audio"] = ds_orig
        min_len = min(min_len, len(ds_orig))
    except Exception as e:
        print(f"[WARN] Cannot load original audio: {e}")
    
    try:
        ds_cached = CachedWavDataset(cached_dir)
        datasets["cached_wav"] = ds_cached
        min_len = min(min_len, len(ds_cached))
    except Exception as e:
        print(f"[WARN] Cannot load cached wav: {e}")
    
    try:
        ds_mmap = MemmapDataset(mmap_dir)
        datasets["mmap"] = ds_mmap
        min_len = min(min_len, len(ds_mmap))
    except Exception as e:
        print(f"[WARN] Cannot load mmap: {e}")
    
    # Use uniform sample count
    num_samples = min(MIN_SAMPLES_PER_TEST, int(min_len))
    print(f"\n[INFO] Using {num_samples} samples per test (uniform)")
    
    # Run benchmarks
    results = {}
    
    # Test configurations
    configs = [
        # (name, dataset_key, batch_size, num_workers)
        ("original_audio", "original_audio", 8, 0),
        ("cached_wav", "cached_wav", 8, 0),
        ("mmap", "mmap", 8, 0),
    ]
    
    for name, ds_key, batch_size, num_workers in configs:
        if ds_key not in datasets:
            continue
            
        ds = datasets[ds_key]
        test_name = f"{name}_b{batch_size}_w{num_workers}"
        
        print(f"\n[RUNNING] {test_name} ({NUM_RUNS} runs)...")
        
        result = run_rigorous_benchmark(
            dataset=ds,
            num_samples=num_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            warmup_runs=WARMUP_RUNS,
            num_runs=NUM_RUNS,
            test_name=test_name,
            format_type=name,
        )
        
        results[name] = result
        
        print(f"  Mean: {result.mean_throughput:.2f} samples/s")
        print(f"  95% CI: [{result.ci_95_lower:.2f}, {result.ci_95_upper:.2f}]")
        print(f"  Outliers removed: {result.outliers_removed}")
    
    # Generate outputs
    print(format_comparison_table(results))
    print(generate_summary_report(results, system_info))
    
    # Save results
    output_data = {
        "metadata": {
            "version": "2.0",
            "random_seed": RANDOM_SEED,
            "num_runs": NUM_RUNS,
            "warmup_runs": WARMUP_RUNS,
            "num_samples": num_samples,
            "confidence_level": CONFIDENT_LEVEL,
            "timestamp": datetime.now().isoformat(),
        },
        "system_info": system_info,
        "results": {name: res.to_dict() for name, res in results.items()},
    }
    
    output_path = Path("benchmark_test/io_benchmark_rigorous_v2.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n[Saved] Results to: {output_path}")


if __name__ == "__main__":
    main()