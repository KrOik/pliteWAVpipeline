import os
import json
import random
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plitewavpipeline import load_audio_robust, resample_if_needed, ensure_channels

def calculate_snr(ref, target):
    noise = ref - target
    ref_power = torch.mean(ref**2)
    noise_power = torch.mean(noise**2)
    if noise_power < 1e-10:
        return float('inf')
    return 10 * torch.log10(ref_power / noise_power).item()

def calculate_mse(ref, target):
    return torch.mean((ref - target)**2).item()

def plot_spectrograms(original, processed, title_id, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Use torchaudio for spectrogram
    spec_transform = torchaudio.transforms.Spectrogram(n_fft=1024)
    
    # Original (taking first channel)
    spec_orig = spec_transform(original[0:1])
    axes[0].imshow(torch.log10(spec_orig[0] + 1e-9).numpy(), aspect='auto', origin='lower')
    axes[0].set_title(f"Original Spectrogram - Seg {title_id}")
    
    # Processed (taking first channel)
    spec_proc = spec_transform(processed[0:1])
    axes[1].imshow(torch.log10(spec_proc[0] + 1e-9).numpy(), aspect='auto', origin='lower')
    axes[1].set_title(f"Processed (Mmap) Spectrogram - Seg {title_id}")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    mmap_root = Path("D:/data_out/verify_run")
    work_root = Path("D:/data_out/work_verify")
    
    mmap_index_path = mmap_root / "mmap_index.json"
    segments_manifest_path = work_root / "segments/manifest.jsonl"
    cache_manifest_path = work_root / "cache/manifest.jsonl"
    
    if not mmap_index_path.exists():
        print(f"Error: {mmap_index_path} not found")
        return

    # Load metadata
    with open(mmap_index_path, "r") as f:
        mmap_meta = json.load(f)
    
    sample_rate = mmap_meta["sample_rate"]
    channels = mmap_meta["channels"]
    segment_samples = mmap_meta["segment_samples"]
    count = mmap_meta["count"]
    pcm_scale = mmap_meta["pcm_scale"]
    
    # Load manifests
    seg_manifest = []
    with open(segments_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            seg_manifest.append(json.loads(line))
            
    cache_manifest = {}
    with open(cache_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cache_manifest[item["path"]] = item["src"]

    # Randomly pick 3 indices
    test_indices = random.sample(range(count), min(3, count))
    
    mmap_file = mmap_root / "mmap/audio_i16.mmap"
    mmap_data = np.memmap(mmap_file, dtype='int16', mode='r', shape=(count, channels, segment_samples))
    
    results = []
    
    for idx in test_indices:
        print(f"\nVerifying segment {idx}...")
        
        # 1. Get mmap data
        mmap_seg_np = mmap_data[idx]
        mmap_seg = torch.from_numpy(mmap_seg_np).float() / pcm_scale
        
        # 2. Find source
        seg_info = seg_manifest[idx]
        cached_rel_path = seg_info["src"].replace(str(work_root / "cache") + os.sep, "").replace("\\", "/")
        original_src = cache_manifest.get(cached_rel_path)
        
        if not original_src:
            print(f"Warning: Could not find original source for {cached_rel_path}")
            continue
            
        print(f"Original source: {original_src}")
        
        # 3. Load and process original
        orig_wav, orig_sr = load_audio_robust(original_src, sample_rate)
        orig_wav, _ = resample_if_needed(orig_wav, orig_sr, sample_rate)
        orig_wav = ensure_channels(orig_wav, channels)
        
        # Crop to the same segment
        start = seg_info["start"]
        end = seg_info["end"]
        orig_seg = orig_wav[:, start:end]
        
        # Ensure length matches
        if orig_seg.shape[1] < segment_samples:
            padding = segment_samples - orig_seg.shape[1]
            orig_seg = torch.nn.functional.pad(orig_seg, (0, padding))
        elif orig_seg.shape[1] > segment_samples:
            orig_seg = orig_seg[:, :segment_samples]
            
        # 4. Compare
        snr = calculate_snr(orig_seg, mmap_seg)
        mse = calculate_mse(orig_seg, mmap_seg)
        
        print(f"  SNR: {snr:.2f} dB")
        print(f"  MSE: {mse:.2e}")
        
        # 5. Spectrogram
        stem = Path(original_src).stem
        # Use ASCII-only filename and title ID to avoid character encoding issues
        plot_path = f"spectrogram_seg_{idx}.png"
        plot_spectrograms(orig_seg, mmap_seg, f"{idx}", plot_path)
        print(f"  Spectrogram saved to {plot_path}")
        
        results.append({
            "index": idx,
            "source": original_src,
            "snr": snr,
            "mse": mse,
            "plot": plot_path
        })

    # Summary
    print("\n=== Verification Summary ===")
    avg_snr = sum(r["snr"] for r in results) / len(results)
    print(f"Average SNR: {avg_snr:.2f} dB")
    
    # Save results to json
    with open("verification_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
