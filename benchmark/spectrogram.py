#!/usr/bin/env python3
"""
Generate spectrogram comparison for pipeline verification.
Processes a single FLAC file through the pipeline and compares original vs processed audio.
"""

import os
import sys
import json
import tempfile
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plitewavpipeline import (
    cache_audio_files,
    cut_segments,
    pack_memmap,
    MemmapDataset,
    load_audio_robust,
    resample_if_needed,
    ensure_channels,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_MIN_SEGMENT_S,
    DEFAULT_MAX_SEGMENT_S,
)

SAMPLE_RATE = 48000
MIN_SEG = 5.0
MAX_SEG = 5.0


def create_spectrogram(wav, sample_rate=48000):
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        window_fn=torch.hann_window
    )
    spec = spec_transform(wav)
    spec_db = 10 * torch.log10(spec + 1e-9)
    return spec_db, spec_transform.n_fft, sample_rate


def plot_spectrogram_comparison(original_wav, processed_wav, sample_rate, title_id, save_path, track_title=""):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    orig_spec, n_fft, sr = create_spectrogram(original_wav, sample_rate)
    proc_spec, _, _ = create_spectrogram(processed_wav, sample_rate)
    
    vmin = min(orig_spec.min().item(), proc_spec.min().item())
    vmax = max(orig_spec.max().item(), proc_spec.max().item())
    
    freq_bins = orig_spec.shape[0]
    max_freq = sr // 2
    tick_interval = 2000
    tick_positions = list(range(0, max_freq + 1, tick_interval))
    bin_positions = [int(freq * n_fft / sr) for freq in tick_positions]
    
    title_with_track = f'{track_title}' if track_title else ''
    
    im1 = axes[0].imshow(orig_spec[0].numpy(), aspect='auto', origin='lower',
                         cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Original: {track_title}', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[0].set_xlabel('Time Frame', fontsize=12)
    axes[0].yaxis.set_major_locator(FixedLocator(bin_positions))
    axes[0].set_yticklabels([f'{freq}' for freq in tick_positions])
    plt.colorbar(im1, ax=axes[0], label='dB')
    
    im2 = axes[1].imshow(proc_spec[0].numpy(), aspect='auto', origin='lower',
                         cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Processed (Memmap): {track_title}', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[1].set_xlabel('Time Frame', fontsize=12)
    axes[1].yaxis.set_major_locator(FixedLocator(bin_positions))
    axes[1].set_yticklabels([f'{freq}' for freq in tick_positions])
    plt.colorbar(im2, ax=axes[1], label='dB')
    
    plt.suptitle(f'Spectrogram Comparison - Segment {title_id}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def calculate_metrics(original, processed):
    noise = original - processed
    ref_power = torch.mean(original ** 2)
    noise_power = torch.mean(noise ** 2)
    
    if noise_power < 1e-10:
        snr = float('inf')
    else:
        snr = 10 * torch.log10(ref_power / noise_power).item()
    
    mse = torch.mean((original - processed) ** 2).item()
    max_diff = torch.max(torch.abs(original - processed)).item()
    
    return {
        'snr': snr,
        'mse': mse,
        'max_diff': max_diff
    }


def main():
    flac_file = r"C:\Users\29668\Documents\VmMix4\perthlitePP2L\pliteWAVpipeline\tmp\perthlite\testWav\Give Me Something (for Arknights Endfield) - OneRepublic.flac"
    track_title = "Give Me Something (for Arknights Endfield) - OneRepublic"
    
    if not os.path.exists(flac_file):
        print(f"Error: FLAC file not found: {flac_file}")
        return
    
    with tempfile.TemporaryDirectory() as work_dir:
        cache_dir = os.path.join(work_dir, "cache")
        segments_dir = os.path.join(work_dir, "segments")
        mmap_dir = os.path.join(work_dir, "mmap")
        output_dir = r"C:\Users\29668\Documents\VmMix4\perthlitePP2L\pliteWAVpipeline\benchmark\spectrogram_output"
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(segments_dir, exist_ok=True)
        os.makedirs(mmap_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Input: {flac_file}")
        print(f"Working directory: {work_dir}")
        
        print("\n[Step 1] Caching audio...")
        files = [flac_file]
        cached_count = cache_audio_files(
            files,
            cache_dir,
            sample_rate=SAMPLE_RATE,
            force_stereo=True
        )
        print(f"Cached {cached_count} files")
        
        print("\n[Step 2] Slicing audio...")
        result = cut_segments(
            data_dirs=cache_dir,
            output_dir=segments_dir,
            sample_rate=SAMPLE_RATE,
            min_segment_s=MIN_SEG,
            max_segment_s=MAX_SEG,
            output_format='pt',
            resume=True
        )
        print(f"Created {result['total_segments']} segments")
        
        print("\n[Step 3] Packing to memmap...")
        pack_result = pack_memmap(
            segments_dir=segments_dir,
            output_dir=mmap_dir,
            quality_sort=False
        )
        print(f"Packed {pack_result['count']} segments to {pack_result['output_path']}")
        
        print("\n[Step 4] Loading and comparing...")
        
        mmap_index_path = os.path.join(mmap_dir, "mmap_index.json")
        with open(mmap_index_path, 'r') as f:
            mmap_meta = json.load(f)
        
        sample_rate = mmap_meta["sample_rate"]
        channels = mmap_meta["channels"]
        segment_samples = mmap_meta["segment_samples"]
        count = mmap_meta["count"]
        pcm_scale = mmap_meta.get("pcm_scale", 32768.0)
        
        mmap_file = os.path.join(mmap_dir, mmap_meta["path"])
        mmap_data = np.memmap(mmap_file, dtype='int16', mode='r', 
                              shape=(count, channels, segment_samples))
        
        segments_manifest = os.path.join(segments_dir, "manifest.jsonl")
        seg_manifest = []
        with open(segments_manifest, 'r', encoding='utf-8') as f:
            for line in f:
                seg_manifest.append(json.loads(line))
        
        cache_manifest = os.path.join(cache_dir, "manifest.jsonl")
        cache_info = {}
        with open(cache_manifest, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                cache_info[os.path.basename(item['path'])] = item['src']
        
        print(f"Found {count} segments in mmap")
        
        all_metrics = []
        
        mid_start = max(0, (count - 3) // 2)
        test_indices = list(range(mid_start, mid_start + 3))
        print(f"Testing middle segments: {test_indices}")
        
        for idx in test_indices:
            print(f"\n--- Segment {idx} ---")
            
            mmap_seg_np = mmap_data[idx]
            mmap_seg = torch.from_numpy(np.array(mmap_seg_np, copy=True)).float() / pcm_scale
            
            seg_info = seg_manifest[idx]
            cached_filename = os.path.basename(seg_info['src'])
            original_src = cache_info.get(cached_filename)
            
            if not original_src:
                print(f"Warning: Could not find original source")
                continue
            
            orig_wav, orig_sr = load_audio_robust(original_src, sample_rate)
            orig_wav, _ = resample_if_needed(orig_wav, orig_sr, sample_rate)
            orig_wav = ensure_channels(orig_wav, channels)
            
            start = seg_info['start']
            end = seg_info['end']
            orig_seg = orig_wav[:, start:end]
            
            if orig_seg.shape[1] < segment_samples:
                padding = segment_samples - orig_seg.shape[1]
                orig_seg = torch.nn.functional.pad(orig_seg, (0, padding))
            elif orig_seg.shape[1] > segment_samples:
                orig_seg = orig_seg[:, :segment_samples]
            
            metrics = calculate_metrics(orig_seg, mmap_seg)
            all_metrics.append(metrics)
            
            print(f"  SNR: {metrics['snr']:.2f} dB")
            print(f"  MSE: {metrics['mse']:.2e}")
            print(f"  Max Diff: {metrics['max_diff']:.2e}")
            
            plot_path = os.path.join(output_dir, f"spectrogram_comparison_seg_{idx}.png")
            plot_spectrogram_comparison(orig_seg, mmap_seg, sample_rate, idx, plot_path, track_title)
        
        if all_metrics:
            avg_snr = sum(m['snr'] for m in all_metrics) / len(all_metrics)
            avg_mse = sum(m['mse'] for m in all_metrics) / len(all_metrics)
            avg_max_diff = sum(m['max_diff'] for m in all_metrics) / len(all_metrics)
            
            print(f"\n=== Overall Metrics ===")
            print(f"Average SNR: {avg_snr:.2f} dB")
            print(f"Average MSE: {avg_mse:.2e}")
            print(f"Average Max Diff: {avg_max_diff:.2e}")
            
            summary = {
                'input_file': flac_file,
                'segments_tested': len(all_metrics),
                'avg_snr_db': avg_snr,
                'avg_mse': avg_mse,
                'avg_max_diff': avg_max_diff,
                'per_segment': all_metrics
            }
            
            summary_path = os.path.join(output_dir, "verification_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\nSummary saved to: {summary_path}")
            
            print(f"\nGenerated {len(all_metrics)} spectrogram comparison images in:")
            print(f"  {output_dir}")


if __name__ == "__main__":
    main()