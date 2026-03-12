"""
pliteWAVpipeline - Audio Data Processing Pipeline

A comprehensive audio data processing pipeline for machine learning applications.
Supports audio caching, intelligent slicing, memmap packing, and efficient data loading.

Usage:
    # Cache audio files
    from plitewavpipeline import cache_audio_files
    cache_audio_files(files, "output_dir", sample_rate=48000)
    
    # Slice audio into segments
    from plitewavpipeline import cut_segments
    cut_segments("input_dir", "output_dir", min_segment_s=5.0, max_segment_s=5.0)
    
    # Pack into memmap
    from plitewavpipeline import pack_memmap
    pack_memmap("segments_dir", "output_dir")
    
    # Load with dataset
    from plitewavpipeline import MemmapDataset
    ds = MemmapDataset("output_dir/mmap")
"""

__version__ = "1.0.0"
__author__ = "pliteWAVpipeline Team"

# Core modules
from .utils import (
    # Constants
    DEFAULT_SAMPLE_RATE,
    DEFAULT_MIN_SEGMENT_S,
    DEFAULT_MAX_SEGMENT_S,
    DEFAULT_SILENCE_THRESHOLD_DB,
    DEFAULT_ANALYSIS_FRAME_MS,
    DEFAULT_ANALYSIS_HOP_MS,
    DEFAULT_MIN_SILENCE_MS,
    PT_PCM_SCALE,
    # Functions
    is_fake_stereo,
    ensure_channels,
    force_stereo,
    resample_if_needed,
    atomic_write_text,
    atomic_torch_save,
    get_stable_name,
    scan_files,
    load_audio_robust,
)

from .caching import (
    cache_audio_files,
    cache_audio_directory,
)

from .slicing import (
    EnergyVAD,
    FenwickTree,
    cut_segments,
)

from .packing import (
    pack_memmap,
    parse_size,
)

from .dataset import (
    # Datasets
    AudioDataset,
    SegmentShardDataset,
    MemmapDataset,
    WavSegmentDataset,
    MemmapSegmentDataset,
    # Samplers
    ShardGroupedBatchSampler,
    # Utilities
    collate_audio_batch,
    scan_files as scan_audio_files,
    scan_wav_segments,
    create_dataloader,
)

# Convenience imports
__all__ = [
    # Version
    "__version__",
    # Utils
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_MIN_SEGMENT_S",
    "DEFAULT_MAX_SEGMENT_S",
    "DEFAULT_SILENCE_THRESHOLD_DB",
    "DEFAULT_ANALYSIS_FRAME_MS",
    "DEFAULT_ANALYSIS_HOP_MS",
    "DEFAULT_MIN_SILENCE_MS",
    "PT_PCM_SCALE",
    "is_fake_stereo",
    "ensure_channels",
    "force_stereo",
    "resample_if_needed",
    "atomic_write_text",
    "atomic_torch_save",
    "get_stable_name",
    "scan_files",
    "load_audio_robust",
    # Caching
    "cache_audio_files",
    "cache_audio_directory",
    # Slicing
    "EnergyVAD",
    "FenwickTree",
    "cut_segments",
    # Packing
    "pack_memmap",
    "parse_size",
    # Dataset
    "AudioDataset",
    "SegmentShardDataset",
    "MemmapDataset",
    "WavSegmentDataset",
    "MemmapSegmentDataset",
    "ShardGroupedBatchSampler",
    "collate_audio_batch",
    "scan_audio_files",
    "scan_wav_segments",
    "create_dataloader",
]