#!/usr/bin/env python3
"""
CLI entry point for pliteWAVpipeline.

This module provides a unified command-line interface to run the full data processing pipeline
or individual steps (cache, slice, pack).

Usage:
    # Run the complete pipeline
    plitewav-run run --input_dir /path/to/audio --output_dir /path/to/output

    # Or run step by step
    plitewav-run cache --input_dir /path/to/audio --output_dir /path/to/cache
    plitewav-run slice --input_dir /path/to/cache --output_dir /path/to/segments
    plitewav-run pack --input_dir /path/to/segments --output_dir /path/to/output
"""

import argparse
import sys
from pathlib import Path

from . import (
    DEFAULT_MAX_SEGMENT_S,
    DEFAULT_MIN_SEGMENT_S,
    DEFAULT_SAMPLE_RATE,
    cache_audio_files,
    cut_segments,
    pack_memmap,
    scan_files,
)


def cmd_cache(args):
    """Run audio caching step."""
    files = scan_files(args.input_dir)
    print(f"Found {len(files)} audio files in {args.input_dir}")

    count = cache_audio_files(
        files,
        args.output_dir,
        sample_rate=args.sample_rate,
        force_stereo=not args.mono,
    )
    print(f"Successfully cached {count} files to {args.output_dir}")


def cmd_slice(args):
    """Run audio slicing step."""
    result = cut_segments(
        data_dirs=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        min_segment_s=args.min_segment,
        max_segment_s=args.max_segment,
        output_format=args.format,
        overwrite=args.overwrite,
        resume=args.resume,
        silence_threshold_db=args.silence_threshold_db,
    )
    print(f"Created {result['shards']} shards with {result['total_segments']} total segments.")


def cmd_pack(args):
    """Run memory mapping step."""
    result = pack_memmap(
        segments_dir=args.input_dir,
        output_dir=args.output_dir,
        max_size=args.max_size,
        quality_sort=args.quality_sort,
    )
    print(f"\nOutput: {result['count']} segments, {result['channels']} channels, {result['segment_samples']} samples/segment")


def cmd_run(args):
    """Run the complete pipeline."""
    import shutil

    src_dir = Path(args.input_dir).resolve()
    dst_dir = Path(args.output_dir).resolve()
    work_dir = Path(args.work_dir).resolve()

    work_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Cache
    if not args.skip_cache:
        print("\n=== Step 1: Caching Audio ===")
        cache_dir = work_dir / "cache"
        cache_dir.mkdir(exist_ok=True)

        files = scan_files(str(src_dir))
        print(f"Found {len(files)} audio files in {src_dir}")

        count = cache_audio_files(
            files,
            str(cache_dir),
            sample_rate=args.sample_rate,
            force_stereo=not args.mono,
        )
        print(f"Successfully cached {count} files")
        input_for_slice = str(cache_dir)
    else:
        print("\n=== Skipping Cache ===")
        input_for_slice = str(src_dir)

    # Step 2: Slice
    if not args.skip_slice:
        print("\n=== Step 2: Slicing Audio ===")
        segments_dir = work_dir / "segments"
        segments_dir.mkdir(exist_ok=True)

        result = cut_segments(
            data_dirs=input_for_slice,
            output_dir=str(segments_dir),
            sample_rate=args.sample_rate,
            min_segment_s=args.min_segment,
            max_segment_s=args.max_segment,
            overwrite=args.overwrite,
            resume=not args.overwrite,
        )
        print(f"Created {result['shards']} shards with {result['total_segments']} total segments.")
        input_for_pack = str(segments_dir)
    else:
        print("\n=== Skipping Slice ===")
        input_for_pack = input_for_slice

    # Step 3: Pack
    if not args.skip_pack:
        print("\n=== Step 3: Packing Memmap ===")
        result = pack_memmap(
            segments_dir=input_for_pack,
            output_dir=str(dst_dir),
            max_size=args.max_size,
        )
        print(f"Output: {result['count']} segments, {result['channels']} channels")
    else:
        print("\n=== Skipping Pack ===")

    print(f"\n✅ Pipeline completed! Output: {dst_dir}")

    # Cleanup if requested
    if args.cleanup and work_dir.exists():
        print(f"\nCleaning up work directory: {work_dir}")
        shutil.rmtree(work_dir)


def main():
    parser = argparse.ArgumentParser(
        description="pliteWAVpipeline - Audio Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Cache command
    cache_parser = subparsers.add_parser("cache", help="Cache audio files")
    cache_parser.add_argument("--input_dir", "-i", required=True, help="Input directory with audio files")
    cache_parser.add_argument("--output_dir", "-o", required=True, help="Output directory for cached files")
    cache_parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Target sample rate")
    cache_parser.add_argument("--mono", action="store_true", help="Keep mono audio instead of forcing stereo")

    # Slice command
    slice_parser = subparsers.add_parser("slice", help="Slice audio into segments")
    slice_parser.add_argument("--input_dir", "-i", required=True, help="Input directory with cached audio")
    slice_parser.add_argument("--output_dir", "-o", required=True, help="Output directory for segments")
    slice_parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Target sample rate")
    slice_parser.add_argument("--min_segment", type=float, default=DEFAULT_MIN_SEGMENT_S, help="Min segment seconds")
    slice_parser.add_argument("--max_segment", type=float, default=DEFAULT_MAX_SEGMENT_S, help="Max segment seconds")
    slice_parser.add_argument("--format", choices=["pt", "wav", "both"], default="pt", help="Output format")
    slice_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    slice_parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    slice_parser.add_argument("--silence_threshold_db", type=float, default=-30.0, help="Silence threshold in dB")

    # Pack command
    pack_parser = subparsers.add_parser("pack", help="Pack segments into memmap")
    pack_parser.add_argument("--input_dir", "-i", required=True, help="Input directory with segments")
    pack_parser.add_argument("--output_dir", "-o", required=True, help="Output directory for mmap")
    pack_parser.add_argument("--max_size", type=str, help="Max output size (e.g., 1.5G)")
    pack_parser.add_argument("--quality_sort", action="store_true", help="Sort by quality score")

    # Run (full pipeline) command
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("--input_dir", "-i", required=True, help="Source audio directory")
    run_parser.add_argument("--output_dir", "-o", required=True, help="Final output directory")
    run_parser.add_argument("--work_dir", "-w", default="work_dir", help="Working directory for intermediate files")
    run_parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Target sample rate")
    run_parser.add_argument("--min_segment", type=float, default=DEFAULT_MIN_SEGMENT_S, help="Min segment seconds")
    run_parser.add_argument("--max_segment", type=float, default=DEFAULT_MAX_SEGMENT_S, help="Max segment seconds")
    run_parser.add_argument("--max_size", type=str, help="Max mmap size (e.g., 1.5G)")
    run_parser.add_argument("--mono", action="store_true", help="Keep mono instead of stereo")
    run_parser.add_argument("--cleanup", action="store_true", help="Clean up intermediate files")
    run_parser.add_argument("--skip_cache", action="store_true", help="Skip cache step")
    run_parser.add_argument("--skip_slice", action="store_true", help="Skip slice step")
    run_parser.add_argument("--skip_pack", action="store_true", help="Skip pack step")
    run_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to appropriate command handler
    if args.command == "cache":
        cmd_cache(args)
    elif args.command == "slice":
        cmd_slice(args)
    elif args.command == "pack":
        cmd_pack(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()