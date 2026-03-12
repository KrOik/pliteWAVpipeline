#!/usr/bin/env python3
"""
Unified CLI for pliteWAVpipeline.

This script provides a unified command-line interface to run the full data processing pipeline
or individual steps (cache, slice, pack).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plitewavpipeline import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_MIN_SEGMENT_S,
    DEFAULT_MAX_SEGMENT_S,
)


def run_command(cmd, env=None):
    """Run a command and check for errors."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, env=env)
    return result.returncode


def check_disk_space(src_dir, work_dir, dst_dir):
    """Check available disk space before processing."""
    import shutil
    
    try:
        src_usage = shutil.disk_usage(src_dir)
        work_usage = shutil.disk_usage(work_dir)
        dst_usage = shutil.disk_usage(dst_dir)
        
        print(f"\n=== Disk Space Check ===")
        print(f"Source: {src_usage.free / (1024**3):.2f} GB free")
        print(f"Work: {work_usage.free / (1024**3):.2f} GB free")
        print(f"Dest: {dst_usage.free / (1024**3):.2f} GB free")
        
    except Exception as e:
        print(f"Warning: Could not check disk space: {e}")


def cmd_cache(args):
    """Run audio caching step."""
    from plitewavpipeline import cache_audio_files, scan_files
    
    files = scan_files(args.input_dir)
    print(f"Found {len(files)} audio files in {args.input_dir}")
    
    from plitewavpipeline import cache_audio_files
    count = cache_audio_files(
        files,
        args.output_dir,
        sample_rate=args.sample_rate,
        force_stereo=not args.mono,
    )
    print(f"Successfully cached {count} files to {args.output_dir}")


def cmd_slice(args):
    """Run audio slicing step."""
    from plitewavpipeline import cut_segments
    
    result = cut_segments(
        data_dirs=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        min_segment_s=args.min_segment,
        max_segment_s=args.max_segment,
        output_format=args.format,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    print(f"Created {result['shards']} shards with {result['total_segments']} segments")


def cmd_pack(args):
    """Run memmap packing step."""
    from plitewavpipeline import pack_memmap
    
    result = pack_memmap(
        segments_dir=args.input_dir,
        output_dir=args.output_dir,
        max_size=args.max_size,
        quality_sort=args.quality_sort,
    )
    print(f"Packed {result['count']} segments into {result['output_path']}")


def cmd_run(args):
    """Run full pipeline."""
    src_dir = Path(args.input_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    dst_dir = Path(args.output_dir).resolve()
    
    # Create directories
    work_dir.mkdir(parents=True, exist_ok=True)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Check disk space
    if not args.skip_space_check:
        check_disk_space(src_dir, work_dir, dst_dir)
    
    # Set up environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    
    # Step 1: Cache (if not skipped)
    if not args.skip_cache:
        print("\n=== Step 1: Caching Audio ===")
        cache_dir = work_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable, "-m", "plitewavpipeline.caching",
            "--data_dirs", str(src_dir),
            "--output_dir", str(cache_dir),
            "--sample_rate", str(args.sample_rate),
        ]
        if args.mono:
            cmd.append("--mono")
        
        run_command(cmd, env)
        input_for_slice = str(cache_dir)
    else:
        print("\n=== Skipping Cache (using existing cache) ===")
        input_for_slice = str(src_dir)
    
    # Step 2: Slice
    if not args.skip_slice:
        print("\n=== Step 2: Slicing Audio ===")
        segments_dir = work_dir / "segments"
        
        cmd = [
            sys.executable, "-m", "plitewavpipeline.slicing",
            "--data_dirs", input_for_slice,
            "--output_dir", str(segments_dir),
            "--sample_rate", str(args.sample_rate),
            "--min_segment_s", str(args.min_segment),
            "--max_segment_s", str(args.max_segment),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        else:
            cmd.append("--resume")
        
        run_command(cmd, env)
        input_for_pack = str(segments_dir)
    else:
        print("\n=== Skipping Slice (using existing segments) ===")
        input_for_pack = str(src_dir)
    
    # Step 3: Pack
    if not args.skip_pack:
        print("\n=== Step 3: Packing Memmap ===")
        
        cmd = [
            sys.executable, "-m", "plitewavpipeline.packing",
            "--segments_dir", input_for_pack,
            "--output_dir", str(dst_dir),
        ]
        if args.max_size:
            cmd.extend(["--max_size", args.max_size])
        
        run_command(cmd, env)
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
    run_parser.add_argument("--skip_space_check", action="store_true", help="Skip disk space check")
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()