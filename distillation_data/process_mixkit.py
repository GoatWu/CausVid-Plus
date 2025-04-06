# the following code is taken from FastVideo https://github.com/hao-ai-lab/FastVideo/tree/main
# Apache-2.0 License

import argparse
import logging
import time
import math
import psutil
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from moviepy.editor import VideoFileClip
from skimage.transform import resize
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('video_processing.log')])

def check_memory():
    """Check current process memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def is_16_9_ratio(width: int, height: int, tolerance: float = 0.1) -> bool:
    target_ratio = 16 / 9
    actual_ratio = width / height
    return abs(actual_ratio - target_ratio) <= (target_ratio * tolerance)

def resize_video(args_tuple):
    """
    Resize a single video file while preserving all frames using OpenCV.
    args_tuple: (input_file, output_dir, width, height, fps)
    """
    input_file, output_dir, width, height, fps = args_tuple
    output_file = output_dir / f"{input_file.stem}.mp4"  # Ensure .mp4 extension

    if output_file.exists():
        output_file.unlink()

    try:
        # Open input video
        cap = cv2.VideoCapture(str(input_file))
        if not cap.isOpened():
            return (input_file.name, "failed", "Could not open video file")

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not is_16_9_ratio(original_width, original_height):
            cap.release()
            return (input_file.name, "skipped", "Not 16:9")

        # Create output video writer with MP4 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 codec
        out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Use cv2.resize instead of skimage for better performance
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # Write frame
            out.write(resized)
            frame_count += 1

        # Release resources
        cap.release()
        out.release()
        
        if frame_count != total_frames:
            return (input_file.name, "failed", f"Frame count mismatch: original {total_frames}, processed {frame_count}")
            
        return (input_file.name, "success", None)
            
    except Exception as e:
        logging.error(f"Error processing {input_file.name}: {str(e)}")
        return (input_file.name, "failed", str(e))
    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()

def process_folder(args):
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    all_video_files = [f for f in input_path.iterdir() if f.is_file()
                   and f.suffix.lower() in video_extensions]

    if not all_video_files:
        print(f"No video files found in {args.input_dir}")
        return

    # Filter out already processed files
    video_files = [f for f in all_video_files if not (output_path / f"{f.stem}.mp4").exists()]
    total_videos = len(all_video_files)
    existing_videos = total_videos - len(video_files)
    
    print(f"\nVideo Processing Summary:")
    print(f"Total videos found: {total_videos}")
    print(f"Already processed: {existing_videos}")
    print(f"Videos to process: {len(video_files)}")
    
    if not video_files:
        print(f"All videos in {args.input_dir} have been processed")
        return

    print(f"\nTarget resolution: {args.width}x{args.height} at {args.fps}fps")

    # Process all videos in parallel
    max_workers = getattr(args, 'max_workers', 4)
    successful = 0
    skipped = 0
    failed = []

    process_args = [(video_file, output_path, args.width,
                    args.height, args.fps) for video_file in video_files]

    with tqdm(total=len(video_files), desc="Processing videos", dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(resize_video, arg): arg[0] for arg in process_args}

            for future in as_completed(future_to_file):
                filename, status, message = future.result()
                if status == "success":
                    successful += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed.append((filename, message))
                pbar.update(1)
                
                # Log memory usage
                logging.info(f"Current memory usage: {check_memory()} MB")

    # Print final summary
    print(f"\nProcessing Summary:")
    print(f"Total videos: {total_videos}")
    print(f"Already processed: {existing_videos}")
    print(f"Newly processed: {successful}")
    print(f"Skipped (not 16:9): {skipped}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed files:")
        for fname, error in failed:
            print(f"- {fname}: {error}")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch resize videos to specified resolution and FPS (16:9 only)')
    parser.add_argument('--input_dir', required=True,
                        help='Input directory containing video files')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for processed videos')
    parser.add_argument('--width', type=int, default=832,
                        help='Target width in pixels (default: 832)')
    parser.add_argument('--height', type=int, default=480,
                        help='Target height in pixels (default: 480)')
    parser.add_argument('--fps', type=int, default=16,
                        help='Target frames per second (default: 16)')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of worker threads (default: 4)')
    parser.add_argument('--log-level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO',
                        help='Set the logging level (default: INFO)')
    return parser.parse_args()

def main():
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if not Path(args.input_dir).exists():
        logging.error(f"Input directory not found: {args.input_dir}")
        return

    start_time = time.time()
    process_folder(args)
    duration = time.time() - start_time
    logging.info(f"Batch processing completed in {duration:.2f} seconds")

if __name__ == "__main__":
    main()
