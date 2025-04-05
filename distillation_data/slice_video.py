import os
import json
import argparse
import cv2
from tqdm import tqdm
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Process videos based on annotation json')
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing input videos')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save processed videos')
    parser.add_argument('--anno_json', type=str, required=True, help='Path to annotation json file')
    parser.add_argument('--output_json', type=str, default='cap_to_video.json', help='Name of the output json file')
    return parser.parse_args()

def is_ffmpeg_available():
    """Check if ffmpeg is available on the system"""
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE,
                      check=False)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def cut_video_ffmpeg(input_path, output_path, start_frame, end_frame, fps):
    """Slice video using ffmpeg with [start_frame, end_frame) range"""
    try:
        # Calculate start and end times in seconds (end_frame is exclusive)
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps
        
        # Use -ss before -i for accurate seeking
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "medium", 
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-loglevel", "error",
            output_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            # Verify the output file
            verify_cap = cv2.VideoCapture(output_path)
            if not verify_cap.isOpened():
                return False
            verify_cap.release()
            return True
        else:
            error = result.stderr.decode()
            if error:
                tqdm.write(f"FFmpeg error: {error[:100]}...")
            return False
    except Exception as e:
        tqdm.write(f"Error using ffmpeg: {str(e)}")
        return False

def process_video_segments(input_path, output_folder, segments):
    """Process multiple segments from the same video file using ffmpeg"""
    # Check if input video exists
    if not os.path.exists(input_path):
        tqdm.write(f"Error: Input video not found: {input_path}")
        return {}

    # Extract base name for output files
    video_name = os.path.basename(input_path)
    base_name = os.path.splitext(video_name)[0]
    
    # Get video properties using FFmpeg
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-select_streams", "v:0", 
        "-show_entries", "stream=r_frame_rate,nb_frames", 
        "-of", "csv=p=0", 
        input_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        fps_str, frames_str = result.stdout.strip().split(',')
        
        # Parse FPS (which may be in form "30000/1001")
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den
        else:
            fps = float(fps_str)
        
        total_frames = int(frames_str) if frames_str else 0
        
        # If couldn't get frames, use cv2 as fallback just for counting frames
        if total_frames <= 0:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                tqdm.write(f"Error: Cannot open video: {input_path}")
                return {}
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0:
                fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
    except Exception as e:
        # Fallback to OpenCV for just getting video properties
        tqdm.write(f"Warning: FFprobe failed, using OpenCV to get video properties: {str(e)}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            tqdm.write(f"Error: Cannot open video: {input_path}")
            return {}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    
    # Results mapping
    results = {}
    
    # Process each segment with FFmpeg
    for segment in tqdm(segments, desc=f"Processing segments", leave=False):
        # Get frame range [start_frame, end_frame)
        frame_idx = segment['frame_idx']
        start_frame, end_frame = map(int, frame_idx.split(':'))
        
        # Validate frame range
        if start_frame >= total_frames:
            tqdm.write(f"Error: Start frame {start_frame} exceeds total frames {total_frames}")
            continue
        
        if end_frame > total_frames:
            tqdm.write(f"Warning: End frame {end_frame} exceeds total frames {total_frames}, clamping to {total_frames}")
            end_frame = total_frames
        
        # Generate output path
        output_name = f"{base_name}_{start_frame}_{end_frame}.mp4"
        output_path = os.path.join(output_folder, output_name)
        
        # Skip if already exists
        if os.path.exists(output_path):
            results[segment['cap']] = output_name
            continue
        
        # Process with FFmpeg
        if cut_video_ffmpeg(input_path, output_path, start_frame, end_frame, fps):
            results[segment['cap']] = output_name
    
    return results

def main():
    args = parse_args()
    
    # Verify FFmpeg is available
    if not is_ffmpeg_available():
        print("Error: FFmpeg is required but not found in the system. Please install FFmpeg and try again.")
        sys.exit(1)
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Read annotations
    with open(args.anno_json, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Caption to video mapping
    cap_to_video = {}
    
    # Group annotations by original video
    video_segments = {}
    for annotation in annotations:
        full_path = annotation['path']
        video_name = os.path.basename(full_path)
        if video_name not in video_segments:
            video_segments[video_name] = []
        video_segments[video_name].append(annotation)
    
    print(f"Processing {len(video_segments)} videos")
    
    # Process each video
    stats = {"processed": 0, "skipped": 0, "failed": 0}
    
    with tqdm(total=len(video_segments), desc="Processing videos") as pbar:
        for video_name, segments in video_segments.items():
            input_video_path = os.path.join(args.input_folder, video_name)
            
            # Process all segments of this video at once
            pbar.set_description(f"Video: {video_name[:20]} ({len(segments)} segments)")
            
            # Check for pre-existing segments
            pre_existing = 0
            for segment in segments:
                frame_idx = segment['frame_idx']
                start_frame, end_frame = map(int, frame_idx.split(':'))
                output_name = f"{os.path.splitext(video_name)[0]}_{start_frame}_{end_frame}.mp4"
                output_path = os.path.join(args.output_folder, output_name)
                
                if os.path.exists(output_path):
                    cap_to_video[segment['cap']] = output_name
                    pre_existing += 1
            
            # Skip processing if all segments already exist
            if pre_existing == len(segments):
                stats["skipped"] += len(segments)
                pbar.update(1)
                continue
            
            # Process remaining segments
            segment_results = process_video_segments(
                input_video_path, 
                args.output_folder, 
                [s for s in segments if s['cap'] not in cap_to_video]
            )
            
            # Update mappings and stats
            for cap, video_file in segment_results.items():
                cap_to_video[cap] = video_file
                stats["processed"] += 1
            
            # Count failed segments
            failed = len(segments) - pre_existing - len(segment_results)
            stats["skipped"] += pre_existing
            stats["failed"] += failed
            
            # Update progress
            pbar.update(1)
    
    # Save caption to video mapping
    output_json_path = args.output_json
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(cap_to_video, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print(f"Processed: {stats['processed']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
    print(f"Processing complete. Mapping saved to: {output_json_path}")

if __name__ == "__main__":
    main()
