import os
import json
import argparse
import cv2
from tqdm import tqdm
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Process videos based on annotation json')
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing input videos')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save processed videos')
    parser.add_argument('--anno_json', type=str, required=True, help='Path to annotation json file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output folder (if it doesn't exist)
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Read anno_json file
    with open(args.anno_json, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Dictionary to store caption to video filename mapping
    cap_to_video = {}
    
    # Check if ffmpeg is available
    ffmpeg_available = is_ffmpeg_available()
    if not ffmpeg_available:
        print("Warning: ffmpeg not found, will use OpenCV for video processing")
    
    # Group annotations by original video to process each video only once
    video_segments = {}
    for annotation in annotations:
        full_path = annotation['path']
        video_name = os.path.basename(full_path)
        if video_name not in video_segments:
            video_segments[video_name] = []
        video_segments[video_name].append(annotation)
    
    # Process each video with all its segments
    print(f"Processing {len(video_segments)} original videos")
    
    # Create the main progress bar for processing videos
    with tqdm(total=len(video_segments), desc="Processing videos") as pbar:
        # Counter for processed and skipped segments
        processed_count = 0
        skipped_count = 0
        
        for video_name, segments in video_segments.items():
            input_video_path = os.path.join(args.input_folder, video_name)
            
            # Process all segments of this video
            for annotation in segments:
                # Get frame range
                frame_idx = annotation['frame_idx']
                start_frame, end_frame = map(int, frame_idx.split(':'))
                
                # Update progress bar description
                pbar.set_description(f"Video: {video_name[:20]}... Frame: {start_frame}-{end_frame}")
                
                # Generate output video filename, ensure mp4 extension
                output_video_name = f"{os.path.splitext(video_name)[0]}_{start_frame}_{end_frame}.mp4"
                output_video_path = os.path.join(args.output_folder, output_video_name)
                
                # Check if the video segment already exists in the output folder
                if not os.path.exists(output_video_path):
                    # Process the video
                    if ffmpeg_available:
                        success = cut_video_ffmpeg(input_video_path, output_video_path, start_frame, end_frame)
                        if not success:
                            tqdm.write(f"Failed with ffmpeg, using OpenCV for {video_name} ({start_frame}:{end_frame})")
                            cut_video_opencv(input_video_path, output_video_path, start_frame, end_frame)
                    else:
                        cut_video_opencv(input_video_path, output_video_path, start_frame, end_frame)
                    
                    processed_count += 1
                else:
                    skipped_count += 1
                
                # Update mapping
                cap_to_video[annotation['cap']] = output_video_name
            
            # Update the progress bar after processing all segments of this video
            pbar.update(1)
    
    # Final statistics
    print(f"Processed {processed_count} segments, skipped {skipped_count} existing segments")
    
    # Save caption to video filename mapping as json file in the current directory
    output_json_path = "cap_to_video.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(cap_to_video, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete. The mapping file has been saved to: {output_json_path}")

def is_ffmpeg_available():
    """Check if ffmpeg is available on the system"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=False
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def cut_video_ffmpeg(input_path, output_path, start_frame, end_frame):
    """Slice video using ffmpeg to ensure compatibility with PyAV"""
    try:
        # Get the FPS of the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            tqdm.write(f"Error: Unable to open video {input_path}")
            import time
            time.sleep(10)
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Calculate start and end times in seconds
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps
        
        # First try using direct stream copy for best performance and quality
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "copy",  # Copy video codec
            "-c:a", "copy",  # Copy audio codec if any
            "-avoid_negative_ts", "1",
            "-loglevel", "error",  # Reduce ffmpeg output
            output_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # If direct copy fails, use encoding with h264 (most compatible with PyAV)
        if result.returncode != 0:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", input_path,
                "-t", str(duration),
                "-c:v", "libx264",  # Use H.264 codec for compatibility
                "-preset", "medium", 
                "-crf", "23",       # Reasonable quality
                "-pix_fmt", "yuv420p",  # Widely supported pixel format
                "-movflags", "+faststart",  # Optimize for web streaming
                "-loglevel", "error",  # Reduce ffmpeg output
                output_path
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            # Verify the output file can be opened with OpenCV (as a proxy for PyAV compatibility)
            verify_cap = cv2.VideoCapture(output_path)
            if verify_cap.isOpened():
                verify_cap.release()
                return True
            else:
                tqdm.write(f"Warning: Created file cannot be opened with OpenCV")
                return False
        else:
            error = result.stderr.decode()
            if error:
                tqdm.write(f"FFmpeg error: {error[:100]}...")
            return False
            
    except Exception as e:
        tqdm.write(f"Error using ffmpeg: {str(e)}")
        return False

def cut_video_opencv(input_path, output_path, start_frame, end_frame):
    """Slice video according to frame indices using OpenCV"""
    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        tqdm.write(f"Error: Unable to open video {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Use mp4v codec for compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        tqdm.write("Failed to create video writer.")
        return
    
    # Jump to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Calculate total frames to process
    total_frames = end_frame - start_frame
    
    # Read and write the specified range of frames
    frame_count = 0
    while cap.isOpened() and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
    
    # Release resources
    cap.release()
    out.release()
    
    # If OpenCV method is used, convert the output to h264 for better compatibility with PyAV
    if is_ffmpeg_available():
        temp_path = output_path + ".temp.mp4"
        os.rename(output_path, temp_path)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_path,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-loglevel", "error",  # Reduce ffmpeg output
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.remove(temp_path)
        except:
            # If conversion fails, revert to original file
            tqdm.write("Conversion failed, keeping original OpenCV output")
            os.rename(temp_path, output_path)

if __name__ == "__main__":
    main()
