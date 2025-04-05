#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Get video fps and other basic information')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--use-ffmpeg', action='store_true', help='Use FFmpeg for more detailed information')
    return parser.parse_args()

def get_video_info_opencv(video_path):
    """Get video information using OpenCV"""
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist")
        return False
    
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_path}'")
            return False
        
        # Get basic video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate video duration
        duration_sec = frame_count / fps if fps > 0 else 0
        mins = int(duration_sec // 60)
        secs = duration_sec % 60
        
        # Print video information
        print("\n=== Video Information (OpenCV) ===")
        print(f"File: {os.path.basename(video_path)}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.2f}")
        print(f"Frame Count: {frame_count}")
        print(f"Duration: {mins} min {secs:.2f} sec")
        
        # Check codec (4-character code)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = chr(fourcc_int & 0xFF) + chr((fourcc_int >> 8) & 0xFF) + chr((fourcc_int >> 16) & 0xFF) + chr((fourcc_int >> 24) & 0xFF)
        print(f"Codec (FourCC): {fourcc}")
        
        # Release the video capture object
        cap.release()
        return True
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def get_video_info_ffmpeg(video_path):
    """Get detailed video information using FFmpeg"""
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist")
        return False
    
    try:
        # Check if ffmpeg is available
        ffmpeg_path = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True).stdout.strip()
        if not ffmpeg_path:
            print("FFmpeg not found. Please install FFmpeg to use this feature.")
            return False
        
        print("\n=== Video Information (FFmpeg) ===")
        
        # Run ffprobe to get video information in JSON format
        cmd = [
            "ffprobe", 
            "-v", "quiet", 
            "-print_format", "json", 
            "-show_format", 
            "-show_streams", 
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running FFmpeg: {result.stderr}")
            return False
        
        # Print raw ffprobe output
        print(result.stdout)
        
        # Run ffprobe for a more human-readable format
        print("\n=== Video Summary ===")
        cmd = ["ffprobe", video_path]
        subprocess.run(cmd, capture_output=False)
        
        return True
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    args = parse_args()
    
    # Always try OpenCV method
    opencv_result = get_video_info_opencv(args.video_path)
    
    # Use FFmpeg for more detailed information if requested
    if args.use_ffmpeg:
        ffmpeg_result = get_video_info_ffmpeg(args.video_path)
    
    if not opencv_result and (not args.use_ffmpeg or not ffmpeg_result):
        sys.exit(1)

if __name__ == "__main__":
    main() 