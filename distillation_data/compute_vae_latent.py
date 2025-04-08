from causvid.models.wan.wan_wrapper import WanVAEWrapper
from causvid.util import launch_distributed_job
import torch.distributed as dist
import imageio.v3 as iio
from tqdm import tqdm
import argparse
import torch
import json
import math
import os

torch.set_grad_enabled(False)


def video_to_numpy(video_path):
    """
    Reads a video file and returns a NumPy array containing all frames.

    :param video_path: Path to the video file.
    :return: NumPy array of shape (num_frames, height, width, channels)
    """
    return iio.imread(video_path, plugin="pyav")  # Reads the entire video as a NumPy array


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_folder", type=str,
                        help="Path to the folder containing input videos.")
    parser.add_argument("--output_latent_folder", type=str,
                        help="Path to the folder where output latents will be saved.")
    parser.add_argument("--model_type", type=str, default="T2V-1.3B",
                        help="Name of the model to use.")
    parser.add_argument("--info_path", type=str,
                        help="Path to the info file containing video metadata.")

    args = parser.parse_args()

    # Setup environment
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    # Initialize distributed environment
    launch_distributed_job()
    device = torch.cuda.current_device()

    # Load prompt:video mapping
    with open(args.info_path, "r") as f:
        prompt_info = json.load(f)

    model = WanVAEWrapper(model_type=args.model_type).to(device=device, dtype=torch.bfloat16)
    os.makedirs(args.output_latent_folder, exist_ok=True)

    # Dictionary to store video_path:latent_file_path mapping
    video_latent_map = {}
    
    # Initialize counters
    total_videos = 0
    skipped_videos = 0
    successful_encodings = 0
    failed_encodings = 0

    # Process each prompt:video pair
    prompt_items = list(prompt_info.items())
    for index in tqdm(range(int(math.ceil(len(prompt_items) / dist.get_world_size()))), disable=dist.get_rank() != 0):
        global_index = index * dist.get_world_size() + dist.get_rank()
        if global_index >= len(prompt_items):
            break

        prompt, video_path = prompt_items[global_index]
        output_path = os.path.join(args.output_latent_folder, f"{global_index:08d}.pt")
        
        # Check if video file exists
        full_path = os.path.join(args.input_video_folder, video_path)
        if not os.path.exists(full_path):
            skipped_videos += 1
            continue

        # Check if we've already processed this video
        if video_path in video_latent_map:
            # If video was processed before, copy the latent to new file
            existing_dict = torch.load(video_latent_map[video_path])
            # Get the latent from the dictionary (it's the only value)
            existing_latent = next(iter(existing_dict.values()))
            torch.save({prompt: existing_latent}, output_path)
            continue

        total_videos += 1
        try:
            # Read and process video
            array = video_to_numpy(full_path)
        except Exception as e:
            print(f"Failed to read video: {video_path}")
            print(f"Error details: {str(e)}")
            failed_encodings += 1
            continue

        # Convert video to tensor and normalize
        video_tensor = torch.tensor(array, dtype=torch.float32, device=device).unsqueeze(0).permute(
            0, 4, 1, 2, 3
        ) / 255.0
        video_tensor = video_tensor * 2 - 1
        video_tensor = video_tensor.to(torch.bfloat16)

        # Encode video to latent
        encoded_latents = encode(model, video_tensor).transpose(2, 1)
        latent = encoded_latents.cpu().detach()

        # Save prompt:latent mapping
        torch.save({prompt: latent}, output_path)
        
        # Update video:latent_file mapping
        video_latent_map[video_path] = output_path
        successful_encodings += 1

    # Convert counters to tensors for all_reduce
    total_videos_tensor = torch.tensor(total_videos, device=device)
    skipped_videos_tensor = torch.tensor(skipped_videos, device=device)
    successful_encodings_tensor = torch.tensor(successful_encodings, device=device)
    failed_encodings_tensor = torch.tensor(failed_encodings, device=device)

    # Sum up counters across all processes
    dist.all_reduce(total_videos_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(skipped_videos_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(successful_encodings_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(failed_encodings_tensor, op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:
        print("\nProcessing Statistics:")
        print(f"Total videos processed: {total_videos_tensor.item()}")
        print(f"Skipped videos (not found): {skipped_videos_tensor.item()}")
        print(f"Successfully encoded: {successful_encodings_tensor.item()}")
        print(f"Failed to encode: {failed_encodings_tensor.item()}")

    dist.barrier()


if __name__ == "__main__":
    main()
