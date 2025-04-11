from causvid.models.wan.causal_inference import InferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
import os
import math

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--prompt_file_path", type=str)
parser.add_argument("--duration", type=int, default=4, help="Video duration in seconds (minimum 4 seconds)")

args = parser.parse_args()

# Validate video duration parameter
if args.duration < 4:
    raise ValueError("Video duration must be at least 4 seconds")

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

# Calculate required number of frames
fps = 16  # Default frame rate
frames_per_latent = 4  # Number of frames per latent state
required_frames = args.duration * fps  # Total frames needed
required_latents = required_frames // frames_per_latent  # Required number of latent states

# Ensure the number of latent states is not less than config.num_frames
initial_latents = max(required_latents, config.num_frames)

# Calculate number of blocks to generate
num_blocks = math.ceil(initial_latents / config.num_frame_per_block)

# Ensure actual_latents is a multiple of num_frame_per_block
actual_latents = num_blocks * config.num_frame_per_block

# Validate the calculated numbers
print(f"Initial required latents: {initial_latents}")
print(f"Number of blocks: {num_blocks}")
print(f"Final actual latents: {actual_latents}")
print(f"Frames per block: {config.num_frame_per_block}")

pipeline = InferencePipeline(config, device="cuda", model_type=config.model_type, num_frames=config.num_frames)
pipeline.to(device="cuda", dtype=torch.bfloat16)

state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")[
    'generator']

pipeline.generator.load_state_dict(
    state_dict, strict=True
)

dataset = TextDataset(args.prompt_file_path)

sampled_noise = torch.randn(
    [1, actual_latents, 16, 60, 104], device="cuda", dtype=torch.bfloat16
)

os.makedirs(args.output_folder, exist_ok=True)

for prompt_index in tqdm(range(len(dataset))):
    prompts = [dataset[prompt_index]]

    video = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        infer_blocks=num_blocks
    )[0].permute(0, 2, 3, 1).cpu().numpy()

    export_to_video(
        video, os.path.join(args.output_folder, f"output_{prompt_index:03d}.mp4"), fps=16)
