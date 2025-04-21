from causvid.models.wan.causal_inference import InferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str)

args = parser.parse_args()

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

pipeline = InferencePipeline(config, device="cuda")
pipeline.to(device="cuda", dtype=torch.bfloat16)

state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")[
    'generator']

pipeline.generator.load_state_dict(
    state_dict, strict=True
)

dataset = TextDataset('sample_dataset/MovieGenVideoBench.txt')

sampled_noise = torch.randn(
    [1, 21, 16, 60, 104], device="cuda", dtype=torch.bfloat16
)

for prompt_index in tqdm(range(len(dataset))):
    prompts = [dataset[prompt_index]]

    video = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts
    )[0].permute(0, 2, 3, 1).cpu().numpy()

    export_to_video(
        video, os.path.join(args.checkpoint_folder, f"output_{prompt_index:03d}.mp4"), fps=16)
