import pdb
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from causvid.dmd import DMD
from PIL import Image
import torch

torch.set_grad_enabled(False)

config = OmegaConf.load("configs/wan_bidirectional_dmd.yaml")

dmd_model = DMD(config, device="cuda")
dmd_model = dmd_model.to(torch.bfloat16).cuda()

conditional_dict = dmd_model.text_encoder(
    text_prompts=[r"""A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."""]
)

unconditional_dict = dmd_model.text_encoder(text_prompts=[config.negative_prompt] * 1)

print("Test 1: Backward Simulation")

image_or_video_shape = [1, 21, 16, 60, 104]

simulated_input = dmd_model._consistency_backward_simulation(
    noise=torch.randn(image_or_video_shape,
                      device="cuda", dtype=torch.bfloat16),
    conditional_dict=conditional_dict
)

# 4 x 1 x 4 x 128 x 128
output = simulated_input[:, -1]
# [B, F, C, H, W] -> [B, C, H, W]
video = dmd_model.vae.decode_to_pixel(output).cpu().detach()

video = ((video + 1.0) / 2.0).clamp(0, 1)[0].permute(0, 2, 3, 1).numpy()

export_to_video(video, "backward_simulated_video.mp4", fps=16)

print("Test 2: Generator Loss")
generator_loss, generator_log_dict = dmd_model.generator_loss(
    image_or_video_shape=image_or_video_shape,
    conditional_dict=conditional_dict,
    unconditional_dict=unconditional_dict,
    clean_latent=None
)

print("Test 3: Critic Loss")
critic_loss, critic_log_dict = dmd_model.critic_loss(
    image_or_video_shape=image_or_video_shape,
    conditional_dict=conditional_dict,
    unconditional_dict=unconditional_dict,
    clean_latent=None
)

print(
    f"Generator Loss: {generator_loss}. dmdtrain_gradient_norm: {generator_log_dict['dmdtrain_gradient_norm']}")

print(
    f"Critic Loss: {critic_loss}.")
