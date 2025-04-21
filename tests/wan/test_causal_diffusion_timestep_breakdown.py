from causvid.models.wan.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from causvid.models.wan.flow_match import FlowMatchScheduler
from diffusers.utils import export_to_video
import torch

torch.set_grad_enabled(False)

model = WanDiffusionWrapper().to("cuda").to(torch.bfloat16)
encoder = WanTextEncoder().to("cuda").to(torch.bfloat16)
vae = WanVAEWrapper().to("cuda").to(torch.bfloat16)

model.set_module_grad(
    {
        "model": False
    }
)

text_prompts = [r"""A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."""] * 2

conditional_dict = encoder(
    text_prompts=text_prompts
)

device = "cuda"

noise = torch.randn(
    1,21, 16, 60, 104, generator=torch.Generator().manual_seed(42), dtype=torch.float32
).to("cuda")

scheduler = FlowMatchScheduler(shift=8.0, sigma_min=0.0, extra_one_step=True)
scheduler.set_timesteps(1000, denoising_strength=1.0)
scheduler.sigmas = scheduler.sigmas.to(device)

data = torch.load("wan_moviegen_ode/00000.pt")
prompt = list(data.keys())[0]
video_latent = data[prompt][:, -1].cuda().to(torch.bfloat16)

for timestep in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
    timestep = timestep * \
        torch.ones([1, 21], device="cuda", dtype=torch.float32)

    noisy_latent = scheduler.add_noise(
        video_latent.flatten(0, 1),
        noise.flatten(0, 1),
        timestep.flatten(0, 1)
    ).unflatten(0, noise.shape[:2]).type_as(video_latent)

    output = model(noisy_latent, conditional_dict, timestep)

    video = vae.decode_to_pixel(output)
    video = (video * 0.5 + 0.5).cpu().detach().to(torch.float32)[0].permute(0, 2, 3, 1).numpy()

    export_to_video(
        video, f"one_stpe_output_t={timestep[0, 0].item()}.mp4", fps=16)
