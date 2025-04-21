from causvid.models.wan.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from causvid.models.wan.flow_match import FlowMatchScheduler
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from tqdm import tqdm
import torch
import os

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model = WanDiffusionWrapper().to("cuda").to(torch.float32)
encoder = WanTextEncoder().to("cuda").to(torch.float32)
vae = WanVAEWrapper().to("cuda").to(torch.float32)

model.set_module_grad(
    {
        "model": False
    }
)

dataset = TextDataset('sample_dataset/MovieGenVideoBench.txt')

os.makedirs('wan_manystep_sampling', exist_ok=True)

device = "cuda"
num_train_timesteps = 1000
sampling_steps = 50
shift = 8

sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

scheduler = FlowMatchScheduler(shift=8.0, sigma_min=0.0, extra_one_step=True)

scheduler.set_timesteps(num_inference_steps=50, denoising_strength=1.0)

scheduler.sigmas = scheduler.sigmas.to(device)

unconditional_dict = encoder(
    text_prompts=[sample_neg_prompt]
)

for index in tqdm(range(len(dataset))):
    prompt = dataset[index]

    conditional_dict = encoder(
        text_prompts=prompt
    )

    latents = torch.randn(
        1, 16, 21, 60, 104, generator=torch.Generator().manual_seed(42), dtype=torch.float32
    ).to("cuda").permute(0, 2, 1, 3, 4)

    with torch.amp.autocast(dtype=torch.float32, device_type=torch.device("cuda:0").type):

        for progress_id, t in enumerate(tqdm(scheduler.timesteps)):
            timestep = t * \
                torch.ones([1, 21], device=device, dtype=torch.float32)

            x0_pred_cond = model(
                latents, conditional_dict, timestep
            )

            x0_pred_uncond = model(
                latents, unconditional_dict, timestep
            )

            x0_pred = x0_pred_uncond + 6 * (
                x0_pred_cond - x0_pred_uncond
            )

            flow_pred = model._convert_x0_to_flow_pred(
                x0_pred=x0_pred.flatten(0, 1),
                xt=latents.flatten(0, 1),
                timestep=timestep.flatten(0, 1),
                scheduler=scheduler
            ).unflatten(0, x0_pred.shape[:2])

            latents = scheduler.step(
                flow_pred.flatten(0, 1),
                scheduler.timesteps[progress_id] * torch.ones(
                    [1, 17], device=device, dtype=torch.long).flatten(0, 1),
                latents.flatten(0, 1)
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])

        decoded_video = vae.decode_to_pixel(latents)[0].permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5

    export_to_video(
        decoded_video, f"test_wanx_wrapper_{index:04d}.mp4", fps=16)
