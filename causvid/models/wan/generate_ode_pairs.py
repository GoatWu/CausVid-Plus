from causvid.models.wan.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from causvid.models.wan.flow_match import FlowMatchScheduler
from causvid.models import get_block_class
from causvid.util import launch_distributed_job, fsdp_wrap
from causvid.data import TextDataset
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
import math
import os


def init_model(device, model_name="T2V-1.3B", use_fsdp=True, config=None):
    # Initialize model
    model = WanDiffusionWrapper(model_name=model_name).to(device).to(torch.float32)
    encoder = WanTextEncoder().to(device).to(torch.float32)
    
    # Wrap with FSDP if enabled
    if use_fsdp and torch.distributed.is_initialized():
        
        model = fsdp_wrap(
            model,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            transformer_module=(get_block_class(config.generator_fsdp_transformer_module),
                                ) if config.generator_fsdp_wrap_strategy == "transformer" else None
        )

        encoder = fsdp_wrap(
            encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            transformer_module=(get_block_class(config.text_encoder_fsdp_transformer_module),
                                ) if config.text_encoder_fsdp_wrap_strategy == "transformer" else None
        )
    
    model.set_module_grad(
        {
            "model": False
        }
    )

    scheduler = FlowMatchScheduler(
        shift=8.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=50, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

    unconditional_dict = encoder(
        text_prompts=[sample_neg_prompt]
    )

    return model, encoder, scheduler, unconditional_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--caption_path", type=str)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--model_name", type=str, default="T2V-1.3B")
    parser.add_argument("--config", type=str, help="Path to config file ")
    parser.add_argument("--fsdp", action="store_true", help="Enable FSDP to save GPU memory")
    args = parser.parse_args()

    # Load FSDP config if provided
    config = None
    if args.fsdp:
        if args.config:
            config = OmegaConf.load(args.config)
        if args.local_rank == 0:
            print("FSDP Configuration:")
            if config:
                print(OmegaConf.to_yaml(config))
            else:
                print("Using default FSDP configuration")

    # Initialize distributed environment
    launch_distributed_job()
    global_rank = dist.get_rank()

    device = torch.cuda.current_device()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, encoder, scheduler, unconditional_dict = init_model(
        device=device, 
        model_name=args.model_name,
        use_fsdp=args.fsdp,
        config=config
    )

    dataset = TextDataset(args.caption_path)

    if global_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)

    for index in tqdm(range(int(math.ceil(len(dataset) / dist.get_world_size()))), disable=dist.get_rank() != 0):
        prompt_index = index * dist.get_world_size() + dist.get_rank()
        if prompt_index >= len(dataset):
            continue
        prompt = dataset[prompt_index]

        conditional_dict = encoder(
            text_prompts=prompt
        )

        latents = torch.randn(
            [1, config.num_frames, 16, 60, 104], dtype=torch.float32, device=device
        )

        noisy_input = []

        for progress_id, t in enumerate(tqdm(scheduler.timesteps)):
            timestep = t * \
                torch.ones([1, config.num_frames], device=device, dtype=torch.float32)

            noisy_input.append(latents)

            x0_pred_cond = model(
                latents, conditional_dict, timestep
            )

            x0_pred_uncond = model(
                latents, unconditional_dict, timestep
            )

            x0_pred = x0_pred_uncond + args.guidance_scale * (
                x0_pred_cond - x0_pred_uncond
            )

            flow_pred = model._convert_x0_to_flow_pred(
                scheduler=scheduler,
                x0_pred=x0_pred.flatten(0, 1),
                xt=latents.flatten(0, 1),
                timestep=timestep.flatten(0, 1)
            ).unflatten(0, x0_pred.shape[:2])

            latents = scheduler.step(
                flow_pred.flatten(0, 1),
                scheduler.timesteps[progress_id] * torch.ones(
                    [1, config.num_frames], device=device, dtype=torch.long).flatten(0, 1),
                latents.flatten(0, 1)
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])

        noisy_input.append(latents)

        noisy_inputs = torch.stack(noisy_input, dim=1)

        noisy_inputs = noisy_inputs[:, [0, 36, 44, -1]]

        stored_data = noisy_inputs

        torch.save(
            {prompt: stored_data.cpu().detach()},
            os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
        )

    dist.barrier()


if __name__ == "__main__":
    main()
