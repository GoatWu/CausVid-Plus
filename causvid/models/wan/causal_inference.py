from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper
)
from typing import List, Optional
import torch


class InferencePipeline(torch.nn.Module):
    def __init__(self, args, device, model_type="T2V-1.3B", num_frames = 21):
        super().__init__()
        # Step 1: Initialize all models
        self.generator_model_name = getattr(
            args, "generator_name", args.model_name)
        self.generator = get_diffusion_wrapper(
            model_name=self.generator_model_name)(model_type=model_type, num_frames=num_frames)
        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)(model_type=model_type)
        self.vae = get_vae_wrapper(model_name=args.model_name)(model_type=model_type)

        # Step 2: Initialize all causal hyperparmeters
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)
        assert self.denoising_step_list[-1] == 0
        # remove the last timestep (which equals zero)
        self.denoising_step_list = self.denoising_step_list[:-1]

        self.scheduler = self.generator.get_scheduler()

        if model_type == "T2V-1.3B":
            self.num_transformer_blocks = 30
            self.num_heads = 12
        elif model_type == "T2V-14B":
            self.num_transformer_blocks = 40
            self.num_heads = 40
        else:
            raise ValueError(f"Model type {model_type} not supported")

        self.frame_seq_length = 1560
        self.num_frames = num_frames

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.args = args
        self.num_frame_per_block = getattr(
            args, "num_frame_per_block", 1)
        
        # Add sliding window related parameters
        self.window_size = self.frame_seq_length * self.num_frames  # Window size based on sequence length
        self.shift_blocks = getattr(args, "shift_blocks", 1)  # Number of blocks to shift when window is full
        self.rope_start = 0  # Starting position for rope encoding
        self.kv_start = 0  # Starting position for KV cache

        print(f"KV inference with {self.num_frame_per_block} frames per block")
        print(f"Sliding window size: {self.window_size}, shift blocks: {self.shift_blocks}")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model with sliding window mechanism.
        The cache size is determined by frame_seq_length * num_frames.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.window_size, self.num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.window_size, self.num_heads, 128], dtype=dtype, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, self.num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, self.num_heads, 128], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache = crossattn_cache  # always store the clean cache

    def _shift_kv_cache(self):
        """
        Shift the KV cache left by shift_blocks * num_frame_per_block * frame_seq_length.
        This is called when kv_start exceeds window_size.
        The first block is preserved, and shifting starts from the second block.
        """
        shift_length = self.shift_blocks * self.num_frame_per_block * self.frame_seq_length
        # first_block_len = self.num_frame_per_block * self.frame_seq_length
        
        for block in self.kv_cache1:
            # # Get the first block's data
            # first_block_data = block["k"][:, :first_block_len].clone()
            # first_block_v = block["v"][:, :first_block_len].clone()
            
            # Shift the rest of the data
            block["k"] = torch.roll(block["k"], shifts=-shift_length, dims=1)
            block["v"] = torch.roll(block["v"], shifts=-shift_length, dims=1)
            
            # # Restore the first block's data
            # block["k"][:, :first_block_len] = first_block_data
            # block["v"][:, :first_block_len] = first_block_v
            
            # Clear the shifted-out part (except the first block)
            block["k"][:, -shift_length:] = 0
            block["v"][:, -shift_length:] = 0
        
        # Update kv_start
        self.kv_start -= shift_length
        return shift_length

    def inference(self, noise: torch.Tensor, text_prompts: List[str], infer_blocks: int) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            infer_blocks (int): Number of blocks to generate.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        output = torch.zeros(
            [batch_size, infer_blocks * self.num_frame_per_block, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )

            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # Reset cross attention cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False

        # Initialize positions
        self.kv_start = 0
        self.rope_start = 0

        # Step 2: Temporal denoising loop
        for block_index in range(infer_blocks):
            noisy_input = noise[:, block_index *
                                self.num_frame_per_block:(block_index + 1) * self.num_frame_per_block]

            # Calculate KV Cache positions
            kv_end = self.kv_start + self.num_frame_per_block * self.frame_seq_length

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # Set current timestep
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block], device=noise.device, dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        kv_start=self.kv_start,
                        kv_end=kv_end,
                        rope_start=self.rope_start
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep *
                        torch.ones([batch_size], device="cuda",
                                   dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # For getting real output
                    denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        kv_start=self.kv_start,
                        kv_end=kv_end,
                        rope_start=self.rope_start
                    )

            output[:, block_index * self.num_frame_per_block:(
                block_index + 1) * self.num_frame_per_block] = denoised_pred
            
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                kv_start=self.kv_start,
                kv_end=kv_end,
                rope_start=self.rope_start
            )
        
            # Update positions for next block
            r_shift_length = self.num_frame_per_block * self.frame_seq_length
            self.kv_start += r_shift_length
            kv_end += r_shift_length
            self.rope_start += r_shift_length
            # Check if we need to shift the cache
            
            if kv_end > self.window_size:
                kv_end -= self._shift_kv_cache()

            # Step 2.2: Rerun with timestep zero to update the cache
            # Only run additional generation if denoising_step_list doesn't contain 0
            # if self.denoising_step_list[-1] != 0:
            # self.generator(
            #     noisy_image_or_video=denoised_pred,
            #     conditional_dict=conditional_dict,
            #     timestep=timestep * 0,
            #     kv_cache=self.kv_cache1,
            #     crossattn_cache=self.crossattn_cache,
            #     kv_start=self.kv_start - r_shift_length,
            #     kv_end=kv_end - r_shift_length,
            #     rope_start=self.rope_start - r_shift_length
            # )

        # Step 3: Decode the output
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        return video
