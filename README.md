# CausVid-Plus

This repository contains an unofficial extension implementation of [CausVid](https://github.com/tianweiy/CausVid). The main extensions include:

- Added support for the Wan2.1-T2V-14B model
- Modified data processing pipeline to generate approximately 50,000 prompt-video training pairs
- Enhanced inference pipeline for generating more coherent long videos

## Environment Setup 

```bash
conda create -n causvid python=3.10 -y
conda activate causvid
pip install torch torchvision 
pip install -r requirements.txt 
python setup.py develop
```

Also download the Wan base models from [here](https://github.com/Wan-Video/Wan2.1) and save it to wan_models/Wan2.1-T2V-1.3B/

## Inference Example 

### Autoregressive 3-step 5-second Video Generation  

```bash 
python minimal_inference/autoregressive_inference.py --config_path configs/wan_causal_dmd.yaml --checkpoint_folder XXX  --output_folder XXX   --prompt_file_path XXX 
```

### Autoregressive 3-step long Video Generation

```bash 
python minimal_inference/longvideo_autoregressive_inference.py --config_path configs/wan_causal_dmd.yaml --checkpoint_folder XXX  --output_folder XXX --prompt_file_path XXX --num_rollout XXX 
```

### Bidirectional 3-step 5-second Video Generation

```bash 
python minimal_inference/bidirectional_inference.py --config_path configs/wan_bidirectional_dmd_from_scratch.yaml --checkpoint_folder XXX --output_folder XXX --prompt_file_path XXX 
```

## Training and Evaluation  

### Dataset Preparation 

We use the [MixKit Dataset](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/all_mixkit) (6K videos, 5w prompts) for distillation. 

To prepare the dataset, follow these steps.

```bash
# download the annotation files
huggingface-cli download LanguageBind/Open-Sora-Plan-v1.1.0 --repo-type dataset --include anno_jsons/video_mixkit_65f_54735.json --local-dir ./sample_dataset

# download and extract video from the Mixkit dataset 
python distillation_data/download_mixkit.py --local_dir ./datasets

# move all the video into a single folder
python distillation_data/move_mixkit.py ./datasets/videos

# slice the videos via the annotation files
python distillation_data/slice_video.py --input_folder ./datasets/videos/ --output_folder ./datasets/sliced_videos/81f/ --anno_json ./sample_dataset/video_mixkit_65f_54735.json --output_json ./sample_dataset/cap_to_video_81f.json

# convert the video to 480x832x81 
python distillation_data/process_mixkit.py --input_dir ./datasets/sliced_videos/81f/ --output_dir ./datasets/converted_videos/81f/ --width 832 --height 480 --fps 16

# precompute the vae latent 
torchrun --nproc_per_node 8 distillation_data/compute_vae_latent.py --input_video_folder ./datasets/converted_videos/81f/ --output_latent_folder ./datasets/latents/81f --info_path ./sample_dataset/cap_to_video_81f.json

# combined everything into a lmdb dataset 
python causvid/ode_data/create_lmdb_iterative.py --data_path ./datasets/latents/81f/ --lmdb_path ./datasets/mixkit_latents_lmdb/81f/

# precompute the ode pairs
# contain step distillation here
torchrun --nnodes 8 --nproc_per_node=8 \
--rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
causvid/models/wan/generate_ode_pairs.py \
--output_folder ./datasets/odes/81f/14b/ \
--caption_path ./sample_dataset/mixkit_prompts_81f_4000.txt \
--model_type T2V-14B \
--config ./configs/wan_14b_causal_ode.yaml \
--fsdp

# combined everything into a lmdb dataset 
python causvid/ode_data/create_lmdb_iterative.py --data_path ./datasets/odes/81f/14b/ --lmdb_path ./datasets/mixkit_odes_lmdb/81f/14b/

```

## Training 

Please first modify the wandb account information in the respective config.

Causal ODE Pretraining. 

```bash
torchrun --nnodes 8 --nproc_per_node=8 \
--rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
causvid/train_ode.py \
--config_path configs/wan_14b_causal_ode.yaml
```

Causal DMD Training.   

```bash
torchrun --nnodes 8 --nproc_per_node=8 \
--rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
causvid/train_distillation.py \
--config_path configs/wan_14b_causal_dmd.yaml \
--no_visualize
```

## Acknowledgments

Our implementation is largely based on the [Wan](https://github.com/Wan-Video/Wan2.1) model suite and the original [CausVid](https://github.com/tianweiy/CausVid) implementation.

