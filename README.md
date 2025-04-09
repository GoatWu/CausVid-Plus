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

First download the checkpoints: [Autoregressive Model](https://huggingface.co/tianweiy/CausVid/tree/main/autoregressive_checkpoint), [Bidirectional Model 1](https://huggingface.co/tianweiy/CausVid/tree/main/bidirectional_checkpoint1) or [Bidirectional Model 2](https://huggingface.co/tianweiy/CausVid/tree/main/bidirectional_checkpoint2) (performs slightly better). 

### Autoregressive 3-step 5-second Video Generation  

```bash 
python minimal_inference/autoregressive_inference.py --config_path configs/wan_causal_dmd.yaml --checkpoint_folder XXX  --output_folder XXX   --prompt_file_path XXX 
```

### Autoregressive 3-step long Video Generation

```bash 
python minimal_inference/longvideo_autoregressive_inference.py --config_path configs/wan_causal_dmd.yaml --checkpoint_folder XXX  --output_folder XXX   --prompt_file_path XXX   --num_rollout XXX 
```

### Bidirectional 3-step 5-second Video Generation

```bash 
python minimal_inference/bidirectional_inference.py --config_path configs/wan_bidirectional_dmd_from_scratch.yaml --checkpoint_folder XXX  --output_folder XXX   --prompt_file_path XXX 
```

## Training and Evaluation  

### Dataset Preparation 

We use the [MixKit Dataset](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/all_mixkit) (6K videos) as a toy example for distillation. 

To prepare the dataset, follow these steps. You can also download the final LMDB dataset from [here](https://huggingface.co/tianweiy/CausVid/tree/main/mixkit_latents_lmdb)

```bash
# download the annotation files
huggingface-cli download LanguageBind/Open-Sora-Plan-v1.1.0 --repo-type dataset --include anno_jsons/video_mixkit_65f_54735.json --local-dir ./sample_dataset

huggingface-cli download LanguageBind/Open-Sora-Plan-v1.1.0 --repo-type dataset --include anno_jsons/video_mixkit_513f_1997.json --local-dir ./sample_dataset

# download and extract video from the Mixkit dataset 
python distillation_data/download_mixkit.py --local_dir ./datasets

# move all the video into a single folder
python distillation_data/move_mixkit.py ./datasets/videos

# slice the videos via the annotation files
python distillation_data/slice_video.py --input_folder ./datasets/videos/ --output_folder ./datasets/sliced_videos/65f/ --anno_json ./sample_dataset/video_mixkit_65f_54735.json --output_json ./sample_dataset/cap_to_video_65f.json

# convert the video to 480x832x81 
python distillation_data/process_mixkit.py --input_dir ./datasets/sliced_videos/65f/ --output_dir ./datasets/converted_videos/65f/ --width 832 --height 480 --fps 16

# precompute the vae latent 
torchrun --nproc_per_node 8 distillation_data/compute_vae_latent.py --input_video_folder ./datasets/onverted_videos/65f/ --output_latent_folder ./datasets/latents/65f --info_path ./sample_dataset/cap_to_video_65f.json

# combined everything into a lmdb dataset 
python causvid/ode_data/create_lmdb_iterative.py --data_path ./datasets/latents/65f/ --lmdb_path ./datasets/mixkit_latent_lmdb/65f/

# precompute the ode pairs
# contain step distillation here
torchrun --nnodes 8 --nproc_per_node=8 \
--rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
causvid/models/wan/generate_ode_pairs.py \
--output_folder ./datasets/odes/65f/ \
--caption_path ./sample_dataset/mixkit_prompts_65f.txt \
--model_type T2V-14B \
--config ./configs/wan_causal_ode.yaml \
--fsdp

# combined everything into a lmdb dataset 
python causvid/ode_data/create_lmdb_iterative.py --data_path ./datasets/odes/65f/ --lmdb_path ./datasets/mixkit_odes_lmdb/65f/

```

## Training 

Please first modify the wandb account information in the respective config.  

Bidirectional DMD Training

```bash
torchrun --nnodes 8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR causvid/train_distillation.py \
    --config_path  configs/wan_bidirectional_dmd_from_scratch.yaml 
```

ODE Dataset Generation. We generate a total of 1.5K dataset pairs, which can also be downloaded from [here](https://huggingface.co/tianweiy/CausVid/tree/main/mixkit_ode_lmdb) 

```bash
torchrun --nproc_per_node 8 causvid/models/wan/generate_ode_pairs.py --output_folder  XXX --caption_path sample_dataset/mixkit_prompts.txt

python causvid/ode_data/create_lmdb_iterative.py  --data_path XXX --lmdb_path XXX 
```

Causal ODE Pretraining. You can also skip this step and download the ode finetuned checkpoint from [here](https://huggingface.co/tianweiy/CausVid/tree/main/wan_causal_ode_checkpoint_model_003000) 

```bash
torchrun --nnodes 8 --nproc_per_node=8 \
--rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
causvid/train_ode.py \
--config_path configs/wan_causal_ode.yaml
```

Causal DMD Training.   

```bash
torchrun --nnodes 8 --nproc_per_node=8 \
--rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \ 
causvid/train_distillation.py \
--config_path configs/wan_causal_dmd.yaml \
--no_visualize
```

## Acknowledgments

Our implementation is largely based on the [Wan](https://github.com/Wan-Video/Wan2.1) model suite and the original [CausVid](https://github.com/tianweiy/CausVid) implementation.

