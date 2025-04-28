import torch
from causvid.models.wan.flow_match import FlowMatchScheduler

def main():
    scheduler = FlowMatchScheduler(
        shift=8.0, 
        sigma_min=0.0, 
        extra_one_step=True
    )
    scheduler.set_timesteps(num_inference_steps=100, denoising_strength=1.0)
    
    print("all timesteps:")
    print(scheduler.timesteps)
    print("all sigmas:")
    print(scheduler.sigmas)
    
    # key_points = [0, 36, 44, 48, -1]
    key_points = [0, 36, 56, 72, 84, 92, 96, 98, 99]
    print("\nkey timesteps:")
    for point in key_points:
        print(f"index {point}: {int(scheduler.timesteps[point].item())}")

if __name__ == "__main__":
    main() 