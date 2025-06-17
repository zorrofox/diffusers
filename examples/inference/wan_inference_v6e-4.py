import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import UMT5EncoderModel, AutoTokenizer
from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanPipeline
from diffusers.utils import export_to_video
import argparse
from pathlib import Path
import time

# Define the device mapping for 4 TPU chips with better memory balance
# NOTE: This device_map is now primarily for reference/text_encoder offloading
# The transformer and VAE will be moved to the local XLA device explicitly below.
device_map = {
    "text_encoder": "cpu",  # Offload text encoder to CPU
    "transformer": {  # Remaining blocks will be loaded on the single device if pipe.to() is called
        "": "xla:0",
        "blocks.0": "xla:1",
        "blocks.1": "xla:2",
        "blocks.2": "xla:3",
        "blocks.3": "xla:0",
        "blocks.4": "xla:1",
        "blocks.5": "xla:2",
    },
    "vae": "xla:3",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.1 Video Generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate") # Further Reduced to 1
    parser.add_argument("--height", type=int, default=432, help="Height of the video") # Further Reduced
    parser.add_argument("--width", type=int, default=640, help="Width of the video") # Further Reduced
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--offload_model", action="store_true", help="Offload text encoder to CPU")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory")
    return parser.parse_args()

# Renamed main to _mp_main and added index argument for xmp.spawn
def _mp_main(index, args):
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model components with device mapping
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    #model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    print(f"Loading model from {model_id}...")

    # Load text encoder - keep on CPU if offload_model is true
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id, 
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu" if args.offload_model else None} # Load to CPU or default
    )

    # Load transformer and VAE without device_map initially
    # They will be loaded to meta device by default, then moved to XLA device below.
    transformer = WanTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        use_gradient_checkpointing=args.use_gradient_checkpointing
    )

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )

    # Create pipeline
    pipe = WanPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16
    )
    
    # Move the entire pipeline to the local XLA device. This will likely OOM.
    current_xla_device = xm.xla_device()
    print(f"Moving pipeline to {current_xla_device} in process {index}")
    pipe.to(current_xla_device)

    print("Model loaded successfully!")
    # Print memory info from the current process's view of the device
    print(f"TPU Memory Info for process {index}: {xm.get_memory_info()}")

    # Set random seed if provided
    if args.seed is not None:
        # Ensure unique seed per process if desired for varied outputs, otherwise use same base seed
        torch.manual_seed(args.seed + index)
        xm.set_rng_state(args.seed + index, device=current_xla_device)

    # Generate video
    print(f"Generating video with prompt: {args.prompt} on TPU core {index}")
    start_time = time.time()
    
    with torch.no_grad():
        video_frames = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            output_type="np"  # Output as numpy array for easier saving
        ).frames

    generation_time = time.time() - start_time
    print(f"Video generation completed in {generation_time:.2f} seconds on TPU core {index}")

    # Save video frames - only save from one process to avoid multiple identical files
    if index == 0: # Only save from the first process
        output_path = output_dir / f"video_{int(time.time())}.mp4"
        export_to_video(video_frames, output_path)
        print(f"Video saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    # Spawn 4 processes, one for each TPU core
    xmp.spawn(_mp_main, args=(args,), nprocs=None) 
