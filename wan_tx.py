import math
import torch
import torchax
import time
import jax


from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from jax.tree_util import register_pytree_node

from transformers import modeling_outputs

def flatten_model_output(obj):
  return obj.to_tuple(), type(obj)

def unflatten_model_output(aux, children):
  return aux(*children)

register_pytree_node(
  modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
  flatten_model_output,
  unflatten_model_output)

def main():
  torch.set_default_dtype(torch.bfloat16)
  # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
  #model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
  vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
  flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
  scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
  pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
  pipe.scheduler = scheduler

  torchax.enable_globally()

  pipe.to('jax')
  pipe.vae = torchax.compile(pipe.vae)
  pipe.text_encoder = torchax.compile(pipe.text_encoder)
  pipe.transformer.to('jax')
  # the param below is not declared as param or buffer so the module.to('jax') didnt work
  pipe.transformer.rope.freqs = pipe.transformer.rope.freqs.to('jax')
  options = torchax.CompileOptions(
      jax_jit_kwargs={'static_argnames': ('return_dict',)}
  )
  pipe.transformer = torchax.compile(pipe.transformer, options)

  def move_scheduler(scheduler):
    for k, v in scheduler.__dict__.items():
      if isinstance(v, torch.Tensor):
        setattr(scheduler, k, v.to('jax'))

  #move_scheduler(pipe.scheduler)

  def module_size(module):
    size = 0
    for k, v in module.state_dict().items():
      size += math.prod(v.shape) * v.dtype.itemsize
    return size

  for m in dir(pipe):
      module = getattr(pipe, m, None)
      if isinstance(module, torch.nn.Module):
          print(m, module_size(module) / (1024 * 1024 * 1024), 'G')


  prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
  negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

  outputs = []
  for i in range(5):
    start = time.perf_counter()
    if i == 3:
      jax.profiler.start_trace('/tmp/tensorboard')
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=384,
        width=640,
        num_inference_steps=50,
        #height=720,
        #width=1280,
        num_frames=41,
        guidance_scale=5.0,
        ).frames[0]
    if i == 4:
      jax.profiler.stop_trace()
    end = time.perf_counter()  
    print(f'Iteration {i}: {end - start:.6f}s')
    outputs.append(output)

  export_to_video(outputs[0], "output.mp4", fps=8)
  print('DONE')

  #print(f'生成视频时长= {(num_frams-1)/fps} - 目前针对1.3B生成5s = (41-1)/8)


if __name__ == '__main__':
  main()