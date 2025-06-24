import re
import math
import torch
import torchax
import time
import jax

from jax.sharding import NamedSharding, PartitionSpec as P


from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from jax.tree_util import register_pytree_node

from transformers import modeling_outputs

from datetime import datetime

# import torchax.ops.jtorch


#### SETTINGS
# 1.3B
# MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
# 14B
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

# 384p
# FLOW_SHIFT = 3.0 # 5.0 for 720P, 3.0 for 480P
# WIDTH = 640
# HEIGHT = 384
# 480p
# FLOW_SHIFT = 3.0 # 5.0 for 720P, 3.0 for 480P
# WIDTH = 832
# HEIGHT = 480
# 720p
FLOW_SHIFT = 5.0 # 5.0 for 720P, 3.0 for 480P
WIDTH = 1280
HEIGHT = 720

# 41 frames
# FRAMES = 41
# FPS = 8

# 81 frames
FRAMES = 81
FPS = 16

# step
NUM_STEP = 50
# NUM_STEP = 1

####


axis = 'axis'

# Sharding for tranformers, all the replicated are commented out for speed
transformer_shardings = {
# 'scale_shift_table': (), # (torch.Size([1, 2, 1536]), torch.float32)
# 'patch_embedding.weight': (), # (torch.Size([1536, 16, 1, 2, 2]), torch.bfloat16)
# 'patch_embedding.bias': (), # (torch.Size([1536]), torch.bfloat16)
r'condition_embedder.time_embedder.linear_1.weight': (axis, None), # (torch.Size([1536, 256]), torch.float32)
r'condition_embedder.time_embedder.linear_1.bias': (axis,), # (torch.Size([1536]), torch.float32)
r'condition_embedder.time_embedder.linear_2.weight': (None, axis), # (torch.Size([1536, 1536]), torch.float32)
# 'condition_embedder.time_embedder.linear_2.bias': (), # (torch.Size([1536]), torch.float32)
# 'condition_embedder.time_proj.weight': (), # (torch.Size([9216, 1536]), torch.bfloat16)
# 'condition_embedder.time_proj.bias': (), # (torch.Size([9216]), torch.bfloat16)
r'condition_embedder.text_embedder.linear_1.weight': (axis, None), # (torch.Size([1536, 4096]), torch.bfloat16)
r'condition_embedder.text_embedder.linear_1.bias': (axis, ), # (torch.Size([1536]), torch.bfloat16)
r'condition_embedder.text_embedder.linear_2.weight': (None, axis), # (torch.Size([1536, 1536]), torch.bfloat16)
# 'condition_embedder.text_embedder.linear_2.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.scale_shift_table': (), # (torch.Size([1, 6, 1536]), torch.float32)
# 'blocks.\d+.attn1.norm_q.weight': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.attn1.norm_k.weight': (), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn1.to_q.weight': (axis, None), # (torch.Size([1536, 1536]), torch.bfloat16)
r'blocks.\d+.attn1.to_q.bias': (axis, ), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn1.to_k.weight': (axis, ), # (torch.Size([1536, 1536]), torch.bfloat16)
r'blocks.\d+.attn1.to_k.bias': (axis, ), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn1.to_v.weight': (axis, ), # (torch.Size([1536, 1536]), torch.bfloat16)
r'blocks.\d+.attn1.to_v.bias': (axis, ), # (torch.Size([1536]), torch.bfloat16)
# to_out has 2 submodules, the first is the Linear and second is dropout
r'blocks.\d+.attn1.to_out.0.weight': (None, axis), # (torch.Size([1536, 1536]), torch.bfloat16)
# 'blocks.\d+.attn1.to_out.0.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.attn1.to_out.1.weight': (), # (torch.Size([1536, 1536]), torch.bfloat16)
# 'blocks.\d+.attn1.to_out.1.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.attn2.norm_q.weight': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.attn2.norm_k.weight': (), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_q.weight': (axis, ), # (torch.Size([1536, 1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_q.bias': (axis, ), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_k.weight': (axis, ), # (torch.Size([1536, 1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_k.bias': (axis, ), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_v.weight': (axis, ), # (torch.Size([1536, 1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_v.bias': (axis, ), # (torch.Size([1536]), torch.bfloat16)
r'blocks.\d+.attn2.to_out.0.weight': (None, axis), # (torch.Size([1536, 1536]), torch.bfloat16)
# 'blocks.\d+.attn2.to_out.0.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.attn2.to_out.1.weight': (), # (torch.Size([1536, 1536]), torch.bfloat16)
# 'blocks.\d+.attn2.to_out.1.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'blocks.\d+.norm2.weight': (), # (torch.Size([1536]), torch.float32)
# 'blocks.\d+.norm2.bias': (), # (torch.Size([1536]), torch.float32)
r'blocks.\d+.ffn.net.0.proj.weight': (axis,), # (torch.Size([8960, 1536]), torch.bfloat16)
r'blocks.\d+.ffn.net.0.proj.bias': (axis, ), # (torch.Size([8960]), torch.bfloat16)
r'blocks.\d+.ffn.net.2.weight': (None, axis), # (torch.Size([1536, 8960]), torch.bfloat16)
# 'blocks.\d+.ffn.net.2.bias': (), # (torch.Size([1536]), torch.bfloat16)
# 'proj_out.weight': (), # (torch.Size([64, 1536]), torch.bfloat16)
# 'proj_out.bias': (), # (torch.Size([64]), torch.bfloat16)
}

text_encoder_shardings = {
  'shared.weight': (axis, ), # (torch.Size([256384, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.SelfAttention.q.weight': (axis, ), # (torch.Size([4096, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.SelfAttention.k.weight': (axis, ), # (torch.Size([4096, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.SelfAttention.v.weight': (axis, ), # (torch.Size([4096, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.SelfAttention.o.weight': (None, axis), # (torch.Size([4096, 4096]), torch.bfloat16)
  # 'encoder.block.*.layer.*.SelfAttention.relative_attention_bias.weight': (), # (torch.Size([32, 64]), torch.bfloat16)
  # 'encoder.block.*.layer.*.layer_norm.weight': (), # (torch.Size([4096]), torch.bfloat16)
  'encoder.block.*.layer.*.DenseReluDense.wi_0.weight': (axis, ), # (torch.Size([10240, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.DenseReluDense.wi_1.weight': (axis, ), # (torch.Size([10240, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.DenseReluDense.wo.weight': (None, axis), # (torch.Size([4096, 10240]), torch.bfloat16)
  # 'encoder.final_layer_norm.weight': (), # (torch.Size([4096]), torch.bfloat16)
}


def _shard_weight_dict(weight_dict, sharding_dict, mesh):
  result = {}
  for k, v in weight_dict.items():
    for target, sharding in sharding_dict.items():
      if re.fullmatch(target, k) is not None:
        v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
        break
    else:
      # replicate
      v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))

    result[k] = v
  return result


def flatten_model_output(obj):
  return obj.to_tuple(), type(obj)

def unflatten_model_output(aux, children):
  return aux(*children)

register_pytree_node(
  modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
  flatten_model_output,
  unflatten_model_output)

def make_key(name):
  return re.sub('\.\d+\.', '.*.', name)

  
def _get_weights_of_linear(module):

  result = {}

  def fn(start_path, module):
    if isinstance(module, torch.nn.Linear):
      for k, v in module.named_parameters():
        start_path.append(k)
        key = '.'.join(start_path)
        result[key] = v
        start_path.pop()
    else:
      for name, child in module.named_children():
        start_path.append(name)
        fn(start_path, child)
        start_path.pop()
  fn([], module)
  return result


def _print_weights(module):
  all_buffers = dict(module.named_parameters())
  all_buffers.update(module.named_buffers())
  result = {}
  for k, v in all_buffers.items():
    result[make_key(k)] = (v.shape, v.dtype)
  print('{')
  for k, v in result.items():
    print(f"'{k}': (), # {v}")
  print('}')

def main():
  torch.set_default_dtype(torch.bfloat16)
  # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  #model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
  # model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
  model_id = MODEL_ID
  vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
  # flow_shift = 5.0 # 5.0 for 720P, 3.0 for 480P
  flow_shift = FLOW_SHIFT
  scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
  pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
  pipe.scheduler = scheduler

  # print('vae=====')
  # _print_weights(pipe.vae)
  # print('trans===')
  # print(_get_weights_of_linear(pipe.transformer).keys())
  # print('encoder===')
  # _print_weights(pipe.text_encoder)
  # return

  def _move_module(module):
    with jax.default_device('cpu'):
      state_dict  = module.state_dict()
      state_dict = env.to_xla(state_dict)
      module.load_state_dict(state_dict, assign=True)

  torchax.enable_globally()
  env = torchax.default_env()
  mesh = jax.make_mesh((len(jax.devices()),), (axis,))
  env.default_device_or_sharding = NamedSharding(mesh, P())

  env._mesh = mesh
  env.config.use_tpu_flash_attention = True
  env.config.shmap_flash_attention = True


  vae_options = torchax.CompileOptions(
    methods_to_compile=['decode']
  )
  _move_module(pipe.vae)
  pipe.vae = torchax.compile(pipe.vae)
  _move_module(pipe.text_encoder)
  pipe.text_encoder = torchax.compile(pipe.text_encoder)

  # the param below is not declared as param or buffer so the module.to('jax') didnt work
  _move_module(pipe.transformer)
  pipe.transformer.rope.freqs = pipe.transformer.rope.freqs.to('jax')
  options = torchax.CompileOptions(
      jax_jit_kwargs={'static_argnames': ('return_dict',)}
  )
  pipe.transformer = torchax.compile(pipe.transformer, options)

  #pipe.to('jax')
  print('Number of devices is:, ', len(jax.devices()))

  

  pipe.transformer.params = _shard_weight_dict(pipe.transformer.params, 
                                               transformer_shardings,
                                               mesh)
  pipe.transformer.buffers = _shard_weight_dict(pipe.transformer.buffers, 
                                               transformer_shardings,
                                               mesh)
  pipe.text_encoder.params = _shard_weight_dict(pipe.text_encoder.params, 
                                               text_encoder_shardings,
                                               mesh)
  pipe.text_encoder.buffers = _shard_weight_dict(pipe.text_encoder.buffers, 
                                               text_encoder_shardings,
                                               mesh)


  # NOTE this will effectively replicate vae
  pipe.vae.params = _shard_weight_dict(pipe.vae.params, {}, mesh)
  pipe.vae.buffers = _shard_weight_dict(pipe.vae.buffers, {}, mesh)

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
  # prompt = "Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach.The crashing blue waters create white-tipped waves,while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and greenshrubbery covers the cliffs edge. The steep drop from the road down to the beach is adramatic feat, with the cliff's edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway."
  negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


  with mesh:
    outputs = []
    for i in range(5):
      start = time.perf_counter()
      if i == 3:
        jax.profiler.start_trace('/tmp/profiler/wan')
      output = pipe(
          prompt=prompt,
          negative_prompt=negative_prompt,
          height=HEIGHT,
          width=WIDTH,
          # num_inference_steps=50,
          num_inference_steps=NUM_STEP,
          # height=720,
          # width=1280,
          num_frames=FRAMES, ### YYY: OOM use 41, need 81
          guidance_scale=5.0,
          ).frames[0]
      if i == 4:
        jax.effects_barrier()
        jax.profiler.stop_trace()
        break
      end = time.perf_counter()  
      print(f'Iteration {i}: {end - start:.6f}s')
      outputs.append(output)

      if i == 0:
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"/tmp/{current_datetime}.mp4"
        export_to_video(output, file_name, fps=FPS)
        print(f"output video done. {file_name}")
    print('DONE')

  #print(f'生成视频时长= {(num_frams-1)/fps} - 目前针对1.3B生成5s = (41-1)/8)


if __name__ == '__main__':
  main()