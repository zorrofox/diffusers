import functools
import re
import math
import torch
import torchax
from torchax.ops import ops_registry
import time
import jax
import jax.numpy as jnp
import numpy as np

from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.sharding import Mesh
from jax.experimental import mesh_utils

# Add JAX CogVideoX VAE imports
from flax import nnx
from diffusers.pipelines.cogvideo.modeling_flax_cogvideox_vae import (
    FlaxAutoencoderKLCogVideoXConfig,
    FlaxAutoencoderKLCogVideoX,
    FlaxAutoencoderKLCogVideoXCache,
    load_cogvideox_vae,
)
from flax.linen import partitioning as nn_partitioning

from diffusers.utils import export_to_video
from diffusers import CogVideoXPipeline

from jax.tree_util import register_pytree_node

from transformers import modeling_outputs

from datetime import datetime

import traceback
import types
import argparse

#### SETTINGS
# Models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
MODEL_ID = "THUDM/CogVideoX-2b"

WIDTH = 832
HEIGHT = 480

# 49 frames for CogVideoX
FRAMES = 49
FPS = 8  # CogVideoX default is 8

# step for CogVideoX
NUM_STEP = 5  # Reduced for faster testing

BQSIZE = 3024
BKVSIZE = 2048
BKVCOMPUTESIZE = 1024

WINDOW_SIZE = None

PROFILE_OUT_PATH = "/dev/shm/tensorboard"

USE_DP = True
SP_NUM = 1
USE_FSDP = True

# for shard vae
LOGICAL_AXIS_RULES = (
                    ('conv_out', ('axis','dp','sp')),
                    ('conv_in', ('axis','dp','sp'))
                  )

USE_K_SMOOTH = True

####


axis = 'axis'

# Sharding for CogVideoX transformer
# These are common patterns for attention and feedforward layers
transformer_shardings_fsdp = {
    # Attention layers - shard output dimension
    r'.*\.to_q\.weight': (None, ('axis','sp')),
    r'.*\.to_k\.weight': (None, ('axis','sp')),
    r'.*\.to_v\.weight': (None, ('axis','sp')),
    r'.*\.to_out.*\.weight': (('axis','sp'), None),
    # Feedforward layers
    r'.*\.ff\.net\.0\.weight': (None, ('axis','sp')),
    r'.*\.ff\.net\.2\.weight': (('axis','sp'), None),
}

transformer_shardings_tp = {
    # Attention layers - shard input dimension for TP
    r'.*\.to_q\.weight': (('axis','sp'), None),
    r'.*\.to_k\.weight': (('axis','sp'), None), 
    r'.*\.to_v\.weight': (('axis','sp'), None),
    r'.*\.to_out.*\.weight': (None, ('axis','sp')),
    # Feedforward layers
    r'.*\.ff\.net\.0\.weight': (('axis','sp'), None),
    r'.*\.ff\.net\.2\.weight': (None, ('axis','sp')),
}

text_encoder_shardings = {
  'shared.weight': ((axis,'dp','sp'), ), # (torch.Size([256384, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.SelfAttention.q.weight': ((axis,'dp','sp'), ), # (torch.Size([4096, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.SelfAttention.k.weight': ((axis,'dp','sp'), ), # (torch.Size([4096, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.SelfAttention.v.weight': ((axis,'dp','sp'), ), # (torch.Size([4096, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.SelfAttention.o.weight': (None, (axis,'dp','sp')), # (torch.Size([4096, 4096]), torch.bfloat16)
  # 'encoder.block.*.layer.*.SelfAttention.relative_attention_bias.weight': (), # (torch.Size([32, 64]), torch.bfloat16)
  # 'encoder.block.*.layer.*.layer_norm.weight': (), # (torch.Size([4096]), torch.bfloat16)
  'encoder.block.*.layer.*.DenseReluDense.wi_0.weight': ((axis,'dp','sp'), ), # (torch.Size([10240, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.DenseReluDense.wi_1.weight': ((axis,'dp','sp'), ), # (torch.Size([10240, 4096]), torch.bfloat16)
  'encoder.block.*.layer.*.DenseReluDense.wo.weight': (None, (axis,'dp','sp')), # (torch.Size([4096, 10240]), torch.bfloat16)
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


### Splash attention ###

def _sdpa_reference(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
) -> torch.Tensor:
  L, S = query.size(-2), key.size(-2)
  scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
  attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
  if is_causal:
    assert attn_mask is None
    temp_mask = torch.ones(
        L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_bias.to(query.dtype)
  if attn_mask is not None:
    if attn_mask.dtype == torch.bool:
      attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    else:
      attn_bias += attn_mask
  if enable_gqa:
    key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
    value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

  attn_weight = query @ key.transpose(-2, -1) * scale_factor
  attn_weight += attn_bias
  attn_weight = torch.softmax(attn_weight, dim=-1)
  if dropout_p > 0:
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
  return attn_weight @ value


# <--- MODIFIED: Added window_size parameter to the function signature --->
def _tpu_splash_attention(query, key, value, env, scale=None, is_causal=False, window_size=None):
    import jax
    import math
    mesh = env._mesh
    num_heads = query.shape[1]

    # The function that will be sharded across devices.
    def _attention_on_slices(q, k, v):
        import jax.numpy as jnp
        # Scale the query tensor. This happens on each device with its slice of data.
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        q = q * scale_factor

        # Helper to pad to next multiple
        def pad_to_multiple(x, multiple, axis):
            seq_len = x.shape[axis]
            pad_len = (multiple - seq_len % multiple) % multiple
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len

        # This function operates on a single item from the batch.
        def kernel_3d(q_3d, k_3d, v_3d):
            q_seq_len = q_3d.shape[1]
            kv_seq_len = k_3d.shape[1]
            num_heads_on_device = q_3d.shape[0]

            # CogVideoX has different sequence lengths.
            # Self-attention in CogVideoX is on spatial dimensions, cross-attention is on text.
            # The sequence length will be different from Wan.
            # Let's keep the logic but be aware it might need tuning.
            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, k_orig_len = pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, v_orig_len = pad_to_multiple(v_3d, BKVSIZE, axis=1)

            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]

            # ======================= NEW MASK LOGIC =======================
            if window_size is not None:
                mask_class = functools.partial(splash_attention.LocalMask, window_size=window_size, offset=0)
            else:
                mask_class = splash_attention.FullMask

            mask = splash_attention.MultiHeadMask(
                [mask_class((padded_q_seq_len, padded_kv_seq_len)) for _ in range(num_heads_on_device)]
            )
            # =============================================================

            block_sizes = splash_attention.BlockSizes(
                block_q=min(BQSIZE, padded_q_seq_len),
                block_kv=min(BKVSIZE, padded_kv_seq_len),
                block_kv_compute=min(BKVCOMPUTESIZE, padded_kv_seq_len),
            )
            splash_kernel = splash_attention.make_splash_mha(
                mask=mask, block_sizes=block_sizes, head_shards=1, q_seq_shards=1
            )
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            # Remove padding if any
            return out[:, :q_orig_len, ...]

        # Map the kernel over the batch dimension.
        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    # Sharding logic for CogVideoX might be different.
    # It depends on how attention is used (self vs cross).
    if num_heads < mesh.size:
        q_partition_spec = P()
        kv_partition_spec = P()
    else:
        # This logic is specific to Wan's transformer structure.
        # A simple heuristic: if q and k have same seq len, it's self-attention.
        if query.shape[2] == key.shape[2]:  # Self-attention
          q_partition_spec = P('dp', 'axis', 'sp', None)
          kv_partition_spec = P('dp', 'axis', None, None)
        else:  # Cross-attention
          q_partition_spec = P('dp', None, ('axis', 'sp'), None)
          kv_partition_spec = P('dp', None, None, None)

    # ALWAYS use shard_map. The partition_spec will control the behavior.
    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_rep=False,
    )
    out = sharded_fn(query, key, value)
    out = jax.lax.with_sharding_constraint(out, P('dp', None, ('axis', 'sp'), None))
    return out


# <--- MODIFIED: Added window_size parameter to the function signature --->
def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    env=None,
    window_size=None, # <--- NEW
) -> torch.Tensor:

  if env.config.use_tpu_splash_attention:
    jquery, jkey, jvalue = env.t2j_iso((query, key, value))
    if USE_K_SMOOTH:
      key_mean = jnp.mean(jkey, axis=2, keepdims=True)
      # jkey_smoothed
      jkey = jkey - key_mean
    # <--- MODIFIED: Pass window_size to the backend function --->
    res = _tpu_splash_attention(jquery, jkey, jvalue, env, scale=scale, is_causal=is_causal, window_size=window_size)
    return env.j2t_iso(res)

  return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal,
                         scale, enable_gqa)

### Sharding VAE ###

def _add_sharding_rule(vs: nnx.VariableState, logical_axis_rules) -> nnx.VariableState:
  vs.sharding_rules = logical_axis_rules
  return vs

@nnx.jit(static_argnums=(1,), donate_argnums=(0,))
def create_sharded_logical_model(model, logical_axis_rules):
  graphdef, state, rest_of_state = nnx.split(model, nnx.Param, ...)
  p_add_sharding_rule = functools.partial(_add_sharding_rule, logical_axis_rules=logical_axis_rules)
  state = jax.tree.map(p_add_sharding_rule, state, is_leaf=lambda x: isinstance(x, nnx.VariableState))
  pspecs = nnx.get_partition_spec(state)
  sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
  model = nnx.merge(graphdef, sharded_state, rest_of_state)
  return model

#####################


# --- Config Wrapper ---
class ConfigWrapper:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        setattr(self, key, value)

def to_torch_recursive(x):
    import torch
    import numpy as np
    if 'ArrayImpl' in str(type(x)) or isinstance(x, jnp.ndarray):
        # Handle JAX arrays
        np_array = np.array(x)
        # If the array is bfloat16, convert through float32
        if hasattr(x, 'dtype') and x.dtype == jnp.bfloat16:
            return torch.from_numpy(np_array.astype(np.float32)).to(torch.bfloat16)
        else:
            return torch.from_numpy(np_array)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_torch_recursive(xx) for xx in x)
    elif isinstance(x, dict):
        return {k: to_torch_recursive(v) for k, v in x.items()}
    elif hasattr(x, 'sample'):
        sample = to_torch_recursive(x.sample)
        if hasattr(x, 'replace'):
            return x.replace(sample=sample)
        else:
            return sample
    else:
        return x

def to_jax_recursive(x):
    import torch
    if isinstance(x, torch.Tensor):
        # Handle BFloat16 specially since numpy doesn't support it
        if x.dtype == torch.bfloat16:
            # Convert to float32 first, then to JAX array
            return jnp.array(x.detach().to(torch.float32).cpu().numpy()).astype(jnp.bfloat16)
        else:
            return jnp.array(x.detach().cpu().numpy())
    elif isinstance(x, (list, tuple)):
        return type(x)(to_jax_recursive(xx) for xx in x)
    elif isinstance(x, dict):
        return {k: to_jax_recursive(v) for k, v in x.items()}
    else:
        return x

class VAEProxy:
    def __init__(self, vae, vae_cache, dtype, config, mesh=None):
        self._vae = vae
        self.vae_cache = vae_cache
        self.dtype = dtype
        self.config = config
        self._original_sample = None
        self._mesh = mesh
    def __getattr__(self, name):
        return getattr(self._vae, name)
    def encode(self, sample, *args, **kwargs):
        # Store the original sample for use in decode
        self._original_sample = sample
        # Convert to JAX and encode
        jax_sample = to_jax_recursive(sample)
        out = self._vae.encode(jax_sample, *args, **kwargs)
        return to_torch_recursive(out)
    def decode(self, latents, *args, **kwargs):
        # The JAX VAE doesn't need a cache like the maxdiffusion one.
        # So we remove the feat_cache logic.
        # CogVideoX VAE needs the original sample for spatial conditioning
        
        # Apply scaling factor before decoding (CogVideoX uses 1/0.7)
        scaling_factor = self.config.scaling_factor if hasattr(self.config, 'scaling_factor') else 0.7
        latents = latents / scaling_factor
        
        # Convert latents to JAX format
        jax_latents = to_jax_recursive(latents)
        
        # Debug: Check the shape of latents
        print(f"DEBUG: VAE decode input shape: {jax_latents.shape}")
        print(f"DEBUG: Expected latent channels: {self.config.latent_channels}")
        print(f"DEBUG: Latent stats - min: {float(jnp.min(jax_latents)):.4f}, max: {float(jnp.max(jax_latents)):.4f}, mean: {float(jnp.mean(jax_latents)):.4f}, std: {float(jnp.std(jax_latents)):.4f}")
        
        # Convert from PyTorch format (NCDHW) to JAX format (NDHWC)
        if len(jax_latents.shape) == 5:
            B, C, T, H, W = jax_latents.shape
            print(f"DEBUG: Input shape (PyTorch format BCTHW): [B={B}, C={C}, T={T}, H={H}, W={W}]")
            # Convert from BCTHW to BTHWC for JAX
            jax_latents = jax_latents.transpose(0, 2, 3, 4, 1)
            print(f"DEBUG: After transpose to JAX format (BTHWC): {jax_latents.shape}")
            
            # Apply sharding to the latents if mesh is available
            if self._mesh is not None:
                from jax.sharding import NamedSharding, PartitionSpec as P
                try:
                    # Shard latents along spatial dimensions
                    latent_sharding = NamedSharding(self._mesh, P(None, None, 'sp', 'axis', None))
                    jax_latents = jax.device_put(jax_latents, latent_sharding)
                    print(f"DEBUG: Applied sharding to latents")
                except Exception as e:
                    print(f"DEBUG: Could not apply sharding to latents: {e}")
            else:
                print(f"DEBUG: No mesh available for sharding latents")
        
        # For spatial conditioning in CogVideoX, we need to use latent space representation
        # not RGB space. The 'sample' should be a latent tensor, not an RGB image
        if self._original_sample is None:
            # No original sample available (generation mode)
            # For generation, we can use the input latents themselves as spatial conditioning
            # This is a common approach when no reference is available
            jax_sample = jax_latents
            print(f"DEBUG: Using input latents as spatial conditioning: {jax_sample.shape}")
        else:
            # Use the stored original sample (should be in latent space)
            sample = self._original_sample
            jax_sample = to_jax_recursive(sample) if hasattr(sample, 'numpy') else sample
            
            # Convert sample to JAX format if it's 5D
            if hasattr(jax_sample, 'shape') and len(jax_sample.shape) == 5:
                print(f"DEBUG: Sample shape before transpose: {jax_sample.shape}")
                # Assume it's in BCTHW format, convert to BTHWC
                jax_sample = jax_sample.transpose(0, 2, 3, 4, 1)
                print(f"DEBUG: Sample shape after transpose to JAX format: {jax_sample.shape}")
        
        out = self._vae.decode(jax_latents, sample=jax_sample)
        
        # Convert output back from JAX format (NDHWC) to PyTorch format (NCDHW)
        if hasattr(out, 'shape') and len(out.shape) == 5:
            print(f"DEBUG: VAE output shape (JAX format): {out.shape}")
            # Convert from BTHWC to BCTHW
            out = out.transpose(0, 4, 1, 2, 3)
            print(f"DEBUG: After transpose to PyTorch format: {out.shape}")
        
        # Convert to torch tensor
        torch_out = to_torch_recursive(out)
        
        # Debug: Check output statistics
        print(f"DEBUG: VAE output stats - min: {torch_out.min().item():.4f}, max: {torch_out.max().item():.4f}, mean: {torch_out.mean().item():.4f}")
        print(f"DEBUG: VAE output std: {torch_out.std().item():.4f}")
        
        # Import DecoderOutput from diffusers
        from diffusers.models.autoencoders.vae import DecoderOutput
        
        # Wrap in DecoderOutput to match PyTorch VAE interface
        return DecoderOutput(sample=torch_out)

def prepare_video_for_export(video):
    import torch
    import numpy as np
    if isinstance(video, (list, tuple)):
        return [prepare_video_for_export(v) for v in video]
    if isinstance(video, torch.Tensor):
        # Handle 5D tensor (batch, channels, frames, height, width)
        if video.dim() == 5:
            video = video[0]
        
        # Now we should have 4D tensor
        # Check if it's (frames, channels, height, width) format
        if video.dim() == 4:
            # Assume channels is dimension 1 if it's 3
            if video.shape[1] == 3:
                # Convert from (frames, channels, height, width) to (frames, height, width, channels)
                video = video.permute(0, 2, 3, 1)
            # If shape[0] is not frames, might be (channels, frames, height, width)
            elif video.shape[0] == 3:
                # Convert from (channels, frames, height, width) to (frames, height, width, channels)
                video = video.permute(1, 2, 3, 0)
        
        # Convert BFloat16 to Float32 before converting to numpy
        if video.dtype == torch.bfloat16:
            video = video.float()
        video = video.cpu().numpy()
        video = (video * 255).round().astype("uint8")
        return video
    if isinstance(video, np.ndarray):
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        return video
    return video

def sharded_device_put(tensor, sharding):
  if isinstance(tensor, tuple):
    return tuple(sharded_device_put(t, sharding) for t in tensor)
  num_global_devices = jax.device_count()
  num_local_devices = jax.local_device_count()

  if num_global_devices == num_local_devices:
    return jax.device_put(tensor, sharding)

  shape = tensor.shape
  x_split = [
    jax.device_put(tensor[i], device)
    for device, i in sharding.addressable_devices_indices_map(shape).items()
  ]
  return jax.make_array_from_single_device_arrays(shape, sharding, x_split)

def main():
  # Set JAX config to enable compilation cache
  jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
  jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
  jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

  torch.set_default_dtype(torch.bfloat16)
  model_id = args.model_id
  
  tp_dim, dp_dim, sp_dim = len(jax.devices()), 1, 1
  if args.use_dp:
    print(f"{args.use_dp=}")
    tp_dim //= 2
    dp_dim = 2
  
  if args.sp_num > 1:
    print(f"{args.sp_num=}")
    tp_dim //= args.sp_num
    sp_dim = args.sp_num

  print(f"{tp_dim=}, {dp_dim=}, {sp_dim=}")
  # mesh = jax.make_mesh((len(jax.devices()), 1), (axis, 'fsdp'))
  mesh_devices = mesh_utils.create_device_mesh((tp_dim, dp_dim, sp_dim), allow_split_physical_axes=True)
  mesh = Mesh(mesh_devices, (axis,'dp','sp'))

  # Initialize JAX VAE
  key = jax.random.key(0)
  rngs = nnx.Rngs(key)
  
  # Create JAX VAE from config
  # Load config BEFORE enabling torchax, as it can interfere with from_pretrained
  vae_config = FlaxAutoencoderKLCogVideoX.config_class.from_config(FlaxAutoencoderKLCogVideoX.config_class.load_config(model_id, subfolder="vae"))
  
  with mesh:
    # Create model inside mesh context
    cog_vae = FlaxAutoencoderKLCogVideoX(config=vae_config, rngs=rngs)
    
    # Debug: Check parameter count right after creation
    graphdef_initial, state_initial = nnx.split(cog_vae)
    params_initial = state_initial.to_pure_dict()
    initial_count = len(jax.tree_util.tree_leaves(params_initial))
    print(f"Initial model parameter count (right after creation): {initial_count}")
    
    # Create VAE cache
    vae_cache = FlaxAutoencoderKLCogVideoXCache(cog_vae)
    
    # Load pretrained weights
    graphdef, state = nnx.split(cog_vae)
    params = state.to_pure_dict()
    
    # Debug: Print model parameter structure
    print("Model parameters structure (before loading weights):")
    model_leaf_count = len(jax.tree_util.tree_leaves(params))
    print(f"Total model parameters (leaf nodes): {model_leaf_count}")
    
    try:
        loaded_weights = load_cogvideox_vae(model_id, {}, "tpu")
        
        # Debug: Print loaded parameter structure  
        print("Loaded parameters structure:")
        # Count leaf nodes
        leaf_count = len(jax.tree_util.tree_leaves(loaded_weights))
        print(f"Total loaded parameters (leaf nodes): {leaf_count}")
        
        # CogVideoX doesn't use quant_conv and post_quant_conv
        print("Note: CogVideoX doesn't use quant_conv and post_quant_conv layers")
        
        # No need to add dummy weights
        updated_leaf_count = leaf_count
        print(f"Loaded parameters ready for use: {updated_leaf_count}")
        
        # 保证全部 replicate 到 mesh 上所有 device
        sharding = NamedSharding(mesh, P())
        params = jax.tree_util.tree_map(lambda x: sharded_device_put(x, sharding), loaded_weights)
        params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
        
        print("Attempting to merge parameters...")
        print(f"DEBUG: Model expects {model_leaf_count} parameters, checkpoint provides {updated_leaf_count}")
        
        # Try to merge anyway and see which parameters are missing
        try:
            cog_vae = nnx.merge(graphdef, params)
            print("Merge successful!")
        except ValueError as e:
            print(f"Merge failed: {e}")
            
            # Debug: Let's see what parameters are missing
            model_params = state.to_pure_dict()
            loaded_params = params
            
            # Compare keys
            model_flat, _ = jax.tree_util.tree_flatten(model_params)
            loaded_flat, _ = jax.tree_util.tree_flatten(loaded_params)
            
            # For debugging, let's just count the leaves
            model_keys_count = len(model_flat)
            loaded_keys_count = len(loaded_flat)
            
            print(f"\nDEBUG: Model parameter count: {model_keys_count}")
            print(f"DEBUG: Loaded parameter count: {loaded_keys_count}")
            print(f"DEBUG: Difference: {model_keys_count - loaded_keys_count}")
            
            # Try to use the loaded parameters even if merge fails
            print("\nAttempting to update model with loaded parameters...")
            state_dict = state.to_pure_dict()
            
            # Debug: Print some actual keys to understand the structure
            print("\nDEBUG: Sample model keys:")
            for i, (k, v) in enumerate(jax.tree_util.tree_flatten_with_path(state_dict)[0][:10]):
                key_str = '.'.join(str(part.key) if hasattr(part, 'key') else str(part) for part in k)
                print(f"  {key_str}")
            
            print("\nDEBUG: Sample loaded keys:")
            for i, (k, v) in enumerate(jax.tree_util.tree_flatten_with_path(loaded_params)[0][:10]):
                key_str = '.'.join(str(part.key) if hasattr(part, 'key') else str(part) for part in k)
                print(f"  {key_str}")
            
            # First, let's try a simpler approach - directly use loaded params
            print("\nDirect merge attempt with loaded params...")
            try:
                # Simply use the loaded params, ignoring the model's initial state
                cog_vae = nnx.merge(graphdef, loaded_params)
                print("Direct merge successful with loaded params!")
            except Exception as e2:
                print(f"Direct merge also failed: {e2}")
                
                # If that fails too, we need to understand the structure difference
                # For now, let's proceed with the model as-is
                print("\nWARNING: Using model with default initialization!")
                print("This will produce incorrect results!")
                
                # Debug: Look for conv_shortcut keys in both model and loaded params
                def find_conv_shortcut_keys(params_dict, path=""):
                    keys = []
                    for key, value in params_dict.items():
                        current_path = f"{path}.{key}" if path else key
                        if isinstance(value, dict):
                            keys.extend(find_conv_shortcut_keys(value, current_path))
                        elif "conv_shortcut" in current_path:
                            keys.append(current_path)
                    return keys
                
                model_shortcut_keys = find_conv_shortcut_keys(state_dict)
                loaded_shortcut_keys = find_conv_shortcut_keys(loaded_params)
                
                print("\nDEBUG: conv_shortcut keys in model:")
                for k in sorted(model_shortcut_keys):
                    print(f"  {k}")
                    
                print("\nDEBUG: conv_shortcut keys in loaded params:")
                for k in sorted(loaded_shortcut_keys):
                    print(f"  {k}")
                
                # Update state_dict with loaded parameters where keys match
                def update_nested_dict(target, source, path="", show_warnings=True):
                    updated_count = 0
                    for key, value in source.items():
                        if key in target:
                            if isinstance(value, dict) and isinstance(target[key], dict):
                                sub_count = update_nested_dict(target[key], value, f"{path}.{key}", show_warnings)
                                updated_count += sub_count
                            else:
                                target[key] = value
                                updated_count += 1
                        else:
                            if not isinstance(value, dict) and show_warnings:  # Only warn for leaf nodes
                                print(f"Warning: Key not found in model: {path}.{key}")
                    return updated_count
                
                # Also check which model keys are not in loaded params
                def find_missing_in_loaded(target, source, path=""):
                    missing = []
                    for key, value in target.items():
                        current_path = f"{path}.{key}" if path else key
                        if key in source:
                            if isinstance(value, dict) and isinstance(source[key], dict):
                                sub_missing = find_missing_in_loaded(value, source[key], current_path)
                                missing.extend(sub_missing)
                        else:
                            if isinstance(value, dict):
                                # All leaves under this dict are missing
                                def get_all_leaves(d, p):
                                    leaves = []
                                    for k, v in d.items():
                                        if isinstance(v, dict):
                                            leaves.extend(get_all_leaves(v, f"{p}.{k}"))
                                        else:
                                            leaves.append(f"{p}.{k}")
                                    return leaves
                                missing.extend(get_all_leaves(value, current_path))
                            else:
                                missing.append(current_path)
                    return missing
                
                missing_in_loaded = find_missing_in_loaded(state_dict, loaded_params)
                print(f"\nDEBUG: Model parameters not found in loaded weights ({len(missing_in_loaded)}):")
                for k in missing_in_loaded[:10]:  # Show first 10
                    print(f"  {k}")
            
                updated = update_nested_dict(state_dict, loaded_params)
                print(f"\nTotal parameters updated: {updated}")
                
                # Recreate the model with updated parameters
                cog_vae = nnx.merge(graphdef, state_dict)
                print("Model updated with partially loaded parameters!")
        
        # Apply sharding to the VAE model
        print("Applying sharding to VAE...")
        
        # Debug: Check parameter count before sharding
        graphdef_before_shard, state_before_shard = nnx.split(cog_vae)
        params_before_shard = state_before_shard.to_pure_dict()
        before_shard_count = len(jax.tree_util.tree_leaves(params_before_shard))
        print(f"Parameter count before sharding: {before_shard_count}")
        
        cog_vae = create_sharded_logical_model(cog_vae, LOGICAL_AXIS_RULES)
        
        # Debug: Check parameter count after sharding
        graphdef_after_shard, state_after_shard = nnx.split(cog_vae)
        params_after_shard = state_after_shard.to_pure_dict()
        after_shard_count = len(jax.tree_util.tree_leaves(params_after_shard))
        print(f"Parameter count after sharding: {after_shard_count}")
        
    except Exception as e:
        print(f"Error during weight loading/merging: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise

    # Load the CogVideoX pipeline BEFORE enabling torchax
    # This avoids the UntypedStorage error
    print(f"Loading CogVideoX pipeline from {model_id}...")
    pipe = CogVideoXPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    print("Pipeline loaded successfully!")
    
    # NOW enable torchax after all models are loaded
    torchax.enable_globally()
    
    # Set torchax to be more permissive with tensor types
    import os
    os.environ['TORCHAX_ALLOW_TORCH_FALLBACK'] = '1'
  
    # Setup torchax environment
    env = torchax.default_env()
    env.default_device_or_sharding = NamedSharding(mesh, P())
    env._mesh = mesh
    env.config.use_tpu_splash_attention = True

  # Replace the VAE in the pipeline with our JAX VAE
  vae_config_proxy = ConfigWrapper(
      scaling_factor=cog_vae.config.scaling_factor,
      block_out_channels=cog_vae.config.block_out_channels,
      temporal_compression_ratio=cog_vae.config.temporal_compression_ratio,
      latent_channels=cog_vae.config.latent_channels,
  )
  pipe.vae = VAEProxy(cog_vae, vae_cache, torch.bfloat16, vae_config_proxy, mesh=mesh)

  # 伪装 config
  vae_config = ConfigWrapper(
      scaling_factor=cog_vae.config.scaling_factor,
      block_out_channels=cog_vae.config.block_out_channels,
      temporal_compression_ratio=cog_vae.config.temporal_compression_ratio,
      latent_channels=cog_vae.config.latent_channels,
  )
  pipe.vae.config = vae_config
  
  # Helper function to move module weights to XLA
  def _move_module(module):
    with jax.default_device('cpu'):
      state_dict  = module.state_dict()
      state_dict = env.to_xla(state_dict)
      module.load_state_dict(state_dict, assign=True)

  # <--- MODIFIED: Override flash attention with custom function, now with window_size --->
  custom_attention = functools.partial(
      scaled_dot_product_attention,
      env=env,
      window_size=args.window_size # Inject the global window size setting here
  )
  # Workaround for the function lack is_view_op argument
  # env.override_op_definition(torch.nn.functional.scaled_dot_product_attention, custom_attention)
  op_to_override = torch.nn.functional.scaled_dot_product_attention
  op_impl = custom_attention
  env._ops[op_to_override] = ops_registry.Operator(
        op_to_override,
        op_impl,
        is_jax_function=False,
        is_user_defined=True,
        needs_env=False,
        is_view_op=False,
    )

  # Compile modules with torchax (skip VAE as it's already JAX)
  
  if args.t5_cpu:
    # 只把 text_encoder 移到 CPU，不做 compile 和 shard
    pipe.text_encoder.to("cpu")
  else:
    # TPU 路径，做 compile 和 shard
    # First move to XLA, then compile
    _move_module(pipe.text_encoder)
    
    # Try to compile text_encoder with error handling
    try:
      print("Attempting to compile text_encoder...")
      pipe.text_encoder = torchax.nn_module.compile(
        pipe.text_encoder, 
        is_lazy=True,
        mark_dynamic=['input_ids'],
      )
      print("Text encoder compilation successful!")
    except Exception as e:
      print(f"Text encoder compilation failed: {e}")
      print("Continuing without compilation...")

  # Move transformer module to XLA before compilation
  _move_module(pipe.transformer)
  # CogVideoX transformer doesn't have `rope.freqs`
  # pipe.transformer.rope.freqs = pipe.transformer.rope.freqs.to('jax')
  
  # Try to compile transformer with error handling
  try:
    print("Attempting to compile transformer...")
    pipe.transformer = torchax.nn_module.compile(
      pipe.transformer,
      is_lazy=True,
      mark_dynamic=['timestep', 'encoder_hidden_states'],
    )
    print("Transformer compilation successful!")
  except Exception as e:
    print(f"Transformer compilation failed: {e}")
    print("Continuing without compilation...")
  
  # Apply sharding to transformer weights even without compilation
  if args.use_fsdp:
    transformer_shardings = transformer_shardings_fsdp
  else:
    transformer_shardings = transformer_shardings_tp
  
  # Shard transformer parameters
  try:
    transformer_state_dict = pipe.transformer.state_dict()
    # Apply sharding to each parameter
    for name, param in transformer_state_dict.items():
      for pattern, sharding_spec in transformer_shardings.items():
        if re.match(pattern, name):
          # Move parameter to XLA with sharding
          sharding = NamedSharding(mesh, P(*sharding_spec))
          param.data = torch.from_numpy(
            np.array(jax.device_put(param.detach().cpu().numpy(), sharding))
          ).to(param.device)
          print(f"Sharded transformer param: {name} with spec {sharding_spec}")
          break
    print("Transformer sharding applied")
  except Exception as e:
    print(f"Failed to shard transformer: {e}")

  #pipe.to('jax')
  print('Number of devices is:, ', len(jax.devices()))
  
  # Debug: Check transformer and VAE channel configuration
  print(f"DEBUG: Transformer in_channels: {pipe.transformer.config.in_channels}")
  print(f"DEBUG: VAE latent_channels: {pipe.vae.config.latent_channels}")

  # Sharding rules for CogVideoX transformer will be different.
  # Disabling for now.
  # if args.use_fsdp:
  #   transformer_shardings = transformer_shardings_fsdp
  # else:
  #   transformer_shardings = transformer_shardings_tp
  #
  # pipe.transformer.params = _shard_weight_dict(pipe.transformer.params,
  #                                              transformer_shardings,
  #                                              mesh)
  # pipe.transformer.buffers = _shard_weight_dict(pipe.transformer.buffers,
  #                                              transformer_shardings,
  #                                              mesh)

  # Move scheduler tensors to XLA
  def move_scheduler(scheduler):
    for k, v in scheduler.__dict__.items():
      if isinstance(v, torch.Tensor):
        setattr(scheduler, k, env.to_xla(v))
  
  move_scheduler(pipe.scheduler)

  def module_size(module):
    size = 0
    for _, v in module.state_dict().items():
      size += math.prod(v.shape) * v.dtype.itemsize
    return size

  for m in dir(pipe):
      module = getattr(pipe, m, None)
      if isinstance(module, torch.nn.Module):
          print(m, module_size(module) / (1024 * 1024 * 1024), 'G')
      elif m == 'vae':
          print(f"{m} (JAX VAE) - size calculation not implemented")


  # Ensure all pipeline components are ready for torchax
  # Move all components to the proper device
  pipe.vae_scale_factor = 2 ** len(pipe.vae.config.block_out_channels)
  
  # For multi-device sharding, we shouldn't hardcode a single device
  # Instead, let's ensure the pipeline can work with sharded models
  # The _execution_device should be handled by the pipeline's logic
  # but we need to make sure our wrapped modules are recognized
  
  # Store references to the wrapped modules so the pipeline can find them
  if hasattr(pipe, 'transformer') and hasattr(pipe.transformer, 'compiled_module'):
    # Make sure the pipeline can access the underlying module for device detection
    pipe._transformer_module = pipe.transformer.compiled_module
  
  if hasattr(pipe, 'text_encoder') and hasattr(pipe.text_encoder, 'compiled_module'):
    pipe._text_encoder_module = pipe.text_encoder.compiled_module
    
  # Override _execution_device to handle our special case
  # In a sharded environment, we use the torchax environment's default
  original_execution_device = pipe.__class__._execution_device
  
  @property  
  def custom_execution_device(self):
    # In a sharded environment, we return a generic XLA device
    # The actual device placement is handled by JAX/torchax sharding
    return torch.device("xla")
  
  pipe.__class__._execution_device = custom_execution_device
  
  # Patch the pipeline's prepare_latents method if it exists
  if hasattr(pipe, 'prepare_latents'):
    original_prepare_latents = pipe.prepare_latents
    
    @functools.wraps(original_prepare_latents)
    def wrapped_prepare_latents(*args, **kwargs):
      # Remove generator to avoid issues
      if 'generator' in kwargs:
        kwargs['generator'] = None
      return original_prepare_latents(*args, **kwargs)
    
    pipe.prepare_latents = wrapped_prepare_latents

  # Move the entire pipeline to XLA
  print("Moving pipeline components to XLA...")
  for name in dir(pipe):
    module = getattr(pipe, name, None)
    if isinstance(module, torch.nn.Module) and name not in ['vae']:  # Skip VAE as it's JAX
      try:
        module.to('xla')
        print(f"  Moved {name} to XLA")
      except Exception as e:
        print(f"  Failed to move {name}: {e}")

  prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
  negative_prompt = ""

  # Set random seed for reproducibility
  import random
  import numpy as np
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  
  # Generator might not work with torchax, but we can try
  try:
    generator = torch.Generator(device='xla')
    generator.manual_seed(42)
  except:
    generator = None
    print("Generator creation failed, proceeding without it")
  
  with mesh, nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
    # warm up and save video
    pipe_kwargs = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'height': args.height,
        'width': args.width,
        'num_inference_steps': args.num_inference_steps,
        'num_frames': args.frames,
        'guidance_scale': 6.0,
        'use_dynamic_cfg': False, # Set to True for dynamic guidance
        'output_type': 'pt',  # Return PyTorch tensor instead of PIL images
    }
    
    # Add generator if available
    if generator is not None:
        pipe_kwargs['generator'] = generator
    
    pipeline_output = pipe(**pipe_kwargs)
    output = pipeline_output.frames[0]
    
    # Debug: Check output statistics before export
    print(f"\nDEBUG: Pipeline output type: {type(pipeline_output)}")
    print(f"DEBUG: Frames type: {type(pipeline_output.frames)}")
    print(f"DEBUG: Output shape: {output.shape}")
    print(f"DEBUG: Output stats - min: {output.min().item():.4f}, max: {output.max().item():.4f}, mean: {output.mean().item():.4f}")
    
    output = prepare_video_for_export(output)
    
    # Debug: Check output after prepare_video_for_export
    print(f"DEBUG: Prepared output shape: {output.shape}")
    print(f"DEBUG: Prepared output stats - min: {output.min()}, max: {output.max()}, mean: {output.mean():.4f}")
    print(f"DEBUG: Expected shape for export_to_video: (num_frames, height, width, channels)")
    print(f"DEBUG: Current shape: {output.shape}, ndim: {output.ndim}")
    
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{current_datetime}.mp4"
    export_to_video(output, file_name, fps=args.fps)
    print(f"output video done. {file_name}")
    jax.effects_barrier()
    
    if args.profile:
      jax.profiler.start_trace(PROFILE_OUT_PATH)
      # CogVideoX doesn't support output_type="latent" in the same way.
      # We will profile the full pipeline.
      pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=3,
        num_frames=args.frames,
        guidance_scale=6.0,
        # generator=generator,  # Skip for torchax
        output_type='pt',  # Return PyTorch tensor
      )
      jax.effects_barrier()
      jax.profiler.stop_trace()
      print("profile done")
    
    # Benchmark loop
    for i in range(1):
      start = time.perf_counter()
      output = pipe(**pipe_kwargs)
      # make sure all computation done
      jax.effects_barrier()
      end = time.perf_counter()  
      print(f'Iteration {i} BKVCOMPUTESIZE={BKVCOMPUTESIZE} BKVSIZE={BKVSIZE}, BQSIZE={BQSIZE}: {end - start:.6f}s')
        
  print('DONE')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    # CogVideoX doesn't use flow_shift
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_STEP)
    parser.add_argument("--window_size", type=int, nargs=2, default=None)
    parser.add_argument("--use_dp", action="store_true", default=USE_DP)
    parser.add_argument("--sp_num", type=int, default=SP_NUM)
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Offload T5 text_encoder to CPU")
    parser.add_argument("--bqsize", type=int, default=BQSIZE, help="Block Q size")
    parser.add_argument("--bkvsize", type=int, default=BKVSIZE, help="Block KV size")
    parser.add_argument("--bkvcomputesize", type=int, default=BKVCOMPUTESIZE, help="Block KV compute size")
    parser.add_argument("--profile", action="store_true", default=False, help="Add profiler")
    parser.add_argument("--use_fsdp", type=bool, default=USE_FSDP, help="Use FSDP")
    parser.add_argument("--use_k_smooth", type=bool, default=USE_K_SMOOTH, help="Use K smooth")
    return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  print(args)
  BQSIZE = args.bqsize
  BKVSIZE = args.bkvsize
  BKVCOMPUTESIZE = args.bkvcomputesize
  USE_K_SMOOTH = args.use_k_smooth
  main()
