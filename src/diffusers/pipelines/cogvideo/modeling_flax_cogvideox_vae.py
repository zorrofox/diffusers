# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx

from ...configuration_utils import ConfigMixin
from ...models.modeling_flax_utils import FlaxModelMixin


class FlaxAutoencoderKLCogVideoXConfig(ConfigMixin):
    """
    Configuration class for FlaxAutoencoderKLCogVideoX.
    """
    config_name = "config.json"

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "DownEncoderBlock3D",
            "DownEncoderBlock3D",
            "DownEncoderBlock3D",
            "DownEncoderBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "UpDecoderBlock3D",
            "UpDecoderBlock3D",
            "UpDecoderBlock3D",
            "UpDecoderBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        latent_channels: int = 16,
        norm_num_groups: int = 32,
        sample_size: int = 512,
        scaling_factor: float = 0.7,
        temporal_compression_ratio: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_block_types = down_block_types
        self.up_block_types = up_block_types
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.act_fn = act_fn
        self.latent_channels = latent_channels
        self.norm_num_groups = norm_num_groups
        self.sample_size = sample_size
        self.scaling_factor = scaling_factor
        self.temporal_compression_ratio = temporal_compression_ratio


class FlaxConv3d(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = (3, 3, 3),
        stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        padding: Union[int, Tuple[int, int, int]] = 1,
        rngs: nnx.Rngs = None,
    ):
        # Handle padding - Flax uses string or tuple, PyTorch uses int
        # For symmetric padding, convert int to 'SAME' or explicit tuple
        if isinstance(padding, int):
            if padding == 0:
                padding_mode = ((0, 0), (0, 0), (0, 0))  # No padding
            else:
                # For padding=1, use explicit padding to match PyTorch behavior
                padding_mode = ((padding, padding), (padding, padding), (padding, padding))
        elif isinstance(padding, tuple) and len(padding) == 3:
            padding_mode = tuple((p, p) for p in padding)
        else:
            padding_mode = padding
            
        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding_mode,
            rngs=rngs,
        )

    def __call__(self, x):
        return self.conv(x)


class FlaxConv2d(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        padding: Union[int, str] = 1,
        rngs: nnx.Rngs = None,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
            
        # Handle padding
        if isinstance(padding, int):
            if padding == 0:
                padding_mode = ((0, 0), (0, 0))
            else:
                padding_mode = ((padding, padding), (padding, padding))
        else:
            padding_mode = padding
            
        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding_mode,
            rngs=rngs,
        )

    def __call__(self, x):
        return self.conv(x)


class FlaxCogVideoXSpatialNorm3D(nnx.Module):
    """
    Spatially conditioned normalization for CogVideoX.
    Matches the PyTorch implementation structure.
    """
    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        groups: int = 32,
        rngs: nnx.Rngs = None,
    ):
        self.norm_layer = nnx.GroupNorm(
            num_features=f_channels,
            num_groups=groups,
            epsilon=1e-6,
            rngs=rngs
        )
        self.conv_y = FlaxConv3d(zq_channels, f_channels, kernel_size=(1, 1, 1), padding=0, rngs=rngs)
        self.conv_b = FlaxConv3d(zq_channels, f_channels, kernel_size=(1, 1, 1), padding=0, rngs=rngs)

    def __call__(self, f, zq):
        # Normalize
        f = self.norm_layer(f)
        
        # Apply spatial conditioning
        # Note: zq should already be at the correct resolution
        # as it's adjusted in the decoder
        y = self.conv_y(zq)
        b = self.conv_b(zq)
        
        # Scale and shift
        return f * (1 + y) + b


class FlaxGroupNorm(nnx.Module):
    """Regular GroupNorm for encoder that matches PyTorch structure."""
    def __init__(self, num_groups: int, num_channels: int, epsilon: float = 1e-5, rngs: nnx.Rngs = None):
        # Instead of using nnx.GroupNorm, let's create our own parameters
        # to match PyTorch exactly
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.epsilon = epsilon
        
        # Create weight and bias parameters directly
        # In PyTorch, GroupNorm has weight and bias of shape (num_channels,)
        key = rngs.params() if rngs else jax.random.PRNGKey(0)
        self.scale = nnx.Param(jnp.ones((num_channels,)))  # This will be mapped from 'weight'
        self.bias = nnx.Param(jnp.zeros((num_channels,)))  # This will be mapped from 'bias'

    def __call__(self, x):
        # Implement GroupNorm manually
        # x shape: (batch, channels, depth, height, width)
        N, C, D, H, W = x.shape
        assert C == self.num_channels
        
        # Reshape to group
        x = x.reshape(N, self.num_groups, C // self.num_groups, D, H, W)
        
        # Compute mean and variance
        mean = jnp.mean(x, axis=(2, 3, 4, 5), keepdims=True)
        var = jnp.var(x, axis=(2, 3, 4, 5), keepdims=True)
        
        # Normalize
        x = (x - mean) / jnp.sqrt(var + self.epsilon)
        
        # Reshape back
        x = x.reshape(N, C, D, H, W)
        
        # Apply scale and bias
        x = x * self.scale.value.reshape(1, C, 1, 1, 1) + self.bias.value.reshape(1, C, 1, 1, 1)
        
        return x


class FlaxResnetBlock3D(nnx.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: Optional[int], 
        temb_channels: int, 
        groups: int, 
        spatial_norm_dim: Optional[int] = None,
        use_encoder_norm: bool = True,
        rngs: nnx.Rngs = None
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv_shortcut = self.in_channels != self.out_channels
        # Debug
        if self.use_conv_shortcut:
            print(f"DEBUG: FlaxResnetBlock3D creating conv_shortcut: {self.in_channels} -> {self.out_channels}")

        # Choose normalization type based on whether this is encoder or decoder
        if use_encoder_norm:
            # Encoder uses regular GroupNorm
            self.norm1 = FlaxGroupNorm(groups, in_channels, rngs=rngs)
            self.norm2 = FlaxGroupNorm(groups, self.out_channels, rngs=rngs)
        else:
            # Decoder uses SpatialNorm3D
            self.norm1 = FlaxCogVideoXSpatialNorm3D(
                f_channels=in_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
                rngs=rngs
            )
            self.norm2 = FlaxCogVideoXSpatialNorm3D(
                f_channels=self.out_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
                rngs=rngs
            )

        self.conv1 = FlaxConv3d(in_channels, self.out_channels, rngs=rngs)
        self.conv2 = FlaxConv3d(self.out_channels, self.out_channels, rngs=rngs)
        
        if self.use_conv_shortcut:
            self.conv_shortcut = FlaxConv3d(in_channels, self.out_channels, kernel_size=(1, 1, 1), padding=0, rngs=rngs)

    def __call__(self, x, temb=None, zq=None):
        h = x
        
        # First conv block
        if hasattr(self.norm1, 'conv_y'):  # SpatialNorm3D
            h = self.norm1(h, zq)
        else:  # Regular GroupNorm
            h = self.norm1(h)
        h = jax.nn.silu(h)
        h = self.conv1(h)

        # Add time embedding if provided
        if temb is not None:
            h = h + temb[:, :, None, None, None]

        # Second conv block
        if hasattr(self.norm2, 'conv_y'):  # SpatialNorm3D
            h = self.norm2(h, zq)
        else:  # Regular GroupNorm
            h = self.norm2(h)
        h = jax.nn.silu(h)
        h = self.conv2(h)

        # Skip connection
        if self.use_conv_shortcut:
            x = self.conv_shortcut(x)

        return x + h


class FlaxDownEncoderBlock3D(nnx.Module):
    def __init__(self, config, in_channels, out_channels, rngs):
        self.layers = []
        for i in range(config.layers_per_block):
            res_in = in_channels if i == 0 else out_channels
            # Debug: Print when we expect a shortcut
            if i == 0 and in_channels != out_channels:
                print(f"DEBUG: DownEncoderBlock3D expects conv_shortcut: in={in_channels}, out={out_channels}")
            layer = FlaxResnetBlock3D(
                res_in, 
                out_channels, 
                temb_channels=0, 
                groups=config.norm_num_groups,
                use_encoder_norm=True,  # Encoder uses regular norm
                rngs=rngs
            )
            self.layers.append(layer)
            setattr(self, f'layer_{i}', layer)
        
        # Add downsampler
        self.downsamplers = [
            FlaxConv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, rngs=rngs)
        ]
        self.downsampler_0 = self.downsamplers[0]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        for downsampler in self.downsamplers:
            x = downsampler(x)
        
        return x


class FlaxUpDecoderBlock3D(nnx.Module):
    def __init__(self, config, in_channels, out_channels, rngs, add_upsampler=True):
        self.layers = []
        
        # CogVideoX decoder uses layers_per_block + 1 resnets per block
        num_layers = config.layers_per_block + 1
        for i in range(num_layers):
            res_in = in_channels if i == 0 else out_channels
            layer = FlaxResnetBlock3D(
                res_in, 
                out_channels, 
                temb_channels=0, 
                groups=config.norm_num_groups,
                spatial_norm_dim=config.latent_channels,  # Use latent channels for spatial norm
                use_encoder_norm=False,  # Decoder uses spatial norm
                rngs=rngs
            )
            self.layers.append(layer)
            setattr(self, f'layer_{i}', layer)
        
        # Add upsampler only if requested
        self.add_upsampler = add_upsampler
        if add_upsampler:
            # CogVideoX uses 2D convolution for upsampling
            self.upsamplers = [
                FlaxConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, rngs=rngs)
            ]
            self.upsampler_0 = self.upsamplers[0]
        else:
            self.upsamplers = []

    def __call__(self, x, zq):
        for layer in self.layers:
            x = layer(x, zq=zq)
        
        # Upsample only if we have upsamplers
        if self.add_upsampler:
            for upsampler in self.upsamplers:
                # CogVideoX upsampling: reshape to 2D, upsample, apply conv2d, reshape back
                # Input shape: (B, T, H, W, C) - JAX format (NTHWC)
                B, T, H, W, C = x.shape
                
                # Reshape to combine batch and time: (B*T, H, W, C)
                x_2d = x.reshape(B * T, H, W, C)
                
                # Manual 2D upsampling (nearest neighbor)
                x_2d = x_2d.reshape(B * T, H, 1, W, 1, C)
                x_2d = jnp.tile(x_2d, (1, 1, 2, 1, 2, 1))
                x_2d = x_2d.reshape(B * T, H * 2, W * 2, C)
                
                # Apply 2D convolution
                x_2d = upsampler(x_2d)
                
                # Reshape back to 5D: (B, T, H*2, W*2, C)
                x = x_2d.reshape(B, T, H * 2, W * 2, C)
        
        return x


class FlaxEncoder(nnx.Module):
    def __init__(self, config, rngs):
        self.conv_in = FlaxConv3d(config.in_channels, config.block_out_channels[0], rngs=rngs)
        
        # Create down blocks
        self.down_blocks = []
        in_channels = config.block_out_channels[0]
        for i, out_channels in enumerate(config.block_out_channels):
            # Skip downsampler for last block
            has_downsample = i < len(config.block_out_channels) - 1
            
            if has_downsample:
                block = FlaxDownEncoderBlock3D(config, in_channels, out_channels, rngs)
                self.down_blocks.append(block)
                setattr(self, f'down_block_{i}', block)
            else:
                # Last block doesn't have downsampler
                class LastBlock(nnx.Module):
                    def __init__(self, config, in_channels, out_channels, rngs):
                        self.resnets = []
                        for j in range(config.layers_per_block):  # Use layers_per_block from config
                            res_in = in_channels if j == 0 else out_channels
                            resnet = FlaxResnetBlock3D(
                                res_in, 
                                out_channels, 
                                temb_channels=0, 
                                groups=config.norm_num_groups,
                                use_encoder_norm=True,
                                rngs=rngs
                            )
                            self.resnets.append(resnet)
                            setattr(self, f'resnet_{j}', resnet)
                
                block = LastBlock(config, in_channels, out_channels, rngs)
                self.down_blocks.append(block)
                setattr(self, f'down_block_{i}', block)
            
            in_channels = out_channels
        
        # Mid block
        self.mid_block_resnet_0 = FlaxResnetBlock3D(
            config.block_out_channels[-1], 
            config.block_out_channels[-1], 
            temb_channels=0, 
            groups=config.norm_num_groups,
            use_encoder_norm=True,
            rngs=rngs
        )
        self.mid_block_resnet_1 = FlaxResnetBlock3D(
            config.block_out_channels[-1], 
            config.block_out_channels[-1], 
            temb_channels=0, 
            groups=config.norm_num_groups,
            use_encoder_norm=True,
            rngs=rngs
        )
        
        # Output layers
        self.norm_out = FlaxGroupNorm(config.norm_num_groups, config.block_out_channels[-1], rngs=rngs)
        self.conv_out = FlaxConv3d(config.block_out_channels[-1], 2 * config.latent_channels, padding=0, rngs=rngs)

    def __call__(self, x):
        # Initial conv
        x = self.conv_in(x)
        
        # Down blocks
        for i, block in enumerate(self.down_blocks):
            if hasattr(block, 'layers'):
                x = block(x)
            else:
                # Last block
                for resnet in block.resnets:
                    x = resnet(x)
        
        # Mid block
        x = self.mid_block_resnet_0(x)
        x = self.mid_block_resnet_1(x)
        
        # Output
        x = self.norm_out(x)
        x = jax.nn.silu(x)
        x = self.conv_out(x)
        
        return x


class FlaxDecoder(nnx.Module):
    def __init__(self, config, rngs):
        # Store config for later use
        self.config = config
        # Input conv
        self.conv_in = FlaxConv3d(config.latent_channels, config.block_out_channels[-1], rngs=rngs)
        
        # Mid block with spatial norm
        self.mid_block_resnet_0 = FlaxResnetBlock3D(
            config.block_out_channels[-1], 
            config.block_out_channels[-1], 
            temb_channels=0, 
            groups=config.norm_num_groups,
            spatial_norm_dim=config.latent_channels,
            use_encoder_norm=False,
            rngs=rngs
        )
        self.mid_block_resnet_1 = FlaxResnetBlock3D(
            config.block_out_channels[-1], 
            config.block_out_channels[-1], 
            temb_channels=0, 
            groups=config.norm_num_groups,
            spatial_norm_dim=config.latent_channels,
            use_encoder_norm=False,
            rngs=rngs
        )
        
        # Up blocks - same number as block_out_channels
        self.up_blocks = []
        reversed_block_out_channels = list(reversed(config.block_out_channels))
        # Create same number of blocks as PyTorch version
        for i in range(len(reversed_block_out_channels)):
            if i == 0:
                in_channels = reversed_block_out_channels[0]
                out_channels = reversed_block_out_channels[0]
            else:
                in_channels = reversed_block_out_channels[i-1]
                out_channels = reversed_block_out_channels[i]
            
            # Last block shouldn't have upsampler
            is_last_block = (i == len(reversed_block_out_channels) - 1)
            add_upsampler = not is_last_block
            
            block = FlaxUpDecoderBlock3D(config, in_channels, out_channels, rngs, add_upsampler=add_upsampler)
            self.up_blocks.append(block)
            setattr(self, f'up_block_{i}', block)
        
        # Output layers with spatial norm
        self.norm_out = FlaxCogVideoXSpatialNorm3D(
            reversed_block_out_channels[-1], 
            config.latent_channels, 
            groups=config.norm_num_groups,
            rngs=rngs
        )
        self.conv_out = FlaxConv3d(reversed_block_out_channels[-1], config.out_channels, rngs=rngs)

    def _adjust_zq_resolution(self, zq, target_h, target_w):
        """Adjust zq resolution to match target dimensions."""
        _, _, h, w, _ = zq.shape
        if h == target_h and w == target_w:
            return zq
        
        # Calculate downsampling factors
        h_factor = h // target_h
        w_factor = w // target_w
        
        if h_factor > 1 or w_factor > 1:
            # Downsample using average pooling
            import jax.lax
            zq = jax.lax.reduce_window(
                zq, 
                init_value=0.0,
                computation=jax.lax.add,
                window_dimensions=(1, 1, h_factor, w_factor, 1),
                window_strides=(1, 1, h_factor, w_factor, 1),
                padding='VALID'
            )
            zq = zq / (h_factor * w_factor)
        elif h < target_h or w < target_w:
            # Upsample using nearest neighbor
            # This happens in decoder when moving through up_blocks
            h_factor = target_h // h
            w_factor = target_w // w
            
            # Reshape for upsampling: add new dimensions for repeating
            b, t, h, w, c = zq.shape
            zq = zq.reshape(b, t, h, 1, w, 1, c)
            # Repeat along spatial dimensions
            zq = jnp.tile(zq, (1, 1, 1, h_factor, 1, w_factor, 1))
            # Reshape back
            zq = zq.reshape(b, t, h * h_factor, w * w_factor, c)
        
        return zq
    
    def __call__(self, z, zq):
        # Initial conv
        x = self.conv_in(z)
        
        # Mid block - adjust zq to match x's resolution
        _, _, h, w, _ = x.shape
        zq_mid = self._adjust_zq_resolution(zq, h, w)
        x = self.mid_block_resnet_0(x, zq=zq_mid)
        x = self.mid_block_resnet_1(x, zq=zq_mid)
        
        # Up blocks - need to adjust zq for each resolution
        for i, block in enumerate(self.up_blocks):
            # Get current resolution
            _, _, h, w, c = x.shape
            print(f"DEBUG: Up block {i} input shape: {x.shape}")
            zq_block = self._adjust_zq_resolution(zq, h, w)
            x = block(x, zq=zq_block)
            print(f"DEBUG: Up block {i} output shape: {x.shape}")
        
        # Output - x should now be at original resolution
        _, _, h, w, _ = x.shape
        zq_out = self._adjust_zq_resolution(zq, h, w)
        x = self.norm_out(x, zq_out)
        x = jax.nn.silu(x)
        x = self.conv_out(x)
        
        return x


class FlaxAutoencoderKLCogVideoX(nnx.Module, FlaxModelMixin):
    config_class = FlaxAutoencoderKLCogVideoXConfig

    def __init__(self, config: FlaxAutoencoderKLCogVideoXConfig, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.config = config
        self.encoder = FlaxEncoder(config, rngs)
        self.decoder = FlaxDecoder(config, rngs)
        # CogVideoX doesn't use quant_conv and post_quant_conv by default
        # self.quant_conv = FlaxConv3d(2 * config.latent_channels, 2 * config.latent_channels, (1, 1, 1), rngs=rngs)
        # self.post_quant_conv = FlaxConv3d(config.latent_channels, config.latent_channels, (1, 1, 1), rngs=rngs)
        self.dtype = dtype

    def encode(self, x):
        h = self.encoder(x)
        # CogVideoX doesn't use quant_conv
        # moments = self.quant_conv(h)
        # Instead, directly split the encoder output
        mean, logvar = jnp.split(h, 2, axis=1)
        return mean, logvar

    def decode(self, z, sample=None):
        """Decode latent to video.
        
        Args:
            z: Latent representation
            sample: Original input sample for spatial conditioning
        """
        # CogVideoX doesn't use post_quant_conv
        # z = self.post_quant_conv(z)
        # Use the original sample as spatial conditioning
        zq = sample if sample is not None else z
        return self.decoder(z, zq)

    def __call__(self, x):
        mean, logvar = self.encode(x)
        z = mean  # No sampling in inference
        decoded = self.decode(z, sample=x)
        return decoded


class FlaxAutoencoderKLCogVideoXCache(nnx.Cache):
    def __init__(self, model: FlaxAutoencoderKLCogVideoX):
        super().__init__(model)


def load_cogvideox_vae(
    pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True
):
    """Fixed version of load_cogvideox_vae that handles the correct model structure"""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from flax.traverse_util import unflatten_dict

    device_obj = jax.local_devices(backend=device)[0]
    with jax.default_device(device_obj):
        if hf_download:
            ckpt_path = hf_hub_download(
                pretrained_model_name_or_path, subfolder="vae", filename="diffusion_pytorch_model.safetensors"
            )
        print(f"Load and port CogVideoX VAE on {device}")

        if ckpt_path is not None:
            tensors = {}
            with safe_open(ckpt_path, framework="np") as f:
                for k in f.keys():
                    tensors[k] = jnp.array(f.get_tensor(k))

                ## Debug:
                #print("Keys in checkpoint file:")
                #for key in sorted(tensors.keys()):
                #    print(f"  {key}")
                #print(f"Total keys: {len(tensors)}")

            flax_state_dict = {}

            for pt_key, tensor in tensors.items():
                # Skip '_orig_mod' prefix if present
                if pt_key.startswith("_orig_mod."):
                    pt_key = pt_key[len("_orig_mod."):]
                
                # Convert PyTorch key to Flax key
                flax_key = pt_key
                
                # Map block structure
                # Map down_blocks.X.Y to down_block_X
                import re
                
                # Encoder down blocks
                if m := re.match(r'encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.*)', flax_key):
                    block_idx, resnet_idx, rest = m.groups()
                    if int(block_idx) < 3:  # First 3 blocks have downsamplers
                        flax_key = f'encoder.down_block_{block_idx}.layer_{resnet_idx}.{rest}'
                    else:  # Last block
                        flax_key = f'encoder.down_block_{block_idx}.resnet_{resnet_idx}.{rest}'
                        
                elif m := re.match(r'encoder\.down_blocks\.(\d+)\.downsamplers\.0\.(.*)', flax_key):
                    block_idx, rest = m.groups()
                    flax_key = f'encoder.down_block_{block_idx}.downsampler_0.{rest}'
                
                # Encoder mid block
                elif m := re.match(r'encoder\.mid_block\.resnets\.(\d+)\.(.*)', flax_key):
                    resnet_idx, rest = m.groups()
                    flax_key = f'encoder.mid_block_resnet_{resnet_idx}.{rest}'
                
                # Decoder mid block
                elif m := re.match(r'decoder\.mid_block\.resnets\.(\d+)\.(.*)', flax_key):
                    resnet_idx, rest = m.groups()
                    flax_key = f'decoder.mid_block_resnet_{resnet_idx}.{rest}'
                
                # Decoder up blocks
                elif m := re.match(r'decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.*)', flax_key):
                    block_idx, resnet_idx, rest = m.groups()
                    # Direct mapping now that we have 4 blocks
                    flax_key = f'decoder.up_block_{block_idx}.layer_{resnet_idx}.{rest}'
                    
                elif m := re.match(r'decoder\.up_blocks\.(\d+)\.upsamplers\.0\.(.*)', flax_key):
                    block_idx, rest = m.groups()
                    # Direct mapping for upsamplers
                    flax_key = f'decoder.up_block_{block_idx}.upsampler_0.{rest}'
                
                # Our FlaxConv3d wraps the conv in a .conv attribute
                # So we need to add .conv to the path for conv layers
                # Do this BEFORE replacing weight->kernel
                # But check if .conv.weight or .conv.bias is already in the path
                needs_conv = False
                if any(pattern in flax_key for pattern in ['.conv_in.', '.conv_out.', '.conv1.', '.conv2.', 
                                                           '.conv_shortcut.', '.conv_y.', '.conv_b.', 
                                                           '.downsampler_', '.upsampler_']):
                    # Check if it already ends with .conv.weight or .conv.bias
                    if not (flax_key.endswith('.conv.weight') or flax_key.endswith('.conv.bias')):
                        if flax_key.endswith('.weight') or flax_key.endswith('.bias'):
                            needs_conv = True
                
                if needs_conv:
                    # Insert .conv before .weight or .bias
                    parts = flax_key.rsplit('.', 1)
                    flax_key = f"{parts[0]}.conv.{parts[1]}"
                
                # Handle conv weights - PyTorch uses (out, in, t, h, w), Flax uses (t, h, w, in, out)
                if "conv" in flax_key and "weight" in flax_key:
                    flax_key = flax_key.replace(".weight", ".kernel")
                    # Debug upsampler weights
                    if "upsampler" in flax_key:
                        print(f"DEBUG: Upsampler weight shape: {flax_key} -> {tensor.shape}")
                    if len(tensor.shape) == 5:  # 3D conv
                        tensor = tensor.transpose(2, 3, 4, 1, 0)
                    elif len(tensor.shape) == 4:  # 2D conv (out, in, h, w) -> (h, w, in, out)
                        tensor = tensor.transpose(2, 3, 1, 0)
                    elif len(tensor.shape) == 3:  # 1D conv
                        tensor = tensor.transpose(1, 2, 0)
                
                # Handle norm layers
                if "norm_layer.weight" in flax_key:
                    flax_key = flax_key.replace("norm_layer.weight", "norm_layer.scale")
                    
                # Handle GroupNorm in encoder - now they map directly to scale and bias
                if "encoder" in flax_key and ".norm" in flax_key:
                    if ".weight" in flax_key:
                        # encoder.down_blocks.0.resnets.0.norm1.weight -> encoder.down_block_0.layer_0.norm1.scale
                        flax_key = flax_key.replace(".weight", ".scale")
                    elif ".bias" in flax_key:
                        # encoder.down_blocks.0.resnets.0.norm1.bias -> encoder.down_block_0.layer_0.norm1.bias
                        # No change needed for bias
                        pass
                
                flax_state_dict[flax_key] = tensor

            # Convert to nested dict
            flax_state_dict = unflatten_dict(flax_state_dict, sep=".")
            
            return flax_state_dict
        else:
            return {}