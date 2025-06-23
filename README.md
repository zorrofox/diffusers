Original readme moved to README_original.md

# install
```
pip install -e .
pip install transformers accelerate

# install torchax
pip install git+https://github.com/pytorch/xla.git@hanq_wan_changes#subdirectory=torchax
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jax[tpu]
```

To run:

```
python wan_tx.py
```

# Progress:

(Jun 17)
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Device ┃ Memory usage         ┃ Duty cycle ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0      │ 2.09 GiB / 31.25 GiB │      0.00% │
│ 1      │ 2.08 GiB / 31.25 GiB │      0.00% │
│ 2      │ 2.08 GiB / 31.25 GiB │      0.00% │
│ 3      │ 2.08 GiB / 31.25 GiB │      0.00% │
│ 4      │ 2.08 GiB / 31.25 GiB │      0.00% │
│ 5      │ 2.08 GiB / 31.25 GiB │      0.00% │
│ 6      │ 2.08 GiB / 31.25 GiB │      0.00% │
│ 7      │ 2.08 GiB / 31.25 GiB │      0.00% │


## sizes:
### wan 1.3B:
text_encoder 12.537574768066406 G
transformer 2.64891254901886 G
vae 0.23635575734078884 G

### wan 14B
text_encoder 12.537574768066406 G
transformer 26.66874897480011 G
vae 0.23635575734078884 G

## Shapes of weights for 1.3B model:
### vae
```
encoder.conv_in.weight : # (torch.Size([96, 3, 3, 3, 3]), torch.bfloat16)
encoder.conv_in.bias : # (torch.Size([96]), torch.bfloat16)
encoder.down_blocks.*.norm*.gamma : # (torch.Size([384, 1, 1, 1]), torch.bfloat16)
encoder.down_blocks.*.conv*.weight : # (torch.Size([384, 384, 3, 3, 3]), torch.bfloat16)
encoder.down_blocks.*.conv*.bias : # (torch.Size([384]), torch.bfloat16)
encoder.down_blocks.*.resample.*.weight : # (torch.Size([384, 384, 3, 3]), torch.bfloat16)
encoder.down_blocks.*.resample.*.bias : # (torch.Size([384]), torch.bfloat16)
encoder.down_blocks.*.conv_shortcut.weight : # (torch.Size([384, 192, 1, 1, 1]), torch.bfloat16)
encoder.down_blocks.*.conv_shortcut.bias : # (torch.Size([384]), torch.bfloat16)
encoder.down_blocks.*.time_conv.weight : # (torch.Size([384, 384, 3, 1, 1]), torch.bfloat16)
encoder.down_blocks.*.time_conv.bias : # (torch.Size([384]), torch.bfloat16)
encoder.mid_block.attentions.*.norm.gamma : # (torch.Size([384, 1, 1]), torch.bfloat16)
encoder.mid_block.attentions.*.to_qkv.weight : # (torch.Size([1152, 384, 1, 1]), torch.bfloat16)
encoder.mid_block.attentions.*.to_qkv.bias : # (torch.Size([1152]), torch.bfloat16)
encoder.mid_block.attentions.*.proj.weight : # (torch.Size([384, 384, 1, 1]), torch.bfloat16)
encoder.mid_block.attentions.*.proj.bias : # (torch.Size([384]), torch.bfloat16)
encoder.mid_block.resnets.*.norm*.gamma : # (torch.Size([384, 1, 1, 1]), torch.bfloat16)
encoder.mid_block.resnets.*.conv*.weight : # (torch.Size([384, 384, 3, 3, 3]), torch.bfloat16)
encoder.mid_block.resnets.*.conv*.bias : # (torch.Size([384]), torch.bfloat16)
encoder.norm_out.gamma : # (torch.Size([384, 1, 1, 1]), torch.bfloat16)
encoder.conv_out.weight : # (torch.Size([32, 384, 3, 3, 3]), torch.bfloat16)
encoder.conv_out.bias : # (torch.Size([32]), torch.bfloat16)
quant_conv.weight : # (torch.Size([32, 32, 1, 1, 1]), torch.bfloat16)
quant_conv.bias : # (torch.Size([32]), torch.bfloat16)
post_quant_conv.weight : # (torch.Size([16, 16, 1, 1, 1]), torch.bfloat16)
post_quant_conv.bias : # (torch.Size([16]), torch.bfloat16)
decoder.conv_in.weight : # (torch.Size([384, 16, 3, 3, 3]), torch.bfloat16)
decoder.conv_in.bias : # (torch.Size([384]), torch.bfloat16)
decoder.mid_block.attentions.*.norm.gamma : # (torch.Size([384, 1, 1]), torch.bfloat16)
decoder.mid_block.attentions.*.to_qkv.weight : # (torch.Size([1152, 384, 1, 1]), torch.bfloat16)
decoder.mid_block.attentions.*.to_qkv.bias : # (torch.Size([1152]), torch.bfloat16)
decoder.mid_block.attentions.*.proj.weight : # (torch.Size([384, 384, 1, 1]), torch.bfloat16)
decoder.mid_block.attentions.*.proj.bias : # (torch.Size([384]), torch.bfloat16)
decoder.mid_block.resnets.*.norm*.gamma : # (torch.Size([384, 1, 1, 1]), torch.bfloat16)
decoder.mid_block.resnets.*.conv*.weight : # (torch.Size([384, 384, 3, 3, 3]), torch.bfloat16)
decoder.mid_block.resnets.*.conv*.bias : # (torch.Size([384]), torch.bfloat16)
decoder.up_blocks.*.resnets.*.norm*.gamma : # (torch.Size([96, 1, 1, 1]), torch.bfloat16)
decoder.up_blocks.*.resnets.*.conv*.weight : # (torch.Size([96, 96, 3, 3, 3]), torch.bfloat16)
decoder.up_blocks.*.resnets.*.conv*.bias : # (torch.Size([96]), torch.bfloat16)
decoder.up_blocks.*.upsamplers.*.resample.*.weight : # (torch.Size([96, 192, 3, 3]), torch.bfloat16)
decoder.up_blocks.*.upsamplers.*.resample.*.bias : # (torch.Size([96]), torch.bfloat16)
decoder.up_blocks.*.upsamplers.*.time_conv.weight : # (torch.Size([768, 384, 3, 1, 1]), torch.bfloat16)
decoder.up_blocks.*.upsamplers.*.time_conv.bias : # (torch.Size([768]), torch.bfloat16)
decoder.up_blocks.*.resnets.*.conv_shortcut.weight : # (torch.Size([384, 192, 1, 1, 1]), torch.bfloat16)
decoder.up_blocks.*.resnets.*.conv_shortcut.bias : # (torch.Size([384]), torch.bfloat16)
decoder.norm_out.gamma : # (torch.Size([96, 1, 1, 1]), torch.bfloat16)
decoder.conv_out.weight : # (torch.Size([3, 96, 3, 3, 3]), torch.bfloat16)
decoder.conv_out.bias : # (torch.Size([3]), torch.bfloat16)
```

### transformer
```
scale_shift_table : # (torch.Size([1, 2, 1536]), torch.float32)
patch_embedding.weight : # (torch.Size([1536, 16, 1, 2, 2]), torch.bfloat16)
patch_embedding.bias : # (torch.Size([1536]), torch.bfloat16)
condition_embedder.time_embedder.linear_*.weight : # (torch.Size([1536, 1536]), torch.float32)
condition_embedder.time_embedder.linear_*.bias : # (torch.Size([1536]), torch.float32)
condition_embedder.time_proj.weight : # (torch.Size([9216, 1536]), torch.bfloat16)
condition_embedder.time_proj.bias : # (torch.Size([9216]), torch.bfloat16)
condition_embedder.text_embedder.linear_*.weight : # (torch.Size([1536, 1536]), torch.bfloat16)
condition_embedder.text_embedder.linear_*.bias : # (torch.Size([1536]), torch.bfloat16)
blocks.*.scale_shift_table : # (torch.Size([1, 6, 1536]), torch.float32)
blocks.*.attn*.norm_q.weight : # (torch.Size([1536]), torch.bfloat16)
blocks.*.attn*.norm_k.weight : # (torch.Size([1536]), torch.bfloat16)
blocks.*.attn*.to_q.weight : # (torch.Size([1536, 1536]), torch.bfloat16)
blocks.*.attn*.to_q.bias : # (torch.Size([1536]), torch.bfloat16)
blocks.*.attn*.to_k.weight : # (torch.Size([1536, 1536]), torch.bfloat16)
blocks.*.attn*.to_k.bias : # (torch.Size([1536]), torch.bfloat16)
blocks.*.attn*.to_v.weight : # (torch.Size([1536, 1536]), torch.bfloat16)
blocks.*.attn*.to_v.bias : # (torch.Size([1536]), torch.bfloat16)
blocks.*.attn*.to_out.*.weight : # (torch.Size([1536, 1536]), torch.bfloat16)
blocks.*.attn*.to_out.*.bias : # (torch.Size([1536]), torch.bfloat16)
blocks.*.norm*.weight : # (torch.Size([1536]), torch.float32)
blocks.*.norm*.bias : # (torch.Size([1536]), torch.float32)
blocks.*.ffn.net.*.proj.weight : # (torch.Size([8960, 1536]), torch.bfloat16)
blocks.*.ffn.net.*.proj.bias : # (torch.Size([8960]), torch.bfloat16)
blocks.*.ffn.net.*.weight : # (torch.Size([1536, 8960]), torch.bfloat16)
blocks.*.ffn.net.*.bias : # (torch.Size([1536]), torch.bfloat16)
proj_out.weight : # (torch.Size([64, 1536]), torch.bfloat16)
proj_out.bias : # (torch.Size([64]), torch.bfloat16)
```

### text encoder
```
shared.weight : # (torch.Size([256384, 4096]), torch.bfloat16)
encoder.block.*.layer.*.SelfAttention.q.weight : # (torch.Size([4096, 4096]), torch.bfloat16)
encoder.block.*.layer.*.SelfAttention.k.weight : # (torch.Size([4096, 4096]), torch.bfloat16)
encoder.block.*.layer.*.SelfAttention.v.weight : # (torch.Size([4096, 4096]), torch.bfloat16)
encoder.block.*.layer.*.SelfAttention.o.weight : # (torch.Size([4096, 4096]), torch.bfloat16)
encoder.block.*.layer.*.SelfAttention.relative_attention_bias.weight : # (torch.Size([32, 64]), torch.bfloat16)
encoder.block.*.layer.*.layer_norm.weight : # (torch.Size([4096]), torch.bfloat16)
encoder.block.*.layer.*.DenseReluDense.wi_*.weight : # (torch.Size([10240, 4096]), torch.bfloat16)
encoder.block.*.layer.*.DenseReluDense.wo.weight : # (torch.Size([4096, 10240]), torch.bfloat16)
encoder.final_layer_norm.weight : # (torch.Size([4096]), torch.bfloat16)
```

### adding flash attention inconsistant issue unit test

```
python compare_fa_sharding_consistency.py
...
==============================
      FINAL CONCLUSION
==============================
❌❌❌ [FAILED] The outputs are different.
This provides strong evidence that the problem IS inside the WanTransformer3DModel when Flash Attention is used.
The root cause is likely the self-attention calculation itself being altered by torchax's Flash Attention implementation, when used with this specific tensor sharding strategy.
==============================

```

### support flash attention

Patch the torachax/ops/jtorch.py into the pytorch/xla/torachax repo.

Current support flash attention to generate correct normal 14B model, 81 frames videos.
Flash attention prevent the huge attention weight which cause OOM.

1.3B model is not yet ready using flash attention since kv_head = 12 cannot divide by 8 tpus.
Disable flash attention for VAE for now since kv_head = 1 in VAE.

Iteration 0: 952.361927s
Iteration 1: 675.130811s
