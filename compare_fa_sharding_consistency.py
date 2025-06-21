import pickle
import torch
import torchax
import jax
import numpy as np
import re

from jax.sharding import NamedSharding, PartitionSpec as P
from diffusers.models import WanTransformer3DModel
from diffusers import WanPipeline  # 用来方便地获取初始模型和配置

axis = 'axis'

# 复用sharding配置
transformer_shardings = {
    r'condition_embedder.time_embedder.linear_1.weight': (axis, None),
    r'condition_embedder.time_embedder.linear_1.bias': (axis,),
    r'condition_embedder.time_embedder.linear_2.weight': (None, axis),
    r'condition_embedder.text_embedder.linear_1.weight': (axis, None),
    r'condition_embedder.text_embedder.linear_1.bias': (axis, ),
    r'condition_embedder.text_embedder.linear_2.weight': (None, axis),
    r'blocks.\d+.attn1.to_q.weight': (axis, None),
    r'blocks.\d+.attn1.to_q.bias': (axis, ),
    r'blocks.\d+.attn1.to_k.weight': (axis, ),
    r'blocks.\d+.attn1.to_k.bias': (axis, ),
    r'blocks.\d+.attn1.to_v.weight': (axis, ),
    r'blocks.\d+.attn1.to_v.bias': (axis, ),
    r'blocks.\d+.attn1.to_out.0.weight': (None, axis),
    r'blocks.\d+.attn2.to_q.weight': (axis, ),
    r'blocks.\d+.attn2.to_q.bias': (axis, ),
    r'blocks.\d+.attn2.to_k.weight': (axis, ),
    r'blocks.\d+.attn2.to_k.bias': (axis, ),
    r'blocks.\d+.attn2.to_v.weight': (axis, ),
    r'blocks.\d+.attn2.to_v.bias': (axis, ),
    r'blocks.\d+.attn2.to_out.0.weight': (None, axis),
    r'blocks.\d+.ffn.net.0.proj.weight': (axis,),
    r'blocks.\d+.ffn.net.0.proj.bias': (axis, ),
    r'blocks.\d+.ffn.net.2.weight': (None, axis),
}

def _shard_weight_dict(weight_dict, sharding_dict, mesh):
    print(f"Sharding {len(weight_dict)} parameters...")
    result = {}
    for k, v in weight_dict.items():
        sharded = False
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                print(f"  - Applying sharding rule '{target}' to '{k}'")
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                sharded = True
                break
        if not sharded:
            # Replicate if no rule was applied
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        result[k] = v
    print(f"Sharding completed for {len(result)} parameters")
    return result

def setup_transformer(state_path, enable_flash_attention):
    """
    设置并编译一个WanTransformer3DModel实例。
    """
    print(f"\n--- Setting up Transformer (Flash Attention: {enable_flash_attention}) ---")

    # 步骤 1: 在干净的PyTorch环境中从Hub加载模型结构
    print("Loading initial model structure from Hub (torchax disabled)...")
    try:
        torchax.disable_globally()
    except Exception:
        pass

    model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    transformer = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).transformer
    print("Initial model loaded.")

    # 步骤 2: 现在启用并配置torchax
    torchax.enable_globally()
    env = torchax.default_env()
    
    env.config.use_tpu_flash_attention = enable_flash_attention
    env.config.shmap_flash_attention = enable_flash_attention
    print(f"torchax.config.use_tpu_flash_attention = {env.config.use_tpu_flash_attention}")
    
    mesh = jax.make_mesh((len(jax.devices()), ), (axis, ))
    env.default_device_or_sharding = NamedSharding(mesh, P())

    # 关键修复：torchax的TPU flash attention实现需要显式地设置mesh
    if enable_flash_attention:
        env._mesh = mesh

    # 步骤 3: 使用pickle文件中的精确状态覆盖权重
    print(f"Loading transformer state from pickle file: {state_path}...")
    with open(state_path, 'rb') as f:
        state = pickle.load(f)
    
    # 修正键名：移除'_model.'前缀
    pickle_state_dict = state['transformer_params']
    corrected_state_dict = {k.removeprefix('_model.'): v for k, v in pickle_state_dict.items()}

    transformer.load_state_dict(corrected_state_dict)
    print("Transformer state loaded from pickle (with corrected keys).")
    
    # 步骤 4: 将模型的张量移动到JAX设备
    def _move_module_to_jax(module):
        with jax.default_device('cpu'):
            state_dict = module.state_dict()
            state_dict = env.to_xla(state_dict)
            module.load_state_dict(state_dict, assign=True)
    
    print("Moving transformer module to JAX devices...")
    _move_module_to_jax(transformer)
    transformer.rope.freqs = transformer.rope.freqs.to('jax')
    print("Module moved.")

    # 步骤 5: 编译JAX化的模块
    print("Compiling transformer...")
    options = torchax.CompileOptions(jax_jit_kwargs={'static_argnames': ('return_dict',)})
    compiled_transformer = torchax.compile(transformer, options)
    print("Transformer compiled.")

    # 步骤 6: 对编译后模型的参数和缓冲区进行分片
    print("Sharding transformer parameters and buffers...")
    compiled_transformer.params = _shard_weight_dict(compiled_transformer.params, transformer_shardings, mesh)
    compiled_transformer.buffers = _shard_weight_dict(compiled_transformer.buffers, {}, mesh) # Buffers are usually replicated
    print("Parameters and buffers sharded.")
    
    return compiled_transformer, mesh

def compare_tensors(tensor1, tensor2, tolerance=1e-5):
    """比较两个tensor"""
    if tensor1.shape != tensor2.shape:
        print(f"❌ 形状不匹配: {tensor1.shape} vs {tensor2.shape}")
        return False
    
    diff = torch.abs(tensor1 - tensor2)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    
    print(f"  - 最大差异: {max_diff}")
    print(f"  - 平均差异: {mean_diff}")
    
    if max_diff > tolerance:
        print(f"❌ 差异超过容差 ({tolerance})")
        return False
    else:
        print(f"✅ 差异在容差内")
        return True

def main():
    torch.set_default_dtype(torch.bfloat16)
    
    # --- 1. 设置确定性输入 ---
    print("--- 1. Preparing inputs ---")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建一个固定的 encoder_hidden_states (模拟T5 embeds)
    # 不再需要从外部文件加载，使脚本自包含
    encoder_hidden_states_shape = (1, 512, 4096)
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape, dtype=torch.bfloat16)
    print(f"Created dummy encoder_hidden_states with shape: {encoder_hidden_states.shape}")

    # 创建一个固定的latents输入
    latents_shape = (1, 16, 16, 96, 128)
    latents = torch.randn(latents_shape, dtype=torch.bfloat16)
    print(f"Created latents with shape: {latents.shape}")

    # 创建一个固定的timestep
    timestep = torch.tensor([999], dtype=torch.int64)
    print(f"Using timestep: {timestep.item()}")

    state_path = "debug_state_before_inference_0.pkl"

    # --- 2. 设置和运行Normal Transformer (基准) ---
    transformer_normal, mesh_normal = setup_transformer(state_path, enable_flash_attention=False)
    
    # 将输入移动到JAX设备
    latents_jax_normal = latents.to('jax')
    encoder_hidden_states_jax_normal = encoder_hidden_states.to('jax')
    timestep_jax_normal = timestep.to('jax')

    with mesh_normal:
        print("Running forward pass on Normal Transformer...")
        output_normal = transformer_normal(
            hidden_states=latents_jax_normal,
            timestep=timestep_jax_normal,
            encoder_hidden_states=encoder_hidden_states_jax_normal,
            return_dict=False
        )[0]
    output_normal_cpu = output_normal.cpu()
    print("Normal forward pass complete.")
    
    # 清理内存
    del transformer_normal, mesh_normal, latents_jax_normal, encoder_hidden_states_jax_normal, timestep_jax_normal
    import gc
    gc.collect()

    # --- 3. 设置和运行Flash Attention Transformer (实验) ---
    transformer_flash, mesh_flash = setup_transformer(state_path, enable_flash_attention=True)

    # 将同样的输入移动到JAX设备
    latents_jax_flash = latents.to('jax')
    encoder_hidden_states_jax_flash = encoder_hidden_states.to('jax')
    timestep_jax_flash = timestep.to('jax')

    with mesh_flash:
        print("Running forward pass on Flash Attention Transformer...")
        output_flash = transformer_flash(
            hidden_states=latents_jax_flash,
            timestep=timestep_jax_flash,
            encoder_hidden_states=encoder_hidden_states_jax_flash,
            return_dict=False
        )[0]
    output_flash_cpu = output_flash.cpu()
    print("Flash Attention forward pass complete.")

    # --- 4. 对比结果 ---
    print("\n--- 4. Comparing outputs ---")
    result = compare_tensors(output_normal_cpu, output_flash_cpu)

    print("\n" + "="*30)
    print("      FINAL CONCLUSION")
    print("="*30)
    if result:
        print("✅✅✅ [PASSED] The outputs are identical.")
    else:
        print("❌❌❌ [FAILED] The outputs are different.")
        print("This provides strong evidence that the problem IS inside the WanTransformer3DModel when Flash Attention is used.")
        print("The root cause is likely the self-attention calculation itself being altered by torchax's Flash Attention implementation, when used with this specific tensor sharding strategy.")
    print("="*30)


if __name__ == '__main__':
    main() 