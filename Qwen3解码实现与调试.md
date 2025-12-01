# Qwen3解码实现与调试

## 1. Qwen3解码实现与调试

**system**: 

```
You are a helpful assistant.
```

**user**: 

```
python_vm.py包含使用指令融合进行decode计算的指令和计算方式，scheduler.py中包含融合指令的全局内存管理和指令调度，qwen3.py是decode的torch实现，qwen3.py的计算结果是正确的，而python_vm.py的计算结果是错误的，现在我们需要通过比较这两个文件在python_vm.py中每个指令算子的计算结果判断python_vm.py在哪个部分出错，请在python_vm.py和qwen3.py中分别将关键结算结果存储在json格式文件中
```

Attachments:
- python_vm.py
- qwen3.py
- scheduler.py

**assistant**: 

```
这里是修改后的代码。我们需要通过在关键计算步骤（Pre-Norm, Projection, Norm, RoPE, Attention）插入数据记录代码，将中间结果的统计信息（均值、求和、前几个数值）保存到 JSON 文件中，以便对比查找错误。

主要修改思路：
1.  **引入 `log_debug_data` 函数**：用于将 Tensor 的统计信息追加写入 JSON 文件。
2.  **在 `python_vm.py` 中**：重点监测 `layer_norm_matvec_rope_append` 函数，因为这里包含了 QKV Projection、Q/K Norm（嫌疑最大的部分）和 RoPE。同时监测 `partial_attention` 的输出。
3.  **在 `qwen3.py` 中**：在 `Qwen3Attention` 的 `forward` 函数中，记录对应的中间 Tensor 状态。

请使用以下两个文件的内容替换原文件。

### 文件 1: `python_vm.py` (添加了调试记录)


import math
import json
import os

import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor

from megakernels.demos.qwen3_0_6b.instructions import (
    AttentionReduction,
    DownProjResidual,
    Globals,
    LayerNorm_QKV_MatVecRopeAppend,
    LayerNormDoubleMatVecSiLU,
    O_ProjResidual,
    PartialAttention,
    RMS_LM_Head,
)

from megakernels.qwen3 import (
    apply_rotary_pos_emb_interleaved,
)

# ================= DEBUG UTILS =================
DEBUG_FILE = "python_vm_debug.json"
if os.path.exists(DEBUG_FILE):
    os.remove(DEBUG_FILE)

def log_debug_data(tag, tensor):
    """记录 Tensor 的关键统计信息到 JSON 文件"""
    if tensor is None:
        return
    
    # 为了避免数据量过大，我们只记录：形状、均值、和、以及前 16 个值
    t_flat = tensor.detach().float().flatten().cpu()
    data = {
        "tag": tag,
        "shape": list(tensor.shape),
        "mean": float(t_flat.mean().item()),
        "sum": float(t_flat.sum().item()),
        "head_values": t_flat[:16].tolist()
    }
    
    with open(DEBUG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
# ===============================================

def get_start_end(block_size: int, block_idx: int):
    start = block_size * block_idx
    end = start + block_size
    return start, end


def matvec(
    mat: Tensor,
    vec: Tensor,
    block_size: int,
    block_idx: int,
    reduce: bool = False,
    reduction_size: int = 0,
    reduction_idx: int = 0,
):
    start, end = get_start_end(block_size, block_idx)
    if reduce:
        red_start, red_end = get_start_end(reduction_size, reduction_idx)
        mat = mat[start:end, red_start:red_end]
        vec = vec[red_start:red_end]
    else:
        mat = mat[start:end]

    out = einsum(mat, vec, "o i, i -> o")
    return out, start, end


def rms_norm(inp: Tensor, weight: Tensor, eps: float):
    input_dtype = inp.dtype
    inp = inp.to(torch.float32)
    variance = inp.pow(2).mean(-1, keepdim=True)
    inp = inp * torch.rsqrt(variance + eps)

    return weight * inp.to(input_dtype)


def matvec_with_residual(
    mat: Tensor,
    vec: Tensor,
    residual: Tensor,
    block_size: int,
    start_block_idx: int,
    end_block_idx: int,
    reduction_size: int,
    reduction_block_idx: int,
):
    for block_idx in range(start_block_idx, end_block_idx):
        matvec_out, start, end = matvec(
            mat=mat,
            vec=vec,
            block_size=block_size,
            block_idx=block_idx,
            reduce=True,
            reduction_size=reduction_size,
            reduction_idx=reduction_block_idx,
        )

        residual[start:end] += matvec_out.to(residual.dtype)


def o_proj_residual(globals: Globals, instruction: O_ProjResidual):
    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == globals.num_attention_heads

    assert instruction.start_block_idx == instruction.end_block_idx - 1
    assert instruction.reduction_block_idx == 0

    # DEBUG: Log input to O-Proj (Attention Output)
    if instruction.layer_idx == 0 and instruction.start_block_idx == 0:
         log_debug_data(f"layer_{instruction.layer_idx}_attn_out_buffer", globals.attn_out)

    matvec_with_residual(
        mat=globals.o_proj_weights[instruction.layer_idx],
        vec=globals.attn_out,
        residual=globals.hidden_states,
        block_size=globals.o_proj_block_size,
        start_block_idx=instruction.start_block_idx,
        end_block_idx=instruction.end_block_idx,
        reduction_size=globals.matvec_reduction_size,
        reduction_block_idx=instruction.reduction_block_idx,
    )

    # Barrier update
    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += instruction.end_block_idx - instruction.start_block_idx
    
    # DEBUG: Log final output of O-Proj (Partial Residual)
    if instruction.layer_idx == 0 and instruction.start_block_idx == 0:
        log_debug_data(f"layer_{instruction.layer_idx}_hidden_states_post_oproj_partial", globals.hidden_states[0:16])


def down_proj_residual(globals: Globals, instruction: DownProjResidual):
    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == 192  # 8192 / 16

    matvec_with_residual(
        mat=globals.down_proj_weights[instruction.layer_idx],
        vec=globals.silu_out,
        residual=globals.hidden_states,
        block_size=globals.down_proj_block_size,
        start_block_idx=instruction.start_block_idx,
        end_block_idx=instruction.end_block_idx,
        reduction_size=globals.matvec_reduction_size,
        reduction_block_idx=instruction.reduction_block_idx,
    )

    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += instruction.end_block_idx - instruction.start_block_idx


def layer_norm_double_matvec_silu(
    globals: Globals, instruction: LayerNormDoubleMatVecSiLU
):
    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == 64

    post_ln = rms_norm(
        inp=globals.hidden_states,
        weight=globals.mlp_ln_weights[instruction.layer_idx],
        eps=globals.rms_norm_eps,
    )
    
    # DEBUG
    if instruction.layer_idx == 0:
         log_debug_data(f"layer_{instruction.layer_idx}_mlp_post_ln", post_ln)

    block_size = globals.up_gate_proj_block_size

    barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]

    for block_idx in instruction.block_idxs:
        start, end = get_start_end(block_size, block_idx)

        up_matvec, start, end = matvec(
            mat=globals.up_proj_weights[instruction.layer_idx],
            vec=post_ln,
            block_size=block_size,
            block_idx=block_idx,
        )

        gate_matvec, _, _ = matvec(
            mat=globals.gate_proj_weights[instruction.layer_idx],
            vec=post_ln,
            block_size=block_size,
            block_idx=block_idx,
        )

        post_silu = F.silu(gate_matvec) * up_matvec

        globals.silu_out[start:end] = post_silu

        barriers[0] += 1


def layer_norm_matvec_rope_append(
    globals: Globals, instruction: LayerNorm_QKV_MatVecRopeAppend
):
    layer_idx = instruction.layer_idx

    # Barrier check
    if layer_idx > 0:
        op_barriers = globals.barriers[layer_idx - 1, instruction.prev_opcode() - 1]
        assert op_barriers[0] == 64

    # Pre-Attention LayerNorm
    post_ln = rms_norm(
        inp=globals.hidden_states,
        weight=globals.attn_ln_weights[layer_idx],
        eps=globals.rms_norm_eps,
    )
    
    # DEBUG: Log Pre-Norm Output
    if layer_idx == 0 and instruction.start_output_block_idx == 0:
        log_debug_data(f"layer_{layer_idx}_attn_post_ln", post_ln)

    pos_id = globals.pos_id

    k_start = globals.num_attention_heads * globals.head_dim
    v_start = k_start + globals.num_kv_heads * globals.head_dim

    barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]

    for block_idx in range(
        instruction.start_output_block_idx, instruction.end_output_block_idx
    ):
        start, end = get_start_end(globals.qkv_block_size, block_idx)

        if start < k_start:
            mode = "q"
        elif start < v_start:
            mode = "k"
        else:
            mode = "v"

        # 1. QKV Projection
        matmul_output = einsum(
            globals.qkv_proj_weights[layer_idx][start:end],
            post_ln,
            "o i, i -> o",
        )

        out = matmul_output
        
        # DEBUG: Log Projection Output (Raw Q/K/V)
        if layer_idx == 0 and block_idx == 0:
             log_debug_data(f"layer_{layer_idx}_block_{block_idx}_{mode}_proj_raw", out)

        # ================= Modified Section =================
        if mode in ["q", "k"]:
            full_weight = None

            # 1. Determine weight tensor
            if mode == "q":
                full_weight = globals.q_norm_weights[layer_idx].view(-1)
                weight_len = full_weight.shape[0]
                idx_start = start % weight_len
                idx_end = idx_start + (end - start)

            elif mode == "k":
                full_weight = globals.k_norm_weights[layer_idx].view(-1)
                weight_len = full_weight.shape[0]
                rel_start = start - k_start
                idx_start = rel_start % weight_len
                idx_end = idx_start + (end - start)

            # 2. Safe Slicing
            if idx_end > full_weight.shape[0]:
                part1 = full_weight[idx_start:]
                part2 = full_weight[: idx_end - full_weight.shape[0]]
                norm_weight = torch.cat([part1, part2], dim=0)
            else:
                norm_weight = full_weight[idx_start:idx_end]

            # 3. Execute Norm (Potential Bug Here: RMSNorm on partial block?)
            out = rms_norm(inp=out, weight=norm_weight, eps=globals.rms_norm_eps)
            
            # DEBUG: Log Q/K Norm Output
            if layer_idx == 0 and block_idx == 0:
                 log_debug_data(f"layer_{layer_idx}_block_{block_idx}_{mode}_norm_out", out)

        # ================= End Modified Section =================

        if mode in ["q", "k"]:
            full_head = torch.zeros(
                1,
                globals.head_dim,
                device=globals.hidden_states.device,
                dtype=out.dtype,
            )
            head_segment = start % globals.head_dim
            full_head_start = head_segment
            full_head_end = full_head_start + (end - start)

            full_head[:, full_head_start:full_head_end] = out

            # 2. RoPE
            full_head_with_rope, _ = apply_rotary_pos_emb_interleaved(
                q=full_head,
                k=full_head,
                cos=globals.rope_cos[pos_id],
                sin=globals.rope_sin[pos_id],
                unsqueeze_dim=0,
            )
            out = full_head_with_rope[:, full_head_start:full_head_end].view(-1)
            
            # DEBUG: Log RoPE Output
            if layer_idx == 0 and block_idx == 0:
                 log_debug_data(f"layer_{layer_idx}_block_{block_idx}_{mode}_rope_out", out)

        # Store results
        match mode:
            case "q":
                globals.post_ln_rope_q[start:end] = out
            case "k":
                start_in_k = start - k_start
                end_in_k = end - k_start
                globals.k_cache[layer_idx, :, pos_id].view(-1)[
                    start_in_k:end_in_k
                ] = out
            case "v":
                start_in_v = start - v_start
                end_in_v = end - v_start
                globals.v_cache[layer_idx, :, pos_id].view(-1)[
                    start_in_v:end_in_v
                ] = out

        barriers[block_idx // 8] += 1


def rms_lm_head(globals: Globals, instruction: RMS_LM_Head):
    op_barriers = globals.barriers[
        globals.num_hidden_layers - 1, instruction.prev_opcode() - 1
    ]
    assert op_barriers[0] == 64

    post_ln = rms_norm(
        inp=globals.hidden_states,
        weight=globals.lm_head_norm_weights,
        eps=globals.rms_norm_eps,
    )

    for block_idx in range(
        instruction.start_output_block_idx, instruction.end_output_block_idx
    ):
        start, end = get_start_end(globals.lm_head_block_size, block_idx)

        matmul_output = einsum(
            globals.lm_head_weights[start:end],
            post_ln,
            "o i, i -> o",
        )

        globals.logits[start:end] = matmul_output


def partial_attention(globals: Globals, instruction: PartialAttention):
    gqa_ratio = globals.num_attention_heads // globals.num_kv_heads

    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    for i in range(gqa_ratio):
        assert op_barriers[instruction.kv_head_idx * gqa_ratio + i] == 8
    assert op_barriers[globals.num_attention_heads + instruction.kv_head_idx] == 8
    assert (
        op_barriers[
            globals.num_attention_heads + globals.num_kv_heads + instruction.kv_head_idx
        ]
        == 8
    )

    kv_block_size = globals.attn_kv_block_size
    seq_len = globals.pos_id + 1
    layer_idx = instruction.layer_idx
    kv_head_idx = instruction.kv_head_idx

    total_blocks = math.ceil(seq_len / kv_block_size)
    blocks_per_partial = math.ceil(total_blocks / instruction.num_partials)

    start_block = instruction.partial_idx * blocks_per_partial
    end_block = min(start_block + blocks_per_partial, total_blocks)

    start_token = start_block * kv_block_size
    end_token = min(end_block * kv_block_size, seq_len)

    k = globals.k_cache[layer_idx, 0, start_token:end_token, kv_head_idx]
    v = globals.v_cache[layer_idx, 0, start_token:end_token, kv_head_idx]

    head_start = kv_head_idx * gqa_ratio
    head_end = head_start + gqa_ratio

    q = globals.post_ln_rope_q.view(globals.num_attention_heads, -1)[
        head_start:head_end
    ]

    qk = einsum(q.float(), k.float(), "h i, k i -> h k")
    scaled_qk = qk * globals.attn_scale

    # casting the output of the softmax to 16-bit causes small numerical differences
    softmax = torch.softmax(scaled_qk, dim=-1)

    # lse = torch.logsumexp(scaled_qk, dim=-1)
    lse = torch.log2(torch.sum(torch.exp(scaled_qk), dim=-1))

    out = einsum(softmax.float(), v.float(), "h k, k o -> h o")
    # print(out.shape)
    
    # DEBUG
    if instruction.layer_idx == 0 and instruction.partial_idx == 0:
        log_debug_data(f"layer_{layer_idx}_head_grp_{kv_head_idx}_attn_out", out)


    if globals.skip_attn_reduction:
        globals.attn_out.view(globals.num_attention_heads, -1)[
            head_start:head_end, :
        ] = out
        barriers = globals.barriers[
            instruction.layer_idx, AttentionReduction.opcode() - 1
        ]
        barriers[0] += head_end - head_start

    else:
        globals.attn_lse_intermediates[head_start:head_end, instruction.partial_idx] = (
            lse
        )
        globals.attn_out_intermediates[head_start:head_end, instruction.partial_idx] = (
            out
        )

        # Barrier update
        barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
        barriers[head_start:head_end] += 1


def attention_reduction(globals: Globals, instruction: AttentionReduction):
    head_start_idx = instruction.head_start_idx

    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[head_start_idx] == instruction.num_partials

    indices_to_reduce = torch.tensor(
        instruction.reduction_list,
        dtype=torch.long,
        device=globals.hidden_states.device,
    )

    lses = globals.attn_lse_intermediates[
        head_start_idx : head_start_idx + globals.attn_reduction_size, indices_to_reduce
    ]
    outs = globals.attn_out_intermediates[
        head_start_idx : head_start_idx + globals.attn_reduction_size, indices_to_reduce
    ]

    max_lse = torch.max(lses, dim=-1, keepdim=True).values

    # adjusted_factors = (lses - max_lse).exp()
    adjusted_factors = (lses - max_lse).exp2()
    new_denominator = adjusted_factors.sum(dim=-1, keepdim=True)

    reduced = (outs * adjusted_factors.unsqueeze(-1)).sum(dim=1) / new_denominator

    if instruction.is_terminal:
        globals.attn_out.view(globals.num_attention_heads, -1)[
            head_start_idx : head_start_idx + globals.attn_reduction_size
        ] = reduced
    else:
        new_lse = new_denominator.log()
        output_slot = instruction.output_partial_idx
        globals.attn_lse_intermediates[
            head_start_idx : head_start_idx + globals.attn_reduction_size, output_slot
        ] = new_lse
        globals.attn_out_intermediates[
            head_start_idx : head_start_idx + globals.attn_reduction_size, output_slot
        ] = reduced

    # Barrier update
    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += globals.attn_reduction_size  # the dumb way


INSTRUCTION_TO_SOLVER = {
    O_ProjResidual: o_proj_residual,
    DownProjResidual: down_proj_residual,
    LayerNormDoubleMatVecSiLU: layer_norm_double_matvec_silu,
    LayerNorm_QKV_MatVecRopeAppend: layer_norm_matvec_rope_append,
    RMS_LM_Head: rms_lm_head,
    PartialAttention: partial_attention,
    AttentionReduction: attention_reduction,
}


### 文件 2: `qwen3.py` (添加了调试记录)


from dataclasses import dataclass
from pathlib import Path
import json
import os

import huggingface_hub
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from einops import rearrange
from torch import Tensor, nn
from torch.distributed import _functional_collectives as funcol
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from megakernels.model_types import (
    BatchState,
    DeviceType,
    ExtraModelConfig,
)
from megakernels.utils import (
    load_safetensors_repo,
)

# ================= DEBUG UTILS =================
DEBUG_FILE = "qwen3_torch_debug.json"
if os.path.exists(DEBUG_FILE):
    os.remove(DEBUG_FILE)

def log_debug_data(tag, tensor):
    """记录 Tensor 的关键统计信息到 JSON 文件"""
    if tensor is None:
        return
    # 同样记录形状、均值、和、前16个值
    t_flat = tensor.detach().float().flatten().cpu()
    data = {
        "tag": tag,
        "shape": list(tensor.shape),
        "mean": float(t_flat.mean().item()),
        "sum": float(t_flat.sum().item()),
        "head_values": t_flat[:16].tolist()
    }
    with open(DEBUG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
# ===============================================

KV_Cache = tuple[Tensor, Tensor]


class RMSNorm(nn.Module):
    def __init__(self, config: Qwen3Config, dim):
        """
        Taken from Qwen3RMSNorm.
        """
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.config.rms_norm_eps)

        if self.weight is not None:
            return self.weight * hidden_states.to(input_dtype)
        else:
            return hidden_states.to(input_dtype)


def all_gather(x: Tensor, extra_config: ExtraModelConfig):
    if extra_config.tp_size == 1:
        return x

    assert extra_config.tp_group is not None

    # funcol + no compile + tp > 1 and pp > 1 leads to some nccl crash
    if extra_config.torch_compile:
        return funcol.all_gather_tensor(x, gather_dim=0, group=extra_config.tp_group)

    out = torch.empty(
        (extra_config.tp_size * x.shape[0], *x.shape[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    torch.distributed.all_gather_into_tensor(out, x, group=extra_config.tp_group)
    return out


def reduce_scatter(x: Tensor, extra_config: ExtraModelConfig):
    if extra_config.tp_size == 1:
        return x

    assert extra_config.tp_group is not None

    # funcol + no compile + tp > 1 and pp > 1 leads to some nccl crash
    if extra_config.torch_compile:
        return funcol.reduce_scatter_tensor(
            x, reduceOp="sum", scatter_dim=0, group=extra_config.tp_group
        )

    out = torch.empty(
        (x.shape[0] // extra_config.tp_size, *x.shape[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    torch.distributed.reduce_scatter_tensor(out, x, group=extra_config.tp_group)
    return out


def attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    kv_cache: KV_Cache,
    position_ids: Tensor,
    seq_len: int,
) -> Tensor:
    bsz, new_tok_seq_len = query_states.shape[:2]

    k_cache, v_cache = kv_cache

    k_cache[:, position_ids] = key_states
    v_cache[:, position_ids] = value_states

    def shape_for_sdpa(x: Tensor):
        return rearrange(x, "b l h d -> b h l d")

    def unshape_for_sdpa(x: Tensor):
        return rearrange(x, "b l h d -> b l h d")

    if new_tok_seq_len > 1:
        k_for_sdpa = shape_for_sdpa(key_states)
        v_for_sdpa = shape_for_sdpa(value_states)

        q_for_sdpa = shape_for_sdpa(query_states)

        # assume running prefill from scratch
        attn_output = F.scaled_dot_product_attention(
            q_for_sdpa, k_for_sdpa, v_for_sdpa, is_causal=True, enable_gqa=True
        )
    else:
        # decode
        k_for_sdpa = shape_for_sdpa(k_cache[:, :seq_len])
        v_for_sdpa = shape_for_sdpa(v_cache[:, :seq_len])
        q_for_sdpa = shape_for_sdpa(query_states)

        attn_output = F.scaled_dot_product_attention(
            q_for_sdpa, k_for_sdpa, v_for_sdpa, is_causal=False, enable_gqa=True
        )
    reshaped_attn_output = unshape_for_sdpa(attn_output)
    return reshaped_attn_output


def rotate_half_interleaved(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    new_x1 = -x2
    new_x2 = x1

    stacked = torch.stack((new_x1, new_x2), dim=-1)
    return stacked.view_as(x)


def apply_rotary_pos_emb_interleaved(
    q, k, cos, sin, position_ids=None, unsqueeze_dim=1
):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


class Qwen3Attention(nn.Module):
    """Qwen3 Attention with QK-Norm"""

    def __init__(
        self, config: Qwen3Config, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config, dim=config.hidden_size)

        self.tp_size = extra_config.tp_size or 1

        assert config.num_attention_heads % self.tp_size == 0

        self.head_dim = config.head_dim

        assert self.config.num_attention_heads % self.tp_size == 0
        assert (
            self.config.num_key_value_heads % self.tp_size == 0
            or self.config.num_key_value_heads == 1
        )

        self.num_attention_heads = config.num_attention_heads // self.tp_size
        self.num_kv_heads = (
            config.num_key_value_heads // self.tp_size
            if config.num_key_value_heads > 1
            else 1
        )

        self.q_proj = nn.Linear(
            self.config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )

        self.q_norm = RMSNorm(config, dim=self.head_dim)
        self.k_norm = RMSNorm(config, dim=self.head_dim)

        self.kv_cache: KV_Cache | None = None

    def forward(
        self,
        batch_state: BatchState,
    ):
        assert batch_state.hidden_states is not None
        assert batch_state.position_embeddings is not None
        assert batch_state.position_ids is not None
        assert self.kv_cache is not None
        assert batch_state.seq_len is not None

        inp = batch_state.hidden_states
        residual = inp

        hidden_states = self.input_layernorm(inp)
        
        # DEBUG: Log Post-Norm (Input to QKV Proj)
        if self.layer_idx == 0:
            log_debug_data(f"layer_{self.layer_idx}_attn_post_ln", hidden_states)

        hidden_states = all_gather(hidden_states, self.extra_config)
        bsz, seq_len = hidden_states.shape[:2]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, seq_len, self.num_attention_heads, -1)
        key_states = key_states.view(bsz, seq_len, self.num_kv_heads, -1)
        value_states = value_states.view(bsz, seq_len, self.num_kv_heads, -1)
        
        # DEBUG: Log Raw Projections
        if self.layer_idx == 0:
             # Log first head block equivalent
             log_debug_data(f"layer_{self.layer_idx}_block_0_q_proj_raw", query_states[0, 0, 0, :16]) 
             log_debug_data(f"layer_{self.layer_idx}_block_0_k_proj_raw", key_states[0, 0, 0, :16])

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # DEBUG: Log Normed Projections
        if self.layer_idx == 0:
             log_debug_data(f"layer_{self.layer_idx}_block_0_q_norm_out", query_states[0, 0, 0, :16])
             log_debug_data(f"layer_{self.layer_idx}_block_0_k_norm_out", key_states[0, 0, 0, :16])

        cos, sin = batch_state.position_embeddings

        dtype = query_states.dtype

        if self.extra_config.interleave_rope:
            rope_fn = apply_rotary_pos_emb_interleaved
        else:
            rope_fn = apply_rotary_pos_emb

        query_states, key_states = rope_fn(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=-2,  # unsqueeze dim = head dim on q/k
        )

        query_states = query_states.to(dtype)
        key_states = key_states.to(dtype)
        
        # DEBUG: Log RoPE Output
        if self.layer_idx == 0:
             log_debug_data(f"layer_{self.layer_idx}_block_0_q_rope_out", query_states[0, 0, 0, :16])
             log_debug_data(f"layer_{self.layer_idx}_block_0_k_rope_out", key_states[0, 0, 0, :16])

        raw_attn_output = attention(
            query_states,
            key_states,
            value_states,
            self.kv_cache,
            batch_state.position_ids,
            seq_len=batch_state.seq_len,
        )
        # print(raw_attn_output.shape)

        attn_output = raw_attn_output.reshape(bsz, seq_len, -1)

        # print(attn_output.shape)
        
        # DEBUG: Log Attention Output
        if self.layer_idx == 0:
            # 模拟 VM 中的 partial attention 输出
            log_debug_data(f"layer_{self.layer_idx}_attn_out_buffer", attn_output)

        o_proj = self.o_proj(attn_output)

        o_proj = reduce_scatter(o_proj, self.extra_config)

        with_residual = residual + o_proj
        
        # DEBUG: Log Output of Attention Block
        if self.layer_idx == 0:
             log_debug_data(f"layer_{self.layer_idx}_hidden_states_post_oproj_partial", with_residual)

        batch_state.hidden_states = with_residual
        return batch_state


class Qwen3MLP(nn.Module):
    def __init__(
        self, config: Qwen3Config, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.tp_size = extra_config.tp_size
        assert self.config.intermediate_size % self.tp_size == 0
        self.intermediate_size = self.config.intermediate_size // self.tp_size

        self.up_proj = nn.Linear(
            self.config.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            config.hidden_size,
            bias=False,
        )

        self.input_layernorm = RMSNorm(config, dim=config.hidden_size)

    def forward(
        self,
        batch_state: BatchState,
    ):
        inp = batch_state.hidden_states
        assert inp is not None
        hidden_states = self.input_layernorm(inp)

        hidden_states = all_gather(hidden_states, self.extra_config)
        
        # DEBUG: MLP Post Norm
        if self.layer_idx == 0:
            log_debug_data(f"layer_{self.layer_idx}_mlp_post_ln", hidden_states)

        up = self.up_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        prod = F.silu(gate) * up
        down = self.down_proj(prod)

        down = reduce_scatter(down, self.extra_config)

        with_residual = inp + down

        batch_state.hidden_states = with_residual
        return batch_state


class Qwen3Block(nn.Module):
    def __init__(
        self, config: Qwen3Config, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.self_attn = Qwen3Attention(config, extra_config, layer_idx)
        self.mlp = Qwen3MLP(config, extra_config, layer_idx)

    def forward(self, batch_state: BatchState):
        out = self.self_attn(batch_state)
        out = self.mlp(out)
        return out


class Qwen3LMHead(nn.Module):
    def __init__(self, config: Qwen3Config, extra_config: ExtraModelConfig):
        super().__init__()
        self.config = config
        self.extra_config = extra_config

        self.input_norm = RMSNorm(config, dim=config.hidden_size)

        self.tp_size = extra_config.tp_size or 1

        assert config.vocab_size % self.tp_size == 0
        head_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, head_size, bias=False)

    def forward(self, batch_state: BatchState):
        assert batch_state.hidden_states is not None

        hidden_states = batch_state.hidden_states

        if self.extra_config.tp_size > 1:
            hidden_states = all_gather(hidden_states, self.extra_config)

        hidden_states = self.input_norm(hidden_states)

        logits = self.lm_head(hidden_states)

        next_token_ids = logits.argmax(dim=-1)

        if self.tp_size > 1:
            # TODO: fusion
            next_token_ids = all_gather(next_token_ids, self.extra_config)

        batch_state.output_ids = next_token_ids
        return batch_state


class Qwen3Embeddings(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, batch_state: BatchState):
        hidden_states = self.embed_tokens(batch_state.input_ids)

        batch_state.hidden_states = hidden_states
        return batch_state


class Qwen3Model(nn.Module):
    rope_cos: Tensor
    rope_sin: Tensor

    def __init__(
        self,
        config: Qwen3Config,
        extra_config: ExtraModelConfig,
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.embed_tokens = Qwen3Embeddings(config)

        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(Qwen3Block(config, extra_config, i))

        self.layers = nn.ModuleList(layers)

        self.rope = Qwen3RotaryEmbedding(
            config=config,
        )

        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        dummy_float_input = torch.empty((0, config.hidden_size), dtype=torch.float32)

        cos, sin = self.rope(dummy_float_input, position_ids)
        assert cos.dtype == torch.float32
        assert sin.dtype == torch.float32
        self.register_buffer("rope_cos", cos.squeeze(0), persistent=False)
        self.register_buffer("rope_sin", sin.squeeze(0), persistent=False)

    def interleave_rope(self):
        indices_for_q_list = []
        half_head_dim = self.config.head_dim // 2
        for n in range(self.config.num_attention_heads):
            offset = n * self.config.head_dim
            for i in range(half_head_dim):
                indices_for_q_list.append(i + offset)
                indices_for_q_list.append(i + half_head_dim + offset)

        indices_for_q = torch.tensor(indices_for_q_list, device=self.rope_cos.device)
        one_head_indices = indices_for_q[: self.config.head_dim]

        self.rope_cos = self.rope_cos[..., one_head_indices]
        self.rope_sin = self.rope_sin[..., one_head_indices]

        indices_for_k = indices_for_q[
            : self.config.head_dim * self.config.num_key_value_heads
        ]

        for mod in self.modules():
            if isinstance(mod, Qwen3Attention):
                mod.q_proj.weight[:] = mod.q_proj.weight[indices_for_q]
                mod.k_proj.weight[:] = mod.k_proj.weight[indices_for_k]

    def forward(self, batch_state: BatchState):
        out: BatchState = self.embed_tokens(batch_state)
        assert self.rope_cos.dtype == torch.float32
        assert self.rope_sin.dtype == torch.float32
        cos = self.rope_cos[batch_state.position_ids]
        sin = self.rope_sin[batch_state.position_ids]
        out.position_embeddings = (cos, sin)

        for layer in self.layers:
            out = layer(out)
        return out


@dataclass
class StackedParams:
    qkv_proj_weight: Tensor
    q_norm_weight: Tensor
    k_norm_weight: Tensor
    o_proj: Tensor
    attn_ln_weight: Tensor
    mlp_ln_weight: Tensor
    up_proj: Tensor
    gate_proj: Tensor
    down_proj: Tensor


class Qwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        extra_config: ExtraModelConfig,
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.device = torch.get_default_device()
        self.dtype = torch.get_default_dtype()

        self.model = Qwen3Model(config, extra_config)

        self.lm_head = Qwen3LMHead(config, extra_config)

    def num_kv_heads(self):
        all_heads = self.config.num_key_value_heads
        tp_size = self.extra_config.tp_size
        assert all_heads % tp_size == 0
        return all_heads // tp_size

    def num_qo_heads(self):
        all_heads = self.config.num_attention_heads
        tp_size = self.extra_config.tp_size
        assert all_heads % tp_size == 0
        return all_heads // tp_size

    def forward(
        self,
        batch_state: BatchState,
    ):
        input_ids = batch_state.input_ids
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        position_ids = batch_state.position_ids
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)

        out = BatchState(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=batch_state.hidden_states,
            seq_len=batch_state.seq_len,
        )

        out = self.model(out)
        out = self.lm_head(out)

        return out

    def setup_caches(self):
        k_cache = torch.zeros(
            (
                self.config.num_hidden_layers,
                self.extra_config.max_batch_size,
                self.extra_config.max_len_override
                or self.config.max_position_embeddings,
                self.config.num_key_value_heads,
                self.config.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        v_cache = k_cache.clone()

        self.stacked_kv_cache = (k_cache, v_cache)

        for layer_idx in range(self.config.num_hidden_layers):
            layer: Qwen3Block = self.model.layers[layer_idx]  # type: ignore
            layer.self_attn.kv_cache = (
                self.stacked_kv_cache[0][layer_idx],
                self.stacked_kv_cache[1][layer_idx],
            )

    def to(self, device: DeviceType | None = None, dtype: torch.dtype | None = None):  # type: ignore
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return super().to(device=device, dtype=dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        extra_config: ExtraModelConfig | None = None,
        device: DeviceType | None = None,
        dtype: torch.dtype | None = None,
    ):
        if extra_config is None:
            extra_config = ExtraModelConfig()

        config: Qwen3Config = Qwen3Config.from_pretrained(model_name_or_path)  # type: ignore
        if extra_config.rope_scaling is not None:
            config.rope_scaling = extra_config.rope_scaling

        if dtype is None:
            dtype = config.torch_dtype

        with init_empty_weights(include_buffers=False):
            model = cls(
                config,
                extra_config,
            )
        model.dtype = dtype
        model.device = device

        if (as_path := Path(model_name_or_path)).exists():
            model_path = as_path
        else:
            snapshot_path_str = huggingface_hub.snapshot_download(
                model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
            )

            model_path = Path(snapshot_path_str)

        model.load_from_safetensors(model_path)

        # SE (10/18/24): It is important not to call model.to(device, dtype) because
        # this will convert the `inv_freq` buffer in the rotary embeddings to fp16
        # the HF load from pretrained is careful to not do this and keeps it in fp32.
        # The dtype for the parameters is already handled by the load calls above, but
        # it's possible that there are other buffers which *should* be converted to fp16.
        # TODO: it's probably easiest to figure out how we can just use HFs `load_from_pretrained`
        # to load the model weights so we can ensure that there are no other subtle differences
        model.to(device=device)

        model.requires_grad_(False)

        if extra_config.interleave_rope:
            model.model.interleave_rope()

        model.stack_params()
        model.setup_caches()

        return model

    def make_name_to_hf_name(self):
        keys = self.state_dict().keys()

        name_to_hf_name = {k: k for k in keys}

        for layer_idx in range(self.config.num_hidden_layers):
            name_to_hf_name[
                f"model.layers.{layer_idx}.self_attn.input_layernorm.weight"
            ] = f"model.layers.{layer_idx}.input_layernorm.weight"
            name_to_hf_name[f"model.layers.{layer_idx}.mlp.input_layernorm.weight"] = (
                f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            )

        name_to_hf_name["model.embed_tokens.embed_tokens.weight"] = (
            "model.embed_tokens.weight"
        )
        name_to_hf_name["lm_head.input_norm.weight"] = "model.norm.weight"

        if self.config.tie_word_embeddings:
            name_to_hf_name["lm_head.lm_head.weight"] = "model.embed_tokens.weight"
        else:
            name_to_hf_name["lm_head.lm_head.weight"] = "lm_head.weight"

        return name_to_hf_name

    def make_tp_map(self):
        """
        Maps parameter names to the dimension they should be split on.
        Parameters that are not included in the map should not be split.
        """

        tp_map = {}
        for param_name, _ in self.named_parameters():
            if any(
                param_name.endswith(suffix)
                for suffix in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "up_proj.weight",
                    "gate_proj.weight",
                ]
            ):
                tp_map[param_name] = 0

            elif any(
                param_name.endswith(suffix)
                for suffix in ["o_proj.weight", "down_proj.weight"]
            ):
                tp_map[param_name] = 1

        return tp_map

    def load_from_safetensors(
        self,
        model_path: Path,
    ):
        name_to_hf_name = self.make_name_to_hf_name()
        all_hf_names = set(name_to_hf_name.values())

        hf_state_dict = load_safetensors_repo(
            model_path,
            include_parameters=all_hf_names,
            device=self.device,
            tp_rank=self.extra_config.tp_rank,
            tp_size=self.extra_config.tp_size,
            tp_map=self.make_tp_map(),
        )

        state_dict = {k: hf_state_dict[v] for k, v in name_to_hf_name.items()}

        self.load_state_dict(state_dict, assign=True, strict=True)

    def stack_params(self):
        def stack_and_reassign(modules, prop: str):
            params = [getattr(m, prop) for m in modules]
            stacked = torch.stack(params, dim=0)
            for i, m in enumerate(modules):
                getattr(m, prop)[:] = stacked[i]
            return stacked

        layers: list[Qwen3Block] = self.model.layers
        self_attns = [x.self_attn for x in layers]

        # Qwen3 Specific: Stack Q-Norm and K-Norm weights
        # Provided param: model.layers.0.self_attn.q_norm.weight [128]
        q_norms = [x.q_norm for x in self_attns]
        k_norms = [x.k_norm for x in self_attns]

        mlps = [x.mlp for x in layers]

        o_projs = [x.o_proj for x in self_attns]
        self_attn_lns = [x.input_layernorm for x in self_attns]

        mlp_lns = [x.input_layernorm for x in mlps]
        up_projs = [x.up_proj for x in mlps]
        gate_projs = [x.gate_proj for x in mlps]
        down_projs = [x.down_proj for x in mlps]

        stacked_q_norm_weights = stack_and_reassign(q_norms, "weight")
        stacked_k_norm_weights = stack_and_reassign(k_norms, "weight")
        stacked_o_proj = stack_and_reassign(o_projs, "weight")
        stacked_self_attn_ln_weights = stack_and_reassign(self_attn_lns, "weight")
        stacked_mlp_ln_weights = stack_and_reassign(mlp_lns, "weight")
        stacked_up_proj = stack_and_reassign(up_projs, "weight")
        stacked_gate_proj = stack_and_reassign(gate_projs, "weight")
        stacked_down_proj = stack_and_reassign(down_projs, "weight")

        qkv_weights = []
        for self_attn in self_attns:
            cat_weight = torch.cat(
                [
                    self_attn.q_proj.weight,
                    self_attn.k_proj.weight,
                    self_attn.v_proj.weight,
                ],
                dim=0,
            )
            qkv_weights.append(cat_weight)

        stacked_qkv_weights = torch.stack(qkv_weights, dim=0)

        for i, self_attn in enumerate(self_attns):
            qkv_weight = stacked_qkv_weights[i]
            q_weight, k_weight, v_weight = qkv_weight.split(
                [
                    self.config.num_attention_heads * self.config.head_dim,
                    self.config.num_key_value_heads * self.config.head_dim,
                    self.config.num_key_value_heads * self.config.head_dim,
                ],
                dim=0,
            )

            self_attn.q_proj.weight[:] = q_weight
            self_attn.k_proj.weight[:] = k_weight
            self_attn.v_proj.weight[:] = v_weight
        self.stacked_params = StackedParams(
            qkv_proj_weight=stacked_qkv_weights,
            o_proj=stacked_o_proj,
            attn_ln_weight=stacked_self_attn_ln_weights,
            mlp_ln_weight=stacked_mlp_ln_weights,
            up_proj=stacked_up_proj,
            gate_proj=stacked_gate_proj,
            down_proj=stacked_down_proj,
            q_norm_weight=stacked_q_norm_weights,
            k_norm_weight=stacked_k_norm_weights,
        )

```

**user**: 

```
qwen3_repl.py中是调用python_vm.py和qwen3.py中的方法进行生成的文件，以此为基础，生成一个能够输出两种模式的调试信息并比较的脚本。
```

Attachments:
- qwen3_repl.py

**assistant**: 

```
这是一个基于 `qwen3_repl.py` 修改的调试与对比脚本 `debug_compare.py`。

它的主要功能是：
1.  **自动化执行**：分别以 `torch` 模式和 `pyvm` 模式运行模型进行一次 Decode 步进。
2.  **环境隔离**：在两次运行之间重置模型状态，确保 KV Cache 为空或一致。
3.  **日志对齐与对比**：读取上一步修改代码生成的 `qwen3_torch_debug.json` 和 `python_vm_debug.json`，针对相同的 Tag 进行数值对比（均值、和、前16位数值），并高亮显示第一个出现误差的地方。

请将此脚本保存在与 `python_vm.py`, `qwen3.py`, `scheduler.py` 同级的目录下运行。

### 文件: `debug_compare.py`


import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer

# 引入原始项目的依赖
from megakernels.dispatch import (
    make_schedule_builder,
    make_pyvm_interpreter,
)
from megakernels.generators_qwen3 import (
    PyTorchGenerator,
    PyVM_Generator,
)
from megakernels.qwen3 import Qwen3ForCausalLM
from megakernels.model_types import BatchState, ExtraModelConfig
from megakernels.scheduler import (
    assign_to_sms,
    tensorize_instructions,
)

# 配置类
class DebugConfig:
    # 请根据实际路径修改模型地址
    model: str = "/home/ziming/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B" 
    device: str = "cuda:0"
    setting: str = "qwen3"
    sched: str = "rr"
    memory_fraction: float | None = None
    
    # 调试输出文件路径 (对应 python_vm.py 和 qwen3.py 中的定义)
    torch_log: str = "qwen3_torch_debug.json"
    vm_log: str = "python_vm_debug.json"

def run_generation_step(mode: str, config: DebugConfig):
    """
    运行一次生成步骤 (Prefill + 1 Decode step)
    """
    print(f"\n[INFO] Starting execution in mode: {mode}...")
    
    torch.cuda.set_device(config.device)
    
    # 1. 清理旧日志 (根据模式清理对应的)
    target_log = config.torch_log if mode == "torch" else config.vm_log
    if os.path.exists(target_log):
        os.remove(target_log)

    # 2. 加载模型 (每次重新加载以确保状态纯净)
    print("  Loading model...")
    extra_config = ExtraModelConfig(
        interleave_rope=True,
        max_batch_size=1,
    )
    model = Qwen3ForCausalLM.from_pretrained(
        config.model, device=config.device, extra_config=extra_config
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    # 3. 准备输入
    prompt = "Hello"
    messages = [{"role": "user", "content": prompt}]
    tok_inp = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(tok_inp, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(model.device)
    
    prompt_len = input_ids.shape[-1]
    position_ids = torch.arange(prompt_len).to(model.device)

    # 4. 初始化生成器
    gen = None
    if mode == "torch":
        gen = PyTorchGenerator(model)
    elif mode == "pyvm":
        # 设置调度器
        schedule_builder = make_schedule_builder(config.setting)
        schedule = schedule_builder.build(model)
        assigned_to_sms = assign_to_sms(
            config.sched, schedule=schedule, memory_fraction=config.memory_fraction
        )
        tensorize_instructions(schedule.globs, assigned_to_sms)
        
        # 设置解释器
        interpreter = make_pyvm_interpreter(config.setting)
        gen = PyVM_Generator(model, interpreter, schedule)

    # 5. 运行 Prefill
    print("  Running Prefill...")
    prefill_inp = BatchState(
        input_ids=input_ids,
        position_ids=position_ids,
    )
    prefill_output: BatchState = model(prefill_inp)
    
    # 准备 Decode 输入 (1个token)
    new_input_token = prefill_output.output_ids[:, -1:]
    output_tokens = torch.zeros(
        1, 128, device=model.device, dtype=torch.long
    )
    output_tokens[:, 0] = new_input_token

    # 6. 运行 Decode (仅跑1步以触发 log)
    print("  Running Decode (1 step)...")
    gen.generate_with_eos(
        output_tokens=output_tokens,
        prompt_len=prompt_len,
        ntok=1, # 只生成1个token
        eos_token_ids=[151643], # 随便填一个EOS
        eos_token_check_interval=1,
    )
    
    print(f"  Finished {mode} execution. Log generated at {target_log}")
    
    # 显式清理显存
    del model
    del gen
    torch.cuda.empty_cache()

def load_log(filepath):
    data = []
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    data.append(item)
                except:
                    pass
    # 转为字典，key为tag
    return {item['tag']: item for item in data}

def compare_logs(config: DebugConfig):
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60)

    torch_data = load_log(config.torch_log)
    vm_data = load_log(config.vm_log)

    if not torch_data:
        print("[ERROR] No Torch logs found.")
        return
    if not vm_data:
        print("[ERROR] No VM logs found.")
        return

    # 按 Torch 的顺序进行对比
    all_match = True
    
    # 定义比较的 Tag 列表 (根据 python_vm.py 中添加的 log tag)
    # 注意：顺序很重要，按计算图流向
    ordered_tags = [
        "layer_0_attn_post_ln",
        "layer_0_block_0_q_proj_raw",
        "layer_0_block_0_k_proj_raw",
        "layer_0_block_0_q_norm_out",
        "layer_0_block_0_k_norm_out",
        "layer_0_block_0_q_rope_out",
        "layer_0_block_0_k_rope_out",
        "layer_0_head_grp_0_attn_out", # 部分 Attention 输出
        "layer_0_attn_out_buffer",     # 完整 Attention 输出
        "layer_0_hidden_states_post_oproj_partial", # O-Proj 残差后
        "layer_0_mlp_post_ln"
    ]

    for tag in ordered_tags:
        # 如果 Log 中没有这个 Tag，尝试模糊匹配（因为 block index 可能是 0）
        t_item = torch_data.get(tag)
        v_item = vm_data.get(tag)
        
        if not t_item:
            # 可能是部分日志未触发
            continue

        print(f"Checking: {tag:<45}", end="")

        if not v_item:
            print(f" -> [MISSING IN VM]")
            all_match = False
            continue

        # 对比 Sum 和 Mean
        t_sum, v_sum = t_item['sum'], v_item['sum']
        t_mean, v_mean = t_item['mean'], v_item['mean']
        
        # 简单的相对误差计算
        diff_sum = abs(t_sum - v_sum)
        rel_err = diff_sum / (abs(t_sum) + 1e-6)
        
        threshold = 1e-3 # 宽松一点，fp16/bf16 误差

        if rel_err > threshold:
            print(f" -> [FAIL]")
            print(f"    Torch Sum: {t_sum:.6f}, Mean: {t_mean:.6f}")
            print(f"    VM    Sum: {v_sum:.6f}, Mean: {v_mean:.6f}")
            print(f"    Diff  Sum: {diff_sum:.6f}, Rel Err: {rel_err:.6f}")
            
            print(f"    Torch Head: {t_item['head_values'][:4]}...")
            print(f"    VM    Head: {v_item['head_values'][:4]}...")
            
            print("\n[CONCLUSION] Divergence detected at:", tag)
            print("Check python_vm.py logic corresponding to this operator.")
            all_match = False
            break # 找到第一个错误后停止，因为后面的都会错
        else:
            print(f" -> [PASS]")

    if all_match:
        print("\n[SUCCESS] All monitored tensors match within threshold.")
    else:
        print("\n[FAIL] Comparison failed.")

if __name__ == "__main__":
    cfg = DebugConfig()
    
    # 如果参数中有路径，覆盖默认模型路径
    if len(sys.argv) > 1:
        cfg.model = sys.argv[1]

    try:
        # 1. 运行 Torch 获取基准
        run_generation_step("torch", cfg)
        
        # 2. 运行 VM 获取测试结果
        run_generation_step("pyvm", cfg)
        
        # 3. 对比
        compare_logs(cfg)
        
    except Exception as e:
        print(f"\n[ERROR] Execution failed: {e}")
        import traceback
        traceback.print_exc()


### 使用方法

1.  确保你已经按照上一步的指示，修改了 `python_vm.py` 和 `qwen3.py` 并添加了 `log_debug_data` 函数。
2.  修改脚本中的 `DebugConfig` 类中的 `model` 路径为你本地的模型路径。
3.  在终端运行：
    
    python debug_compare.py
    
    或者传入模型路径作为参数：
    
    python debug_compare.py /path/to/your/Qwen3-0.6B
    

### 预期输出

脚本将按顺序执行：
1.  QKV Projection 原始值
2.  Q/K Norm 输出
3.  RoPE 输出
4.  Attention 输出

通过观察 `[FAIL]` 出现的位置，你可以精确定位是 `python_vm.py` 中的哪一行代码（例如是否是 Norm 的切片逻辑有问题，还是 RoPE 的计算逻辑有问题）导致了数值偏差。
```

**user**: 

```
[ERROR] No VM logs found.
```

**assistant**: 

```
出现 "[ERROR] No VM logs found." 错误，原因极有可能是Python环境导入的是已安装的 `megakernels` 库中的原始文件，而不是你本地修改后的 `python_vm.py` 和 `qwen3.py`。

为了强制脚本使用你当前目录下的修改版文件（包含日志记录代码），我们需要在脚本最开始使用 **Module Patching（模块补丁）** 技术，将本地文件注入到 Python 的模块系统中。

请使用下面的代码完全替换 `debug_compare.py`。

### 修改后的 `debug_compare.py`

主要变更：
1.  添加了 `patch_local_modules()` 函数，利用 `importlib` 强制加载当前目录下的 `python_vm.py`, `qwen3.py`, `scheduler.py`。
2.  将这些本地模块注入到 `sys.modules` 中对应的 `megakernels` 路径下，欺骗后续的 `import` 语句使用你的本地文件。
3.  **注意**：这段代码假设你的文件结构是扁平的（所有文件都在同一层级）。


import json
import os
import sys
import importlib.util
from pathlib import Path
import torch
from transformers import AutoTokenizer

# ================= MODULE PATCHING START =================
# 必须在导入 megakernels 之前执行，强制使用当前目录下的本地文件
def load_local_module(local_filename, target_module_name):
    """加载本地文件并伪装成指定的模块名"""
    current_dir = Path(__file__).parent
    file_path = current_dir / local_filename
    
    if not file_path.exists():
        print(f"[WARN] Local file {local_filename} not found, skipping patch.")
        return False

    print(f"[PATCH] Injecting local '{local_filename}' as '{target_module_name}'...")
    spec = importlib.util.spec_from_file_location(target_module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[target_module_name] = module
        spec.loader.exec_module(module)
        return True
    return False

# 尝试注入本地修改过的文件
# 注意：这里的 target_module_name 必须匹配 megakernels 内部 import 的路径
# 1. qwen3.py -> megakernels.qwen3
load_local_module("qwen3.py", "megakernels.qwen3")

# 2. python_vm.py -> megakernels.demos.qwen3_0_6b.python_vm
# 注意：如果不确定具体路径，我们尝试注入到最可能的路径，并修补 sys.path
# 为了保险起见，我们也把当前目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent))

# 我们还需要手动 patch python_vm，因为 dispatch 中可能是动态导入的
# 尝试查找 python_vm 在 megakernels 中的标准引用路径
# 通常是 megakernels.demos.qwen3_0_6b.python_vm 或者 megakernels.python_vm
# 根据原文件内容，它似乎位于 demos 下
patch_success = load_local_module("python_vm.py", "megakernels.demos.qwen3_0_6b.python_vm")
if not patch_success:
    # 尝试另一个可能的路径，以防万一
    load_local_module("python_vm.py", "megakernels.python_vm")

# 3. scheduler.py -> megakernels.scheduler
load_local_module("scheduler.py", "megakernels.scheduler")

# ================= MODULE PATCHING END =================

# 现在可以安全导入 megakernels 的组件了，它们应该会使用上面注入的模块
from megakernels.dispatch import (
    make_schedule_builder,
    make_pyvm_interpreter,
)
from megakernels.generators_qwen3 import (
    PyTorchGenerator,
    PyVM_Generator,
)
# 从我们注入的模块中导入 (虽然 sys.modules 已变，但显式导入更安全)
import megakernels.qwen3 as local_qwen3
Qwen3ForCausalLM = local_qwen3.Qwen3ForCausalLM
from megakernels.model_types import BatchState, ExtraModelConfig
from megakernels.scheduler import (
    assign_to_sms,
    tensorize_instructions,
)

# 如果 python_vm 导入成功，我们需要确保解释器使用的是它
try:
    import megakernels.demos.qwen3_0_6b.python_vm as local_vm
    print("[INFO] Local python_vm loaded successfully.")
except ImportError:
    local_vm = None
    print("[WARN] Could not import local python_vm via package path.")


class DebugConfig:
    # 请根据实际路径修改模型地址
    model: str = "/home/ziming/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B" 
    device: str = "cuda:0"
    setting: str = "qwen3"
    sched: str = "rr"
    memory_fraction: float | None = None
    
    torch_log: str = "qwen3_torch_debug.json"
    vm_log: str = "python_vm_debug.json"

def run_generation_step(mode: str, config: DebugConfig):
    print(f"\n[INFO] Starting execution in mode: {mode}...")
    
    torch.cuda.set_device(config.device)
    target_log = config.torch_log if mode == "torch" else config.vm_log
    
    # 清理旧日志
    if os.path.exists(target_log):
        os.remove(target_log)

    print("  Loading model...")
    extra_config = ExtraModelConfig(
        interleave_rope=True,
        max_batch_size=1,
    )
    model = Qwen3ForCausalLM.from_pretrained(
        config.model, device=config.device, extra_config=extra_config
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    prompt = "Hello"
    messages = [{"role": "user", "content": prompt}]
    tok_inp = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(tok_inp, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(model.device)
    
    prompt_len = input_ids.shape[-1]
    position_ids = torch.arange(prompt_len).to(model.device)

    gen = None
    if mode == "torch":
        gen = PyTorchGenerator(model)
    elif mode == "pyvm":
        schedule_builder = make_schedule_builder(config.setting)
        schedule = schedule_builder.build(model)
        assigned_to_sms = assign_to_sms(
            config.sched, schedule=schedule, memory_fraction=config.memory_fraction
        )
        tensorize_instructions(schedule.globs, assigned_to_sms)
        
        interpreter = make_pyvm_interpreter(config.setting)
        
        # ====== 关键修正：强制解释器使用本地 VM 的指令映射 ======
        if local_vm and hasattr(local_vm, 'INSTRUCTION_TO_SOLVER'):
            print("[PATCH] Overriding interpreter solvers with local python_vm...")
            # 假设 interpreter 有一个 solvers 属性或者是类似的机制
            # 如果没有直接暴露，我们可能需要查看 interpreter 的类型
            # 这里做一种通用尝试：假设 interpreter 内部通过查找 type(inst) -> function
            
            # 检查 interpreter 是否有 solver_map 属性 (这取决于 megakernels 的具体实现)
            if hasattr(interpreter, 'solver_map'):
                interpreter.solver_map.update(local_vm.INSTRUCTION_TO_SOLVER)
            elif hasattr(interpreter, 'solvers'):
                interpreter.solvers.update(local_vm.INSTRUCTION_TO_SOLVER)
            else:
                # 如果 interpreter 是一个简单的函数或对象，我们可能无法直接修补
                # 但如果我们之前的 sys.modules 注入成功，make_pyvm_interpreter 内部
                # 应该已经使用了正确的 python_vm 模块，所以这里可能不需要做任何事。
                print("  (Interpreter structure unknown, relying on sys.modules patch)")
        # ==========================================================
        
        gen = PyVM_Generator(model, interpreter, schedule)

    print("  Running Prefill...")
    prefill_inp = BatchState(input_ids=input_ids, position_ids=position_ids)
    prefill_output: BatchState = model(prefill_inp)
    
    new_input_token = prefill_output.output_ids[:, -1:]
    output_tokens = torch.zeros(1, 128, device=model.device, dtype=torch.long)
    output_tokens[:, 0] = new_input_token

    print("  Running Decode (1 step)...")
    # 使用 generate_with_eos 触发 decode 循环
    gen.generate_with_eos(
        output_tokens=output_tokens,
        prompt_len=prompt_len,
        ntok=1, 
        eos_token_ids=[151643],
        eos_token_check_interval=1,
    )
    
    # 检查文件是否生成
    if os.path.exists(target_log) and os.path.getsize(target_log) > 0:
        print(f"  [OK] Log file generated: {target_log} ({os.path.getsize(target_log)} bytes)")
    else:
        print(f"  [ERROR] Log file NOT generated or empty: {target_log}")
    
    del model
    del gen
    torch.cuda.empty_cache()

def load_log(filepath):
    if not os.path.exists(filepath):
        return {}
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    data[item['tag']] = item
                except:
                    pass
    return data

def compare_logs(config: DebugConfig):
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60)

    torch_data = load_log(config.torch_log)
    vm_data = load_log(config.vm_log)

    if not torch_data:
        print("[ERROR] No Torch logs found. (Did run_generation_step('torch') fail?)")
        return
    if not vm_data:
        print("[ERROR] No VM logs found. (Did run_generation_step('pyvm') fail?)")
        print("Check if python_vm.py is in the same directory and module patching succeeded.")
        return

    all_match = True
    ordered_tags = [
        "layer_0_attn_post_ln",
        "layer_0_block_0_q_proj_raw",
        "layer_0_block_0_k_proj_raw",
        "layer_0_block_0_q_norm_out",
        "layer_0_block_0_k_norm_out",
        "layer_0_block_0_q_rope_out",
        "layer_0_block_0_k_rope_out",
        "layer_0_attn_out_buffer", 
        "layer_0_hidden_states_post_oproj_partial",
        "layer_0_mlp_post_ln"
    ]

    print(f"{'TAG':<45} | {'STATUS':<10} | {'DIFF INFO'}")
    print("-" * 80)

    for tag in ordered_tags:
        t_item = torch_data.get(tag)
        v_item = vm_data.get(tag)
        
        if not t_item: continue # Skip if Torch didn't log it

        status = "PASS"
        diff_info = ""
        
        if not v_item:
            status = "MISSING"
            diff_info = "Not found in VM logs"
            all_match = False
        else:
            t_sum = t_item['sum']
            v_sum = v_item['sum']
            diff_sum = abs(t_sum - v_sum)
            # 使用更宽容的误差阈值，防止累积误差干扰
            if diff_sum > 0.1: 
                status = "FAIL"
                diff_info = f"Sum Diff: {diff_sum:.4f} (T:{t_sum:.2f} V:{v_sum:.2f})"
                all_match = False
            else:
                diff_info = f"Diff: {diff_sum:.6f}"

        print(f"{tag:<45} | {status:<10} | {diff_info}")
        
        if status == "FAIL":
            print(f"   >>> Values (Torch): {t_item['head_values'][:4]}")
            print(f"   >>> Values (VM)   : {v_item['head_values'][:4]}")
            print("   >>> STOPPING COMPARISON HERE due to divergence.")
            break

    if all_match:
        print("\n[SUCCESS] Logic aligns perfectly!")
    else:
        print("\n[FAIL] Logic divergence detected. Inspect the first FAIL tag in python_vm.py.")

if __name__ == "__main__":
    cfg = DebugConfig()
    if len(sys.argv) > 1:
        cfg.model = sys.argv[1]

    try:
        run_generation_step("torch", cfg)
        run_generation_step("pyvm", cfg)
        compare_logs(cfg)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

```

**user**: 

```
> 这是一个基于 `qwen3_repl.py` 修改的调试与对比脚本 `debug_compare.py`。
> 
> 它的主要功能是：
> 1.  **自动化执行**：分别以 `torch` 模式和 `pyvm` 模式运行模型进行一次 Decode 步进。
> 2.  **环境隔离**：在两次运行之间重置模型状态，确保 KV Cache 为空或一致。
> 3.  **日志对齐与对比**：读取上一步修改代码生成的 `qwen3_torch_debug.json` 和 `python_vm_debug.json`，针对相同的 Tag 进行数值对比（均值、和、前16位数值），并高亮显示第一个出现误差的地方。
> 
> 请将此脚本保存在与 `python_vm.py`, `qwen3.py`, `scheduler.py` 同级的目录下运行。
> 
> ### 文件: `debug_compare.py`
> 
> 
> import json
> import os
> import sys
> from pathlib import Path
> import numpy as np
> import torch
> from transformers import AutoTokenizer
> 
> # 引入原始项目的依赖
> from megakernels.dispatch import (
>     make_schedule_builder,
>     make_pyvm_interpreter,
> )
> from megakernels.generators_qwen3 import (
>     PyTorchGenerator,
>     PyVM_Generator,
> )
> from megakernels.qwen3 import Qwen3ForCausalLM
> from megakernels.model_types import BatchState, ExtraModelConfig
> from megakernels.scheduler import (
>     assign_to_sms,
>     tensorize_instructions,
> )
> 
> # 配置类
> class DebugConfig:
>     # 请根据实际路径修改模型地址
>     model: str = "/home/ziming/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B" 
>     device: str = "cuda:0"
>     setting: str = "qwen3"
>     sched: str = "rr"
>     memory_fraction: float | None = None
>     
>     # 调试输出文件路径 (对应 python_vm.py 和 qwen3.py 中的定义)
>     torch_log: str = "qwen3_torch_debug.json"
>     vm_log: str = "python_vm_debug.json"
> 
> def run_generation_step(mode: str, config: DebugConfig):
>     """
>     运行一次生成步骤 (Prefill + 1 Decode step)
>     """
>     print(f"\n[INFO] Starting execution in mode: {mode}...")
>     
>     torch.cuda.set_device(config.device)
>     
>     # 1. 清理旧日志 (根据模式清理对应的)
>     target_log = config.torch_log if mode == "torch" else config.vm_log
>     if os.path.exists(target_log):
>         os.remove(target_log)
> 
>     # 2. 加载模型 (每次重新加载以确保状态纯净)
>     print("  Loading model...")
>     extra_config = ExtraModelConfig(
>         interleave_rope=True,
>         max_batch_size=1,
>     )
>     model = Qwen3ForCausalLM.from_pretrained(
>         config.model, device=config.device, extra_config=extra_config
>     )
>     tokenizer = AutoTokenizer.from_pretrained(config.model)
> 
>     # 3. 准备输入
>     prompt = "Hello"
>     messages = [{"role": "user", "content": prompt}]
>     tok_inp = tokenizer.apply_chat_template(
>         messages, tokenize=False, add_generation_prompt=True
>     )
>     input_ids = tokenizer(tok_inp, return_tensors="pt", add_special_tokens=False)[
>         "input_ids"
>     ].to(model.device)
>     
>     prompt_len = input_ids.shape[-1]
>     position_ids = torch.arange(prompt_len).to(model.device)
> 
>     # 4. 初始化生成器
>     gen = None
>     if mode == "torch":
>         gen = PyTorchGenerator(model)
>     elif mode == "pyvm":
>         # 设置调度器
>         schedule_builder = make_schedule_builder(config.setting)
>         schedule = schedule_builder.build(model)
>         assigned_to_sms = assign_to_sms(
>             config.sched, schedule=schedule, memory_fraction=config.memory_fraction
>         )
>         tensorize_instructions(schedule.globs, assigned_to_sms)
>         
>         # 设置解释器
>         interpreter = make_pyvm_interpreter(config.setting)
>         gen = PyVM_Generator(model, interpreter, schedule)
> 
>     # 5. 运行 Prefill
>     print("  Running Prefill...")
>     prefill_inp = BatchState(
>         input_ids=input_ids,
>         position_ids=position_ids,
>     )
>     prefill_output: BatchState = model(prefill_inp)
>     
>     # 准备 Decode 输入 (1个token)
>     new_input_token = prefill_output.output_ids[:, -1:]
>     output_tokens = torch.zeros(
>         1, 128, device=model.device, dtype=torch.long
>     )
>     output_tokens[:, 0] = new_input_token
> 
>     # 6. 运行 Decode (仅跑1步以触发 log)
>     print("  Running Decode (1 step)...")
>     gen.generate_with_eos(
>         output_tokens=output_tokens,
>         prompt_len=prompt_len,
>         ntok=1, # 只生成1个token
>         eos_token_ids=[151643], # 随便填一个EOS
>         eos_token_check_interval=1,
>     )
>     
>     print(f"  Finished {mode} execution. Log generated at {target_log}")
>     
>     # 显式清理显存
>     del model
>     del gen
>     torch.cuda.empty_cache()
> 
> def load_log(filepath):
>     data = []
>     if not os.path.exists(filepath):
>         return {}
>     with open(filepath, 'r') as f:
>         for line in f:
>             if line.strip():
>                 try:
>                     item = json.loads(line)
>                     data.append(item)
>                 except:
>                     pass
>     # 转为字典，key为tag
>     return {item['tag']: item for item in data}
> 
> def compare_logs(config: DebugConfig):
>     print("\n" + "="*60)
>     print("COMPARISON REPORT")
>     print("="*60)
> 
>     torch_data = load_log(config.torch_log)
>     vm_data = load_log(config.vm_log)
> 
>     if not torch_data:
>         print("[ERROR] No Torch logs found.")
>         return
>     if not vm_data:
>         print("[ERROR] No VM logs found.")
>         return
> 
>     # 按 Torch 的顺序进行对比
>     all_match = True
>     
>     # 定义比较的 Tag 列表 (根据 python_vm.py 中添加的 log tag)
>     # 注意：顺序很重要，按计算图流向
>     ordered_tags = [
>         "layer_0_attn_post_ln",
>         "layer_0_block_0_q_proj_raw",
>         "layer_0_block_0_k_proj_raw",
>         "layer_0_block_0_q_norm_out",
>         "layer_0_block_0_k_norm_out",
>         "layer_0_block_0_q_rope_out",
>         "layer_0_block_0_k_rope_out",
>         "layer_0_head_grp_0_attn_out", # 部分 Attention 输出
>         "layer_0_attn_out_buffer",     # 完整 Attention 输出
>         "layer_0_hidden_states_post_oproj_partial", # O-Proj 残差后
>         "layer_0_mlp_post_ln"
>     ]
> 
>     for tag in ordered_tags:
>         # 如果 Log 中没有这个 Tag，尝试模糊匹配（因为 block index 可能是 0）
>         t_item = torch_data.get(tag)
>         v_item = vm_data.get(tag)
>         
>         if not t_item:
>             # 可能是部分日志未触发
>             continue
> 
>         print(f"Checking: {tag:<45}", end="")
> 
>         if not v_item:
>             print(f" -> [MISSING IN VM]")
>             all_match = False
>             continue
> 
>         # 对比 Sum 和 Mean
>         t_sum, v_sum = t_item['sum'], v_item['sum']
>         t_mean, v_mean = t_item['mean'], v_item['mean']
>         
>         # 简单的相对误差计算
>         diff_sum = abs(t_sum - v_sum)
>         rel_err = diff_sum / (abs(t_sum) + 1e-6)
>         
>         threshold = 1e-3 # 宽松一点，fp16/bf16 误差
> 
>         if rel_err > threshold:
>             print(f" -> [FAIL]")
>             print(f"    Torch Sum: {t_sum:.6f}, Mean: {t_mean:.6f}")
>             print(f"    VM    Sum: {v_sum:.6f}, Mean: {v_mean:.6f}")
>             print(f"    Diff  Sum: {diff_sum:.6f}, Rel Err: {rel_err:.6f}")
>             
>             print(f"    Torch Head: {t_item['head_values'][:4]}...")
>             print(f"    VM    Head: {v_item['head_values'][:4]}...")
>             
>             print("\n[CONCLUSION] Divergence detected at:", tag)
>             print("Check python_vm.py logic corresponding to this operator.")
>             all_match = False
>             break # 找到第一个错误后停止，因为后面的都会错
>         else:
>             print(f" -> [PASS]")
> 
>     if all_match:
>         print("\n[SUCCESS] All monitored tensors match within threshold.")
>     else:
>         print("\n[FAIL] Comparison failed.")
> 
> if __name__ == "__main__":
>     cfg = DebugConfig()
>     
>     # 如果参数中有路径，覆盖默认模型路径
>     if len(sys.argv) > 1:
>         cfg.model = sys.argv[1]
> 
>     try:
>         # 1. 运行 Torch 获取基准
>         run_generation_step("torch", cfg)
>         
>         # 2. 运行 VM 获取测试结果
>         run_generation_step("pyvm", cfg)
>         
>         # 3. 对比
>         compare_logs(cfg)
>         
>     except Exception as e:
>         print(f"\n[ERROR] Execution failed: {e}")
>         import traceback
>         traceback.print_exc()
> 
> 
> ### 使用方法
> 
> 1.  确保你已经按照上一步的指示，修改了 `python_vm.py` 和 `qwen3.py` 并添加了 `log_debug_data` 函数。
> 2.  修改脚本中的 `DebugConfig` 类中的 `model` 路径为你本地的模型路径。
> 3.  在终端运行：
>     
>     python debug_compare.py
>     
>     或者传入模型路径作为参数：
>     
>     python debug_compare.py /path/to/your/Qwen3-0.6B
>     
> 
> ### 预期输出
> 
> 脚本将按顺序执行：
> 1.  QKV Projection 原始值
> 2.  Q/K Norm 输出
> 3.  RoPE 输出
> 4.  Attention 输出
> 
> 通过观察 `[FAIL]` 出现的位置，你可以精确定位是 `python_vm.py` 中的哪一行代码（例如是否是 Norm 的切片逻辑有问题，还是 RoPE 的计算逻辑有问题）导致了数值偏差。

-------------------

这里给出的插桩结果有几个问题：1.pyvm模式没有k_proj、k_norm、k_rope的结果。2.pyvm模式和torch模式的插桩命名不统一。3.pyvm的mlp_post_ln的结果重复输出。
修复上述问题，并检查是否存在其他问题，给出修复后的python_vm.py和qwen3.py文件
```

**assistant**: 

```
针对您提出的问题，我修复了两个文件中的插桩逻辑：

1.  **修复 K 向量漏记问题**：在 `python_vm.py` 的 `layer_norm_matvec_rope_append` 中添加了针对 K 起始位置的检测逻辑，确保能记录到 K 的 Projection、Norm 和 RoPE 结果。
2.  **统一命名与对齐**：将 `torch` 和 `vm` 的 tag 命名完全统一（例如 `layer_0_q_proj_raw`, `layer_0_k_norm_out`），并且确保 Torch 侧记录数据时切取前 16 个元素，与 VM 的 Block 大小对齐。
3.  **消除重复日志**：在 `layer_norm_double_matvec_silu` (MLP) 中添加了 `block_idx` 检测，防止因多指令并行导致同一个 Layer Norm 结果被重复记录。
4.  **调试信息增强**：在关键位置（如 Attention Output）确保记录的数据是可比对的。

请使用以下文件替换原有文件。

### 文件 1: `python_vm.py` (修复版)


import math
import json
import os

import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor

from megakernels.demos.qwen3_0_6b.instructions import (
    AttentionReduction,
    DownProjResidual,
    Globals,
    LayerNorm_QKV_MatVecRopeAppend,
    LayerNormDoubleMatVecSiLU,
    O_ProjResidual,
    PartialAttention,
    RMS_LM_Head,
)

from megakernels.qwen3 import (
    apply_rotary_pos_emb_interleaved,
)

# ================= DEBUG UTILS =================
DEBUG_FILE = "python_vm_debug.json"
# 如果是第一次加载模块，尝试清理旧日志，防止追加混淆（实际运行通常由 runner 清理）
# if os.path.exists(DEBUG_FILE):
#     os.remove(DEBUG_FILE)

def log_debug_data(tag, tensor):
    """记录 Tensor 的关键统计信息到 JSON 文件"""
    if tensor is None:
        return
    
    # 确保数据在 CPU 上并转为 float
    t_flat = tensor.detach().float().flatten().cpu()
    
    # 为了避免数据量过大，我们只记录：形状、均值、和、以及前 16 个值
    # 这 16 个值对应于 VM 中的 block_size，方便直接对比
    data = {
        "tag": tag,
        "shape": list(tensor.shape),
        "mean": float(t_flat.mean().item()),
        "sum": float(t_flat.sum().item()),
        "head_values": t_flat[:16].tolist()
    }
    
    with open(DEBUG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
# ===============================================

def get_start_end(block_size: int, block_idx: int):
    start = block_size * block_idx
    end = start + block_size
    return start, end


def matvec(
    mat: Tensor,
    vec: Tensor,
    block_size: int,
    block_idx: int,
    reduce: bool = False,
    reduction_size: int = 0,
    reduction_idx: int = 0,
):
    start, end = get_start_end(block_size, block_idx)
    if reduce:
        red_start, red_end = get_start_end(reduction_size, reduction_idx)
        mat = mat[start:end, red_start:red_end]
        vec = vec[red_start:red_end]
    else:
        mat = mat[start:end]

    out = einsum(mat, vec, "o i, i -> o")
    return out, start, end


def rms_norm(inp: Tensor, weight: Tensor, eps: float):
    input_dtype = inp.dtype
    inp = inp.to(torch.float32)
    variance = inp.pow(2).mean(-1, keepdim=True)
    inp = inp * torch.rsqrt(variance + eps)

    return weight * inp.to(input_dtype)


def matvec_with_residual(
    mat: Tensor,
    vec: Tensor,
    residual: Tensor,
    block_size: int,
    start_block_idx: int,
    end_block_idx: int,
    reduction_size: int,
    reduction_block_idx: int,
):
    for block_idx in range(start_block_idx, end_block_idx):
        matvec_out, start, end = matvec(
            mat=mat,
            vec=vec,
            block_size=block_size,
            block_idx=block_idx,
            reduce=True,
            reduction_size=reduction_size,
            reduction_idx=reduction_block_idx,
        )

        residual[start:end] += matvec_out.to(residual.dtype)


def o_proj_residual(globals: Globals, instruction: O_ProjResidual):
    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == globals.num_attention_heads

    assert instruction.start_block_idx == instruction.end_block_idx - 1
    assert instruction.reduction_block_idx == 0
    
    # DEBUG: 记录进入 O-Proj 的数据 (即 Attention 的完整输出)
    # 仅在第0层第0块记录
    if instruction.layer_idx == 0 and instruction.start_block_idx == 0:
         log_debug_data(f"layer_{instruction.layer_idx}_attn_out_buffer", globals.attn_out)

    with open("output_pyvm.txt", "a", encoding="utf-8") as f:
        f.writelines("layer idx: " + instruction.layer_idx.__str__() + "\n")
        f.writelines(globals.attn_out.__str__() + "\n")

    matvec_with_residual(
        mat=globals.o_proj_weights[instruction.layer_idx],
        vec=globals.attn_out,
        residual=globals.hidden_states,
        block_size=globals.o_proj_block_size,
        start_block_idx=instruction.start_block_idx,
        end_block_idx=instruction.end_block_idx,
        reduction_size=globals.matvec_reduction_size,
        reduction_block_idx=instruction.reduction_block_idx,
    )

    # Barrier update
    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += instruction.end_block_idx - instruction.start_block_idx
    
    # DEBUG: 记录 O-Proj 之后的 Residual 状态 (部分)
    if instruction.layer_idx == 0 and instruction.start_block_idx == 0:
        # 这里的 globals.hidden_states 是被 o_proj 累加后的结果
        log_debug_data(f"layer_{instruction.layer_idx}_hidden_states_post_oproj_partial", globals.hidden_states[0:16])


def down_proj_residual(globals: Globals, instruction: DownProjResidual):
    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == 192  # 8192 / 16

    matvec_with_residual(
        mat=globals.down_proj_weights[instruction.layer_idx],
        vec=globals.silu_out,
        residual=globals.hidden_states,
        block_size=globals.down_proj_block_size,
        start_block_idx=instruction.start_block_idx,
        end_block_idx=instruction.end_block_idx,
        reduction_size=globals.matvec_reduction_size,
        reduction_block_idx=instruction.reduction_block_idx,
    )

    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += instruction.end_block_idx - instruction.start_block_idx


def layer_norm_double_matvec_silu(
    globals: Globals, instruction: LayerNormDoubleMatVecSiLU
):
    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == 64

    post_ln = rms_norm(
        inp=globals.hidden_states,
        weight=globals.mlp_ln_weights[instruction.layer_idx],
        eps=globals.rms_norm_eps,
    )
    
    # DEBUG: 记录 MLP Pre-LN 的结果
    # 注意：这个指令会被分片执行多次，我们只在处理第0个block时记录一次
    if instruction.layer_idx == 0:
        first_block = instruction.block_idxs[0]
        # up_gate_proj_block_size = 16. Check if this instruction covers the start.
        if first_block == 0:
             log_debug_data(f"layer_{instruction.layer_idx}_mlp_pre_ln_out", post_ln)

    block_size = globals.up_gate_proj_block_size

    barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]

    for block_idx in instruction.block_idxs:
        start, end = get_start_end(block_size, block_idx)

        up_matvec, start, end = matvec(
            mat=globals.up_proj_weights[instruction.layer_idx],
            vec=post_ln,
            block_size=block_size,
            block_idx=block_idx,
        )

        gate_matvec, _, _ = matvec(
            mat=globals.gate_proj_weights[instruction.layer_idx],
            vec=post_ln,
            block_size=block_size,
            block_idx=block_idx,
        )

        post_silu = F.silu(gate_matvec) * up_matvec

        globals.silu_out[start:end] = post_silu

        barriers[0] += 1


def layer_norm_matvec_rope_append(
    globals: Globals, instruction: LayerNorm_QKV_MatVecRopeAppend
):
    layer_idx = instruction.layer_idx

    # Barrier check (保持不变)
    if layer_idx > 0:
        op_barriers = globals.barriers[layer_idx - 1, instruction.prev_opcode() - 1]
        assert op_barriers[0] == 64

    # Pre-Attention LayerNorm (保持不变)
    post_ln = rms_norm(
        inp=globals.hidden_states,
        weight=globals.attn_ln_weights[layer_idx],
        eps=globals.rms_norm_eps,
    )

    # DEBUG: 记录 Attention Pre-LN
    # 仅在处理这一层的第一个 output block 时记录，避免重复
    if layer_idx == 0 and instruction.start_output_block_idx == 0:
        log_debug_data(f"layer_{layer_idx}_attn_pre_ln_out", post_ln)

    pos_id = globals.pos_id

    k_start = globals.num_attention_heads * globals.head_dim
    v_start = k_start + globals.num_kv_heads * globals.head_dim

    barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]

    for block_idx in range(
        instruction.start_output_block_idx, instruction.end_output_block_idx
    ):
        start, end = get_start_end(globals.qkv_block_size, block_idx)

        if start < k_start:
            mode = "q"
        elif start < v_start:
            mode = "k"
        else:
            mode = "v"

        # 1. QKV 投影计算 (保持不变)
        matmul_output = einsum(
            globals.qkv_proj_weights[layer_idx][start:end],
            post_ln,
            "o i, i -> o",
        )

        out = matmul_output
        
        # DEBUG: 记录 Projection 原始输出
        # 判断是否是 Q 的第一个 Block (start == 0)
        # 判断是否是 K 的第一个 Block (start == k_start)
        if layer_idx == 0:
            if start == 0:
                log_debug_data(f"layer_{layer_idx}_q_proj_raw", out)
            elif start == k_start:
                log_debug_data(f"layer_{layer_idx}_k_proj_raw", out)

        # ================= 修改开始 =================
        if mode in ["q", "k"]:
            full_weight = None

            # 1. 确定使用哪个权重张量
            if mode == "q":
                full_weight = globals.q_norm_weights[layer_idx].view(-1)
                # 计算相对于权重张量的偏移量
                weight_len = full_weight.shape[0]
                idx_start = start % weight_len
                idx_end = idx_start + (end - start)

            elif mode == "k":
                full_weight = globals.k_norm_weights[layer_idx].view(-1)
                weight_len = full_weight.shape[0]
                # K 的全局 start 需要先减去 k_start 得到 K 区域内的相对偏移
                rel_start = start - k_start
                idx_start = rel_start % weight_len
                idx_end = idx_start + (end - start)

            # 2. 安全切片处理
            if idx_end > full_weight.shape[0]:
                part1 = full_weight[idx_start:]
                part2 = full_weight[: idx_end - full_weight.shape[0]]
                norm_weight = torch.cat([part1, part2], dim=0)
            else:
                norm_weight = full_weight[idx_start:idx_end]

            # 3. 执行 Norm
            out = rms_norm(inp=out, weight=norm_weight, eps=globals.rms_norm_eps)
            
            # DEBUG: 记录 Norm 后的结果
            if layer_idx == 0:
                if start == 0:
                    log_debug_data(f"layer_{layer_idx}_q_norm_out", out)
                elif start == k_start:
                    log_debug_data(f"layer_{layer_idx}_k_norm_out", out)
        # ================= 修改结束 =================

        if mode in ["q", "k"]:
            full_head = torch.zeros(
                1,
                globals.head_dim,
                device=globals.hidden_states.device,
                dtype=out.dtype,
            )
            head_segment = start % globals.head_dim
            full_head_start = head_segment
            full_head_end = full_head_start + (end - start)

            # 将经过 Norm 的数据填入 full_head
            full_head[:, full_head_start:full_head_end] = out

            # 2. RoPE 计算 (保持不变)
            full_head_with_rope, _ = apply_rotary_pos_emb_interleaved(
                q=full_head,
                k=full_head,
                cos=globals.rope_cos[pos_id],
                sin=globals.rope_sin[pos_id],
                unsqueeze_dim=0,
            )
            out = full_head_with_rope[:, full_head_start:full_head_end].view(-1)
            
            # DEBUG: 记录 RoPE 后的结果
            if layer_idx == 0:
                if start == 0:
                    log_debug_data(f"layer_{layer_idx}_q_rope_out", out)
                elif start == k_start:
                    log_debug_data(f"layer_{layer_idx}_k_rope_out", out)

        # Store results (保持不变)
        match mode:
            case "q":
                globals.post_ln_rope_q[start:end] = out
            case "k":
                start_in_k = start - k_start
                end_in_k = end - k_start
                globals.k_cache[layer_idx, :, pos_id].view(-1)[
                    start_in_k:end_in_k
                ] = out
            case "v":
                start_in_v = start - v_start
                end_in_v = end - v_start
                globals.v_cache[layer_idx, :, pos_id].view(-1)[
                    start_in_v:end_in_v
                ] = out

        barriers[block_idx // 8] += 1


def rms_lm_head(globals: Globals, instruction: RMS_LM_Head):
    op_barriers = globals.barriers[
        globals.num_hidden_layers - 1, instruction.prev_opcode() - 1
    ]
    assert op_barriers[0] == 64

    post_ln = rms_norm(
        inp=globals.hidden_states,
        weight=globals.lm_head_norm_weights,
        eps=globals.rms_norm_eps,
    )

    for block_idx in range(
        instruction.start_output_block_idx, instruction.end_output_block_idx
    ):
        start, end = get_start_end(globals.lm_head_block_size, block_idx)

        matmul_output = einsum(
            globals.lm_head_weights[start:end],
            post_ln,
            "o i, i -> o",
        )

        globals.logits[start:end] = matmul_output


def partial_attention(globals: Globals, instruction: PartialAttention):
    gqa_ratio = globals.num_attention_heads // globals.num_kv_heads

    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    for i in range(gqa_ratio):
        assert op_barriers[instruction.kv_head_idx * gqa_ratio + i] == 8
    assert op_barriers[globals.num_attention_heads + instruction.kv_head_idx] == 8
    assert (
        op_barriers[
            globals.num_attention_heads + globals.num_kv_heads + instruction.kv_head_idx
        ]
        == 8
    )

    kv_block_size = globals.attn_kv_block_size
    seq_len = globals.pos_id + 1
    layer_idx = instruction.layer_idx
    kv_head_idx = instruction.kv_head_idx

    total_blocks = math.ceil(seq_len / kv_block_size)
    blocks_per_partial = math.ceil(total_blocks / instruction.num_partials)

    start_block = instruction.partial_idx * blocks_per_partial
    end_block = min(start_block + blocks_per_partial, total_blocks)

    start_token = start_block * kv_block_size
    end_token = min(end_block * kv_block_size, seq_len)

    k = globals.k_cache[layer_idx, 0, start_token:end_token, kv_head_idx]
    v = globals.v_cache[layer_idx, 0, start_token:end_token, kv_head_idx]

    head_start = kv_head_idx * gqa_ratio
    head_end = head_start + gqa_ratio

    q = globals.post_ln_rope_q.view(globals.num_attention_heads, -1)[
        head_start:head_end
    ]

    qk = einsum(q.float(), k.float(), "h i, k i -> h k")
    scaled_qk = qk * globals.attn_scale

    # casting the output of the softmax to 16-bit causes small numerical differences
    softmax = torch.softmax(scaled_qk, dim=-1)

    # lse = torch.logsumexp(scaled_qk, dim=-1)
    lse = torch.log2(torch.sum(torch.exp(scaled_qk), dim=-1))

    out = einsum(softmax.float(), v.float(), "h k, k o -> h o")
    # print(out.shape)
    
    # DEBUG: 记录 Partial Attention 计算结果 (Head 0)
    if instruction.layer_idx == 0 and kv_head_idx == 0 and instruction.partial_idx == 0:
         # out 的 shape 通常是 [gqa_ratio, head_dim] -> [8, 128]
         # 我们取第一个 head 的前 16 个值
         log_debug_data(f"layer_{layer_idx}_attn_out", out[0, :16])

    if globals.skip_attn_reduction:
        globals.attn_out.view(globals.num_attention_heads, -1)[
            head_start:head_end, :
        ] = out
        barriers = globals.barriers[
            instruction.layer_idx, AttentionReduction.opcode() - 1
        ]
        barriers[0] += head_end - head_start

    else:
        globals.attn_lse_intermediates[head_start:head_end, instruction.partial_idx] = (
            lse
        )
        globals.attn_out_intermediates[head_start:head_end, instruction.partial_idx] = (
            out
        )

        # Barrier update
        barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
        barriers[head_start:head_end] += 1


def attention_reduction(globals: Globals, instruction: AttentionReduction):
    head_start_idx = instruction.head_start_idx

    # Barrier check
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[head_start_idx] == instruction.num_partials

    indices_to_reduce = torch.tensor(
        instruction.reduction_list,
        dtype=torch.long,
        device=globals.hidden_states.device,
    )

    lses = globals.attn_lse_intermediates[
        head_start_idx : head_start_idx + globals.attn_reduction_size, indices_to_reduce
    ]
    outs = globals.attn_out_intermediates[
        head_start_idx : head_start_idx + globals.attn_reduction_size, indices_to_reduce
    ]

    max_lse = torch.max(lses, dim=-1, keepdim=True).values

    # adjusted_factors = (lses - max_lse).exp()
    adjusted_factors = (lses - max_lse).exp2()
    new_denominator = adjusted_factors.sum(dim=-1, keepdim=True)

    reduced = (outs * adjusted_factors.unsqueeze(-1)).sum(dim=1) / new_denominator

    if instruction.is_terminal:
        globals.attn_out.view(globals.num_attention_heads, -1)[
            head_start_idx : head_start_idx + globals.attn_reduction_size
        ] = reduced
    else:
        new_lse = new_denominator.log()
        output_slot = instruction.output_partial_idx
        globals.attn_lse_intermediates[
            head_start_idx : head_start_idx + globals.attn_reduction_size, output_slot
        ] = new_lse
        globals.attn_out_intermediates[
            head_start_idx : head_start_idx + globals.attn_reduction_size, output_slot
        ] = reduced

    # Barrier update
    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += globals.attn_reduction_size  # the dumb way


INSTRUCTION_TO_SOLVER = {
    O_ProjResidual: o_proj_residual,
    DownProjResidual: down_proj_residual,
    LayerNormDoubleMatVecSiLU: layer_norm_double_matvec_silu,
    LayerNorm_QKV_MatVecRopeAppend: layer_norm_matvec_rope_append,
    RMS_LM_Head: rms_lm_head,
    PartialAttention: partial_attention,
    AttentionReduction: attention_reduction,
}


### 文件 2: `qwen3.py` (修复版)


from dataclasses import dataclass
from pathlib import Path
import json
import os

import huggingface_hub
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from einops import rearrange
from torch import Tensor, nn
from torch.distributed import _functional_collectives as funcol
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from megakernels.model_types import (
    BatchState,
    DeviceType,
    ExtraModelConfig,
)
from megakernels.utils import (
    load_safetensors_repo,
)

# ================= DEBUG UTILS =================
DEBUG_FILE = "qwen3_torch_debug.json"
# 逻辑由 runner 控制删除
# if os.path.exists(DEBUG_FILE):
#     os.remove(DEBUG_FILE)

def log_debug_data(tag, tensor):
    """记录 Tensor 的关键统计信息到 JSON 文件"""
    if tensor is None:
        return
    
    # 统一：切取前 16 个元素进行对比，匹配 VM 的 block_size
    t_flat = tensor.detach().float().flatten().cpu()
    if t_flat.numel() > 16:
        t_flat = t_flat[:16]
        
    data = {
        "tag": tag,
        "shape": list(tensor.shape),
        "mean": float(t_flat.mean().item()),
        "sum": float(t_flat.sum().item()),
        "head_values": t_flat.tolist()
    }
    with open(DEBUG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
# ===============================================

KV_Cache = tuple[Tensor, Tensor]


class RMSNorm(nn.Module):
    def __init__(self, config: Qwen3Config, dim):
        """
        Taken from Qwen3RMSNorm.
        """
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.config.rms_norm_eps)

        if self.weight is not None:
            return self.weight * hidden_states.to(input_dtype)
        else:
            return hidden_states.to(input_dtype)


def all_gather(x: Tensor, extra_config: ExtraModelConfig):
    if extra_config.tp_size == 1:
        return x

    assert extra_config.tp_group is not None

    # funcol + no compile + tp > 1 and pp > 1 leads to some nccl crash
    if extra_config.torch_compile:
        return funcol.all_gather_tensor(x, gather_dim=0, group=extra_config.tp_group)

    out = torch.empty(
        (extra_config.tp_size * x.shape[0], *x.shape[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    torch.distributed.all_gather_into_tensor(out, x, group=extra_config.tp_group)
    return out


def reduce_scatter(x: Tensor, extra_config: ExtraModelConfig):
    if extra_config.tp_size == 1:
        return x

    assert extra_config.tp_group is not None

    # funcol + no compile + tp > 1 and pp > 1 leads to some nccl crash
    if extra_config.torch_compile:
        return funcol.reduce_scatter_tensor(
            x, reduceOp="sum", scatter_dim=0, group=extra_config.tp_group
        )

    out = torch.empty(
        (x.shape[0] // extra_config.tp_size, *x.shape[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    torch.distributed.reduce_scatter_tensor(out, x, group=extra_config.tp_group)
    return out


def attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    kv_cache: KV_Cache,
    position_ids: Tensor,
    seq_len: int,
) -> Tensor:
    bsz, new_tok_seq_len = query_states.shape[:2]

    k_cache, v_cache = kv_cache

    k_cache[:, position_ids] = key_states
    v_cache[:, position_ids] = value_states

    def shape_for_sdpa(x: Tensor):
        return rearrange(x, "b l h d -> b h l d")

    def unshape_for_sdpa(x: Tensor):
        return rearrange(x, "b l h d -> b l h d")

    if new_tok_seq_len > 1:
        k_for_sdpa = shape_for_sdpa(key_states)
        v_for_sdpa = shape_for_sdpa(value_states)

        q_for_sdpa = shape_for_sdpa(query_states)

        # assume running prefill from scratch
        attn_output = F.scaled_dot_product_attention(
            q_for_sdpa, k_for_sdpa, v_for_sdpa, is_causal=True, enable_gqa=True
        )
    else:
        # decode
        k_for_sdpa = shape_for_sdpa(k_cache[:, :seq_len])
        v_for_sdpa = shape_for_sdpa(v_cache[:, :seq_len])
        q_for_sdpa = shape_for_sdpa(query_states)

        attn_output = F.scaled_dot_product_attention(
            q_for_sdpa, k_for_sdpa, v_for_sdpa, is_causal=False, enable_gqa=True
        )
    reshaped_attn_output = unshape_for_sdpa(attn_output)
    return reshaped_attn_output


def rotate_half_interleaved(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    new_x1 = -x2
    new_x2 = x1

    stacked = torch.stack((new_x1, new_x2), dim=-1)
    return stacked.view_as(x)


def apply_rotary_pos_emb_interleaved(
    q, k, cos, sin, position_ids=None, unsqueeze_dim=1
):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


class Qwen3Attention(nn.Module):
    """Qwen3 Attention with QK-Norm"""

    def __init__(
        self, config: Qwen3Config, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config, dim=config.hidden_size)

        self.tp_size = extra_config.tp_size or 1

        assert config.num_attention_heads % self.tp_size == 0

        self.head_dim = config.head_dim

        assert self.config.num_attention_heads % self.tp_size == 0
        assert (
            self.config.num_key_value_heads % self.tp_size == 0
            or self.config.num_key_value_heads == 1
        )

        self.num_attention_heads = config.num_attention_heads // self.tp_size
        self.num_kv_heads = (
            config.num_key_value_heads // self.tp_size
            if config.num_key_value_heads > 1
            else 1
        )

        self.q_proj = nn.Linear(
            self.config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )

        self.q_norm = RMSNorm(config, dim=self.head_dim)
        self.k_norm = RMSNorm(config, dim=self.head_dim)

        self.kv_cache: KV_Cache | None = None

    def forward(
        self,
        batch_state: BatchState,
    ):
        assert batch_state.hidden_states is not None
        assert batch_state.position_embeddings is not None
        assert batch_state.position_ids is not None
        assert self.kv_cache is not None
        assert batch_state.seq_len is not None

        inp = batch_state.hidden_states
        residual = inp

        hidden_states = self.input_layernorm(inp)
        
        # DEBUG: Log Pre-LN (Input to QKV Proj)
        if self.layer_idx == 0:
            log_debug_data(f"layer_{self.layer_idx}_attn_pre_ln_out", hidden_states)

        hidden_states = all_gather(hidden_states, self.extra_config)
        bsz, seq_len = hidden_states.shape[:2]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, seq_len, self.num_attention_heads, -1)
        key_states = key_states.view(bsz, seq_len, self.num_kv_heads, -1)
        value_states = value_states.view(bsz, seq_len, self.num_kv_heads, -1)
        
        # DEBUG: Log Raw Projections (Slice 0:16)
        if self.layer_idx == 0:
             # [1, 1, heads, dim] -> [1, 1, 0, :16]
             log_debug_data(f"layer_{self.layer_idx}_q_proj_raw", query_states[0, 0, 0, :16]) 
             log_debug_data(f"layer_{self.layer_idx}_k_proj_raw", key_states[0, 0, 0, :16])

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # DEBUG: Log Normed Projections
        if self.layer_idx == 0:
             log_debug_data(f"layer_{self.layer_idx}_q_norm_out", query_states[0, 0, 0, :16])
             log_debug_data(f"layer_{self.layer_idx}_k_norm_out", key_states[0, 0, 0, :16])

        cos, sin = batch_state.position_embeddings

        dtype = query_states.dtype

        if self.extra_config.interleave_rope:
            rope_fn = apply_rotary_pos_emb_interleaved
        else:
            rope_fn = apply_rotary_pos_emb

        query_states, key_states = rope_fn(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=-2,  # unsqueeze dim = head dim on q/k
        )

        query_states = query_states.to(dtype)
        key_states = key_states.to(dtype)
        
        # DEBUG: Log RoPE Output
        if self.layer_idx == 0:
             log_debug_data(f"layer_{self.layer_idx}_q_rope_out", query_states[0, 0, 0, :16])
             log_debug_data(f"layer_{self.layer_idx}_k_rope_out", key_states[0, 0, 0, :16])

        raw_attn_output = attention(
            query_states,
            key_states,
            value_states,
            self.kv_cache,
            batch_state.position_ids,
            seq_len=batch_state.seq_len,
        )
        # print(raw_attn_output.shape)

        attn_output = raw_attn_output.reshape(bsz, seq_len, -1)

        # print(attn_output.shape)
        
        # DEBUG: Log Attention Output (取前16个值，对应VM的Partial计算结果)
        # 注意：raw_attn_output shape is [1, 1, heads, dim]
        if self.layer_idx == 0:
             log_debug_data(f"layer_{self.layer_idx}_attn_out", raw_attn_output[0, 0, 0, :16])
        
        # DEBUG: Log Full Attention Buffer (进入 O-Proj 前)
        # 对应 VM 的 attn_out_buffer
        if self.layer_idx == 0:
             log_debug_data(f"layer_{self.layer_idx}_attn_out_buffer", attn_output)

        o_proj = self.o_proj(attn_output)

        o_proj = reduce_scatter(o_proj, self.extra_config)

        with_residual = residual + o_proj
        
        # DEBUG: Log Output of O-Proj Block
        if self.layer_idx == 0:
             log_debug_data(f"layer_{self.layer_idx}_hidden_states_post_oproj_partial", with_residual)

        batch_state.hidden_states = with_residual
        return batch_state


class Qwen3MLP(nn.Module):
    def __init__(
        self, config: Qwen3Config, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.tp_size = extra_config.tp_size
        assert self.config.intermediate_size % self.tp_size == 0
        self.intermediate_size = self.config.intermediate_size // self.tp_size

        self.up_proj = nn.Linear(
            self.config.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            config.hidden_size,
            bias=False,
        )

        self.input_layernorm = RMSNorm(config, dim=config.hidden_size)

    def forward(
        self,
        batch_state: BatchState,
    ):
        inp = batch_state.hidden_states
        assert inp is not None
        hidden_states = self.input_layernorm(inp)

        hidden_states = all_gather(hidden_states, self.extra_config)
        
        # DEBUG: MLP Pre Norm (Post LN before Proj)
        if self.layer_idx == 0:
            log_debug_data(f"layer_{self.layer_idx}_mlp_pre_ln_out", hidden_states)

        up = self.up_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        prod = F.silu(gate) * up
        down = self.down_proj(prod)

        down = reduce_scatter(down, self.extra_config)

        with_residual = inp + down

        batch_state.hidden_states = with_residual
        return batch_state


class Qwen3Block(nn.Module):
    def __init__(
        self, config: Qwen3Config, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.self_attn = Qwen3Attention(config, extra_config, layer_idx)
        self.mlp = Qwen3MLP(config, extra_config, layer_idx)

    def forward(self, batch_state: BatchState):
        out = self.self_attn(batch_state)
        out = self.mlp(out)
        return out


class Qwen3LMHead(nn.Module):
    def __init__(self, config: Qwen3Config, extra_config: ExtraModelConfig):
        super().__init__()
        self.config = config
        self.extra_config = extra_config

        self.input_norm = RMSNorm(config, dim=config.hidden_size)

        self.tp_size = extra_config.tp_size or 1

        assert config.vocab_size % self.tp_size == 0
        head_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, head_size, bias=False)

    def forward(self, batch_state: BatchState):
        assert batch_state.hidden_states is not None

        hidden_states = batch_state.hidden_states

        if self.extra_config.tp_size > 1:
            hidden_states = all_gather(hidden_states, self.extra_config)

        hidden_states = self.input_norm(hidden_states)

        logits = self.lm_head(hidden_states)

        next_token_ids = logits.argmax(dim=-1)

        if self.tp_size > 1:
            # TODO: fusion
            next_token_ids = all_gather(next_token_ids, self.extra_config)

        batch_state.output_ids = next_token_ids
        return batch_state


class Qwen3Embeddings(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, batch_state: BatchState):
        hidden_states = self.embed_tokens(batch_state.input_ids)

        batch_state.hidden_states = hidden_states
        return batch_state


class Qwen3Model(nn.Module):
    rope_cos: Tensor
    rope_sin: Tensor

    def __init__(
        self,
        config: Qwen3Config,
        extra_config: ExtraModelConfig,
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.embed_tokens = Qwen3Embeddings(config)

        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(Qwen3Block(config, extra_config, i))

        self.layers = nn.ModuleList(layers)

        self.rope = Qwen3RotaryEmbedding(
            config=config,
        )

        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        dummy_float_input = torch.empty((0, config.hidden_size), dtype=torch.float32)

        cos, sin = self.rope(dummy_float_input, position_ids)
        assert cos.dtype == torch.float32
        assert sin.dtype == torch.float32
        self.register_buffer("rope_cos", cos.squeeze(0), persistent=False)
        self.register_buffer("rope_sin", sin.squeeze(0), persistent=False)

    def interleave_rope(self):
        indices_for_q_list = []
        half_head_dim = self.config.head_dim // 2
        for n in range(self.config.num_attention_heads):
            offset = n * self.config.head_dim
            for i in range(half_head_dim):
                indices_for_q_list.append(i + offset)
                indices_for_q_list.append(i + half_head_dim + offset)

        indices_for_q = torch.tensor(indices_for_q_list, device=self.rope_cos.device)
        one_head_indices = indices_for_q[: self.config.head_dim]

        self.rope_cos = self.rope_cos[..., one_head_indices]
        self.rope_sin = self.rope_sin[..., one_head_indices]

        indices_for_k = indices_for_q[
            : self.config.head_dim * self.config.num_key_value_heads
        ]

        for mod in self.modules():
            if isinstance(mod, Qwen3Attention):
                mod.q_proj.weight[:] = mod.q_proj.weight[indices_for_q]
                mod.k_proj.weight[:] = mod.k_proj.weight[indices_for_k]

    def forward(self, batch_state: BatchState):
        out: BatchState = self.embed_tokens(batch_state)
        assert self.rope_cos.dtype == torch.float32
        assert self.rope_sin.dtype == torch.float32
        cos = self.rope_cos[batch_state.position_ids]
        sin = self.rope_sin[batch_state.position_ids]
        out.position_embeddings = (cos, sin)

        for layer in self.layers:
            out = layer(out)
        return out


@dataclass
class StackedParams:
    qkv_proj_weight: Tensor
    q_norm_weight: Tensor
    k_norm_weight: Tensor
    o_proj: Tensor
    attn_ln_weight: Tensor
    mlp_ln_weight: Tensor
    up_proj: Tensor
    gate_proj: Tensor
    down_proj: Tensor


class Qwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        extra_config: ExtraModelConfig,
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.device = torch.get_default_device()
        self.dtype = torch.get_default_dtype()

        self.model = Qwen3Model(config, extra_config)

        self.lm_head = Qwen3LMHead(config, extra_config)

    def num_kv_heads(self):
        all_heads = self.config.num_key_value_heads
        tp_size = self.extra_config.tp_size
        assert all_heads % tp_size == 0
        return all_heads // tp_size

    def num_qo_heads(self):
        all_heads = self.config.num_attention_heads
        tp_size = self.extra_config.tp_size
        assert all_heads % tp_size == 0
        return all_heads // tp_size

    def forward(
        self,
        batch_state: BatchState,
    ):
        input_ids = batch_state.input_ids
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        position_ids = batch_state.position_ids
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)

        out = BatchState(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=batch_state.hidden_states,
            seq_len=batch_state.seq_len,
        )

        out = self.model(out)
        out = self.lm_head(out)

        return out

    def setup_caches(self):
        k_cache = torch.zeros(
            (
                self.config.num_hidden_layers,
                self.extra_config.max_batch_size,
                self.extra_config.max_len_override
                or self.config.max_position_embeddings,
                self.config.num_key_value_heads,
                self.config.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        v_cache = k_cache.clone()

        self.stacked_kv_cache = (k_cache, v_cache)

        for layer_idx in range(self.config.num_hidden_layers):
            layer: Qwen3Block = self.model.layers[layer_idx]  # type: ignore
            layer.self_attn.kv_cache = (
                self.stacked_kv_cache[0][layer_idx],
                self.stacked_kv_cache[1][layer_idx],
            )

    def to(self, device: DeviceType | None = None, dtype: torch.dtype | None = None):  # type: ignore
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return super().to(device=device, dtype=dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        extra_config: ExtraModelConfig | None = None,
        device: DeviceType | None = None,
        dtype: torch.dtype | None = None,
    ):
        if extra_config is None:
            extra_config = ExtraModelConfig()

        config: Qwen3Config = Qwen3Config.from_pretrained(model_name_or_path)  # type: ignore
        if extra_config.rope_scaling is not None:
            config.rope_scaling = extra_config.rope_scaling

        if dtype is None:
            dtype = config.torch_dtype

        with init_empty_weights(include_buffers=False):
            model = cls(
                config,
                extra_config,
            )
        model.dtype = dtype
        model.device = device

        if (as_path := Path(model_name_or_path)).exists():
            model_path = as_path
        else:
            snapshot_path_str = huggingface_hub.snapshot_download(
                model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
            )

            model_path = Path(snapshot_path_str)

        model.load_from_safetensors(model_path)

        # SE (10/18/24): It is important not to call model.to(device, dtype) because
        # this will convert the `inv_freq` buffer in the rotary embeddings to fp16
        # the HF load from pretrained is careful to not do this and keeps it in fp32.
        # The dtype for the parameters is already handled by the load calls above, but
        # it's possible that there are other buffers which *should* be converted to fp16.
        # TODO: it's probably easiest to figure out how we can just use HFs `load_from_pretrained`
        # to load the model weights so we can ensure that there are no other subtle differences
        model.to(device=device)

        model.requires_grad_(False)

        if extra_config.interleave_rope:
            model.model.interleave_rope()

        model.stack_params()
        model.setup_caches()

        return model

    def make_name_to_hf_name(self):
        keys = self.state_dict().keys()

        name_to_hf_name = {k: k for k in keys}

        for layer_idx in range(self.config.num_hidden_layers):
            name_to_hf_name[
                f"model.layers.{layer_idx}.self_attn.input_layernorm.weight"
            ] = f"model.layers.{layer_idx}.input_layernorm.weight"
            name_to_hf_name[f"model.layers.{layer_idx}.mlp.input_layernorm.weight"] = (
                f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            )

        name_to_hf_name["model.embed_tokens.embed_tokens.weight"] = (
            "model.embed_tokens.weight"
        )
        name_to_hf_name["lm_head.input_norm.weight"] = "model.norm.weight"

        if self.config.tie_word_embeddings:
            name_to_hf_name["lm_head.lm_head.weight"] = "model.embed_tokens.weight"
        else:
            name_to_hf_name["lm_head.lm_head.weight"] = "lm_head.weight"

        return name_to_hf_name

    def make_tp_map(self):
        """
        Maps parameter names to the dimension they should be split on.
        Parameters that are not included in the map should not be split.
        """

        tp_map = {}
        for param_name, _ in self.named_parameters():
            if any(
                param_name.endswith(suffix)
                for suffix in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "up_proj.weight",
                    "gate_proj.weight",
                ]
            ):
                tp_map[param_name] = 0

            elif any(
                param_name.endswith(suffix)
                for suffix in ["o_proj.weight", "down_proj.weight"]
            ):
                tp_map[param_name] = 1

        return tp_map

    def load_from_safetensors(
        self,
        model_path: Path,
    ):
        name_to_hf_name = self.make_name_to_hf_name()
        all_hf_names = set(name_to_hf_name.values())

        hf_state_dict = load_safetensors_repo(
            model_path,
            include_parameters=all_hf_names,
            device=self.device,
            tp_rank=self.extra_config.tp_rank,
            tp_size=self.extra_config.tp_size,
            tp_map=self.make_tp_map(),
        )

        state_dict = {k: hf_state_dict[v] for k, v in name_to_hf_name.items()}

        self.load_state_dict(state_dict, assign=True, strict=True)

    def stack_params(self):
        def stack_and_reassign(modules, prop: str):
            params = [getattr(m, prop) for m in modules]
            stacked = torch.stack(params, dim=0)
            for i, m in enumerate(modules):
                getattr(m, prop)[:] = stacked[i]
            return stacked

        layers: list[Qwen3Block] = self.model.layers
        self_attns = [x.self_attn for x in layers]

        # Qwen3 Specific: Stack Q-Norm and K-Norm weights
        # Provided param: model.layers.0.self_attn.q_norm.weight [128]
        q_norms = [x.q_norm for x in self_attns]
        k_norms = [x.k_norm for x in self_attns]

        mlps = [x.mlp for x in layers]

        o_projs = [x.o_proj for x in self_attns]
        self_attn_lns = [x.input_layernorm for x in self_attns]

        mlp_lns = [x.input_layernorm for x in mlps]
        up_projs = [x.up_proj for x in mlps]
        gate_projs = [x.gate_proj for x in mlps]
        down_projs = [x.down_proj for x in mlps]

        stacked_q_norm_weights = stack_and_reassign(q_norms, "weight")
        stacked_k_norm_weights = stack_and_reassign(k_norms, "weight")
        stacked_o_proj = stack_and_reassign(o_projs, "weight")
        stacked_self_attn_ln_weights = stack_and_reassign(self_attn_lns, "weight")
        stacked_mlp_ln_weights = stack_and_reassign(mlp_lns, "weight")
        stacked_up_proj = stack_and_reassign(up_projs, "weight")
        stacked_gate_proj = stack_and_reassign(gate_projs, "weight")
        stacked_down_proj = stack_and_reassign(down_projs, "weight")

        qkv_weights = []
        for self_attn in self_attns:
            cat_weight = torch.cat(
                [
                    self_attn.q_proj.weight,
                    self_attn.k_proj.weight,
                    self_attn.v_proj.weight,
                ],
                dim=0,
            )
            qkv_weights.append(cat_weight)

        stacked_qkv_weights = torch.stack(qkv_weights, dim=0)

        for i, self_attn in enumerate(self_attns):
            qkv_weight = stacked_qkv_weights[i]
            q_weight, k_weight, v_weight = qkv_weight.split(
                [
                    self.config.num_attention_heads * self.config.head_dim,
                    self.config.num_key_value_heads * self.config.head_dim,
                    self.config.num_key_value_heads * self.config.head_dim,
                ],
                dim=0,
            )

            self_attn.q_proj.weight[:] = q_weight
            self_attn.k_proj.weight[:] = k_weight
            self_attn.v_proj.weight[:] = v_weight
        self.stacked_params = StackedParams(
            qkv_proj_weight=stacked_qkv_weights,
            o_proj=stacked_o_proj,
            attn_ln_weight=stacked_self_attn_ln_weights,
            mlp_ln_weight=stacked_mlp_ln_weights,
            up_proj=stacked_up_proj,
            gate_proj=stacked_gate_proj,
            down_proj=stacked_down_proj,
            q_norm_weight=stacked_q_norm_weights,
            k_norm_weight=stacked_k_norm_weights,
        )

```

**user**: 

```
根据json文件对比结果判断是layer_norm_matvec_rope_append中q_norm和k_norm开始计算出错，结合schduler.py中schedule_qkv对该指令的调度方式，分析错误原因并修改
```

**assistant**: 

```
### 错误分析

根据您的反馈，错误发生在 `layer_norm_matvec_rope_append` 指令中 `q_norm` 和 `k_norm` 的计算环节。

**根本原因分析：**
1.  **RMSNorm 的数学特性**：RMSNorm 的计算公式为 $y = \frac{x}{\text{RMS}(x)} \cdot w$，其中 $\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$。这意味着要正确计算归一化结果，必须获取**完整的一个 Head 向量**（长度为 `head_dim=128`）来计算方差。
2.  **调度器 (Scheduler) 的切分逻辑**：`scheduler.py` 将计算任务切分为较小的 `block`（通常 `block_size=16`）。VM 中的循环是按 block 迭代的。
3.  **VM 实现的逻辑漏洞**：原有的 `python_vm.py` 代码在计算出 16 个元素的投影结果后，立即对这 16 个元素进行了 RMSNorm。这导致方差是基于局部 16 个元素计算的，而不是基于整个 Head 的 128 个元素，从而导致数值完全错误。

**修复方案：**
我们不能在每个 Block 计算完投影后立即进行 Norm。必须修改 VM 的执行逻辑：
1.  **第一步（Projection）**：计算当前 Block 的投影（Linear），并将**原始值**存入全局内存。
2.  **第二步（Check Boundary）**：判断当前 Block 是否是一个 Head 的**最后一个 Block**。
3.  **第三步（Fused Norm & RoPE）**：如果是 Head 的末尾，从全局内存中读取该 Head 的**完整数据**（128个元素），执行 RMSNorm 和 RoPE，然后将最终结果写回全局内存。

---

### 修复后的代码

#### 文件 1: `python_vm.py` (已修复 QK-Norm 逻辑)


import math
import json
import os

import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor

from megakernels.demos.qwen3_0_6b.instructions import (
    AttentionReduction,
    DownProjResidual,
    Globals,
    LayerNorm_QKV_MatVecRopeAppend,
    LayerNormDoubleMatVecSiLU,
    O_ProjResidual,
    PartialAttention,
    RMS_LM_Head,
)

from megakernels.qwen3 import (
    apply_rotary_pos_emb_interleaved,
)

# ================= DEBUG UTILS =================
DEBUG_FILE = "python_vm_debug.json"

def log_debug_data(tag, tensor):
    """记录 Tensor 的关键统计信息到 JSON 文件"""
    if tensor is None:
        return
    
    # 确保数据在 CPU 上并转为 float
    t_flat = tensor.detach().float().flatten().cpu()
    
    # 记录前 16 个值以便直接对比
    data = {
        "tag": tag,
        "shape": list(tensor.shape),
        "mean": float(t_flat.mean().item()),
        "sum": float(t_flat.sum().item()),
        "head_values": t_flat[:16].tolist()
    }
    
    with open(DEBUG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
# ===============================================

def get_start_end(block_size: int, block_idx: int):
    start = block_size * block_idx
    end = start + block_size
    return start, end


def matvec(
    mat: Tensor,
    vec: Tensor,
    block_size: int,
    block_idx: int,
    reduce: bool = False,
    reduction_size: int = 0,
    reduction_idx: int = 0,
):
    start, end = get_start_end(block_size, block_idx)
    if reduce:
        red_start, red_end = get_start_end(reduction_size, reduction_idx)
        mat = mat[start:end, red_start:red_end]
        vec = vec[red_start:red_end]
    else:
        mat = mat[start:end]

    out = einsum(mat, vec, "o i, i -> o")
    return out, start, end


def rms_norm(inp: Tensor, weight: Tensor, eps: float):
    input_dtype = inp.dtype
    inp = inp.to(torch.float32)
    variance = inp.pow(2).mean(-1, keepdim=True)
    inp = inp * torch.rsqrt(variance + eps)

    return weight * inp.to(input_dtype)


def matvec_with_residual(
    mat: Tensor,
    vec: Tensor,
    residual: Tensor,
    block_size: int,
    start_block_idx: int,
    end_block_idx: int,
    reduction_size: int,
    reduction_block_idx: int,
):
    for block_idx in range(start_block_idx, end_block_idx):
        matvec_out, start, end = matvec(
            mat=mat,
            vec=vec,
            block_size=block_size,
            block_idx=block_idx,
            reduce=True,
            reduction_size=reduction_size,
            reduction_idx=reduction_block_idx,
        )

        residual[start:end] += matvec_out.to(residual.dtype)


def o_proj_residual(globals: Globals, instruction: O_ProjResidual):
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == globals.num_attention_heads

    assert instruction.start_block_idx == instruction.end_block_idx - 1
    assert instruction.reduction_block_idx == 0
    
    # DEBUG
    if instruction.layer_idx == 0 and instruction.start_block_idx == 0:
         log_debug_data(f"layer_{instruction.layer_idx}_attn_out_buffer", globals.attn_out)

    matvec_with_residual(
        mat=globals.o_proj_weights[instruction.layer_idx],
        vec=globals.attn_out,
        residual=globals.hidden_states,
        block_size=globals.o_proj_block_size,
        start_block_idx=instruction.start_block_idx,
        end_block_idx=instruction.end_block_idx,
        reduction_size=globals.matvec_reduction_size,
        reduction_block_idx=instruction.reduction_block_idx,
    )

    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += instruction.end_block_idx - instruction.start_block_idx
    
    # DEBUG
    if instruction.layer_idx == 0 and instruction.start_block_idx == 0:
        log_debug_data(f"layer_{instruction.layer_idx}_hidden_states_post_oproj_partial", globals.hidden_states[0:16])


def down_proj_residual(globals: Globals, instruction: DownProjResidual):
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == 192

    matvec_with_residual(
        mat=globals.down_proj_weights[instruction.layer_idx],
        vec=globals.silu_out,
        residual=globals.hidden_states,
        block_size=globals.down_proj_block_size,
        start_block_idx=instruction.start_block_idx,
        end_block_idx=instruction.end_block_idx,
        reduction_size=globals.matvec_reduction_size,
        reduction_block_idx=instruction.reduction_block_idx,
    )

    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += instruction.end_block_idx - instruction.start_block_idx


def layer_norm_double_matvec_silu(
    globals: Globals, instruction: LayerNormDoubleMatVecSiLU
):
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == 64

    post_ln = rms_norm(
        inp=globals.hidden_states,
        weight=globals.mlp_ln_weights[instruction.layer_idx],
        eps=globals.rms_norm_eps,
    )
    
    # DEBUG
    if instruction.layer_idx == 0 and instruction.block_idxs[0] == 0:
         log_debug_data(f"layer_{instruction.layer_idx}_mlp_pre_ln_out", post_ln)

    block_size = globals.up_gate_proj_block_size

    barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]

    for block_idx in instruction.block_idxs:
        start, end = get_start_end(block_size, block_idx)

        up_matvec, start, end = matvec(
            mat=globals.up_proj_weights[instruction.layer_idx],
            vec=post_ln,
            block_size=block_size,
            block_idx=block_idx,
        )

        gate_matvec, _, _ = matvec(
            mat=globals.gate_proj_weights[instruction.layer_idx],
            vec=post_ln,
            block_size=block_size,
            block_idx=block_idx,
        )

        post_silu = F.silu(gate_matvec) * up_matvec

        globals.silu_out[start:end] = post_silu

        barriers[0] += 1


def layer_norm_matvec_rope_append(
    globals: Globals, instruction: LayerNorm_QKV_MatVecRopeAppend
):
    layer_idx = instruction.layer_idx

    # Barrier check
    if layer_idx > 0:
        op_barriers = globals.barriers[layer_idx - 1, instruction.prev_opcode() - 1]
        assert op_barriers[0] == 64

    # Pre-Attention LayerNorm
    post_ln = rms_norm(
        inp=globals.hidden_states,
        weight=globals.attn_ln_weights[layer_idx],
        eps=globals.rms_norm_eps,
    )

    # DEBUG
    if layer_idx == 0 and instruction.start_output_block_idx == 0:
        log_debug_data(f"layer_{layer_idx}_attn_pre_ln_out", post_ln)

    pos_id = globals.pos_id

    k_start = globals.num_attention_heads * globals.head_dim
    v_start = k_start + globals.num_kv_heads * globals.head_dim
    head_dim = globals.head_dim

    barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]

    for block_idx in range(
        instruction.start_output_block_idx, instruction.end_output_block_idx
    ):
        start, end = get_start_end(globals.qkv_block_size, block_idx)

        if start < k_start:
            mode = "q"
        elif start < v_start:
            mode = "k"
        else:
            mode = "v"

        # 1. QKV 投影计算 (保持不变)
        matmul_output = einsum(
            globals.qkv_proj_weights[layer_idx][start:end],
            post_ln,
            "o i, i -> o",
        )
        
        # DEBUG: 记录 Raw Projection (Projection Layer Output)
        if layer_idx == 0:
            if start == 0:
                log_debug_data(f"layer_{layer_idx}_q_proj_raw", matmul_output)
            elif start == k_start:
                log_debug_data(f"layer_{layer_idx}_k_proj_raw", matmul_output)

        # 2. 存储原始投影结果 (Store Raw Projection)
        # QK-Norm 需要完整的 Head 数据，所以我们这里只存储，暂不归一化
        if mode == "q":
            globals.post_ln_rope_q[start:end] = matmul_output
        elif mode == "k":
            start_in_k = start - k_start
            end_in_k = end - k_start
            # 注意：k_cache 是 [layer, bs, seq, head, dim]，这里我们写入对应的 dim 范围
            globals.k_cache[layer_idx, :, pos_id].view(-1)[
                start_in_k:end_in_k
            ] = matmul_output
        elif mode == "v":
            start_in_v = start - v_start
            end_in_v = end - v_start
            globals.v_cache[layer_idx, :, pos_id].view(-1)[
                start_in_v:end_in_v
            ] = matmul_output
            # V 不需要 Norm 和 RoPE，直接结束
        
        # 3. 检查是否完成了一个 Head 的计算 (Fused Norm & RoPE)
        # 如果是 Q 或 K，且当前 Block 是该 Head 的最后一个 Block
        if mode in ["q", "k"]:
            # 计算相对于当前部分 (Q 或 K) 的结束位置
            if mode == "q":
                rel_end = end
            else: # mode == "k"
                rel_end = end - k_start

            # 只有当累积了完整的 head_dim (128) 时，才进行 Norm 和 RoPE
            if rel_end % head_dim == 0:
                # 准备读取完整 Head
                head_start_idx = rel_end - head_dim
                head_end_idx = rel_end
                
                # A. 读取 Full Head Data
                if mode == "q":
                    raw_head = globals.post_ln_rope_q[head_start_idx:head_end_idx]
                else: # k
                    # 映射回 k_cache 的线性索引
                    raw_head = globals.k_cache[layer_idx, :, pos_id].view(-1)[head_start_idx:head_end_idx]

                # B. 执行 RMSNorm (使用该层共享的 Q-Norm 或 K-Norm 权重)
                if mode == "q":
                    norm_weight = globals.q_norm_weights[layer_idx] # Shape: [head_dim]
                else:
                    norm_weight = globals.k_norm_weights[layer_idx] # Shape: [head_dim]
                
                normed_head = rms_norm(raw_head, norm_weight, eps=globals.rms_norm_eps)

                # DEBUG: 记录 Norm 后的结果
                # 由于我们是在 Head 结束时记录，这里会记录该 Head 的所有数据
                # 为了匹配 debug 脚本，我们只在处理第一个 Head 时记录
                if layer_idx == 0:
                    if mode == "q" and head_start_idx == 0:
                        log_debug_data(f"layer_{layer_idx}_q_norm_out", normed_head[:16])
                    elif mode == "k" and head_start_idx == 0:
                        log_debug_data(f"layer_{layer_idx}_k_norm_out", normed_head[:16])

                # C. 执行 RoPE
                # apply_rotary_pos_emb_interleaved 需要输入 shape [batch, seq, head, dim]
                # 这里我们构造 [1, head_dim] 传进去
                full_head_unsqueezed = normed_head.unsqueeze(0)
                
                full_head_with_rope, _ = apply_rotary_pos_emb_interleaved(
                    q=full_head_unsqueezed,
                    k=full_head_unsqueezed, # dummy
                    cos=globals.rope_cos[pos_id],
                    sin=globals.rope_sin[pos_id],
                    unsqueeze_dim=0,
                )
                final_head = full_head_with_rope.view(-1)

                # DEBUG: 记录 RoPE 后的结果
                if layer_idx == 0:
                    if mode == "q" and head_start_idx == 0:
                        log_debug_data(f"layer_{layer_idx}_q_rope_out", final_head[:16])
                    elif mode == "k" and head_start_idx == 0:
                        log_debug_data(f"layer_{layer_idx}_k_rope_out", final_head[:16])

                # D. 写回结果 (覆盖之前存储的 Raw Projection)
                if mode == "q":
                    globals.post_ln_rope_q[head_start_idx:head_end_idx] = final_head
                else: # k
                    globals.k_cache[layer_idx, :, pos_id].view(-1)[head_start_idx:head_end_idx] = final_head

        barriers[block_idx // 8] += 1


def rms_lm_head(globals: Globals, instruction: RMS_LM_Head):
    op_barriers = globals.barriers[
        globals.num_hidden_layers - 1, instruction.prev_opcode() - 1
    ]
    assert op_barriers[0] == 64

    post_ln = rms_norm(
        inp=globals.hidden_states,
        weight=globals.lm_head_norm_weights,
        eps=globals.rms_norm_eps,
    )

    for block_idx in range(
        instruction.start_output_block_idx, instruction.end_output_block_idx
    ):
        start, end = get_start_end(globals.lm_head_block_size, block_idx)

        matmul_output = einsum(
            globals.lm_head_weights[start:end],
            post_ln,
            "o i, i -> o",
        )

        globals.logits[start:end] = matmul_output


def partial_attention(globals: Globals, instruction: PartialAttention):
    gqa_ratio = globals.num_attention_heads // globals.num_kv_heads

    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    for i in range(gqa_ratio):
        assert op_barriers[instruction.kv_head_idx * gqa_ratio + i] == 8
    assert op_barriers[globals.num_attention_heads + instruction.kv_head_idx] == 8
    
    kv_block_size = globals.attn_kv_block_size
    seq_len = globals.pos_id + 1
    layer_idx = instruction.layer_idx
    kv_head_idx = instruction.kv_head_idx

    total_blocks = math.ceil(seq_len / kv_block_size)
    blocks_per_partial = math.ceil(total_blocks / instruction.num_partials)

    start_block = instruction.partial_idx * blocks_per_partial
    end_block = min(start_block + blocks_per_partial, total_blocks)

    start_token = start_block * kv_block_size
    end_token = min(end_block * kv_block_size, seq_len)

    k = globals.k_cache[layer_idx, 0, start_token:end_token, kv_head_idx]
    v = globals.v_cache[layer_idx, 0, start_token:end_token, kv_head_idx]

    head_start = kv_head_idx * gqa_ratio
    head_end = head_start + gqa_ratio

    q = globals.post_ln_rope_q.view(globals.num_attention_heads, -1)[
        head_start:head_end
    ]

    qk = einsum(q.float(), k.float(), "h i, k i -> h k")
    scaled_qk = qk * globals.attn_scale

    softmax = torch.softmax(scaled_qk, dim=-1)
    lse = torch.log2(torch.sum(torch.exp(scaled_qk), dim=-1))
    out = einsum(softmax.float(), v.float(), "h k, k o -> h o")
    
    # DEBUG
    if instruction.layer_idx == 0 and kv_head_idx == 0 and instruction.partial_idx == 0:
         log_debug_data(f"layer_{layer_idx}_head_grp_0_attn_out", out[0, :16])

    if globals.skip_attn_reduction:
        globals.attn_out.view(globals.num_attention_heads, -1)[
            head_start:head_end, :
        ] = out
        barriers = globals.barriers[
            instruction.layer_idx, AttentionReduction.opcode() - 1
        ]
        barriers[0] += head_end - head_start

    else:
        globals.attn_lse_intermediates[head_start:head_end, instruction.partial_idx] = (
            lse
        )
        globals.attn_out_intermediates[head_start:head_end, instruction.partial_idx] = (
            out
        )

        barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
        barriers[head_start:head_end] += 1


def attention_reduction(globals: Globals, instruction: AttentionReduction):
    head_start_idx = instruction.head_start_idx

    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[head_start_idx] == instruction.num_partials

    indices_to_reduce = torch.tensor(
        instruction.reduction_list,
        dtype=torch.long,
        device=globals.hidden_states.device,
    )

    lses = globals.attn_lse_intermediates[
        head_start_idx : head_start_idx + globals.attn_reduction_size, indices_to_reduce
    ]
    outs = globals.attn_out_intermediates[
        head_start_idx : head_start_idx + globals.attn_reduction_size, indices_to_reduce
    ]

    max_lse = torch.max(lses, dim=-1, keepdim=True).values

    adjusted_factors = (lses - max_lse).exp2()
    new_denominator = adjusted_factors.sum(dim=-1, keepdim=True)

    reduced = (outs * adjusted_factors.unsqueeze(-1)).sum(dim=1) / new_denominator

    if instruction.is_terminal:
        globals.attn_out.view(globals.num_attention_heads, -1)[
            head_start_idx : head_start_idx + globals.attn_reduction_size
        ] = reduced
    else:
        new_lse = new_denominator.log()
        output_slot = instruction.output_partial_idx
        globals.attn_lse_intermediates[
            head_start_idx : head_start_idx + globals.attn_reduction_size, output_slot
        ] = new_lse
        globals.attn_out_intermediates[
            head_start_idx : head_start_idx + globals.attn_reduction_size, output_slot
        ] = reduced

    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += globals.attn_reduction_size


INSTRUCTION_TO_SOLVER = {
    O_ProjResidual: o_proj_residual,
    DownProjResidual: down_proj_residual,
    LayerNormDoubleMatVecSiLU: layer_norm_double_matvec_silu,
    LayerNorm_QKV_MatVecRopeAppend: layer_norm_matvec_rope_append,
    RMS_LM_Head: rms_lm_head,
    PartialAttention: partial_attention,
    AttentionReduction: attention_reduction,
}


#### 文件 2: `qwen3.py` (保持 Debug 插桩与命名统一)


from dataclasses import dataclass
from pathlib import Path
import json
import os

import huggingface_hub
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from einops import rearrange
from torch import Tensor, nn
from torch.distributed import _functional_collectives as funcol
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from megakernels.model_types import (
    BatchState,
    DeviceType,
    ExtraModelConfig,
)
from megakernels.utils import (
    load_safetensors_repo,
)

# ================= DEBUG UTILS =================
DEBUG_FILE = "qwen3_torch_debug.json"

def log_debug_data(tag, tensor):
    """记录 Tensor 的关键统计信息到 JSON 文件"""
    if tensor is None:
        return
    
    t_flat = tensor.detach().float().flatten().cpu()
    if t_flat.numel() > 16:
        t_flat = t_flat[:16]
        
    data = {
        "tag": tag,
        "shape": list(tensor.shape),
        "mean": float(t_flat.mean().item()),
        "sum": float(t_flat.sum().item()),
        "head_values": t_flat.tolist()
    }
    with open(DEBUG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
# ===============================================

KV_Cache = tuple[Tensor, Tensor]


class RMSNorm(nn.Module):
    def __init__(self, config: Qwen3Config, dim):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.config.rms_norm_eps)

        if self.weight is not None:
            return self.weight * hidden_states.to(input_dtype)
        else:
            return hidden_states.to(input_dtype)


def all_gather(x: Tensor, extra_config: ExtraModelConfig):
    if extra_config.tp_size == 1:
        return x

    assert extra_config.tp_group is not None

    if extra_config.torch_compile:
        return funcol.all_gather_tensor(x, gather_dim=0, group=extra_config.tp_group)

    out = torch.empty(
        (extra_config.tp_size * x.shape[0], *x.shape[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    torch.distributed.all_gather_into_tensor(out, x, group=extra_config.tp_group)
    return out


def reduce_scatter(x: Tensor, extra_config: ExtraModelConfig):
    if extra_config.tp_size == 1:
        return x

    assert extra_config.tp_group is not None

    if extra_config.torch_compile:
        return funcol.reduce_scatter_tensor(
            x, reduceOp="sum", scatter_dim=0, group=extra_config.tp_group
        )

    out = torch.empty(
        (x.shape[0] // extra_config.tp_size, *x.shape[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    torch.distributed.reduce_scatter_tensor(out, x, group=extra_config.tp_group)
    return out


def attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    kv_cache: KV_Cache,
    position_ids: Tensor,
    seq_len: int,
) -> Tensor:
    bsz, new_tok_seq_len = query_states.shape[:2]

    k_cache, v_cache = kv_cache

    k_cache[:, position_ids] = key_states
    v_cache[:, position_ids] = value_states

    def shape_for_sdpa(x: Tensor):
        return rearrange(x, "b l h d -> b h l d")

    def unshape_for_sdpa(x: Tensor):
        return rearrange(x, "b l h d -> b l h d")

    if new_tok_seq_len > 1:
        k_for_sdpa = shape_for_sdpa(key_states)
        v_for_sdpa = shape_for_sdpa(value_states)
        q_for_sdpa = shape_for_sdpa(query_states)

        attn_output = F.scaled_dot_product_attention(
            q_for_sdpa, k_for_sdpa, v_for_sdpa, is_causal=True, enable_gqa=True
        )
    else:
        k_for_sdpa = shape_for_sdpa(k_cache[:, :seq_len])
        v_for_sdpa = shape_for_sdpa(v_cache[:, :seq_len])
        q_for_sdpa = shape_for_sdpa(query_states)

        attn_output = F.scaled_dot_product_attention(
            q_for_sdpa, k_for_sdpa, v_for_sdpa, is_causal=False, enable_gqa=True
        )
    reshaped_attn_output = unshape_for_sdpa(attn_output)
    return reshaped_attn_output


def rotate_half_interleaved(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    new_x1 = -x2
    new_x2 = x1

    stacked = torch.stack((new_x1, new_x2), dim=-1)
    return stacked.view_as(x)


def apply_rotary_pos_emb_interleaved(
    q, k, cos, sin, position_ids=None, unsqueeze_dim=1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


class Qwen3Attention(nn.Module):
    def __init__(
        self, config: Qwen3Config, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config, dim=config.hidden_size)

        self.tp_size = extra_config.tp_size or 1

        assert config.num_attention_heads % self.tp_size == 0

        self.head_dim = config.head_dim

        self.num_attention_heads = config.num_attention_heads // self.tp_size
        self.num_kv_heads = (
            config.num_key_value_heads // self.tp_size
            if config.num_key_value_heads > 1
            else 1
        )

        self.q_proj = nn.Linear(
            self.config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )

        self.q_norm = RMSNorm(config, dim=self.head_dim)
        self.k_norm = RMSNorm(config, dim=self.head_dim)

        self.kv_cache: KV_Cache | None = None

    def forward(
        self,
        batch_state: BatchState,
    ):
        inp = batch_state.hidden_states
        residual = inp

        hidden_states = self.input_layernorm(inp)
        
        if self.layer_idx == 0:
            log_debug_data(f"layer_{self.layer_idx}_attn_pre_ln_out", hidden_states)

        hidden_states = all_gather(hidden_states, self.extra_config)
        bsz, seq_len = hidden_states.shape[:2]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, seq_len, self.num_attention_heads, -1)
        key_states = key_states.view(bsz, seq_len, self.num_kv_heads, -1)
        value_states = value_states.view(bsz, seq_len, self.num_kv_heads, -1)
        
        # DEBUG: Raw Projection
        if self.layer_idx == 0:
             log_debug_data(f"layer_{self.layer_idx}_q_proj_raw", query_states[0, 0, 0, :16]) 
             log_debug_data(f"layer_{self.layer_idx}_k_proj_raw", key_states[0, 0, 0, :16])

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # DEBUG: Norm Output
        if self.layer_idx == 0:
             log_debug_data(f"layer_{self.layer_idx}_q_norm_out", query_states[0, 0, 0, :16])
             log_debug_data(f"layer_{self.layer_idx}_k_norm_out", key_states[0, 0, 0, :16])

        cos, sin = batch_state.position_embeddings
        dtype = query_states.dtype

        if self.extra_config.interleave_rope:
            rope_fn = apply_rotary_pos_emb_interleaved
        else:
            rope_fn = apply_rotary_pos_emb

        query_states, key_states = rope_fn(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=-2,
        )

        query_states = query_states.to(dtype)
        key_states = key_states.to(dtype)
        
        # DEBUG: RoPE Output
        if self.layer_idx == 0:
             log_debug_data(f"layer_{self.layer_idx}_q_rope_out", query_states[0, 0, 0, :16])
             log_debug_data(f"layer_{self.layer_idx}_k_rope_out", key_states[0, 0, 0, :16])

        raw_attn_output = attention(
            query_states,
            key_states,
            value_states,
            self.kv_cache,
            batch_state.position_ids,
            seq_len=batch_state.seq_len,
        )

        attn_output = raw_attn_output.reshape(bsz, seq_len, -1)
        
        if self.layer_idx == 0:
             # 对应 Partial Attention Output
             log_debug_data(f"layer_{self.layer_idx}_head_grp_0_attn_out", raw_attn_output[0, 0, 0, :16])
             # 对应 Full Buffer
             log_debug_data(f"layer_{self.layer_idx}_attn_out_buffer", attn_output)

        o_proj = self.o_proj(attn_output)
        o_proj = reduce_scatter(o_proj, self.extra_config)
        with_residual = residual + o_proj
        
        if self.layer_idx == 0:
             log_debug_data(f"layer_{self.layer_idx}_hidden_states_post_oproj_partial", with_residual)

        batch_state.hidden_states = with_residual
        return batch_state


class Qwen3MLP(nn.Module):
    def __init__(
        self, config: Qwen3Config, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.tp_size = extra_config.tp_size
        self.intermediate_size = self.config.intermediate_size // self.tp_size

        self.up_proj = nn.Linear(
            self.config.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            config.hidden_size,
            bias=False,
        )

        self.input_layernorm = RMSNorm(config, dim=config.hidden_size)

    def forward(
        self,
        batch_state: BatchState,
    ):
        inp = batch_state.hidden_states
        hidden_states = self.input_layernorm(inp)
        hidden_states = all_gather(hidden_states, self.extra_config)
        
        if self.layer_idx == 0:
            log_debug_data(f"layer_{self.layer_idx}_mlp_pre_ln_out", hidden_states)

        up = self.up_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        prod = F.silu(gate) * up
        down = self.down_proj(prod)

        down = reduce_scatter(down, self.extra_config)

        with_residual = inp + down

        batch_state.hidden_states = with_residual
        return batch_state


class Qwen3Block(nn.Module):
    def __init__(
        self, config: Qwen3Config, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.self_attn = Qwen3Attention(config, extra_config, layer_idx)
        self.mlp = Qwen3MLP(config, extra_config, layer_idx)

    def forward(self, batch_state: BatchState):
        out = self.self_attn(batch_state)
        out = self.mlp(out)
        return out


class Qwen3LMHead(nn.Module):
    def __init__(self, config: Qwen3Config, extra_config: ExtraModelConfig):
        super().__init__()
        self.config = config
        self.extra_config = extra_config

        self.input_norm = RMSNorm(config, dim=config.hidden_size)
        self.tp_size = extra_config.tp_size or 1
        head_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, head_size, bias=False)

    def forward(self, batch_state: BatchState):
        hidden_states = batch_state.hidden_states

        if self.extra_config.tp_size > 1:
            hidden_states = all_gather(hidden_states, self.extra_config)

        hidden_states = self.input_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        next_token_ids = logits.argmax(dim=-1)

        if self.tp_size > 1:
            next_token_ids = all_gather(next_token_ids, self.extra_config)

        batch_state.output_ids = next_token_ids
        return batch_state


class Qwen3Embeddings(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, batch_state: BatchState):
        hidden_states = self.embed_tokens(batch_state.input_ids)
        batch_state.hidden_states = hidden_states
        return batch_state


class Qwen3Model(nn.Module):
    rope_cos: Tensor
    rope_sin: Tensor

    def __init__(
        self,
        config: Qwen3Config,
        extra_config: ExtraModelConfig,
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.embed_tokens = Qwen3Embeddings(config)

        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(Qwen3Block(config, extra_config, i))

        self.layers = nn.ModuleList(layers)

        self.rope = Qwen3RotaryEmbedding(
            config=config,
        )

        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        dummy_float_input = torch.empty((0, config.hidden_size), dtype=torch.float32)

        cos, sin = self.rope(dummy_float_input, position_ids)
        self.register_buffer("rope_cos", cos.squeeze(0), persistent=False)
        self.register_buffer("rope_sin", sin.squeeze(0), persistent=False)

    def interleave_rope(self):
        indices_for_q_list = []
        half_head_dim = self.config.head_dim // 2
        for n in range(self.config.num_attention_heads):
            offset = n * self.config.head_dim
            for i in range(half_head_dim):
                indices_for_q_list.append(i + offset)
                indices_for_q_list.append(i + half_head_dim + offset)

        indices_for_q = torch.tensor(indices_for_q_list, device=self.rope_cos.device)
        one_head_indices = indices_for_q[: self.config.head_dim]

        self.rope_cos = self.rope_cos[..., one_head_indices]
        self.rope_sin = self.rope_sin[..., one_head_indices]

        indices_for_k = indices_for_q[
            : self.config.head_dim * self.config.num_key_value_heads
        ]

        for mod in self.modules():
            if isinstance(mod, Qwen3Attention):
                mod.q_proj.weight[:] = mod.q_proj.weight[indices_for_q]
                mod.k_proj.weight[:] = mod.k_proj.weight[indices_for_k]

    def forward(self, batch_state: BatchState):
        out: BatchState = self.embed_tokens(batch_state)
        cos = self.rope_cos[batch_state.position_ids]
        sin = self.rope_sin[batch_state.position_ids]
        out.position_embeddings = (cos, sin)

        for layer in self.layers:
            out = layer(out)
        return out


@dataclass
class StackedParams:
    qkv_proj_weight: Tensor
    q_norm_weight: Tensor
    k_norm_weight: Tensor
    o_proj: Tensor
    attn_ln_weight: Tensor
    mlp_ln_weight: Tensor
    up_proj: Tensor
    gate_proj: Tensor
    down_proj: Tensor


class Qwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        extra_config: ExtraModelConfig,
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.device = torch.get_default_device()
        self.dtype = torch.get_default_dtype()

        self.model = Qwen3Model(config, extra_config)
        self.lm_head = Qwen3LMHead(config, extra_config)

    def forward(
        self,
        batch_state: BatchState,
    ):
        input_ids = batch_state.input_ids
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        position_ids = batch_state.position_ids
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)

        out = BatchState(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=batch_state.hidden_states,
            seq_len=batch_state.seq_len,
        )

        out = self.model(out)
        out = self.lm_head(out)

        return out

    def setup_caches(self):
        k_cache = torch.zeros(
            (
                self.config.num_hidden_layers,
                self.extra_config.max_batch_size,
                self.extra_config.max_len_override
                or self.config.max_position_embeddings,
                self.config.num_key_value_heads,
                self.config.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        v_cache = k_cache.clone()

        self.stacked_kv_cache = (k_cache, v_cache)

        for layer_idx in range(self.config.num_hidden_layers):
            layer: Qwen3Block = self.model.layers[layer_idx]
            layer.self_attn.kv_cache = (
                self.stacked_kv_cache[0][layer_idx],
                self.stacked_kv_cache[1][layer_idx],
            )

    def to(self, device: DeviceType | None = None, dtype: torch.dtype | None = None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return super().to(device=device, dtype=dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        extra_config: ExtraModelConfig | None = None,
        device: DeviceType | None = None,
        dtype: torch.dtype | None = None,
    ):
        if extra_config is None:
            extra_config = ExtraModelConfig()

        config: Qwen3Config = Qwen3Config.from_pretrained(model_name_or_path)
        if extra_config.rope_scaling is not None:
            config.rope_scaling = extra_config.rope_scaling

        if dtype is None:
            dtype = config.torch_dtype

        with init_empty_weights(include_buffers=False):
            model = cls(
                config,
                extra_config,
            )
        model.dtype = dtype
        model.device = device

        if (as_path := Path(model_name_or_path)).exists():
            model_path = as_path
        else:
            snapshot_path_str = huggingface_hub.snapshot_download(
                model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
            )
            model_path = Path(snapshot_path_str)

        model.load_from_safetensors(model_path)
        model.to(device=device)
        model.requires_grad_(False)

        if extra_config.interleave_rope:
            model.model.interleave_rope()

        model.stack_params()
        model.setup_caches()

        return model

    def make_name_to_hf_name(self):
        keys = self.state_dict().keys()
        name_to_hf_name = {k: k for k in keys}

        for layer_idx in range(self.config.num_hidden_layers):
            name_to_hf_name[
                f"model.layers.{layer_idx}.self_attn.input_layernorm.weight"
            ] = f"model.layers.{layer_idx}.input_layernorm.weight"
            name_to_hf_name[f"model.layers.{layer_idx}.mlp.input_layernorm.weight"] = (
                f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            )

        name_to_hf_name["model.embed_tokens.embed_tokens.weight"] = (
            "model.embed_tokens.weight"
        )
        name_to_hf_name["lm_head.input_norm.weight"] = "model.norm.weight"

        if self.config.tie_word_embeddings:
            name_to_hf_name["lm_head.lm_head.weight"] = "model.embed_tokens.weight"
        else:
            name_to_hf_name["lm_head.lm_head.weight"] = "lm_head.weight"

        return name_to_hf_name

    def make_tp_map(self):
        tp_map = {}
        for param_name, _ in self.named_parameters():
            if any(
                param_name.endswith(suffix)
                for suffix in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "up_proj.weight",
                    "gate_proj.weight",
                ]
            ):
                tp_map[param_name] = 0

            elif any(
                param_name.endswith(suffix)
                for suffix in ["o_proj.weight", "down_proj.weight"]
            ):
                tp_map[param_name] = 1

        return tp_map

    def load_from_safetensors(
        self,
        model_path: Path,
    ):
        name_to_hf_name = self.make_name_to_hf_name()
        all_hf_names = set(name_to_hf_name.values())

        hf_state_dict = load_safetensors_repo(
            model_path,
            include_parameters=all_hf_names,
            device=self.device,
            tp_rank=self.extra_config.tp_rank,
            tp_size=self.extra_config.tp_size,
            tp_map=self.make_tp_map(),
        )

        state_dict = {k: hf_state_dict[v] for k, v in name_to_hf_name.items()}

        self.load_state_dict(state_dict, assign=True, strict=True)

    def stack_params(self):
        def stack_and_reassign(modules, prop: str):
            params = [getattr(m, prop) for m in modules]
            stacked = torch.stack(params, dim=0)
            for i, m in enumerate(modules):
                getattr(m, prop)[:] = stacked[i]
            return stacked

        layers: list[Qwen3Block] = self.model.layers
        self_attns = [x.self_attn for x in layers]
        q_norms = [x.q_norm for x in self_attns]
        k_norms = [x.k_norm for x in self_attns]

        mlps = [x.mlp for x in layers]
        o_projs = [x.o_proj for x in self_attns]
        self_attn_lns = [x.input_layernorm for x in self_attns]
        mlp_lns = [x.input_layernorm for x in mlps]
        up_projs = [x.up_proj for x in mlps]
        gate_projs = [x.gate_proj for x in mlps]
        down_projs = [x.down_proj for x in mlps]

        stacked_q_norm_weights = stack_and_reassign(q_norms, "weight")
        stacked_k_norm_weights = stack_and_reassign(k_norms, "weight")
        stacked_o_proj = stack_and_reassign(o_projs, "weight")
        stacked_self_attn_ln_weights = stack_and_reassign(self_attn_lns, "weight")
        stacked_mlp_ln_weights = stack_and_reassign(mlp_lns, "weight")
        stacked_up_proj = stack_and_reassign(up_projs, "weight")
        stacked_gate_proj = stack_and_reassign(gate_projs, "weight")
        stacked_down_proj = stack_and_reassign(down_projs, "weight")

        qkv_weights = []
        for self_attn in self_attns:
            cat_weight = torch.cat(
                [
                    self_attn.q_proj.weight,
                    self_attn.k_proj.weight,
                    self_attn.v_proj.weight,
                ],
                dim=0,
            )
            qkv_weights.append(cat_weight)

        stacked_qkv_weights = torch.stack(qkv_weights, dim=0)

        for i, self_attn in enumerate(self_attns):
            qkv_weight = stacked_qkv_weights[i]
            q_weight, k_weight, v_weight = qkv_weight.split(
                [
                    self.config.num_attention_heads * self.config.head_dim,
                    self.config.num_key_value_heads * self.config.head_dim,
                    self.config.num_key_value_heads * self.config.head_dim,
                ],
                dim=0,
            )

            self_attn.q_proj.weight[:] = q_weight
            self_attn.k_proj.weight[:] = k_weight
            self_attn.v_proj.weight[:] = v_weight
        self.stacked_params = StackedParams(
            qkv_proj_weight=stacked_qkv_weights,
            o_proj=stacked_o_proj,
            attn_ln_weight=stacked_self_attn_ln_weights,
            mlp_ln_weight=stacked_mlp_ln_weights,
            up_proj=stacked_up_proj,
            gate_proj=stacked_gate_proj,
            down_proj=stacked_down_proj,
            q_norm_weight=stacked_q_norm_weights,
            k_norm_weight=stacked_k_norm_weights,
        )

```



--------------------


<a href="https://chatboxai.app" style="display: flex; align-items: center;">
<img src='https://chatboxai.app/icon.png' style='width: 40px; height: 40px; padding-right: 6px'>
<b style='font-size:30px'>Chatbox AI</b>
</a>
