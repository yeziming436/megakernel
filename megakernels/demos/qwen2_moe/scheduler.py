import math

import torch
import torch.nn.functional as F
from megakernels.demos.qwen2_moe.instructions import (
    DownProjResidual,
    Globals,
    Instruction,
    LayerNorm_QKV_MatVecRopeAppend,
    LayerNormDoubleMatVecSiLU,
    O_ProjResidual,
    Router,
    Dispatch_expert,
    PartialAttention,
    RMS_LM_Head,
)
from megakernels.instructions_moe import NoOp
from megakernels.qwen2_moe import Qwen2MoeForCausalLM
from megakernels.scheduler import DAG_Node, ScheduleBuilder
from megakernels.utils import assert_div, get_sm_count


def pick_num_attention_partitions(prompt_len: int, ntok: int, device: torch.device):
    min_chunk_size = 256
    full_len = prompt_len + ntok

    num_divisions = math.ceil(full_len / min_chunk_size)

    # TODO limitation until we have a better reduction tree
    num_attention_partitions = min(num_divisions, 24)

    # sm_count = get_sm_count(device)
    # num_attention_partitions = min(sm_count, num_divisions)

    assert num_attention_partitions >= 1

    return num_attention_partitions


def make_globals(
    model: Qwen2MoeForCausalLM,
    skip_attn_reduction: bool = True,
):
    config = model.config
    device = model.device
    dtype = model.dtype

    def make_buffer(shape, buffer_dtype=dtype):
        return torch.zeros(shape, device=device, dtype=buffer_dtype)

    stacked_params = model.stacked_params

    max_attn_partitions = get_sm_count(device)

    barriers = torch.zeros(
        [
            config.num_hidden_layers,
            10,  # more than the number of opcodes we have
            config.num_attention_heads + config.num_key_value_heads * 2,
        ],
        dtype=torch.int32,
        device=device,
    )

    return Globals(
        # model params
        qkv_proj_weights=stacked_params.qkv_proj_weights,
        qkv_proj_biases=stacked_params.qkv_proj_biases,
        o_proj_weights=stacked_params.o_proj,
        attn_ln_weights=stacked_params.attn_ln_weight,
        mlp_ln_weights=stacked_params.mlp_ln_weight,
        # new_para
        up_proj=stacked_params.up_proj,
        gate_proj=stacked_params.gate_proj,
        down_proj=stacked_params.down_proj,
        gate=stacked_params.gate,
        shared_up_proj=stacked_params.shared_up_proj,
        shared_gate_proj=stacked_params.shared_gate_proj,
        shared_down_proj=stacked_params.shared_down_proj,
        shared_gate=stacked_params.shared_gate,
        # new_para_end
        lm_head_norm_weights=model.lm_head.input_norm.weight,
        lm_head_weights=model.lm_head.lm_head.weight,
        k_cache=model.stacked_kv_cache[0],
        v_cache=model.stacked_kv_cache[1],
        rope_cos=model.model.rope_cos,
        rope_sin=model.model.rope_sin,
        # activation buffers
        hidden_states=make_buffer(config.hidden_size),
        post_ln_rope_q=make_buffer(config.hidden_size),
        attn_out=make_buffer(config.hidden_size),
        attn_out_intermediates=make_buffer(
            [
                config.num_attention_heads,
                max_attn_partitions,
                (config.hidden_size // config.num_attention_heads),
            ],
            buffer_dtype=torch.float32,
        ),
        attn_lse_intermediates=make_buffer(
            [config.num_attention_heads, max_attn_partitions],
            buffer_dtype=torch.float32,
        ),
        silu_out=make_buffer(config.intermediate_size),
        logits=make_buffer(config.vocab_size),
        # scalars
        pos_id=0,
        attn_scale=1 / math.sqrt(config.hidden_size // config.num_attention_heads),
        rms_norm_eps=config.rms_norm_eps,
        skip_attn_reduction=skip_attn_reduction,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        # block sizes
        up_gate_proj_block_size=16,
        down_proj_block_size=16,
        qkv_block_size=16,
        o_proj_block_size=16,
        lm_head_block_size=16,
        matvec_reduction_size=2048,
        attn_kv_block_size=16,
        attn_reduction_size=4,
        vocab_size=config.vocab_size,
        device=device,
        barriers=barriers,
    )


def schedule_qkv(
    globs: Globals, layer_idx: int
) -> list[LayerNorm_QKV_MatVecRopeAppend]:
    instructions = []

    qkv_outdim = (globs.num_attention_heads + 2 * globs.num_kv_heads) * globs.head_dim

    num_qkv_blocks = assert_div(qkv_outdim, globs.qkv_block_size)
    sm_count = globs.sm_count()

    blocks_per_sm = num_qkv_blocks / sm_count
    assert blocks_per_sm > 1

    for sm_idx in range(sm_count):
        start = round(sm_idx * blocks_per_sm)
        end = round((sm_idx + 1) * blocks_per_sm)
        instructions.append(
            LayerNorm_QKV_MatVecRopeAppend(
                layer_idx=layer_idx,
                start_output_block_idx=start,
                end_output_block_idx=end,
            )
        )

    return instructions


def schedule_upgate(globs: Globals, layer_idx: int):
    instructions: list[Instruction] = []
    
    sm_count = globs.sm_count()

    blocks_per_sm = num_up_gate_blocks / sm_count
    assert blocks_per_sm > 1

    for expert_id in range(globs.expert_hit):
        idx, top_x = torch.where(globs.expert_mask[expert_id].squeeze(0))
        for sm_idx in range(sm_count):
            instructions.append(
                LayerNormDoubleMatVecSiLU(
                    idx=idx,
                    top_x=top_x,
                    expert_idx=expert_id,
                    layer_idx=layer_idx,
                    block_idxs=list(range(sm_idx, num_up_gate_blocks, sm_count)),
                )
            )

    return instructions


def schedule_downproj(globs: Globals, layer_idx: int):
    instructions: list[Instruction] = []
    # mark_maybe_wrong should be added to globals
    num_down_gate_blocks = assert_div(globs.hidden_size, globs.up_gate_proj_block_size)

    sm_count = globs.sm_count()

    blocks_per_sm = num_down_gate_blocks / sm_count
    assert blocks_per_sm > 1

    for expert_id in range(globs.expert_hit):
        idx, top_x = torch.where(globs.expert_mask[expert_id].squeeze(0))
        for sm_idx in range(sm_count):
            instructions.append(
                DownProjResidual(
                    idx=idx,
                    top_x=top_x,
                    expert_idx=expert_id,
                    layer_idx=layer_idx,
                    block_idxs=list(range(sm_idx, num_down_gate_blocks, sm_count)),
                )
            )

    return instructions


def schedule_lm_head(globs: Globals):
    instructions: list[Instruction] = []

    num_logit_blocks = assert_div(globs.vocab_size, globs.lm_head_block_size)
    sm_count = globs.sm_count()

    blocks_per_sm = num_logit_blocks / sm_count
    assert blocks_per_sm > 1

    for sm_idx in range(sm_count):
        start = round(sm_idx * blocks_per_sm)
        end = round((sm_idx + 1) * blocks_per_sm)
        instructions.append(
            RMS_LM_Head(start_output_block_idx=start, end_output_block_idx=end)
        )

    return instructions


def make_dag(
    globs: Globals, stop_after_op: str | None = None, layer_limit: int | None = None
):
    nodes: list[DAG_Node] = []

    if layer_limit is not None:
        nlayers = layer_limit
    else:
        nlayers = globs.num_hidden_layers

    last_outputs = []
    for layer_idx in range(nlayers):
        new_nodes, new_outputs = make_dag_layer(
            globs=globs,
            layer_idx=layer_idx,
            prev_layer_outputs=last_outputs,
            stop_after_op=stop_after_op,
        )
        nodes.extend(new_nodes)
        last_outputs = new_outputs

    if nlayers == globs.num_hidden_layers:
        lm_head_instructions = schedule_lm_head(globs)
        lm_head_nodes: list[DAG_Node] = []
        for ins in lm_head_instructions:
            lm_head_nodes.append(DAG_Node(ins, last_outputs))

        nodes.extend(lm_head_nodes)
        last_outputs = lm_head_nodes

    end_node = DAG_Node(NoOp(), last_outputs)

    return nodes, end_node


def make_dag_layer(
    globs: Globals,
    layer_idx: int,
    prev_layer_outputs: list[DAG_Node],
    stop_after_op: str | None = None,
):
    assert globs.skip_attn_reduction
    num_attention_partitions = 1

    new_nodes: list[DAG_Node] = []

    # qkv
    qkv_instructions = schedule_qkv(globs, layer_idx)
    qkv_nodes: list[DAG_Node] = []
    for ins in qkv_instructions:
        qkv_nodes.append(DAG_Node(ins, prev_layer_outputs))

    qkv_deps = {}

    for node in qkv_nodes:
        ins: LayerNorm_QKV_MatVecRopeAppend = node.instruction
        for block_idx in ins.block_indices():
            qkv_deps[(layer_idx, ins.opcode(), block_idx)] = node

    new_nodes.extend(qkv_nodes)

    if stop_after_op == "qkv":
        return new_nodes, qkv_nodes

    # partial
    partial_nodes: list[DAG_Node] = []

    for kv_head_idx in range(globs.num_kv_heads):
        for partial_idx in range(num_attention_partitions):
            ins = PartialAttention(
                layer_idx=layer_idx,
                kv_head_idx=kv_head_idx,
                num_partials=num_attention_partitions,
                partial_idx=partial_idx,
            )

            block_indices = []

            k_start_dim = (globs.num_attention_heads + kv_head_idx) * globs.head_dim
            v_start_dim = (
                globs.num_attention_heads + globs.num_kv_heads + kv_head_idx
            ) * globs.head_dim

            dims_per_block = assert_div(globs.head_dim, globs.qkv_block_size)

            k_start_block = k_start_dim // globs.qkv_block_size
            v_start_block = v_start_dim // globs.qkv_block_size

            block_indices.extend(
                list(range(k_start_block, k_start_block + dims_per_block))
            )
            block_indices.extend(
                list(range(v_start_block, v_start_block + dims_per_block))
            )

            dep_set = {
                qkv_deps[(layer_idx, PartialAttention.prev_opcode(), block_idx)]
                for block_idx in block_indices
            }
            deps = list(dep_set)

            partial_nodes.append(DAG_Node(ins, deps))

    new_nodes.extend(partial_nodes)

    if stop_after_op == "partial":
        return new_nodes, partial_nodes

    # oproj
    # mark_maybe_wrong o_prpj: 4096 -> 2048; hidden_size -> num_attention_heads * head_dim ?
    num_o_blocks = assert_div(globs.hidden_size, globs.o_proj_block_size)
    o_proj_nodes: list[DAG_Node] = []
    for o_block_idx in range(num_o_blocks):
        ins = O_ProjResidual(
            layer_idx=layer_idx,
            start_block_idx=o_block_idx,
            end_block_idx=o_block_idx + 1,
            reduction_block_idx=0,
        )

        o_proj_nodes.append(DAG_Node(ins, partial_nodes))

    new_nodes.extend(o_proj_nodes)

    if stop_after_op == "oproj":
        return new_nodes, o_proj_nodes

    # router
    # mark_maybe_wrong_fixed o_proj_block_size -> expert_gate_block_size
    num_expert_blocks = assert_div(globs.hidden_size, globs.expert_gate_block_size)
    expert_nodes: list[DAG_Node] = []
    for expert_block_idx in range(num_expert_blocks):
        ins = Router(
            layer_idx=layer_idx,
            start_block_idx=expert_block_idx,
            end_block_idx=expert_block_idx + 1,
            reduction_block_idx=0,
        )

        expert_nodes.append(DAG_Node(ins, o_proj_nodes))

    new_nodes.extend(expert_nodes)

    if stop_after_op == "router":
        return new_nodes, expert_nodes

    # dispatch expert
    ins = Dispatch_expert(layer_idx=layer_idx)
    dispatch_expert_nodes = DAG_Node(ins, expert_nodes)
    new_nodes.append(dispatch_expert_nodes)
    if stop_after_op == "router":
        return new_nodes, dispatch_expert_nodes

    # upgate
    upgate_instructions = schedule_upgate(globs, layer_idx)
    upgate_nodes: list[DAG_Node] = []
    for ins in upgate_instructions:
        upgate_nodes.append(DAG_Node(ins, dispatch_expert_nodes))

    new_nodes.extend(upgate_nodes)

    if stop_after_op == "upgate":
        return new_nodes, upgate_nodes

    # downproj
    # TODO we can do better - we can start a reduction col's work once that fraction of the upgate work is done
    downproj_instructions = schedule_downproj(globs, layer_idx)
    downproj_nodes: list[DAG_Node] = []
    for ins in downproj_instructions:
        downproj_nodes.append(DAG_Node(ins, upgate_nodes))

    new_nodes.extend(downproj_nodes)

    if stop_after_op == "downproj":
        return new_nodes, downproj_nodes

    assert stop_after_op is None

    return new_nodes, downproj_nodes


class Qwen2MoeScheduleBuilder(ScheduleBuilder):
    @classmethod
    def make_globals(cls, model):
        return make_globals(model)

    @classmethod
    def make_dag(
        cls, globs, stop_after_op: str | None = None, layer_limit: int | None = None
    ):
        return make_dag(globs, stop_after_op, layer_limit)
