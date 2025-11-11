from dataclasses import dataclass
from pathlib import Path
import gc
import huggingface_hub
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from einops import rearrange
from torch import Tensor, nn
from torch.distributed import _functional_collectives as funcol

from transformers import Qwen2MoeConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeRotaryEmbedding,
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

KV_Cache = tuple[Tensor, Tensor]


class RMSNorm(nn.Module):
    def __init__(self, config: Qwen2MoeConfig):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

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
        return rearrange(x, "b h l d -> b l h d")

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


class Qwen2MoeAttention(nn.Module):
    def __init__(
        self, config: Qwen2MoeConfig, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config)
        self.tp_size = extra_config.tp_size or 1

        assert config.num_attention_heads % self.tp_size == 0

        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

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
            bias=True,
        )
        self.k_proj = nn.Linear(
            self.config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            self.config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.config.hidden_size,
            bias=False,
        )

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
        hidden_states = all_gather(hidden_states, self.extra_config)
        bsz, seq_len = hidden_states.shape[:2]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, seq_len, self.num_attention_heads, -1)
        key_states = key_states.view(bsz, seq_len, self.num_kv_heads, -1)
        value_states = value_states.view(bsz, seq_len, self.num_kv_heads, -1)

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

        raw_attn_output = attention(
            query_states,
            key_states,
            value_states,
            self.kv_cache,
            batch_state.position_ids,
            seq_len=batch_state.seq_len,
        )

        attn_output = raw_attn_output.reshape(bsz, seq_len, -1)

        o_proj = self.o_proj(attn_output)

        o_proj = reduce_scatter(o_proj, self.extra_config)

        with_residual = residual + o_proj

        batch_state.hidden_states = with_residual
        return batch_state


class Qwen2MoeMLP(nn.Module):
    def __init__(
        self,
        config: Qwen2MoeConfig,
        intermediate_size: int,
        extra_config: ExtraModelConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.tp_size = extra_config.tp_size
        assert self.config.intermediate_size % self.tp_size == 0
        # mark_maybe_wrong
        self.intermediate_size = intermediate_size // self.tp_size

        self.up_proj = nn.Linear(
            self.config.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.gate_proj = nn.Linear(
            self.config.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: Tensor,
    ):
        # inp = hidden_states
        # mark_maybe_wrong
        # assert inp is not None
        # hidden_states = all_gather(inp, self.extra_config)

        up = self.up_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        prod = F.silu(gate) * up
        down = self.down_proj(prod)

        down = reduce_scatter(down, self.extra_config)

        hidden_states = down
        return hidden_states


class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(
        self, config: Qwen2MoeConfig, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                Qwen2MoeMLP(
                    self.config,
                    config.moe_intermediate_size,
                    self.extra_config,
                    self.layer_idx,
                )
                for _ in range(self.num_experts)
            ]
        )

        self.shared_expert = Qwen2MoeMLP(
            self.config,
            config.shared_expert_intermediate_size,
            self.extra_config,
            self.layer_idx,
        )
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

        self.input_layernorm = RMSNorm(config)

    def forward(
        self,
        batch_state: BatchState,
    ) -> tuple[Tensor, Tensor]:
        inp = batch_state.hidden_states
        assert inp is not None
        hidden_states = self.input_layernorm(inp)
        hidden_states = all_gather(hidden_states, self.extra_config)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )

        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )

        with_residual = inp + final_hidden_states

        batch_state.hidden_states = with_residual
        return batch_state


class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(
        self, config: Qwen2MoeConfig, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.self_attn = Qwen2MoeAttention(config, extra_config, layer_idx)
        self.mlp = Qwen2MoeSparseMoeBlock(config, extra_config, layer_idx)

        # mark_print_test
        # print("mlp_only_layer_no: ", config.mlp_only_layers)

        # if (layer_idx not in config.mlp_only_layers) and (
        #     config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        # ):
        #     self.mlp = Qwen2MoeSparseMoeBlock(config, extra_config, layer_idx)
        # else:
        #     self.mlp = Qwen2MoeMLP(config, extra_config, layer_idx)

    def forward(self, batch_state: BatchState):
        out = self.self_attn(batch_state)
        out = self.mlp(out)
        return out


class Qwen2MoeLMHead(nn.Module):
    def __init__(self, config: Qwen2MoeConfig, extra_config: ExtraModelConfig):
        super().__init__()
        self.config = config
        self.extra_config = extra_config

        self.input_norm = RMSNorm(config)

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
            next_token_ids = all_gather(next_token_ids, self.extra_config)

        batch_state.output_ids = next_token_ids
        return batch_state


class Qwen2MoeEmbeddings(nn.Module):
    def __init__(self, config: Qwen2MoeConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

    def forward(self, batch_state: BatchState):
        hidden_states = self.embed_tokens(batch_state.input_ids)

        batch_state.hidden_states = hidden_states
        return batch_state


class Qwen2MoeModel(nn.Module):
    rope_cos: Tensor
    rope_sin: Tensor

    def __init__(
        self,
        config: Qwen2MoeConfig,
        extra_config: ExtraModelConfig,
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.embed_tokens = Qwen2MoeEmbeddings(config)
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(Qwen2MoeDecoderLayer(config, extra_config, i))

        self.layers = nn.ModuleList(layers)

        self.rope = Qwen2MoeRotaryEmbedding(
            config=config,
        )
        # self.head_dim = config.hidden_size // config.num_attention_heads

        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        dummy_float_input = torch.empty((0, config.hidden_size), dtype=torch.float32)

        cos, sin = self.rope(dummy_float_input, position_ids)
        assert cos.dtype == torch.float32
        assert sin.dtype == torch.float32
        self.register_buffer("rope_cos", cos.squeeze(0), persistent=False)
        self.register_buffer("rope_sin", sin.squeeze(0), persistent=False)

    def interleave_rope(self):
        indices_for_q_list = []
        half_head_dim = self.head_dim // 2
        for n in range(self.config.num_attention_heads):
            offset = n * self.head_dim
            for i in range(half_head_dim):
                indices_for_q_list.append(i + offset)
                indices_for_q_list.append(i + half_head_dim + offset)

        indices_for_q = torch.tensor(indices_for_q_list, device=self.rope_cos.device)
        one_head_indices = indices_for_q[: self.head_dim]

        self.rope_cos = self.rope_cos[..., one_head_indices]
        self.rope_sin = self.rope_sin[..., one_head_indices]

        indices_for_k = indices_for_q[: self.head_dim * self.config.num_key_value_heads]

        for mod in self.modules():
            if isinstance(mod, Qwen2MoeAttention):
                mod.q_proj.weight[:] = mod.q_proj.weight[indices_for_q]
                mod.k_proj.weight[:] = mod.k_proj.weight[indices_for_k]
                mod.q_proj.bias[:] = mod.q_proj.bias[indices_for_q]
                mod.k_proj.bias[:] = mod.k_proj.bias[indices_for_k]
                # mark_maybe_wrong
                # mod.q_norm.weight[:] = mod.q_norm.weight[indices_for_q]
                # mod.k_norm.weight[:] = mod.k_norm.weight[indices_for_k]

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
    qkv_proj_weights: Tensor
    qkv_proj_biases: Tensor
    o_proj: Tensor
    attn_ln_weight: Tensor
    mlp_ln_weight: Tensor
    up_proj: Tensor
    gate_proj: Tensor
    down_proj: Tensor
    gate: Tensor
    shared_up_proj: Tensor
    shared_gate_proj: Tensor
    shared_down_proj: Tensor
    shared_gate: Tensor


class Qwen2MoeForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen2MoeConfig,
        extra_config: ExtraModelConfig,
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.device = torch.get_default_device()
        self.dtype = torch.get_default_dtype()

        self.model = Qwen2MoeModel(config, extra_config)

        self.lm_head = Qwen2MoeLMHead(config, extra_config)

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
        self.head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )
        k_cache = torch.zeros(
            (
                self.config.num_hidden_layers,
                self.extra_config.max_batch_size,
                self.extra_config.max_len_override
                or self.config.max_position_embeddings,
                self.config.num_key_value_heads,
                self.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        v_cache = k_cache.clone()

        self.stacked_kv_cache = (k_cache, v_cache)

        for layer_idx in range(self.config.num_hidden_layers):
            layer: Qwen2MoeDecoderLayer = self.model.layers[layer_idx]
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

        config: Qwen2MoeConfig = Qwen2MoeConfig.from_pretrained(model_name_or_path)
        # print(config)
        if extra_config.rope_scaling is not None:
            config.rope_scaling = extra_config.rope_scaling

        if dtype is None:
            # mark_maybe_wrong
            # dtype = getattr(config, "torch_dtype", torch.float16)
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
            # name_to_hf_name[f"model.layers.{layer_idx}.self_attn.q_norm.weight"] = (
            #     f"model.layers.{layer_idx}.self_attn.q_norm.weight"
            # )
            # name_to_hf_name[f"model.layers.{layer_idx}.self_attn.k_norm.weight"] = (
            #     f"model.layers.{layer_idx}.self_attn.k_norm.weight"
            # )

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
                    # mark_maybe_wrong 专家路由线性层可能需要考虑张量并行，但是tp_size就是1，所以应该也无所谓
                    "gate.weight",
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

    # def stack_params(self):
    #     def stack_and_reassign(modules, prop: str):
    #         params = [getattr(m, prop) for m in modules]
    #         stacked = torch.stack(params, dim=0)
    #         for i, m in enumerate(modules):
    #             getattr(m, prop)[:] = stacked[i]
    #         return stacked

    #     layers: list[Qwen2MoeDecoderLayer] = self.model.layers
    #     self_attns = [x.self_attn for x in layers]
    #     mlps = [x.mlp for x in layers]

    #     o_projs = [x.o_proj for x in self_attns]

    #     self_attn_lns = [x.input_layernorm for x in self_attns]
    #     mlp_lns = [x.input_layernorm for x in mlps]

    #     gates = [x.gate for x in mlps]
    #     shared_gates = [x.shared_expert_gate for x in mlps]

    #     shared_expert_up_projs = []
    #     shared_expert_gate_projs = []
    #     shared_expert_down_projs = []

    #     expert_gate_projs = []
    #     expert_up_projs = []
    #     expert_down_projs = []
    #     for mlp in mlps:
    #         for expert in mlp.experts:
    #             expert_gate_projs.append(expert.gate_proj)
    #             expert_up_projs.append(expert.up_proj)
    #             expert_down_projs.append(expert.down_proj)

    #     stacked_o_proj = stack_and_reassign(o_projs, "weight")
    #     # print(o_projs[0].weight)
    #     stacked_q_norm = stack_and_reassign(q_norms, "weight")
    #     stacked_k_norm = stack_and_reassign(k_norms, "weight")
    #     stacked_self_attn_lns_weights = stack_and_reassign(self_attn_lns, "weight")
    #     stacked_mlp_lns_weights = stack_and_reassign(mlp_lns, "weight")
    #     stacked_gate = stack_and_reassign(gates, "weight")

    #     stacked_gate_proj = stack_and_reassign(expert_gate_projs, "weight")
    #     stacked_up_proj = stack_and_reassign(expert_up_projs, "weight")
    #     stacked_down_proj = stack_and_reassign(expert_down_projs, "weight")

    #     qkv_weights = []
    #     for self_attn in self_attns:
    #         cat_weight = torch.cat(
    #             [
    #                 self_attn.q_proj.weight,
    #                 self_attn.k_proj.weight,
    #                 self_attn.v_proj.weight,
    #             ],
    #             dim=0,
    #         )
    #         qkv_weights.append(cat_weight)

    #     stacked_qkv_weights = torch.stack(qkv_weights, dim=0)

    #     for i, self_attn in enumerate(self_attns):
    #         qkv_weight = stacked_qkv_weights[i]
    #         q_weight, k_weight, v_weight = qkv_weight.split(
    #             [
    #                 self.config.num_attention_heads * self.config.head_dim,
    #                 self.config.num_key_value_heads * self.config.head_dim,
    #                 self.config.num_key_value_heads * self.config.head_dim,
    #             ],
    #             dim=0,
    #         )
    #         self_attn.q_proj.weight[:] = q_weight
    #         self_attn.k_proj.weight[:] = k_weight
    #         self_attn.v_proj.weight[:] = v_weight

    #     self.stacked_params = StackedParams(
    #         qkv_proj_weights=stacked_qkv_weights,
    #         o_proj=stacked_o_proj,
    #         q_norm_weight=stacked_q_norm,
    #         k_norm_weight=stacked_k_norm,
    #         attn_ln_weight=stacked_self_attn_lns_weights,
    #         mlp_ln_weight=stacked_mlp_lns_weights,
    #         gate_proj=stacked_gate_proj,
    #         up_proj=stacked_up_proj,
    #         down_proj=stacked_down_proj,
    #         gate=stacked_gate,
    #     )

    def stack_params(self):
        self.head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )

        def stack_and_reassign(modules, prop: str):
            params = [getattr(m, prop) for m in modules]
            stacked = torch.stack(params, dim=0)
            for i, m in enumerate(modules):
                getattr(m, prop)[:] = stacked[i]
            return stacked

        layers: list[Qwen2MoeDecoderLayer] = self.model.layers  # type: ignore
        self_attns = [x.self_attn for x in layers]
        mlps = [x.mlp for x in layers]

        o_projs = [x.o_proj for x in self_attns]
        self_attn_lns = [x.input_layernorm for x in self_attns]
        mlp_lns = [x.input_layernorm for x in mlps]

        gates = [x.gate for x in mlps]
        shared_gates = [x.shared_expert_gate for x in mlps]

        shared_expert_up_projs = [x.shared_expert.up_proj for x in mlps]
        shared_expert_gate_projs = [x.shared_expert.gate_proj for x in mlps]
        shared_expert_down_projs = [x.shared_expert.down_proj for x in mlps]
        up_projs = []
        gate_projs = []
        down_projs = []

        for mlp in mlps:
            for expert in mlp.experts:
                gate_projs.append(expert.gate_proj)
                up_projs.append(expert.up_proj)
                down_projs.append(expert.down_proj)

        stacked_o_proj = stack_and_reassign(o_projs, "weight")
        stacked_self_attn_ln_weights = stack_and_reassign(self_attn_lns, "weight")
        stacked_mlp_ln_weights = stack_and_reassign(mlp_lns, "weight")

        stacked_shared_gate = stack_and_reassign(shared_gates, "weight")
        stacked_gate = stack_and_reassign(gates, "weight")

        stacked_shared_up_proj = stack_and_reassign(shared_expert_up_projs, "weight")
        stacked_shared_gate_proj = stack_and_reassign(
            shared_expert_gate_projs, "weight"
        )
        stacked_shared_down_proj = stack_and_reassign(
            shared_expert_down_projs, "weight"
        )

        stacked_gate_proj = stack_and_reassign(gate_projs, "weight")
        stacked_up_proj = stack_and_reassign(up_projs, "weight")
        stacked_down_proj = stack_and_reassign(down_projs, "weight")

        qkv_weights = []
        qkv_biases = []
        for self_attn in self_attns:
            cat_weight = torch.cat(
                [
                    self_attn.q_proj.weight,
                    self_attn.k_proj.weight,
                    self_attn.v_proj.weight,
                ],
                dim=0,
            )
            cat_bias = torch.cat(
                [
                    self_attn.q_proj.bias,
                    self_attn.k_proj.bias,
                    self_attn.v_proj.bias,
                ],
                dim=0,
            )
            qkv_weights.append(cat_weight)
            qkv_biases.append(cat_bias)

        stacked_qkv_weights = torch.stack(qkv_weights, dim=0)
        stacked_qkv_biases = torch.stack(qkv_biases, dim=0)

        for i, self_attn in enumerate(self_attns):
            qkv_weight = stacked_qkv_weights[i]
            q_weight, k_weight, v_weight = qkv_weight.split(
                [
                    self.config.num_attention_heads * self.head_dim,
                    self.config.num_key_value_heads * self.head_dim,
                    self.config.num_key_value_heads * self.head_dim,
                ],
                dim=0,
            )
            self_attn.q_proj.weight[:] = q_weight
            self_attn.k_proj.weight[:] = k_weight
            self_attn.v_proj.weight[:] = v_weight

            qkv_bias = stacked_qkv_biases[i]
            q_bias, k_bias, v_bias = qkv_bias.split(
                [
                    self.config.num_attention_heads * self.head_dim,
                    self.config.num_key_value_heads * self.head_dim,
                    self.config.num_key_value_heads * self.head_dim,
                ],
                dim=0,
            )
            self_attn.q_proj.bias[:] = q_bias
            self_attn.k_proj.bias[:] = k_bias
            self_attn.v_proj.bias[:] = v_bias

        self.stacked_params = StackedParams(
            qkv_proj_weights=stacked_qkv_weights,
            qkv_proj_biases=stacked_qkv_biases,
            o_proj=stacked_o_proj,
            attn_ln_weight=stacked_self_attn_ln_weights,
            mlp_ln_weight=stacked_mlp_ln_weights,
            up_proj=stacked_up_proj,
            down_proj=stacked_down_proj,
            gate_proj=stacked_gate_proj,
            gate=stacked_gate,
            shared_up_proj=stacked_shared_up_proj,
            shared_down_proj=stacked_shared_down_proj,
            shared_gate_proj=stacked_shared_gate_proj,
            shared_gate=stacked_shared_gate,
        )
