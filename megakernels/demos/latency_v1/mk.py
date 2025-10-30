import torch
from einops import rearrange

from megakernels.demos.latency.instructions import Globals
from megakernels.mk import MK_Interpreter


def interpret_with_mk(
    globs: Globals,
    mk_func,
):
    fourD_k_cache = rearrange(globs.k_cache, "l b t h d -> (l b) t h d")
    fourD_v_cache = rearrange(globs.v_cache, "l b t h d -> (l b) t h d")

    mk_func(
        # vm stuff
        globs.barriers,
        globs.instructions,
        globs.timings,
        # weights
        globs.qkv_proj_weights,
        globs.attn_ln_weights,
        globs.o_proj_weights,
        globs.mlp_ln_weights,
        globs.up_proj_weights,
        globs.gate_proj_weights,
        globs.down_proj_weights,
        globs.lm_head_norm_weights.data,
        globs.lm_head_weights.data,
        fourD_k_cache,
        fourD_v_cache,
        # rope
        globs.rope_cos,
        globs.rope_sin,
        # activations
        globs.hidden_states,
        globs.post_ln_rope_q,
        globs.attn_out,
        globs.attn_lse_intermediates,
        globs.attn_out_intermediates,
        globs.silu_out,
        globs.logits,
        # scalars
        globs.pos_id,
        globs.attn_scale,
        globs.rms_norm_eps,
        globs.skip_attn_reduction,
        stream=torch.cuda.current_stream(),
    )

    # with open('timings_output.txt', 'w') as f:
    #     for sm_idx in range(globs.timings.shape[0]):
    #         f.write(f"===== SM {sm_idx} =====\n")

    #         for instr_idx in range(globs.timings.shape[1]):
    #             f.write(f"----- Instruction {instr_idx} -----\n")

    #             for event_idx in range(globs.timings.shape[2]):
    #                 event_value = globs.timings[sm_idx, instr_idx, event_idx]
    #                 if event_value != 0:  # 只打印不为0的event
    #                     f.write(f"Time event {event_idx}: {event_value}\n")


class LatencyMKV1_Interpreter(MK_Interpreter):
    def interpret(self, globs: Globals):
        interpret_with_mk(globs, self.mk_func)
