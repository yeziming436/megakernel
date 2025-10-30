import os    
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial

try:
    import thunderkittens as tk
    print(f"Successfully imported thunderkittens")
except:
    print("Could not import thunderkittens")

try:
    from layernorm.baselines.layer_norm_triton import layer_norm_fn, RMSNorm
    print(f"Successfully imported layer_norm_fn")
except:
    layer_norm_fn = None
    print("Could not import layer_norm_fn. Please obtain from FlashAttention repo.")


################## Layer norm ######################

def get_flops(batch, seqlen, headdim, nheads):
    f = batch*seqlen*nheads*headdim # compute the mask for dropout 
    f += batch*seqlen*nheads*headdim # add dropout and residual
    f += batch*seqlen*nheads*headdim # compute the mean
    f += batch*seqlen*nheads*headdim # compute the variance
    f += batch*seqlen*nheads*headdim # subtract mean
    f += batch*seqlen*nheads*headdim # divide by variance
    f += batch*seqlen*nheads*headdim # multiply by norm weight 
    f += batch*seqlen*nheads*headdim # add norm bias
    return f


def get_layer_norm_inputs(b, h, n, dv, dt):
    d_model = h * dv
    p = 0.1

    norm = nn.LayerNorm(d_model).cuda()
    dropout = nn.Dropout(p)

    x = torch.randn(b, n, d_model).to(dtype=dt, device='cuda')
    residual = torch.randn(b, n, d_model).to(dtype=dt, device='cuda')
    norm_weight = norm.weight.to(dtype=torch.bfloat16)
    norm_bias = norm.bias.to(dtype=torch.bfloat16)
    out = torch.zeros_like(x)
    out_resid = torch.zeros_like(x)
    return x, residual, norm_weight, norm_bias, out, out_resid, norm, dropout

class pytorch_layernorm(torch.nn.Module):
    def __init__(self, d_model, p):
        super().__init__()
        self.dropout = nn.Dropout(p).cuda()
        self.norm = nn.LayerNorm(d_model).cuda()
        
    def forward(self, x, residual, norm_weight, norm_bias, dropout_p):
        with torch.no_grad():
            dropped = self.dropout(x) 
            residual = (residual + dropped ) if residual is not None else dropped
            y = self.norm(residual.to(dtype=self.norm.weight.dtype))
            residual = residual.to(torch.float32)
        return y, residual
        
def layernorm_test(dt, b, h, n, dv, causal, is_forwards, method_str, num_iters=10, verbose=True, torch_compile=True, **kwargs):
    
    pytorch_method = pytorch_layernorm(d_model=h*dv, p=0.1)
    if torch_compile and method_str == "pytorch_layernorm":
        try:
            pytorch_method = torch.compile(pytorch_method)
        except Exception as e:
            print(f"Could not compile pytorch_layernorm: {e}")
                     
    for stage in ['warmup', 'timed']:
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    
        for i in range(num_iters):
            (
                x, residual, norm_weight, norm_bias, 
                out, out_resid, norm, dropout 
            ) = get_layer_norm_inputs(b, h, n, dv, dt)
            rowscale = torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype, )
            residual_in_fp32 = True
            dtype = dt
            dropout_p = dropout.p
                
            if True:
                if method_str == 'pytorch_layernorm':
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y, residual = pytorch_method(x, residual, norm_weight, norm_bias, dropout_p)
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == 'triton_layernorm':
                    torch.cuda.synchronize()
                    start_events[i].record()
                    with torch.no_grad():
                        y, residual = layer_norm_fn(
                            x,
                            norm.weight,
                            norm.bias,
                            residual=residual,
                            eps=norm.eps,
                            dropout_p=dropout.p,
                            rowscale=rowscale,
                            prenorm=True,
                            residual_in_fp32=residual_in_fp32,
                            is_rms_norm=False
                        )
                    end_events[i].record()
                    torch.cuda.synchronize()
                
                elif method_str == 'tk_layernorm': 
                    torch.cuda.synchronize()
                    start_events[i].record()
                    with torch.no_grad():
                        y, residual = tk.fused_layernorm(
                            x, residual, 
                            norm_weight, norm_bias, 
                            dropout.p,
                        )
                    end_events[i].record()
                    torch.cuda.synchronize()

                else:
                    raise ValueError(f"Unknown method: {method_str}")

            # except Exception as e:
            #     if verbose:
            #         print(f"Error: {e}")
            #     return None, -1

            torch.cuda.empty_cache()

    assert not np.isnan(y.detach().float().cpu()).any(), "NaN values detected in output 'o'"
    assert not np.isinf(y.detach().float().cpu()).any(), "Inf values detected in output 'o'"
    tot = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)])/num_iters
    return y, tot


IMPLEMENTATIONS = {
    'pytorch_layernorm': partial(layernorm_test, causal=True, is_forwards=True, method_str='pytorch_layernorm'),
    'triton_layernorm': partial(layernorm_test, causal=True, is_forwards=True, method_str='triton_layernorm'),
    'tk_layernorm': partial(layernorm_test, causal=True, is_forwards=True, method_str='tk_layernorm'),
}

NAME = "LAYER NORM"
