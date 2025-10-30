import os    
import pandas as pd
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import warnings
warnings.filterwarnings("ignore", message=".*not a leaf Tensor is being accessed.*")
warnings.filterwarnings("ignore", message=".*no current CUDA context.*")

############## Our Imports #############
from utils import efficiency
import attention.implementations as attention
import hedgehog.implementations as hedgehog
import based_attention.implementations as based
import rotary.implementations as rotary
import mamba2.implementations as mamba2
import fftconv.implementations as fftconv
import layernorm.implementations as layernorm
############## Efficiency Measurements #############

b = 16
h = 16
dv = 64

def measure_efficiency(dt, n, method_name, method, verbose=False, torch_compile=True):
    if verbose:
        print(f"{b=}, {n=}, {h=}, {dv=}")

    if 'c=t' in method_name:
        causal = True
        flops = mod.get_flops(b, n, dv, h, causal=('causal'), mode='bwd' if 'bwd' in method_name else 'fwd')
    elif 'c=f' in method_name:
        causal = False
        flops = mod.get_flops(b, n, dv, h, causal=causal, mode='bwd' if 'bwd' in method_name else 'fwd')
    else:
        flops = mod.get_flops(b, n, dv, h)

    outputs, times = method(dt, b, h, n, dv, verbose=verbose, torch_compile=torch_compile)
    times = times * 1000

    eff = efficiency(flops, times)
    if verbose:
        print(f"Method {method_name} -- Efficiency: {eff:.2f} TFLOPS, Time: {times:.4f} us and FLOPS: {flops:.2f}")
    torch.cuda.empty_cache()
    return eff, times

if __name__ == "__main__":
    print("Benchmarking the kernels...")

    verbose = False
    torch_compile = True

    for mod in [
        attention, 
        based, 
        rotary, 
        hedgehog, 
        layernorm, 
        mamba2, 
        fftconv
    ]:
        implementations_list = []
        implementations_fwd = mod.IMPLEMENTATIONS
        implementations_list.append(implementations_fwd)
        name = mod.NAME
        print("============" * 4, name, "============" * 4)

        try:
            implementations_bwd = mod.IMPLEMENTATIONS_BWD
            implementations_list.append(implementations_bwd)
        except:
            pass

        for implementations in implementations_list:
            method2tflops = {}
            method2timing = {}

            for m, method in implementations.items():
                flops_result, timing_result = {},  {}
                if verbose:
                    print(f"Method: {m}")
                for n in [
                    1024 if 'attn' not in m else 768, 
                    2048 if 'attn' not in m else 1536, 
                    4096 if 'attn' in m else 3072,
                    8192 if 'attn' in m else 6144,
                    16384 if 'attn' in m else 12288
                ]:
                    if "conv" in m and n not in [1024, 4096]:
                        # restrict to sizes we have implemented
                        continue
                    if "mamba2_triton" in m and n not in [1024, 2048, 4096, 8192]:
                        # the kernel results in DEVICE_SIDE_ASSERTS
                        continue
                    if "layernorm" in m and dv*h != 1024:
                        # restrict to sizes we have implemented
                        print('skipping layernorm due to incompatible model dim')
                    if verbose:
                        print(f"Sequence Length: {n}")
                    tflops, timing = measure_efficiency(torch.bfloat16, n, m, method, verbose=verbose, torch_compile=torch_compile)
                    if tflops > 0: 
                        flops_result[n] = tflops
                        timing_result[n] = timing
                method2tflops[m] = flops_result
                method2timing[m] = timing_result

            # print table
            df = pd.DataFrame(method2tflops).replace(np.nan, 'OOM', regex=True)
            print(df)
