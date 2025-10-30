"""
This example demonstrates how to provide tokenized ids to LLM as input instead of text prompt, i.e. a token-in-token-out workflow.
"""

import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer
import time

# MODEL_PATH = "/home/weiqiang/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
MODEL_PATH = (
    "/home/ziming/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-1B-Instruct"
)
MAX_TOKENS = 1024
NUM_TRIALS = 10


def main():
    # Sample prompts.
    prompts = ["tell me a 1200-word story about China"]
    # Create a sampling params object.
    sampling_params = {"temperature": 0.6, "top_p": 0.9, "max_new_tokens": MAX_TOKENS}

    # Tokenize inputs
    tokenizer = get_tokenizer(MODEL_PATH)
    token_ids_list = [tokenizer.encode(prompt) for prompt in prompts]

    # Create an LLM.
    llm = sgl.Engine(model_path=MODEL_PATH, skip_tokenizer_init=True)

    total_time = 0.0
    total_tokens = 0

    warm_up_ids = tokenizer.encode("hi")
    warm_up = llm.generate(input_ids=warm_up_ids, sampling_params=sampling_params)

    for i in range(NUM_TRIALS):  # run multiple times to warm up
        start_time = time.perf_counter()
        outputs = llm.generate(
            input_ids=token_ids_list, sampling_params=sampling_params
        )
        elapsed_time = time.perf_counter() - start_time
        total_time += elapsed_time
        # Print the outputs.
        for prompt, output in zip(prompts, outputs):
            decode_output = tokenizer.decode(output["output_ids"])

            output_ids = tokenizer(
                decode_output, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            Output_len = output_ids.shape[-1]
            total_tokens += Output_len

            print("===============================")
            print(
                f"Prompt: {prompt}\nGenerated token ids: {output['output_ids']}\nGenerated text: {decode_output}"
            )
            print(
                f"Response length: {Output_len}, Speed: {Output_len/elapsed_time:.2f} tokens/s"
            )

    print(f"Avg num tokens:    {total_tokens/NUM_TRIALS} tokens")
    print(f"Avg speed:     {total_tokens/total_time:.2f} tokens/s")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()
