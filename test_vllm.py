# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

# MODEL_PATH = "/home/weiqiang/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
MODEL_PATH = (
    "/home/ziming/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-1B-Instruct"
)
MAX_TOKENS = 1024
NUM_TRIALS = 10

# Sample prompts.
prompts = ["tell me a 1200-word story about China"]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=MAX_TOKENS)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def main():
    # Create an LLM.
    llm = LLM(model=MODEL_PATH)
    total_time = 0.0
    total_tokens = 0
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    for i in range(NUM_TRIALS):  # run multiple times to warm up
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed_time = time.perf_counter() - start_time
        total_time += elapsed_time
        # Print the outputs.
        print("\nGenerated Outputs:\n" + "-" * 60)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text

            output_ids = tokenizer(
                generated_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            Output_len = output_ids.shape[-1]

        total_tokens += Output_len

        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)
        print(f"Length:    {Output_len} tokens")

    print(f"Num_tokens:    {total_tokens/NUM_TRIALS} tokens")
    print(f"Speed:     {total_tokens/total_time:.2f} tokens/s")


if __name__ == "__main__":
    main()
