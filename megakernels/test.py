from transformers import AutoModelForCausalLM, AutoConfig

# config = AutoModelForCausalLM.from_pretrained(
#     "/home/ziming/.cache/modelscope/hub/models/facebook/opt-6.7b"
# )

config = AutoConfig.from_pretrained(
    "/home/ziming/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-1B-Instruct"
)
print(config)
