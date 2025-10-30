from pathlib import Path

import pydra
import torch
from art import text2art
from transformers import AutoTokenizer, GenerationConfig

from megakernels.dispatch import (
    make_mk_interpreter,
    make_schedule_builder,
    make_pyvm_interpreter,
)
from megakernels.gen4verifier import MK_Generator, PyTorchGenerator, PyVM_Generator
from megakernels.AutoModel import AutoForCausalLM
from megakernels.model_types import BatchState, ExtraModelConfig
from megakernels.scheduler import (
    assign_to_sms,
    tensorize_instructions,
)


class ScriptConfig(pydra.Config):
    model: str = (
        "/home/ziming/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-1B-Instruct"
    )
    device: str = "cuda:0"
    max_tokens_per_turn: int = 128
    mk_dir: Path = Path(__file__).parent.parent.parent / "demos" / "low-latency-llama"
    mode: str = "mk"
    # mode: str = "pyvm"
    # mode: str = "torch"
    sched: str = "rr"
    setting: str = "latency"
    memory_fraction: float | None = None
    print_prompt_len: bool = False


@torch.inference_mode()
def main(config: ScriptConfig):
    torch.cuda.set_device(config.device)

    tokenizer = AutoTokenizer.from_pretrained(config.model)

    generation_config = GenerationConfig.from_pretrained(config.model)
    eos_token_ids = generation_config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    extra_config = ExtraModelConfig(
        model_type="llama",
        interleave_rope=True,
        max_batch_size=1,
    )
    model = AutoForCausalLM.from_pretrained(
        config.model, device=config.device, extra_config=extra_config
    )

    messages = []

    schedule_builder = make_schedule_builder(config.setting)
    schedule = schedule_builder.build(model)
    assigned_to_sms = assign_to_sms(
        config.sched, schedule=schedule, memory_fraction=config.memory_fraction
    )
    tensorize_instructions(schedule.globs, assigned_to_sms)

    interpreter_test = make_pyvm_interpreter(config.setting)
    gen_test = PyVM_Generator(model, interpreter_test, schedule)

    gen_ver = PyTorchGenerator(model)

    output_tokens_test = torch.zeros(
        1, config.max_tokens_per_turn, device=model.device, dtype=torch.long
    )

    output_tokens_ver = torch.zeros(
        1, config.max_tokens_per_turn, device=model.device, dtype=torch.long
    )

    def generate(messages):
        tok_inp = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        input_ids = tokenizer(tok_inp, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(model.device)
        prompt_len = input_ids.shape[-1]
        if config.print_prompt_len:
            print(f"Prompt length: {prompt_len}")

        position_ids = torch.arange(prompt_len).to(model.device)

        prefill_inp = BatchState(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        prefill_output: BatchState = model(prefill_inp)
        assert prefill_output.output_ids is not None
        new_input_token = prefill_output.output_ids[:, -1:]

        output_tokens_test[:, 0] = new_input_token

        output_tokens_ver[:, 0] = new_input_token

        output_pyvm = gen_test.generate_with_eos(
            output_tokens=output_tokens_test,
            prompt_len=prompt_len,
            ntok=config.max_tokens_per_turn,
            eos_token_ids=eos_token_ids,
            eos_token_check_interval=16,
        )
        torch.cuda.synchronize()

        output_pytorch = gen_ver.generate_with_eos(
            output_tokens=output_tokens_ver,
            prompt_len=prompt_len,
            ntok=config.max_tokens_per_turn,
            eos_token_ids=eos_token_ids,
            eos_token_check_interval=16,
        )
        torch.cuda.synchronize()

        return (
            output_pyvm,
            output_pytorch,
        )

    # warmup
    generate([{"role": "user", "content": "hi"}])

    startup_message = "you have\nbeen granted\nan audience\nwith the\nmegakernel"
    print(text2art(startup_message.replace(" ", "   ")))

    while True:
        user_input = input(">>> ")
        messages.append({"role": "user", "content": user_input})

        (
            output_pyvm,
            output_pytorch,
        ) = generate(messages)

        acc = 0
        total = len(output_pyvm)

        for item in output_pyvm:
            if item in output_pytorch:
                acc += 1
                output_pytorch.remove(item)

        accuracy = acc / total if total > 0 else 0.0
        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    pydra.run(main)
