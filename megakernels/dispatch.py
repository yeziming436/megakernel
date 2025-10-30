from pathlib import Path

from megakernels.demos.latency.mk import LatencyMK_Interpreter
from megakernels.demos.latency.python_vm import (
    INSTRUCTION_TO_SOLVER as LATENCY_INSTRUCTION_TO_SOLVER,
)
from megakernels.demos.latency.scheduler import LatencyScheduleBuilder

from megakernels.demos.throughput.mk import ThroughputMK_Interpreter
from megakernels.demos.throughput.python_vm import (
    INSTRUCTION_TO_SOLVER as THROUGHPUT_INSTRUCTION_TO_SOLVER,
)
from megakernels.demos.throughput.scheduler import ThroughputScheduleBuilder

from megakernels.mk import MK_Interpreter
from megakernels.python_vm import PyVM_Interpreter
from megakernels.scheduler import ScheduleBuilder

from megakernels.demos.llama3b.mk import LLAMA3B_Interpreter
from megakernels.demos.llama3b.python_vm import (
    INSTRUCTION_TO_SOLVER as LLAMA3B_INSTRUCTION_TO_SOLVER,
)
from megakernels.demos.llama3b.scheduler import LLAMA3BScheduleBuilder

from megakernels.demos.qwen1_5b.mk import Qwen1_5B_Interpreter
from megakernels.demos.qwen1_5b.python_vm import (
    INSTRUCTION_TO_SOLVER as QWEN1_5B_INSTRUCTION_TO_SOLVER,
)
from megakernels.demos.qwen1_5b.scheduler import Qwen1_5BScheduleBuilder

from megakernels.demos.qwen3b.mk import Qwen3B_Interpreter
from megakernels.demos.qwen3b.python_vm import (
    INSTRUCTION_TO_SOLVER as QWEN3B_INSTRUCTION_TO_SOLVER,
)
from megakernels.demos.qwen3b.scheduler import Qwen3BScheduleBuilder

from megakernels.demos.latency_v1.mk import LatencyMKV1_Interpreter
from megakernels.demos.latency_v1.python_vm import (
    INSTRUCTION_TO_SOLVER as LATENCY_V1_INSTRUCTION_TO_SOLVER,
)
from megakernels.demos.latency_v1.scheduler import LatencyV1ScheduleBuilder

from megakernels.demos.llama3b_v1.mk import LLAMA3BV1_Interpreter
from megakernels.demos.llama3b_v1.python_vm import (
    INSTRUCTION_TO_SOLVER as LLAMA3B_V1_INSTRUCTION_TO_SOLVER,
)
from megakernels.demos.llama3b_v1.scheduler import LLAMA3BV1ScheduleBuilder

BUILDER_MAP = {
    "latency": LatencyScheduleBuilder,
    "throughput": ThroughputScheduleBuilder,
    "llama3b": LLAMA3BScheduleBuilder,
    "qwen1_5b": Qwen1_5BScheduleBuilder,
    "qwen3b": Qwen3BScheduleBuilder,
    "latency_v1": LatencyV1ScheduleBuilder,
    "llama3b_v1": LLAMA3BV1ScheduleBuilder,
}

MK_INTERPRETER_MAP = {
    "latency": LatencyMK_Interpreter,
    "throughput": ThroughputMK_Interpreter,
    "llama3b": LLAMA3B_Interpreter,
    "qwen1_5b": Qwen1_5B_Interpreter,
    "qwen3b": Qwen3B_Interpreter,
    "latency_v1": LatencyMKV1_Interpreter,
    "llama3b_v1": LLAMA3BV1_Interpreter,
}

INSTRUCTION_TO_SOLVER_MAP = {
    "latency": LATENCY_INSTRUCTION_TO_SOLVER,
    "throughput": THROUGHPUT_INSTRUCTION_TO_SOLVER,
    "llama3b": LLAMA3B_INSTRUCTION_TO_SOLVER,
    "qwen1_5b": QWEN1_5B_INSTRUCTION_TO_SOLVER,
    "qwen3b": QWEN3B_INSTRUCTION_TO_SOLVER,
    "latency_v1": LATENCY_V1_INSTRUCTION_TO_SOLVER,
    "llama3b_v1": LLAMA3B_V1_INSTRUCTION_TO_SOLVER,
}


def make_schedule_builder(mode: str) -> ScheduleBuilder:
    print(f"Using schedule builder: {mode}")
    return BUILDER_MAP[mode]()


def make_mk_interpreter(mode: str, mk_dir: Path) -> MK_Interpreter:
    return MK_INTERPRETER_MAP[mode](mk_dir)


def make_pyvm_interpreter(mode: str) -> PyVM_Interpreter:
    return PyVM_Interpreter(INSTRUCTION_TO_SOLVER_MAP[mode])
