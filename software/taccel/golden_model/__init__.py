"""Golden-model simulators.

W8A16 is the default shipping mode; W8A32 is the FP32 weight-quant
ceiling reference. The base ``MachineState`` / ``Simulator`` classes
remain in this package as internal infrastructure that the mode-specific
sims inherit from for byte-mover ops (LOAD/STORE/BUF_COPY) and control
flow (CONFIG_TILE, SYNC, HALT) — both of which are mode-agnostic.
"""
from .simulator_w8a16 import SimulatorW8A16
from .simulator_w8a32 import SimulatorW8A32
from .state_w8a16 import MachineStateW8A16
from .state_w8a32 import MachineStateW8A32
