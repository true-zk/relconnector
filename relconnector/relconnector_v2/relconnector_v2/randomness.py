"""Guard PyTorch's process-global CPU RNG in the threaded executor.

pyg-lib's sampling operator has no generator argument. Custom trainers may also
draw from this RNG, so those calls must not overlap with sampler save/restore.
"""

from threading import RLock

TORCH_RNG_LOCK = RLock()
