"""2026-11-17 resource policy: available RAM is telemetry, not an execution gate.

Earlier probe modules and their pinned limits remain historical evidence.
Future runners use this explicit policy, retaining worker/GPU/disk supervision.
"""
import os
import subprocess
from .xgb_full_row_sizing_probe import LIMITS as PREVIOUS_LIMITS
from .cuda_readiness_probe import parse_gpu
from experiments.c03.producer.plan import require

LIMITS = {**PREVIOUS_LIMITS, 'start_available': None, 'min_available': None,
          'startup_query_timeout': 10.}


def query_gpu_start():
    """Allow driver wake-up before a worker exists; live polling stays at 2 s."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '0'), 'Unsupported CUDA device remapping')
    result = subprocess.run(['nvidia-smi',
        '--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version',
        '--format=csv,noheader,nounits'], capture_output=True, text=True,
        timeout=LIMITS['startup_query_timeout'], check=True)
    return parse_gpu(result.stdout)


def start_reason(state, gpu=None, *, limits=LIMITS):
    if state['disk_free'] < limits['start_disk']:
        return 'Free disk below start requirement'
    if gpu is not None and gpu['free'] < limits['start_gpu_free']:
        return 'Free GPU memory below start requirement'


def stop_reason(state, gpu, rss, size, elapsed, *, limits=LIMITS):
    if elapsed >= limits['seconds']:
        return 'Shared time budget reached'
    if rss > limits['max_rss']:
        return 'Worker-tree RSS above ceiling'
    if state['disk_free'] < limits['min_disk']:
        return 'Free disk below runtime floor'
    if size > limits['max_artifacts']:
        return 'New artifacts above budget'
    if gpu is not None and gpu['used'] > limits['max_gpu_used']:
        return 'Total device-used VRAM above ceiling'
    if gpu is not None and gpu['free'] < limits['min_gpu_free']:
        return 'Free VRAM below runtime floor'
