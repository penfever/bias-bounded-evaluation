"""Judge interfaces for dynamic sensitivity estimation."""

from .standard_interface import create_judge_function
from .oumi_interface import create_oumi_judge_function

__all__ = [
    'create_judge_function',
    'create_oumi_judge_function'
]