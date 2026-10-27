"""
DendriticMemorySystems
======================

Bio-inspired persistent memory architecture for large language models.

Provides distributed local evaluation, associative retrieval, provenance-aware
storage, and episodic-to-semantic consolidation as an external memory layer.

Author: David Burmeister <davidburm@gmail.com>
GitHub: https://github.com/deadSwank001/DendriticMemorySystems
"""

from dendritic_memory.interfaces.memory_interface import DendriticMemorySystem

__version__ = "0.1.0"
__author__ = "David Burmeister"
__email__ = "davidburm@gmail.com"

__all__ = ["DendriticMemorySystem"]
