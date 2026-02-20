"""
chemberta4 - A minimal library for molecular property prediction with OLMo

Following the nanochat philosophy: explicit over implicit, minimal abstraction.
"""

from .data import PretrainingDataset, InstructionDataset
from .trainer import OLMoPretrainer

