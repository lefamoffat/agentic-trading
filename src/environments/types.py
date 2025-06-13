#!/usr/bin/env python3
"""
Types for reinforcement learning environments.
"""
from enum import Enum


class Position(Enum):
    """
    Trading position.
    """
    SHORT = 0
    FLAT = 1
    LONG = 2 