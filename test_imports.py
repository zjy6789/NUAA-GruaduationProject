#!/usr/bin/env python3
"""Test script to check imports and basic functionality."""

try:
    import sys
    print(f"Python path: {sys.path}")

    print("Testing imports...")
    from harl_wrapper import HARLAgentWrapper
    print("✓ HARLAgentWrapper imported")

    from sc2_env_adapter import SC2MultiAgentEnv
    print("✓ SC2MultiAgentEnv imported")

    from trainer import Trainer
    print("✓ Trainer imported")

    import torch
    print(f"✓ PyTorch version: {torch.__version__}")

    import gym
    print(f"✓ Gym version: {gym.__version__}")

    print("All imports successful!")

except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Other error: {e}")
    sys.exit(1)
