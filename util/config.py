"""Utility functions regarding configuration."""
from importlib.machinery import SourceFileLoader


def load_config(config_path):
    """Loads configuration from python module."""
    return SourceFileLoader('cf', config_path).load_module()
