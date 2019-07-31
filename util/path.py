"""Utility functions regarding paths."""
from pathlib import Path


def get_results_path(results_path, experiment, prefix, mode):
    """Builds appropriate path for storing experiments results."""
    if not experiment:
        raise ArgumentError('experiment cannot be empty')
    if not prefix:
        raise ArgumentError('prefix cannot be empty')
    if not prefix:
        raise ArgumentError('mode cannot be empty')
    return Path(results_path) / experiment / prefix / mode
