"""Configuration loading and validation.

Reads YAML config files and provides defaults for any missing fields.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    yaml.YAMLError
        If the file is not valid YAML.
    """
    # TODO:
    #   1. resolve path
    #   2. open and parse with yaml.safe_load
    #   3. merge with defaults (call _get_defaults)
    #   4. return merged config
    raise NotImplementedError


def _get_defaults() -> dict:
    """Return default values for all config sections.

    These are used as fallbacks when the user's config file omits a field.

    Returns
    -------
    dict
        Full default config.
    """
    # TODO: return a hardcoded dict matching the structure of default.yaml
    raise NotImplementedError


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*.

    Values in *override* take precedence.  Nested dicts are merged
    recursively rather than replaced wholesale.

    Parameters
    ----------
    base : dict
    override : dict

    Returns
    -------
    dict
        Merged dictionary.
    """
    # TODO: recursive dict merge
    raise NotImplementedError
