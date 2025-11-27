"""
Configuration utilities using Hydra
"""

import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Optional


def setup_config(config_path: str = "../configs", config_name: str = "config"):
    """
    Setup Hydra configuration
    
    Args:
        config_path: Path to config directory
        config_name: Name of main config file (without .yaml)
    """
    hydra.initialize(config_path=config_path, version_base=None)
    cfg = hydra.compose(config_name=config_name)
    return cfg


def save_config(cfg: DictConfig, save_path: str):
    """Save configuration to file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        OmegaConf.save(cfg, f)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from file"""
    return OmegaConf.load(config_path)


def merge_configs(base_cfg: DictConfig, override_cfg: DictConfig) -> DictConfig:
    """Merge two configurations"""
    return OmegaConf.merge(base_cfg, override_cfg)


def get_experiment_dir(cfg: DictConfig, base_dir: str = "experiments") -> Path:
    """
    Create experiment directory based on config
    
    Args:
        cfg: Configuration
        base_dir: Base directory for experiments
        
    Returns:
        exp_dir: Path to experiment directory
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg.get('experiment_name', 'exp')
    
    exp_dir = Path(base_dir) / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    save_config(cfg, str(exp_dir / "config.yaml"))
    
    return exp_dir


# TODO: Add config validation
# TODO: Add config templating
# TODO: Add environment variable substitution
