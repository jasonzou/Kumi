"""
Configuration Manager for MD2QA Enhanced Pipeline

Handles loading, merging, and validating configuration from multiple sources:
1. Default configuration (built-in)
2. User configuration file
3. Environment variables
4. Command-line arguments
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigManager:
    """Manages configuration loading and merging from multiple sources"""

    def __init__(self):
        self._config = {}
        self._sources = []

    def load_config(
        self,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        env_prefix: str = "MD2QA_"
    ) -> Dict[str, Any]:
        """
        Load configuration from multiple sources and merge them

        Args:
            config_path: Path to user config file (YAML or JSON)
            overrides: Dictionary of override values
            env_prefix: Prefix for environment variables

        Returns:
            Merged configuration dictionary
        """
        self._config = {}

        # 1. Load default configuration
        default_config = self._load_default_config()
        self._config = default_config
        self._sources.append("default")

        # 2. Load user configuration file
        if config_path and os.path.exists(config_path):
            user_config = self._load_config_file(config_path)
            self._config = self._deep_merge(self._config, user_config)
            self._sources.append(f"file:{config_path}")

        # 3. Load environment variables
        env_config = self._load_env_vars(env_prefix)
        if env_config:
            self._config = self._deep_merge(self._config, env_config)
            self._sources.append("environment")

        # 4. Apply command-line overrides
        if overrides:
            # Apply each override using _set_nested_value for proper nesting
            for key, value in overrides.items():
                self._set_nested_value(self._config, key, value)
            self._sources.append("overrides")

        # 5. Validate configuration
        self._validate_config()

        return self._config

    def _load_default_config(self) -> Dict[str, Any]:
        """Load built-in default configuration"""
        # Default configuration as a dictionary
        return {
            "pipeline": {
                "steps": [1, 2, 3],
                "resume": True,
                "preview": False,
                "version": "1.0"
            },
            "input": {
                "folder": "./input",
                "pattern": "**/*",
                "exclude": ["*.tmp", "*.log", ".DS_Store"],
                "extensions": [".md", ".txt"],
                "recursive": True
            },
            "step1": {
                "enabled": True,
                "chunk_size": 350,
                "min_chunk_size": 100,
                "output_format": "csv",
                "preserve_formatting": True,
                "handle_images": True
            },
            "step2": {
                "enabled": True,
                "rounds": 3,
                "max_workers": 5,
                "model": "qwen/qwen3-vl-235b-a22b-instruct",
                "temperature": 0.7,
                "max_tokens": None,
                "timeout": 300,
                "api_retries": 3,
                "api_retry_delay": 5,
                "api_retry_backoff": 2
            },
            "step3": {
                "enabled": True,
                "merge_strategy": "concat",
                "track_sources": True,
                "source_column": "source_file"
            },
            "output": {
                "folder": "./output",
                "create_subfolders": True,
                "backup_existing": False,
                "encoding": "utf-8-sig",
                "compression": None
            },
            "validation": {
                "check_outputs": True,
                "retry_failed": True,
                "max_retries": 3,
                "retry_delay": 5,
                "retry_backoff": 2,
                "strictness": "normal",
                "skip_invalid": False
            },
            "state": {
                "file": "./state/pipeline_state.json",
                "backup": True,
                "incremental_save": True,
                "format": "json"
            },
            "logging": {
                "level": "INFO",
                "file": "logs/pipeline.log",
                "console": True,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "max_size": 100,
                "backup_count": 5,
                "error_file": "logs/errors.log"
            },
            "metrics": {
                "enabled": True,
                "output": "reports/metrics.json",
                "summary_report": "reports/summary.txt",
                "html_report": None,
                "performance": True,
                "quality": True,
                "resources": True
            },
            "progress": {
                "enabled": True,
                "position": 0,
                "unit": "file",
                "show_eta": True,
                "show_percentage": True
            },
            "resources": {
                "max_memory": 2048,
                "max_cores": None,
                "temp_dir": "./tmp",
                "cleanup_temp": True
            }
        }

    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    def _load_env_vars(self, prefix: str) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}
        prefix_len = len(prefix)

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # Convert MD2QA_STEP1_CHUNK_SIZE to step1.chunk_size
            config_key = key[prefix_len:].lower().replace('_', '.')
            config_key = config_key.replace('..', '.')  # Handle cases like STEP1.ROUNDs

            # Try to parse value
            parsed_value = self._parse_env_value(value)

            # Set nested value
            self._set_nested_value(config, config_key, parsed_value)

        return config

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value"""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        # Boolean values
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False

        # Numeric values
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """Set a nested dictionary value using dot notation"""
        keys = key.split('.')
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _validate_config(self):
        """Validate configuration values"""
        # Validate steps
        if 'pipeline' in self._config:
            steps = self._config['pipeline'].get('steps', [])
            if not all(step in [1, 2, 3] for step in steps):
                raise ValueError("Pipeline steps must be a list of 1, 2, and/or 3")

        # Validate chunk sizes
        if 'step1' in self._config:
            chunk_size = self._config['step1'].get('chunk_size', 0)
            min_chunk_size = self._config['step1'].get('min_chunk_size', 0)
            if chunk_size <= 0:
                raise ValueError("chunk_size must be positive")
            if min_chunk_size < 0:
                raise ValueError("min_chunk_size must be non-negative")
            if min_chunk_size > chunk_size:
                raise ValueError("min_chunk_size cannot be greater than chunk_size")

        # Validate worker count
        if 'step2' in self._config:
            max_workers = self._config['step2'].get('max_workers', 0)
            if max_workers <= 0:
                raise ValueError("max_workers must be positive")

        # Validate retry settings
        if 'validation' in self._config:
            max_retries = self._config['validation'].get('max_retries', 0)
            if max_retries < 0:
                raise ValueError("max_retries must be non-negative")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation"""
        keys = key.split('.')
        current = self._config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def set(self, key: str, value: Any):
        """Set a configuration value using dot notation"""
        self._set_nested_value(self._config, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self._config.copy()

    def save(self, filepath: str):
        """Save current configuration to file"""
        filepath = Path(filepath)

        with open(filepath, 'w', encoding='utf-8') as f:
            if filepath.suffix.lower() == '.yaml':
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
            elif filepath.suffix.lower() == '.json':
                json.dump(self._config, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def get_sources(self) -> list:
        """Return list of configuration sources used"""
        return self._sources.copy()

    def print_config(self):
        """Print current configuration for debugging"""
        print("=" * 60)
        print("CURRENT CONFIGURATION")
        print("=" * 60)
        print(f"Sources: {', '.join(self._sources)}")
        print()
        print(yaml.dump(self._config, default_flow_style=False, sort_keys=False))
