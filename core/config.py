"""
Configuration Manager for Prompt Arsenal
Handles API profiles and settings
"""

import json
import os
from typing import Dict, Optional, List


class Config:
    """Configuration manager for API profiles and settings"""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_path):
            default_config = {
                "profiles": {},
                "default_profile": "",
                "multimodal_settings": {
                    "image": {
                        "epsilon": 0.03,
                        "default_attack": "fgsm"
                    },
                    "audio": {
                        "noise_level": 0.005,
                        "sample_rate": 16000
                    },
                    "video": {
                        "frame_skip": 5,
                        "fps": 30
                    }
                }
            }
            self.save_config(default_config)
            return default_config

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def save_config(self, config: Optional[Dict] = None):
        """Save configuration to JSON file"""
        if config is None:
            config = self.config

        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def get_profile(self, profile_name: str) -> Optional[Dict]:
        """Get API profile by name"""
        return self.config['profiles'].get(profile_name)

    def get_all_profiles(self) -> Dict:
        """Get all API profiles"""
        return self.config['profiles']

    def add_profile(self, name: str, provider: str, model: str, api_key: str, base_url: Optional[str] = None):
        """Add new API profile"""
        self.config['profiles'][name] = {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": base_url
        }
        self.save_config()

    def update_profile(self, name: str, **kwargs):
        """Update existing API profile"""
        if name not in self.config['profiles']:
            raise ValueError(f"Profile '{name}' not found")

        for key, value in kwargs.items():
            if key in ["provider", "model", "api_key", "base_url"]:
                self.config['profiles'][name][key] = value

        self.save_config()

    def delete_profile(self, name: str):
        """Delete API profile"""
        if name in self.config['profiles']:
            del self.config['profiles'][name]
            self.save_config()

    def set_default_profile(self, name: str):
        """Set default API profile"""
        if name not in self.config['profiles']:
            raise ValueError(f"Profile '{name}' not found")

        self.config['default_profile'] = name
        self.save_config()

    def get_default_profile(self) -> Optional[Dict]:
        """Get default API profile"""
        default_name = self.config.get('default_profile')
        if default_name and default_name in self.config['profiles']:
            return self.config['profiles'][default_name]
        return None

    def get_multimodal_settings(self, media_type: str) -> Dict:
        """Get multimodal settings for specific media type"""
        return self.config.get('multimodal_settings', {}).get(media_type, {})

    def update_multimodal_settings(self, media_type: str, settings: Dict):
        """Update multimodal settings"""
        if 'multimodal_settings' not in self.config:
            self.config['multimodal_settings'] = {}

        self.config['multimodal_settings'][media_type] = settings
        self.save_config()
