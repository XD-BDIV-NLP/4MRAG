#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from typing import Dict, Optional, Any

class APIKeysConfig:
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "")
    
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY", "")
    GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "")
    
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    
    LOCAL_QWEN3VL_API_URL = os.getenv("LOCAL_QWEN3VL_API_URL", "http://localhost:8888")
    LOCAL_QWEN3VL_API_KEY = os.getenv("LOCAL_QWEN3VL_API_KEY", "")
    
    @classmethod
    def get_all_keys(cls) -> Dict[str, str]:
        return {
            "OPENAI_API_KEY": cls.OPENAI_API_KEY,
            "OPENAI_ORG_ID": cls.OPENAI_ORG_ID,
            "ANTHROPIC_API_KEY": cls.ANTHROPIC_API_KEY,
            "GOOGLE_API_KEY": cls.GOOGLE_API_KEY,
            "GOOGLE_CLOUD_API_KEY": cls.GOOGLE_CLOUD_API_KEY,
            "GOOGLE_CLOUD_PROJECT_ID": cls.GOOGLE_CLOUD_PROJECT_ID,
            "HUGGINGFACE_API_KEY": cls.HUGGINGFACE_API_KEY,
            "LOCAL_QWEN3VL_API_URL": cls.LOCAL_QWEN3VL_API_URL,
            "LOCAL_QWEN3VL_API_KEY": cls.LOCAL_QWEN3VL_API_KEY
        }
    
    @classmethod
    def get_required_keys(cls) -> Dict[str, str]:
        return {}

    
    @classmethod
    def validate_keys(cls) -> Dict[str, bool]:
        required_keys = cls.get_required_keys()
        validation = {}
        
        for key_name, key_value in required_keys.items():
            validation[key_name] = bool(key_value and key_value.strip())
        
        return validation
    
    @classmethod
    def get_missing_keys(cls) -> list:
        validation = cls.validate_keys()
        missing_keys = [key for key, is_set in validation.items() if not is_set]
        return missing_keys
    
    @classmethod
    def is_configured(cls) -> bool:
        missing_keys = cls.get_missing_keys()
        return len(missing_keys) == 0
    
    @classmethod
    def print_status(cls):
        print("=== API Key Configuration Status ===")
        validation = cls.validate_keys()
        
        for key, is_set in validation.items():
            status = "Configured" if is_set else "Not Configured"
            print(f"{key}: {status}")
        
        missing = cls.get_missing_keys()
        if missing:
            print(f"\nMissing Keys: {', '.join(missing)}")
        else:
            print("\nAll required API keys are configured!")
        
        print("\n=== Local Model Configuration Status ===")
        print(f"LOCAL_QWEN3VL_API_URL: {'Configured' if cls.LOCAL_QWEN3VL_API_URL else 'Not Configured'}")
        print(f"LOCAL_QWEN3VL_API_KEY: {'Configured' if cls.LOCAL_QWEN3VL_API_KEY else 'Optional'}")
    
    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        return {
            "local": {
                "qwen3vl": {
                    "api_url": cls.LOCAL_QWEN3VL_API_URL,
                    "api_key": cls.LOCAL_QWEN3VL_API_KEY,
                    "model_name": "Qwen3-VL-8B-Instruct",
                    "proxy": None
                }
            }
        }

    
    @classmethod
    def get_setup_instructions(cls) -> str:
        return """
=== API Key Setup Instructions ===

1. Set Environment Variables (Recommended):
   
   # Windows (PowerShell)
   $env:OPENAI_API_KEY="your_key"
   $env:ANTHROPIC_API_KEY="your_key"
   $env:GOOGLE_API_KEY="your_key"
   
   # Linux/Mac (bash)
   export OPENAI_API_KEY="your_key"
   export ANTHROPIC_API_KEY="your_key"
   export GOOGLE_API_KEY="your_key"

2. Or create .env file in the project root.

3. Local Model Configuration:
   
   $env:LOCAL_QWEN3VL_API_URL="http://localhost:8888"
   $env:LOCAL_QWEN3VL_API_KEY="your_key_if_needed"

4. Test Configuration:
   
   from mmmrag.config.api_keys_config import APIKeysConfig
   APIKeysConfig.print_status()
        """


if __name__ == "__main__":
    APIKeysConfig.print_status()
    print("\n" + APIKeysConfig.get_setup_instructions())
