#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLMUtilityModule
ProvidesLLM
"""

from typing import Dict, Any, Optional
from .llm_interface import LLMInterface
from ..config.config import Config

_llm_instances: Dict[str, LLMInterface] = {}


def get_llm_interface(agent_name: str = "default") -> LLMInterface:
    """
    LLM
    
    Args:
        agent_name: , Config
        
    Returns:
        LLMInterface
    """
    if agent_name not in _llm_instances:
        # ConfigLLMConfig
        config = Config()
        agent_config = config.AGENT_CONFIGS.get(agent_name, {})
        
        # Use agent specific configuration or default configuration
        # Select API keys and URLs based on provider
        api_key = getattr(config, "LLM_API_KEY", None)
        base_url = getattr(config, "LLM_BASE_URL", None)
        
        llm_config = agent_config.get("llm_config", {
            "provider": config.LLM_PROVIDER,
            "model": config.LLM_MODEL,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": agent_config.get("temperature", config.LLM_TEMPERATURE)
        })
        
        _llm_instances[agent_name] = LLMInterface(
            provider=llm_config["provider"],
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
            temperature=llm_config["temperature"]
        )
    
    return _llm_instances[agent_name]


def call_llm(prompt: str, 
             max_tokens: Optional[int] = 1000,
             temperature: Optional[float] = None,
             agent_name: str = "default",
             **kwargs) -> str:
    """
    LLM
    
    Args:
        prompt: Text
        max_tokens: MaxGenerate
        temperature: Generate(NoneConfig)
        agent_name: , LLMConfig
        **kwargs: LLMParameter
        
    Returns:
        LLMGenerateText
    """
    llm = get_llm_interface(agent_name)
    
    # Parameter
    params = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs
    }
    
    # removeNone
    params = {k: v for k, v in params.items() if v is not None}
    
    # LLMreturnResult
    try:
        result = llm.generate(**params)
        return result
    except Exception as e:
        print(f"LLM call failed ({agent_name}): {e}")
        # returnErrorInfo
        return f"Error: LLM call failed - {str(e)}"


def call_llm_batch(prompts: list, 
                   max_tokens: Optional[int] = 1000,
                   temperature: Optional[float] = None,
                   agent_name: str = "default",
                   **kwargs) -> list:
    """
    LLM
    
    Args:
        prompts: Textlist
        max_tokens: MaxGenerate
        temperature: Generate
        agent_name: 
        **kwargs: LLMParameter
        
    Returns:
        GenerateResultlist
    """
    llm = get_llm_interface(agent_name)
    
    params = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs
    }
    params = {k: v for k, v in params.items() if v is not None}
    
    try:
        results = llm.batch_generate(prompts, **params)
        return results
    except Exception as e:
        print(f"LLM call failed ({agent_name}): {e}")
        # returnErrorInfolist
        return [f"Error: LLM call failed - {str(e)}"] * len(prompts)


def get_embedding(text: str, 
                  agent_name: str = "default",
                  **kwargs) -> list:
    """
    Text
    
    Args:
        text: inputText
        agent_name: 
        **kwargs: Parameter
        
    Returns:
        list
    """
    llm = get_llm_interface(agent_name)
    
    try:
        embedding = llm.embed(text, **kwargs)
        return embedding
    except Exception as e:
        print(f"Failed to get  ({agent_name}): {e}")
        # returnEmpty
        return []


def clear_llm_cache():
    """
    EmptyLLM
    """
    global _llm_instances
    _llm_instances.clear()


def update_agent_llm_config(agent_name: str, llm_config: Dict[str, Any]):
    """
    LLMConfig
    
    Args:
        agent_name: 
        llm_config: LLMConfig
    """
    if agent_name in _llm_instances:
        del _llm_instances[agent_name]
    
    # Config
    config = Config()
    if agent_name in config.AGENT_CONFIGS:
        config.AGENT_CONFIGS[agent_name]["llm_config"] = llm_config
    else:
        config.AGENT_CONFIGS[agent_name] = {"llm_config": llm_config}