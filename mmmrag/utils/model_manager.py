#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, Any, Optional, Union, List
from .llm_interface import LLMInterface, create_llm_interface


class ModelManager:
    
    def __init__(self):
        self.models = {
            "multimodal": None,
            "text": None
        }
        
        from ..utils.logger import get_logger
        self.logger = get_logger("model_manager")
        
        self._initialize_models()
    
    def _initialize_models(self):
        try:
            multimodal_url = os.getenv("LOCAL_QWEN3VL_API_URL", "http://localhost:8888")
            text_url = os.getenv("LOCAL_QWEN_TEXT_API_URL", "http://localhost:8889")
            api_key = os.getenv("LOCAL_QWEN3VL_API_KEY", "")
            
            self.models["multimodal"] = LLMInterface(
                provider="local",
                model_name="Qwen3-VL-8B-Instruct",
                base_url=multimodal_url,
                api_key=api_key
            )
            self.logger.info("Multimodal model Qwen3-VL-8B-Instruct initialized")
            
            self.models["text"] = LLMInterface(
                provider="local",
                model_name="Qwen3-8B",
                base_url=text_url,
                api_key=api_key
            )
            self.logger.info("Text model Qwen3-8B initialized")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise
    
    def get_model(self, task_type: str) -> Optional[LLMInterface]:
        return self.models.get(task_type)
    
    def generate(self, task_type: str, prompt: str, **kwargs) -> Dict[str, Any]:
        model = self.get_model(task_type)
        if model:
            return model.generate(prompt, **kwargs)
        else:
            self.logger.error(f"Model not found for task type: {task_type}")
            return {"text": "", "error": f"Model not found for task type: {task_type}"}
    
    def generate_multimodal(self, prompt: str, image_data: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        model = self.get_model("multimodal")
        if model:
            return model.generate_multimodal(prompt, image_data, **kwargs)
        else:
            self.logger.error("Multimodal model not found")
            return {"text": "", "error": "Multimodal model not found"}
    
    def route_task(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if "image_data" in kwargs and kwargs["image_data"]:
            self.logger.info("Routing to multimodal model: Qwen3-VL-8B-Instruct")
            return self.generate_multimodal(prompt, kwargs["image_data"], **kwargs)
        else:
            self.logger.info("Routing to text model: Qwen3-8B")
            return self.generate("text", prompt, **kwargs)


_global_model_manager = None


def get_model_manager() -> ModelManager:
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
    return _global_model_manager


def create_model_router(config: Optional[Dict[str, Any]] = None) -> ModelManager:
    return get_model_manager()


def create_llm_interface_with_router(config: Optional[Dict[str, Any]] = None) -> LLMInterface:
    if config is None:
        config = {}
    
    model_type = config.get("model_type", "multimodal")
    multimodal_url = os.getenv("LOCAL_QWEN3VL_API_URL", "http://localhost:8888")
    text_url = os.getenv("LOCAL_QWEN_TEXT_API_URL", "http://localhost:8889")
    api_key = os.getenv("LOCAL_QWEN3VL_API_KEY", "")
    
    if model_type == "text":
        return LLMInterface(
            provider="local",
            model_name="Qwen3-8B",
            base_url=text_url,
            api_key=api_key,
            **config
        )
    else:
        return LLMInterface(
            provider="local",
            model_name="Qwen3-VL-8B-Instruct",
            base_url=multimodal_url,
            api_key=api_key,
            **config
        )


__all__ = [
    "ModelManager",
    "get_model_manager",
    "create_model_router",
    "create_llm_interface_with_router"
]
