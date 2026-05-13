#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from typing import Dict, List, Any

class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

    MODEL_BASE_DIR = os.getenv("MMMRAG_MODEL_DIR", os.path.join(PROJECT_ROOT, 'models'))
    DATA_BASE_DIR = os.getenv("MMMRAG_DATA_DIR", os.path.join(PROJECT_ROOT, 'data'))

    LOCAL_MODEL_PATHS = {
        "bge_m3": os.getenv("MMMRAG_BGE_M3_PATH", os.path.join(MODEL_BASE_DIR, 'bge_m3')),
        "bge_reranker_v2_m3": os.getenv("MMMRAG_BGE_RERANKER_PATH", os.path.join(MODEL_BASE_DIR, 'bge_reranker_v2_m3')),
        "bge_vl_base": os.getenv("MMMRAG_BGE_VL_BASE_PATH", os.path.join(MODEL_BASE_DIR, 'BGE-VL-Base')),
        "jina_reranker_m0": os.getenv("MMMRAG_JINA_RERANKER_PATH", os.path.join(MODEL_BASE_DIR, 'jina-reranker-m0'))
    }

    GPU_DEVICES = os.getenv("MMMRAG_GPU_DEVICES", "cuda:0,cuda:1").split(",")
    DEFAULT_GPU = os.getenv("MMMRAG_DEFAULT_GPU", "cuda:0")

    CROSS_MODAL_DIMENSION = int(os.getenv("MMMRAG_CROSS_MODAL_DIM", "512"))
    TEXT_EMBEDDING_DIMENSION = int(os.getenv("MMMRAG_TEXT_EMBED_DIM", "1024"))

    RETRIEVERS = {
        "local_text": {
            "type": "local_text",
            "class": "LocalTextRetriever",
            "params": {
                "model_path": LOCAL_MODEL_PATHS["bge_m3"],
                "index_path": os.getenv("MMMRAG_TEXT_INDEX_PATH", os.path.join(DATA_BASE_DIR, 'ViDoSeek', 'bge_ingestion')),
                "top_k": 10
            }
        },
        "cross_modal": {
            "type": "cross_modal",
            "class": "CrossModalRetrieverAgent",
            "params": {
                "model_path": LOCAL_MODEL_PATHS["bge_vl_base"],
                "index_path": os.getenv("MMMRAG_CROSS_MODAL_INDEX_PATH", os.path.join(DATA_BASE_DIR, 'ViDoSeek', 'bgevlbase_ingestion')),
                "top_k": 10,
                "reranker": {
                    "model_path": LOCAL_MODEL_PATHS["jina_reranker_m0"],
                    "top_k": 10
                }
            }
        }
    }

    RERANKERS = {
        "text": {
            "class": "TextReranker",
            "params": {
                "model_path": LOCAL_MODEL_PATHS["bge_reranker_v2_m3"],
                "top_k": 10
            }
        },
        "multimodal": {
            "class": "MultimodalReranker",
            "params": {
                "model_path": LOCAL_MODEL_PATHS["jina_reranker_m0"],
                "top_k": 10
            }
        }
    }

    def __init__(self, config_path: str = None):
        self.version = "1.0.0"
        self.log_level = "INFO"
        self.log_file = None
        self.file_logging = False

        self.llm_config = {
            "provider": self.__class__.LLM_PROVIDER,
            "model": self.__class__.LLM_MODEL,
            "max_tokens": self.__class__.LLM_MAX_TOKENS,
            "temperature": self.__class__.LLM_TEMPERATURE,
            "timeout": self.__class__.LLM_TIMEOUT,
            "agent_configs": self.__class__.AGENT_LLM_CONFIGS
        }

        self.project_root = self.__class__.PROJECT_ROOT
        self.data_dir = self.__class__.DATA_DIR
        self.log_dir = self.__class__.LOG_DIR

        self.default_config_file = os.path.join(os.path.dirname(__file__), "..", "config.json")

        self.modality_scores = self.__class__.MODALITY_SCORES
        self.multi_hop_scores = self.__class__.MULTI_HOP_SCORES
        self.routing_thresholds = self.__class__.ROUTING_THRESHOLDS
        self.retrievers = self.__class__.RETRIEVERS
        self.agent_configs = self.__class__.AGENT_LLM_CONFIGS

        if config_path:
            pass

    LLM_PROVIDER = "local"
    LLM_MODEL = "Qwen3-VL-8B-Instruct"
    LLM_MAX_TOKENS = 4000
    LLM_TEMPERATURE = 0.3
    LLM_TIMEOUT = 30

    MMMRAG_SYSTEM_PROMPT = "You are the MMMRAG system, a multimodal retrieval augmented generation system. You are designed to handle complex questions using various modalities including text and images."

    AGENT_LLM_CONFIGS = {
        "score_planning": {
            "provider": "local",
            "model": "Qwen3-VL-8B-Instruct",
            "max_tokens": 4000,
            "temperature": 0.2,
            "timeout": 30
        },
        "question_decomposer": {
            "provider": "local",
            "model": "Qwen3-VL-8B-Instruct",
            "max_tokens": 3000,
            "temperature": 0.2,
            "timeout": 30
        },
        "subquery_reviewer": {
            "provider": "local",
            "model": "Qwen3-VL-8B-Instruct",
            "max_tokens": 2000,
            "temperature": 0.1,
            "timeout": 20
        },
        "answer_fuser": {
            "provider": "local",
            "model": "Qwen3-VL-8B-Instruct",
            "max_tokens": 100,
            "length_penalty": 0.8,
            "temperature": 0.1,
            "do_sample": False,
            "timeout": 30
        }
    }

    MODALITY_SCORES = {
        "text-only": 1,
        "text-image unimodal": 2,
        "text-image multimodal": 3
    }

    MULTI_HOP_SCORES = {
        "single-hop": 1,
        "parallel 2-hop": 2,
        "bridge 2-hop": 3,
        "parallel 3+-hop": 4,
        "bridge 3+-hop": 5
    }

    ROUTING_THRESHOLDS = {
        "simple question": {"modality_max": 1, "hop_max": 1},
        "Process": {"modality_max": 3, "hop_min": 2, "hop_max": 4},
        "Process": {"modality_max": 3, "hop_min": 3, "hop_max": 5}
    }

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        return getattr(cls, key, default)

    @classmethod
    def update(cls, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if hasattr(cls, key):
                setattr(cls, key, value)

for dir_path in [Config.DATA_DIR, Config.LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)