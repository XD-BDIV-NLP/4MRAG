#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MMMRAG (Multi-Modal Multi-hop Retrieval Augmented Generation)
多模态多跳检索增强Generate框架
"""

__version__ = "0.1.0"
__author__ = "MMMRAG Team"
__description__ = "多模态多跳检索增强Generate框架"

from .config.config import Config
from .agents.answer_fuser import AnswerFuser
from .agents.question_decomposer import QuestionDecomposer
from .agents.score_planning_agent import ScorePlanningAgent
from .agents.subquery_reviewer import SubqueryReviewer
from .agents.retriever_agent import RetrieverAgent
from .utils.llm_interface import LLMInterface, create_llm_interface
from .utils.text_processor import TextProcessor
from .utils.logger import Logger, get_logger, setup_global_logger


__all__ = [
    # Config
    "Config",
    "ScorePlanningAgent",
    "QuestionDecomposer",
    "SubqueryReviewer",
    "AnswerFuser",
    "RetrieverAgent",
    "LLMInterface",
    "create_llm_interface",
    "TextProcessor",
    # Log工具
    "get_logger",

]

# 包Info
__package_info__ = {
    "name": "MMMRAG",
    "version": __version__,
    "author": __author__,
    "description": __description__,
    "url": "https://github.com/mmrag-team/mmrag",
    "license": "MIT",
    "keywords": ["RAG", "多模态", "多跳", "检索增强Generate", "LLM"]
}