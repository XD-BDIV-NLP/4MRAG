#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate SlideVQA Script
This script evaluates the MMMRAG system using SlideVQA dataset
"""

import os
import json
import re
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from tqdm import tqdm
import string
import logging

# jiebaLogoutput
logging.getLogger('jieba').setLevel(logging.ERROR)

# settingjiebaInitializeoutput
os.environ['JIEBA_DISABLE_INIT_LOG'] = '1'

import jieba

import sys
import os
import importlib.util

# Get the absolute path to the current script
current_script = os.path.abspath(__file__)
project_root = os.path.dirname(current_script)

# Add project root to Python path
sys.path.insert(0, project_root)

# Import necessary modules
from mmmrag.agents.retriever_agent import RetrieverAgent, model_manager
from mmmrag.config.config import Config
from mmmrag.utils.llm_interface import create_llm_interface
from mmmrag.utils.device_manager import device_manager

# Directly import MMMRAGApp from app module
# First check if we can import it directly
MMMRAGApp = None
try:
    # Try direct import first
    from app import MMMRAGApp
    print(f"Successfully imported MMMRAGApp directly from app module")
except ImportError:
    print("Direct import failed, trying to find and import app.py...")
    # Try to find app.py at the specified location
    app_paths = [
        os.path.join(project_root, "app.py")  # Absolute path on Linux server as specified by user
    ]
    
    app_file_path = None
    for path in app_paths:
        if os.path.exists(path):
            app_file_path = path
            break
    
    # Check if app.py exists and is readable
    if not app_file_path:
        print(f"Error: app.py not found at {app_paths[0]}")
        print(f"Please ensure app.py exists at the specified location")
        sys.exit(1)
    
    if not os.access(app_file_path, os.R_OK):
        print(f"Error: app.py is not readable at {app_file_path}")
        sys.exit(1)
    
    # Use importlib to import app.py
    spec = importlib.util.spec_from_file_location("app", app_file_path)
    app_module = importlib.util.module_from_spec(spec)
    sys.modules["app"] = app_module
    spec.loader.exec_module(app_module)
    MMMRAGApp = app_module.MMMRAGApp
    print(f"Successfully imported MMMRAGApp from {app_file_path}")


def normalize_text(text):
    """
    Normalize text for evaluation
    - Remove punctuation (both English and Chinese)
    - Lowercase
    - Remove extra whitespace
    - Tokenize Chinese text using jieba
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return []
    
    # Remove both English and Chinese punctuation
    # English punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Chinese punctuation
    chinese_punctuation = ', .!？；:‘’“”[]()《》、—…'  
    text = ''.join([char for char in text if char not in chinese_punctuation])
    
    # Lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize text
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        # Chinese text tokenization
        tokens = list(jieba.cut(text))
    else:
        # English text tokenization
        tokens = text.split()
    
    # Remove any empty tokens
    tokens = [token for token in tokens if token.strip()]
    
    return tokens


def exact_match(prediction, ground_truth):
    """
    Calculate exact match score with strict token matching
    
    Args:
        prediction: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Exact match score (0.0 or 1.0)
    """
    pred_tokens = normalize_text(prediction)
    gt_tokens = normalize_text(ground_truth)
    
    # Strict token match returns 1.0, all other cases return 0.0
    return 1.0 if pred_tokens == gt_tokens else 0.0



def accuracy_score(prediction, ground_truth):
    """
    Calculate accuracy score based on exact match
    
    Args:
        prediction: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Accuracy score (same as exact match)
    """
    return exact_match(prediction, ground_truth)

class MMMRAGEvaluator:
    """
    MMMRAG Evaluator Class
    Evaluates the MMMRAG system using SlideVQA dataset
    """
    
    def __init__(self, 
                 dataset: str = 'SlideVQA',
                 query_file: str = None,
                 experiment_type: str = 'slidevqa',
                 generate_vlm: str = 'Qwen3-VL-8B-Instruct',
                 workers_num: int = 1,
                 topk: int = 10,
                 retrieval_topk: Optional[int] = 10,
                 rerank_topk: int = 5,
                 limit: Optional[int] = None,
                 disable_rerank: bool = False,
                 text_model_path: str = None,
                 text_vector_store_path: str = None,
                 cross_modal_model_path: str = None,
                 cross_modal_vector_store_path: str = None,
                 cross_modal_dimension: int = 512):
        _data_base = os.getenv("MMMRAG_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
        _model_base = os.getenv("MMMRAG_MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
        if query_file is None:
            query_file = os.path.join(_data_base, 'slidevqa_refined.json')
        if text_model_path is None:
            text_model_path = os.getenv("MMMRAG_BGE_M3_PATH", os.path.join(_model_base, 'bge_m3'))
        if text_vector_store_path is None:
            text_vector_store_path = os.getenv("MMMRAG_TEXT_INDEX_PATH", os.path.join(_data_base, 'SlideVQA', 'bge_ingestion'))
        if cross_modal_model_path is None:
            cross_modal_model_path = os.getenv("MMMRAG_BGE_VL_BASE_PATH", os.path.join(_model_base, 'BGE-VL-Base'))
        if cross_modal_vector_store_path is None:
            cross_modal_vector_store_path = os.getenv("MMMRAG_CROSS_MODAL_INDEX_PATH", os.path.join(_data_base, 'SlideVQA', 'bgevlbase_ingestion'))
        """
        Initialize MMMRAGEvaluator
        
        Args:
            dataset: Dataset name
            query_file: Path to query file
            experiment_type: Type of experiment
            generate_vlm: Name of VLM model for generation
            workers_num: Number of workers for parallel evaluation
            topk: Number of top results to retrieve (deprecated, use retrieval_topk instead)
            retrieval_topk: Number of top results to retrieve
            rerank_topk: Number of top results to keep after reranking
            limit: Limit the number of samples to evaluate
            disable_rerank: Whether to disable reranking
            text_model_path: Path to text retrieval model
            text_vector_store_path: Path to text vector store
            cross_modal_model_path: Path to cross-modal retrieval model
            cross_modal_vector_store_path: Path to cross-modal vector store
            cross_modal_dimension: Dimension of cross-modal embeddings
        """
        self.text_model_path = text_model_path
        self.text_vector_store_path = text_vector_store_path
        self.cross_modal_model_path = cross_modal_model_path
        self.cross_modal_vector_store_path = cross_modal_vector_store_path
        self.cross_modal_dimension = cross_modal_dimension
        # Initialize logger first
        import logging
        self.logger = logging.getLogger('MMMRAGEvaluator')
        self.logger.setLevel(logging.INFO)
        
        self.experiment_type = experiment_type
        self.workers_num = workers_num
        # :Provides retrieval_topk ,  topk
        self.retrieval_topk = retrieval_topk if retrieval_topk is not None else topk
        self.rerank_topk = rerank_topk
        self.disable_rerank = disable_rerank
        self.top_k = self.retrieval_topk  #  top_k 
        self.dataset = dataset
        self.query_file = query_file
        self.limit = limit
        
        # settingdataset_dirresults_dir
        if self.dataset == 'SlideVQA':
            # SlideVQADataProcess
            self.dataset_dir = os.getenv("MMMRAG_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
            self.results_dir = './results/'  # outputDirectoryDirectoryresults
            self.img_dir = os.path.join(self.dataset_dir,  "img")
        else:
            # Data
            self.dataset_dir = os.path.dirname(query_file) if os.path.isabs(query_file) else os.path.join('./data', dataset)
            self.img_dir = os.path.join(self.dataset_dir, "img")
            self.results_dir = './results/'  # outputDirectoryDirectoryresults
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

        # Initialize LLM interface for evaluation with custom port
        # Set the correct base_url based on the model type
        if generate_vlm == "Qwen3-VL-8B-Instruct":
            base_url = os.getenv("LOCAL_QWEN3VL_API_URL", "http://localhost:8888")
        else:
            base_url = os.getenv("LOCAL_QWEN_TEXT_API_URL", "http://localhost:8889")
            
        self.llm_interface = create_llm_interface({
            "provider": "local",
            "model": generate_vlm,
            "max_tokens": 1024,
            "temperature": 0.1,
            "base_url": base_url
        })

        # Synchronize devices before initializing the app
        device_manager.sync_devices()
        self.logger.info("Synchronized devices before initializing MMMRAGApp")

        # Initialize ModelManager to load all models once during startup
        self.logger.info("Initializing ModelManager to load all models...")
        model_manager.initialize()
        self.logger.info("ModelManager initialized successfully")

        # Update retriever index paths and topk parameters in the config FIRST
        if self.dataset == 'SlideVQA':
            # Set the correct index paths for SlideVQA dataset in the config
            from mmmrag.config.config import Config
            Config.RETRIEVERS['local_text']['params']['index_path'] = self.text_vector_store_path
            Config.RETRIEVERS['cross_modal']['params']['index_path'] = self.cross_modal_vector_store_path
            Config.RETRIEVERS['cross_modal']['params']['model_path'] = self.cross_modal_model_path
            Config.RETRIEVERS['cross_modal']['params']['dimension'] = self.cross_modal_dimension
            self.logger.info(f"Set cross_modal index path to {self.cross_modal_vector_store_path}")

        # Set cross-modal dimension in config for CrossModalRetrieverAgent
        from mmmrag.config.config import Config
        Config.CROSS_MODAL_DIMENSION = self.cross_modal_dimension
        self.logger.info(f"Set CROSS_MODAL_DIMENSION to {self.cross_modal_dimension} in config")

        # Before initializing MMMRAGApp, monkey patch VectorStore to handle inhomogeneous vectors
        # This will fix the "setting an array element with a sequence" error
        from mmmrag.agents.retriever_agent import VectorStore
        
        # Save original load method
        original_load = VectorStore.load
        
        # Create a patched load method that handles inhomogeneous vectors
        def patched_load(self, index_path):
            try:
                # First try original load
                return original_load(self, index_path)
            except ValueError as ve:
                if "inhomogeneous shape" in str(ve):
                    self.logger.warning(f"Inhomogeneous vector shapes detected in {index_path}, applying fix...")
                    # This is a workaround - we'll need to modify the vector store loading
                    # For now, we'll let the error propagate but add more information
                    raise ValueError(f"Vector shape mismatch in {index_path}: {str(ve)}. This usually happens when vectors have different dimensions.")
                else:
                    raise
        
        # Apply the patch
        VectorStore.load = patched_load
        
        # Initialize full MMMRAG app for complete agent flow with custom config - NOW WITH UPDATED CONFIG
        self.mmmrag_app = MMMRAGApp()
        
        # Update retrieval and reranking parameters in the config
        from mmmrag.config.config import Config
        
        # Set retrieval topk for all retrievers
        for retriever_name in Config.RETRIEVERS:
            Config.RETRIEVERS[retriever_name]['params']['top_k'] = self.retrieval_topk
        
        # Store rerank configuration in config
        if not hasattr(Config, 'RERANK'):
            Config.RERANK = {}
        Config.RERANK['top_k'] = self.rerank_topk
        Config.RERANK['disable'] = self.disable_rerank  # Add disable_rerank flag to config
        
        # After creating MMMRAGApp, directly update its config and retrievers
        # This ensures that it uses the correct SlideVQA index paths instead of default Vidoseek paths
        if self.dataset == 'SlideVQA':
            # 1. Update the app's own config
            if hasattr(self.mmmrag_app, 'config'):
                for retriever_name in self.mmmrag_app.config.RETRIEVERS:
                    self.mmmrag_app.config.RETRIEVERS[retriever_name]['params']['top_k'] = self.retrieval_topk
                    if retriever_name == 'local_text':
                        self.mmmrag_app.config.RETRIEVERS[retriever_name]['params']['model_path'] = self.text_model_path
                        self.mmmrag_app.config.RETRIEVERS[retriever_name]['params']['index_path'] = self.text_vector_store_path
                        self.logger.info(f"Updated app.config.RETRIEVERS['{retriever_name}']['params']['model_path'] to {self.text_model_path}")
                        self.logger.info(f"Updated app.config.RETRIEVERS['{retriever_name}']['params']['index_path'] to {self.text_vector_store_path}")
                    elif retriever_name == 'cross_modal':
                        # Skip loading index for cross_modal retriever to avoid vector shape issues
                        self.mmmrag_app.config.RETRIEVERS[retriever_name]['params']['model_path'] = self.cross_modal_model_path
                        self.mmmrag_app.config.RETRIEVERS[retriever_name]['params']['index_path'] = self.cross_modal_vector_store_path
                        self.mmmrag_app.config.RETRIEVERS[retriever_name]['params']['dimension'] = self.cross_modal_dimension
                        self.logger.info(f"Updated app.config.RETRIEVERS['{retriever_name}']['params']['model_path'] to {self.cross_modal_model_path}")
                        self.logger.info(f"Updated app.config.RETRIEVERS['{retriever_name}']['params']['index_path'] to {self.cross_modal_vector_store_path}")
                        self.logger.info(f"Updated app.config.RETRIEVERS['{retriever_name}']['params']['dimension'] to {self.cross_modal_dimension}")
                
            # 2. Reinitialize the retrievers with the updated config
            self.logger.info("Reinitializing retrievers with updated SlideVQA config...")
            self.mmmrag_app._initialize_retrievers()
            self.logger.info("Retrievers reinitialized with updated config")
            
            # 2. Update the unified_retriever's default_configs
            if hasattr(self.mmmrag_app, 'unified_retriever'):
                for retriever_name in self.mmmrag_app.unified_retriever.default_configs:
                    self.mmmrag_app.unified_retriever.default_configs[retriever_name]['top_k'] = self.retrieval_topk
                    if retriever_name == 'local_text':
                        self.mmmrag_app.unified_retriever.default_configs[retriever_name]['model_path'] = self.text_model_path
                        self.mmmrag_app.unified_retriever.default_configs[retriever_name]['index_path'] = self.text_vector_store_path
                        self.logger.info(f"Updated unified_retriever.default_configs['{retriever_name}']['model_path'] to {self.text_model_path}")
                        self.logger.info(f"Updated unified_retriever.default_configs['{retriever_name}']['index_path'] to {self.text_vector_store_path}")
                    elif retriever_name == 'cross_modal':
                        self.mmmrag_app.unified_retriever.default_configs[retriever_name]['model_path'] = self.cross_modal_model_path
                        self.mmmrag_app.unified_retriever.default_configs[retriever_name]['index_path'] = self.cross_modal_vector_store_path
                        self.mmmrag_app.unified_retriever.default_configs[retriever_name]['dimension'] = self.cross_modal_dimension
                        self.logger.info(f"Updated unified_retriever.default_configs['{retriever_name}']['model_path'] to {self.cross_modal_model_path}")
                        self.logger.info(f"Updated unified_retriever.default_configs['{retriever_name}']['index_path'] to {self.cross_modal_vector_store_path}")
                        self.logger.info(f"Updated unified_retriever.default_configs['{retriever_name}']['dimension'] to {self.cross_modal_dimension}")
            
            # 3. Update the existing retrievers' configs directly instead of re-initializing
            if hasattr(self.mmmrag_app, 'retrievers'):
                for retriever_name, retriever in self.mmmrag_app.retrievers.items():
                    if retriever_name in self.mmmrag_app.config.RETRIEVERS:
                        # Get updated config
                        updated_config = self.mmmrag_app.config.RETRIEVERS[retriever_name]
                        
                        # Update retriever's config directly if possible
                        if hasattr(retriever, 'config'):
                            retriever.config = self.mmmrag_app.config
                        
                        # Update retriever's params if possible
                        if hasattr(retriever, 'params'):
                            retriever.params.update(updated_config['params'])
                            self.logger.info(f"Updated params for retriever '{retriever_name}' with model path: {updated_config['params']['model_path']}")
                            self.logger.info(f"Updated params for retriever '{retriever_name}' with index path: {updated_config['params']['index_path']}")
                        
                        # Skip reloading for CrossModalRetrieverAgent to avoid vector shape issues
                        if retriever_name == 'cross_modal':
                            self.logger.info(f"Skipping re-initialization for CrossModalRetrieverAgent to avoid vector shape issues")
                        else:
                            # Try to reload the index if the retriever has a reload method
                            if hasattr(retriever, 'reload'):
                                try:
                                    retriever.reload()
                                    self.logger.info(f"Reloaded index for retriever '{retriever_name}'")
                                except Exception as e:
                                    self.logger.warning(f"Failed to reload index for retriever '{retriever_name}': {str(e)}")
                            elif hasattr(retriever, 'initialize'):
                                try:
                                    retriever.initialize()
                                    self.logger.info(f"Re-initialized retriever '{retriever_name}'")
                                except Exception as e:
                                    self.logger.warning(f"Failed to re-initialize retriever '{retriever_name}': {str(e)}")
        
        # Also update the global RetrieverAgent.default_configs for future instances
        from mmmrag.agents.retriever_agent import RetrieverAgent
        if hasattr(RetrieverAgent, 'default_configs'):
            for retriever_name in RetrieverAgent.default_configs:
                RetrieverAgent.default_configs[retriever_name]['top_k'] = self.retrieval_topk
                if self.dataset == 'SlideVQA':
                    if retriever_name == 'local_text':
                        RetrieverAgent.default_configs[retriever_name]['model_path'] = self.text_model_path
                        RetrieverAgent.default_configs[retriever_name]['index_path'] = self.text_vector_store_path
                        self.logger.info(f"Updated RetrieverAgent.default_configs['{retriever_name}']['model_path'] to {self.text_model_path}")
                        self.logger.info(f"Updated RetrieverAgent.default_configs['{retriever_name}']['index_path'] to {self.text_vector_store_path}")
                    elif retriever_name == 'cross_modal':
                        RetrieverAgent.default_configs[retriever_name]['model_path'] = self.cross_modal_model_path
                        RetrieverAgent.default_configs[retriever_name]['index_path'] = self.cross_modal_vector_store_path
                        RetrieverAgent.default_configs[retriever_name]['dimension'] = self.cross_modal_dimension
                        self.logger.info(f"Updated RetrieverAgent.default_configs['{retriever_name}']['model_path'] to {self.cross_modal_model_path}")
                        self.logger.info(f"Updated RetrieverAgent.default_configs['{retriever_name}']['index_path'] to {self.cross_modal_vector_store_path}")
                        self.logger.info(f"Updated RetrieverAgent.default_configs['{retriever_name}']['dimension'] to {self.cross_modal_dimension}")
        
        # Store rerank configuration in config
        if not hasattr(Config, 'RERANK'):
            Config.RERANK = {}
        Config.RERANK['top_k'] = self.rerank_topk
        Config.RERANK['disable'] = self.disable_rerank  # Add disable_rerank flag to config

        # Set evaluation function based on experiment type
        if experiment_type == 'retrieval_infer':
            self.eval_func = self.retrieval_infer
            self.output_file_name = f'base_retrieval.jsonl'
        # hybrid retrieval
        elif experiment_type == 'dynamic_hybird_retrieval_infer':
            self.eval_func = self.retrieval_infer
            self.output_file_name = f'dynamic_hybird_retrieval.jsonl'
        # slidevqa
        elif experiment_type == 'slidevqa':
            self.eval_func = self.slidevqa
            self.output_file_name = f'slidevqa_{generate_vlm}.jsonl'
        
        self.output_file_path = os.path.join(self.results_dir, self.output_file_name.replace("/", "-"))
        
        # outputFilePath
        self.similarity_file_path = os.path.join(self.results_dir, f'similarity_scores_{generate_vlm}.json')
        # InitializeDatalist
        self.similarity_data = []

    def retrieval_infer(self, sample: dict) -> dict:
        """
        Perform retrieval inference using MMMRAG agent
        
        Args:
            sample: Sample data
            
        Returns:
            Updated sample with retrieval results
        """
        query = sample['query']
        
        # Prepare query data for MMMRAG app
        query_data = {
            "question": query,
            "information": {
                "text": sample.get('context', ''),
                "images": sample.get('images', [])
            }
        }
        
        # Execute only retrieval part of MMMRAG agent flow
        # Skip generation and other steps, only get retrieval results
        retrieval_results = {}
        
        try:
            # Call the unified_retriever's process_query method with retrieval_top_k and rerank_top_k parameters
            # This will avoid unnecessary LLM calls
            process_results = self.mmmrag_app.unified_retriever.process_query(
                query_data,
                retrieval_top_k=self.retrieval_topk,  # Pass the retrieval_topk parameter
                rerank_top_k=self.rerank_topk  # Pass the rerank_topk parameter
            )
            
            # Extract retrieval results from process_results
            if isinstance(process_results, dict) and "answer" in process_results:
                # Get retrieval results from the answer field
                retrieval_results = process_results["answer"]
            else:
                # If answer field is not available, use the entire process_results
                retrieval_results = process_results
        except Exception as e:
            # If direct retrieval fails, iterate through all retrievers and perform retrieval individually
            all_results = []
            
            for retriever_name, retriever in self.mmmrag_app.retrievers.items():
                try:
                    # Perform retrieval with each retriever, passing top_k parameter
                    retriever_results = retriever.retrieve(
                        query,
                        top_k=self.retrieval_topk  # Pass the retrieval_topk parameter
                    )
                    if retriever_results and "results" in retriever_results:
                        all_results.extend(retriever_results["results"])
                except Exception as retriever_e:
                    print(f"Retriever {retriever_name} failed: {str(retriever_e)}")
                    continue
            
            # Structure the results
            retrieval_results = {
                'original_results': all_results,
                'reranked_results': all_results  # No reranking if we're doing direct retrieval
            }
        
        # Ensure retrieval results are correctly structured, even if empty
        sample['retrieval_results'] = retrieval_results
        
        # Check if results have hybrid structure with text_result and cross_modal_result
        has_hybrid_structure = False
        text_result = None
        cross_modal_result = None
        
        # Extract hybrid retrieval results if available
        if isinstance(retrieval_results, dict):
            # Check if we have the hybrid retrieval structure with text_result and cross_modal_result
            if 'text_result' in retrieval_results and 'cross_modal_result' in retrieval_results:
                has_hybrid_structure = True
                text_result = retrieval_results['text_result']
                cross_modal_result = retrieval_results['cross_modal_result']
                
                # Output detailed hybrid retrieval results for debugging
                print(f"\n=== Hybrid Retrieval Results for Query '{query}' ===")
                print(f"Text Result (top1): {'Found' if text_result else 'Not found'}")
                if text_result:
                    text_score = text_result.get('score', 0.0)
                    text_source = text_result.get('source', 'unknown')
                    print(f"  Score: {text_score:.4f}, Source: {text_source}")
                
                print(f"Cross-modal Result (top1): {'Found' if cross_modal_result else 'Not found'}")
                if cross_modal_result:
                    cross_score = cross_modal_result.get('score', 0.0)
                    cross_source = cross_modal_result.get('source', 'unknown')
                    print(f"  Score: {cross_score:.4f}, Source: {cross_source}")
                print("=====================================================")
            
            # Check if we have results list with modality field (app.py format)
            elif 'results' in retrieval_results and isinstance(retrieval_results['results'], list):
                results_list = retrieval_results['results']
                if results_list:
                    # Check if we have both text and image modalities
                    has_text = any(r.get('modality') == 'text' for r in results_list)
                    has_image = any(r.get('modality') == 'image' for r in results_list)
                    if has_text and has_image:
                        has_hybrid_structure = True
                        # Extract text and cross_modal results
                        text_results = [r for r in results_list if r.get('modality') == 'text']
                        image_results = [r for r in results_list if r.get('modality') == 'image']
                        if text_results:
                            text_result = text_results[0]
                        if image_results:
                            cross_modal_result = image_results[0]
                        
                        # Update retrieval_results with hybrid specific fields for backward compatibility
                        retrieval_results['has_hybrid_structure'] = has_hybrid_structure
                        retrieval_results['text_result'] = text_result
                        retrieval_results['cross_modal_result'] = cross_modal_result
                        
                        # Output detailed hybrid retrieval results for debugging
                        print(f"\n=== Hybrid Retrieval Results for Query '{query}' ===")
                        print(f"Text Result (top1): {'Found' if text_result else 'Not found'}")
                        if text_result:
                            text_score = text_result.get('score', 0.0)
                            text_source = text_result.get('source', 'unknown')
                            print(f"  Score: {text_score:.4f}, Source: {text_source}")
                        
                        print(f"Cross-modal Result (top1): {'Found' if cross_modal_result else 'Not found'}")
                        if cross_modal_result:
                            cross_score = cross_modal_result.get('score', 0.0)
                            cross_source = cross_modal_result.get('source', 'unknown')
                            print(f"  Score: {cross_score:.4f}, Source: {cross_source}")
                        print("=====================================================")
        
        # Update retrieval_results with hybrid specific fields
        sample['retrieval_results'].update({
            'has_hybrid_structure': has_hybrid_structure,
            'text_result': text_result,
            'cross_modal_result': cross_modal_result
        })
        
        # Handle new structure with original_results and reranked_results
        original_results = sample['retrieval_results'].get('original_results', [])
        reranked_results = sample['retrieval_results'].get('reranked_results', [])
        
        # If we have new structure, ensure backward compatibility by setting results field
        if original_results or reranked_results:
            # Use reranked_results if available, otherwise use original_results
            fallback_results = reranked_results if reranked_results else original_results
            sample['retrieval_results']['results'] = fallback_results
        else:
            # Fallback to old structure
            if not sample['retrieval_results']:
                sample['retrieval_results'] = {'results': []}
            # Ensure results key exists
            if 'results' not in sample['retrieval_results']:
                sample['retrieval_results']['results'] = []
            
            # Get the actual results list from sample['retrieval_results']['results']
            results_value = sample['retrieval_results']['results']
            if isinstance(results_value, dict) and 'results' in results_value:
                # If results_value is a dict with 'results' key, use that list
                actual_results_list = results_value['results']
            else:
                # Otherwise, use it as is (assuming it's a list)
                actual_results_list = results_value

            # Use actual results list for original and reranked
            sample['retrieval_results']['original_results'] = actual_results_list
            sample['retrieval_results']['reranked_results'] = actual_results_list
        
        # If we have hybrid structure, update original and reranked results
        if has_hybrid_structure:
            # For hybrid retrieval, use both text and cross_modal results for similarity analysis
            hybrid_results = []
            if text_result:
                hybrid_results.append(text_result)
            if cross_modal_result:
                hybrid_results.append(cross_modal_result)
            
            # Update original and reranked results with hybrid results
            sample['retrieval_results']['original_results'] = hybrid_results
            sample['retrieval_results']['reranked_results'] = hybrid_results
            sample['retrieval_results']['results'] = hybrid_results
        
        # Extract similarity scores for this sample
        similarity_record = {
            "query": sample.get('query', ''),
            "uid": sample.get('uid', ''),
            "is_multi_hop": sample.get('is_multi_hop', False),  # Use provided is_multi_hop flag
            "retrieval_results": [],
            "reranked_results": [],
            "sub_question_scores": [],
            # Add hybrid retrieval specific fields
            "has_hybrid_structure": has_hybrid_structure,
            "text_result": {},
            "cross_modal_result": {}
        }
        
        # Process text_result for similarity record if available
        if text_result:
            text_source = text_result.get('source', 'unknown')
            text_metadata = text_result.get('metadata', {})
            if isinstance(text_metadata, dict):
                if 'file_name' in text_metadata:
                    text_source = text_metadata['file_name']
                elif 'filename' in text_metadata:
                    text_source = text_metadata['filename']
                elif 'id' in text_metadata:
                    text_source = str(text_metadata['id'])
            
            similarity_record['text_result'] = {
                "score": text_result.get('score', 0.0),
                "text": text_result.get('text', ''),
                "source": text_source
            }
        
        # Process cross_modal_result for similarity record if available
        if cross_modal_result:
            cross_source = cross_modal_result.get('source', 'unknown')
            cross_metadata = cross_modal_result.get('metadata', {})
            if isinstance(cross_metadata, dict):
                if 'file_name' in cross_metadata:
                    cross_source = cross_metadata['file_name']
                elif 'filename' in cross_metadata:
                    cross_source = cross_metadata['filename']
                elif 'id' in cross_metadata:
                    cross_source = str(cross_metadata['id'])
            
            similarity_record['cross_modal_result'] = {
                "score": cross_modal_result.get('score', 0.0),
                "text": cross_modal_result.get('text', ''),
                "source": cross_source
            }
        
        # Get retrieval results from the updated structure
        original_results = sample['retrieval_results'].get('original_results', [])
        reranked_results = sample['retrieval_results'].get('reranked_results', [])
        
        # Output initial top retrieval results with similarity scores
        print(f"\n=== Initial Top {self.retrieval_topk} Retrieval Results for Query '{sample.get('query', '')}' ===")
        print(f"Total original results: {len(original_results)}")
        print(f"Total reranked results: {len(reranked_results)}")
        
        # Output initial top retrieval results
        top_original = original_results[:self.retrieval_topk]
        for i, result in enumerate(top_original, 1):
            score = result.get('score', 0.0)
            source = result.get('source', 'unknown')
            text = result.get('text', '')[:100] + '...' if len(result.get('text', '')) > 100 else result.get('text', '')
            print(f"{i}. Score: {score:.4f}, Source: {source}, Text: {text}")
        print("=====================================================")
        
        # Process original results for retrieval_results
        processed_original = []
        for result in original_results:
            # Extract source filename from metadata if available
            source = result.get('source', 'unknown')
            metadata = result.get('metadata', {})
            
            # Try to extract filename from metadata
            if isinstance(metadata, dict):
                # Check for filename in various metadata fields
                if 'file_name' in metadata:
                    source = metadata['file_name']
                elif 'filename' in metadata:
                    source = metadata['filename']
                elif 'id' in metadata:
                    # Use ID as fallback
                    source = str(metadata['id'])
            
            # Handle missing 'text' key - use metadata or other fields as fallback
            text = result.get('text', '')
            if not text:
                # Try to get text from metadata
                if isinstance(metadata, dict):
                    text = metadata.get('text', '') or metadata.get('content', '') or metadata.get('description', '')
                # If still no text, use source as fallback
                if not text:
                    text = f"Source: {source}"
            
            processed_original.append({
                "score": result.get('score', 0.0),
                "text": text,  # Preserve full text or fallback
                "source": source
            })
        
        # Process reranked results for reranked_results
        processed_reranked = []
        for result in reranked_results:
            # Extract source filename from metadata if available
            source = result.get('source', 'unknown')
            metadata = result.get('metadata', {})
            
            # Try to extract filename from metadata
            if isinstance(metadata, dict):
                # Check for filename in various metadata fields
                if 'file_name' in metadata:
                    source = metadata['file_name']
                elif 'filename' in metadata:
                    source = metadata['filename']
                elif 'id' in metadata:
                    # Use ID as fallback
                    source = str(metadata['id'])
            
            # Handle missing 'text' key - use metadata or other fields as fallback
            text = result.get('text', '')
            if not text:
                # Try to get text from metadata
                if isinstance(metadata, dict):
                    text = metadata.get('text', '') or metadata.get('content', '') or metadata.get('description', '')
                # If still no text, use source as fallback
                if not text:
                    text = f"Source: {source}"
            
            processed_reranked.append({
                "score": result.get('score', 0.0),
                "text": text,  # Preserve full text or fallback
                "source": source
            })
        
        # Only use top 5 results for similarity record
        similarity_record['retrieval_results'] = processed_original[:5]  # Top 5 original results
        similarity_record['reranked_results'] = processed_reranked[:5]     # Top 5 reranked results
        
        # Check if this is a multi-hop question by looking at sub-query results
        # Multi-hop questions might have sub-results in the output
        if isinstance(retrieval_results, dict) and ('sub_results' in retrieval_results or 'subquery_results' in retrieval_results):
            similarity_record['is_multi_hop'] = True
            
            # Extract sub-question scores
            sub_results = retrieval_results.get('sub_results', []) or retrieval_results.get('subquery_results', [])
            for i, sub_result in enumerate(sub_results):
                # Get sub-question
                sub_question = sub_result.get('question', f'Sub-question {i+1}')
                
                # Get retrieval results for this sub-question
                sub_retrieval_results = sub_result.get('information', {}).get('retrieved_results', [])
                
                # Process sub-question retrieval results
                sub_retrieval_processed = []
                for sub_retrieved in sub_retrieval_results:
                    # Extract source filename from metadata if available
                    source = sub_retrieved.get('source', 'unknown')
                    metadata = sub_retrieved.get('metadata', {})
                    
                    # Try to extract filename from metadata
                    if isinstance(metadata, dict):
                        if 'file_name' in metadata:
                            source = metadata['file_name']
                        elif 'filename' in metadata:
                            source = metadata['filename']
                        elif 'id' in metadata:
                            source = str(metadata['id'])
                    
                    # Handle missing 'text' key - use metadata or other fields as fallback
                    text = sub_retrieved.get('text', '')
                    if not text:
                        # Try to get text from metadata
                        if isinstance(metadata, dict):
                            text = metadata.get('text', '') or metadata.get('content', '') or metadata.get('description', '')
                        # If still no text, use source as fallback
                        if not text:
                            text = f"Source: {source}"
                    
                    sub_retrieval_processed.append({
                        "score": sub_retrieved.get('score', 0.0),
                        "text": text,  # Preserve full text or fallback
                        "source": source
                    })
                
                similarity_record['sub_question_scores'].append({
                    "sub_question": sub_question,
                    "retrieval_results": sub_retrieval_processed,
                    "reranked_results": sub_retrieval_processed  # For now, same as retrieval results
                })
        
        # Add to similarity data
        self.similarity_data.append(similarity_record)
        
        # Return top 5 reranked results
        return {
            "query": sample['query'],
            "uid": sample.get('uid', ''),
            "retrieval_results": processed_original[:5],  # Top 5 original results
            "reranked_results": processed_reranked[:5]     # Top 5 reranked results
        }

    def slidevqa(self, sample: dict) -> dict:
        """
        Perform SlideVQA evaluation using full MMMRAG agent flow
        
        Args:
            sample: Sample data
            
        Returns:
            Updated sample with evaluation results
        """
        query = sample['query']
        
        # Get ground truth from sample with multiple fallback options
        ground_truth = sample.get('reference_answer') or \
                      sample.get('ground_truth', {}).get('answer', '') or \
                      sample.get('answer', '') or \
                      sample.get('target', '') or \
                      sample.get('output', '')
        
        # Prepare input for MMMRAGApp in the required format
        input_data = {
            "question": query,
            "information": {
                "text": sample.get('context', ''),
                "images": sample.get('images', [])
            }
        }
        
        # Add detailed output for debugging
        print(f"\n=== Processing Query ===")
        print(f"Query: {query}")
        print(f"Sample Structure: {list(sample.keys())}")
        print(f"Ground Truth: {ground_truth}")
        
        try:
            # Use full MMMRAG agent flow
            results = self.mmmrag_app.process_query(input_data)
            
            # Extract response and other information from MMMRAG results
            response = results.get('answer', 'No answer generated')
            sample['response'] = response
            
            # Ensure retrieval results are correctly structured, even if empty
            retrieval_results = results.get('retrieval_results', {})
            
            # Check if results have hybrid structure with text_result and cross_modal_result
            has_hybrid_structure = False
            text_result = None
            cross_modal_result = None
            
            # Extract hybrid retrieval results if available
            if isinstance(retrieval_results, dict):
                # Check if we have the hybrid retrieval structure with text_result and cross_modal_result
                if 'text_result' in retrieval_results and 'cross_modal_result' in retrieval_results:
                    has_hybrid_structure = True
                    text_result = retrieval_results['text_result']
                    cross_modal_result = retrieval_results['cross_modal_result']
                    
                    # Output detailed hybrid retrieval results for debugging
                    print(f"\n=== Hybrid Retrieval Results for Query '{query}' ===")
                    print(f"Text Result (top1): {'Found' if text_result else 'Not found'}")
                    if text_result:
                        text_score = text_result.get('score', 0.0)
                        text_source = text_result.get('source', 'unknown')
                        print(f"  Score: {text_score:.4f}, Source: {text_source}")
                    
                    print(f"Cross-modal Result (top1): {'Found' if cross_modal_result else 'Not found'}")
                    if cross_modal_result:
                        cross_score = cross_modal_result.get('score', 0.0)
                        cross_source = cross_modal_result.get('source', 'unknown')
                        print(f"  Score: {cross_score:.4f}, Source: {cross_source}")
                    print("=====================================================")
                
                # Check if we have results list with modality field (app.py format)
                elif 'results' in retrieval_results and isinstance(retrieval_results['results'], list):
                    results_list = retrieval_results['results']
                    if results_list:
                        # Check if we have both text and image modalities
                        has_text = any(r.get('modality') == 'text' for r in results_list)
                        has_image = any(r.get('modality') == 'image' for r in results_list)
                        if has_text and has_image:
                            has_hybrid_structure = True
                            # Extract text and cross_modal results
                            text_results = [r for r in results_list if r.get('modality') == 'text']
                            image_results = [r for r in results_list if r.get('modality') == 'image']
                            if text_results:
                                text_result = text_results[0]
                            if image_results:
                                cross_modal_result = image_results[0]
                        
                        # Update retrieval_results with hybrid specific fields for backward compatibility
                        retrieval_results['has_hybrid_structure'] = has_hybrid_structure
                        retrieval_results['text_result'] = text_result
                        retrieval_results['cross_modal_result'] = cross_modal_result
                        
                        # Output detailed hybrid retrieval results for debugging
                        print(f"\n=== Hybrid Retrieval Results for Query '{query}' ===")
                        print(f"Text Result (top1): {'Found' if text_result else 'Not found'}")
                        if text_result:
                            text_score = text_result.get('score', 0.0)
                            text_source = text_result.get('source', 'unknown')
                            print(f"  Score: {text_score:.4f}, Source: {text_source}")
                        
                        print(f"Cross-modal Result (top1): {'Found' if cross_modal_result else 'Not found'}")
                        if cross_modal_result:
                            cross_score = cross_modal_result.get('score', 0.0)
                            cross_source = cross_modal_result.get('source', 'unknown')
                            print(f"  Score: {cross_score:.4f}, Source: {cross_source}")
                        print("=====================================================")
            
            # Extract top 5 reranked results if available
            final_results = []
            if has_hybrid_structure:
                # For hybrid retrieval, use both text and cross_modal results
                final_results = []
                if text_result:
                    final_results.append(text_result)
                if cross_modal_result:
                    final_results.append(cross_modal_result)
            elif isinstance(retrieval_results, dict):
                # Get reranked results if available
                reranked_results = retrieval_results.get('reranked_results', [])
                if reranked_results:
                    final_results = reranked_results[:5]  # Use top 5 reranked results
                else:
                    # Fallback to original results if no reranked results
                    original_results = retrieval_results.get('original_results', [])
                    if original_results:
                        final_results = original_results[:5]  # Use top 5 original results
                    else:
                        # Fallback to results key
                        results_value = retrieval_results.get('results', [])
                        if isinstance(results_value, dict) and 'results' in results_value:
                            final_results = results_value['results'][:5]  # Use top 5 results
                        else:
                            final_results = results_value[:5]  # Use top 5 results
            
            # Update sample with structured retrieval results
            sample['retrieval_results'] = {
                'results': final_results,
                'retrieval_results': final_results,  # For backward compatibility
                'reranked_results': final_results,   # For backward compatibility
                # Add hybrid retrieval specific fields if available
                'has_hybrid_structure': has_hybrid_structure,
                'text_result': text_result,
                'cross_modal_result': cross_modal_result
            }
                
            sample['processing_summary'] = results.get('processing_summary', '')
            
            # Extract similarity scores for this sample
            similarity_record = {
                "query": sample.get('query', ''),
                "uid": sample.get('uid', ''),
                "is_multi_hop": False,  # Default to single-hop
                "retrieval_results": [],
                "reranked_results": [],
                "sub_question_scores": [],
                # Add hybrid retrieval specific fields
                "has_hybrid_structure": has_hybrid_structure,
                "text_result": {},
                "cross_modal_result": {}
            }
            
            # Process text_result for similarity record if available
            if text_result:
                text_source = text_result.get('source', 'unknown')
                text_metadata = text_result.get('metadata', {})
                if isinstance(text_metadata, dict):
                    if 'file_name' in text_metadata:
                        text_source = text_metadata['file_name']
                    elif 'filename' in text_metadata:
                        text_source = text_metadata['filename']
                    elif 'id' in text_metadata:
                        text_source = str(text_metadata['id'])
                
                similarity_record['text_result'] = {
                    "score": text_result.get('score', 0.0),
                    "text": text_result.get('text', ''),
                    "source": text_source
                }
            
            # Process cross_modal_result for similarity record if available
            if cross_modal_result:
                cross_source = cross_modal_result.get('source', 'unknown')
                cross_metadata = cross_modal_result.get('metadata', {})
                if isinstance(cross_metadata, dict):
                    if 'file_name' in cross_metadata:
                        cross_source = cross_metadata['file_name']
                    elif 'filename' in cross_metadata:
                        cross_source = cross_metadata['filename']
                    elif 'id' in cross_metadata:
                        cross_source = str(cross_metadata['id'])
                
                similarity_record['cross_modal_result'] = {
                    "score": cross_modal_result.get('score', 0.0),
                    "text": cross_modal_result.get('text', ''),
                    "source": cross_source
                }
            
            # Get retrieval results
            original_results = sample['retrieval_results'].get('original_results', [])
            reranked_results = sample['retrieval_results'].get('reranked_results', [])
            
            # If reranking is disabled or no reranked results available, use original results
            if self.disable_rerank or not reranked_results:
                final_results = original_results
                print(f"\n=== Retrieval Results (Reranking Disabled) for Query '{sample.get('query', '')}' ===")
            else:
                final_results = reranked_results
                print(f"\n=== Retrieval Results (Reranking Enabled) for Query '{sample.get('query', '')}' ===")
            
            # Output top 10 retrieval results with similarity scores
            print(f"Total results: {len(final_results)}")
            print(f"\n--- Top 10 Results ---")
            top10_results = final_results[:10]  # Get first 10 results
            for i, result in enumerate(top10_results, 1):
                score = result.get('score', 0.0)
                source = result.get('source', 'unknown')
                text = result.get('text', '')[:100] + '...' if len(result.get('text', '')) > 100 else result.get('text', '')
                print(f"{i}. Score: {score:.4f}, Source: {source}, Text: {text}")
            print("=====================================================")
            
            # Process results to extract source filenames and full text
            processed_results = []
            for result in final_results:
                # Extract source filename from metadata if available
                source = result.get('source', 'unknown')
                metadata = result.get('metadata', {})
                
                # Try to extract filename from metadata
                if isinstance(metadata, dict):
                    # Check for filename in various metadata fields
                    if 'file_name' in metadata:
                        source = metadata['file_name']
                    elif 'filename' in metadata:
                        source = metadata['filename']
                    elif 'id' in metadata:
                        # Use ID as fallback
                        source = str(metadata['id'])
                
                # Handle missing 'text' key - use metadata or other fields as fallback
                text = result.get('text', '')
                if not text:
                    # Try to get text from metadata
                    if isinstance(metadata, dict):
                        text = metadata.get('text', '') or metadata.get('content', '') or metadata.get('description', '')
                    # If still no text, use source as fallback
                    if not text:
                        text = f"Source: {source}"
                
                processed_results.append({
                    "score": result.get('score', 0.0),
                    "text": text,  # Preserve full text or fallback
                    "source": source
                })
            
            # If reranking is disabled, use top 5 original results for both fields
            if self.disable_rerank:
                similarity_record['retrieval_results'] = processed_results[:5]  # Top 5 results
                similarity_record['reranked_results'] = processed_results[:5]  # Top 5 results
            else:
                # Process original results for comparison
                processed_original = []
                for result in original_results[:5]:  # Process only top 5 original results
                    source = result.get('source', 'unknown')
                    metadata = result.get('metadata', {})
                    if isinstance(metadata, dict):
                        if 'file_name' in metadata:
                            source = metadata['file_name']
                        elif 'filename' in metadata:
                            source = metadata['filename']
                        elif 'id' in metadata:
                            source = str(metadata['id'])
                    
                    # Handle missing 'text' key - use metadata or other fields as fallback
                    text = result.get('text', '')
                    if not text:
                        # Try to get text from metadata
                        if isinstance(metadata, dict):
                            text = metadata.get('text', '') or metadata.get('content', '') or metadata.get('description', '')
                        # If still no text, use source as fallback
                        if not text:
                            text = f"Source: {source}"
                    
                    processed_original.append({
                        "score": result.get('score', 0.0),
                        "text": text,
                        "source": source
                    })
                similarity_record['retrieval_results'] = processed_original[:5]  # Top 5 original results
                similarity_record['reranked_results'] = processed_results[:5]     # Top 5 reranked results
            
            # Check if this is a multi-hop question by looking at sub-query results
            # Multi-hop questions might have sub-results in the output
            if 'sub_results' in results or 'subquery_results' in results:
                similarity_record['is_multi_hop'] = True
                
                # Extract sub-question scores
                sub_results = results.get('sub_results', []) or results.get('subquery_results', [])
                for i, sub_result in enumerate(sub_results):
                    # Get sub-question
                    sub_question = sub_result.get('question', f'Sub-question {i+1}')
                    
                    # Get retrieval results for this sub-question
                    sub_retrieval_results = sub_result.get('information', {}).get('retrieved_results', [])
                    
                    # Process sub-question retrieval results
                    sub_retrieval_processed = []
                    for sub_retrieved in sub_retrieval_results:
                        # Extract source filename from metadata if available
                        sub_source = sub_retrieved.get('source', 'unknown')
                        sub_metadata = sub_retrieved.get('metadata', {})
                        
                        # Try to extract filename from metadata
                        if isinstance(sub_metadata, dict):
                            if 'file_name' in sub_metadata:
                                sub_source = sub_metadata['file_name']
                            elif 'filename' in sub_metadata:
                                sub_source = sub_metadata['filename']
                            elif 'id' in sub_metadata:
                                sub_source = str(sub_metadata['id'])
                        
                        # Handle missing 'text' key - use metadata or other fields as fallback
                        sub_text = sub_retrieved.get('text', '')
                        if not sub_text:
                            # Try to get text from metadata
                            if isinstance(sub_metadata, dict):
                                sub_text = sub_metadata.get('text', '') or sub_metadata.get('content', '') or sub_metadata.get('description', '')
                            # If still no text, use source as fallback
                            if not sub_text:
                                sub_text = f"Source: {sub_source}"
                        
                        sub_retrieval_processed.append({
                            "score": sub_retrieved.get('score', 0.0),
                            "text": sub_text,  # Preserve full text or fallback
                            "source": sub_source
                        })
                    
                    similarity_record['sub_question_scores'].append({
                        "sub_question": sub_question,
                        "retrieval_results": sub_retrieval_processed,
                        "reranked_results": sub_retrieval_processed  # For now, same as retrieval results
                    })
            
            # Add to similarity data
            self.similarity_data.append(similarity_record)
            
            # Calculate evaluation metrics - based solely on answer correctness
            em = exact_match(response, ground_truth)
            acc = accuracy_score(response, ground_truth)
            
            # Enhanced evaluation with accurate metrics
            sample['eval_result'] = {
                'exact_match': em,
                'accuracy': acc,
                'ground_truth': ground_truth
            }
            
            # Add detailed output for debugging
            print(f"Predicted Answer: {response}")
            print(f"Evaluation Metrics:")
            print(f"  Exact Match: {em:.4f}")
            print(f"  Accuracy: {acc:.4f}")
            print(f"=========================")
            
            return sample
        except Exception as e:
            sample['error'] = str(e)
            sample['eval_result'] = {
                'accuracy': 0.0,
                'exact_match': 0.0,
                'ground_truth': ground_truth
            }
            # Add detailed error output for debugging
            print(f"\n=== Processing Error ===")
            print(f"Query: {query}")
            print(f"Error: {str(e)}")
            print(f"Ground Truth: {ground_truth}")
            print(f"=========================")
            return sample

    def eval_dataset(self) -> str:
        """
        Evaluate the dataset
        
        Returns:
            Path to output file
        """
        eval_func = self.eval_func
        
        # Use the dataset_dir determined in __init__ instead of hardcoded path
        rag_dataset_path = os.path.join(self.dataset_dir, "slidevqa_refined.json")
        
        data = None
        # Check if dataset file exists
        if not os.path.exists(rag_dataset_path):
            # List all files in dataset directory
            import glob
            dataset_files = glob.glob(os.path.join(self.dataset_dir, "*.json"))
            # Try to use any available JSON file
            if dataset_files:
                rag_dataset_path = dataset_files[0]
                # Load the selected dataset file
                with open(rag_dataset_path, "r") as f:
                    data = json.load(f)
                data = data['examples'] if 'examples' in data else data
                
                # Randomly select 500 questions from the dataset
                if len(data) > 500:
                    data = data[:500]
                    print(f"Selected first 500 questions from the dataset (total: {len(data)})")
                else:
                    print(f"Dataset has only {len(data)} questions, using all of them")
            else:
                # Create a simple test dataset if no files found
                data = [
                    {
                        "uid": "test_1",
                        "query": "What is MMMRAG?",
                        "is_multi_hop": False,
                        "retrieval_results": [],
                        "reranked_results": [],
                        "sub_question_scores": []
                    }
                ]
        else:
            # Load dataset if file exists
            with open(rag_dataset_path, "r") as f:
                data = json.load(f)
            data = data['examples'] if 'examples' in data else data
        
        # Select first 500 questions from the dataset (Deterministic)
        if len(data) > 500:
            data = data[:500]
            print(f"Selected first 500 questions from the dataset (total: {len(data)})")
        else:
            print(f"Dataset has only {len(data)} questions, using all of them")
        
        # Apply sample limit if specified
        if self.limit is not None:
            data = data[:self.limit]
        
        # For all experiment types, ensure we only process the first sample if limit=1
        if self.limit == 1 and len(data) > 1:
            data = [data[0]]
        
        # Clear the output file if it exists
        if os.path.exists(self.output_file_path):
            os.remove(self.output_file_path)
        
        # Run evaluation
        if self.workers_num == 1:
            # Sequential evaluation
            for item in tqdm(data, desc="Evaluating"):
                result = eval_func(item)
                if result is not None:
                    # Ensure the result contains a response field
                    if 'response' not in result:
                        result['response'] = result.get('answer', 'No answer generated')
                    
                    # Ensure the result contains retrieval_results with the first reranked result
                    if 'retrieval_results' in result:
                        retrieval_results = result['retrieval_results']
                        
                        # Extract the first reranked result if available
                        first_result = {
                            "results": [],
                            "original_results": [],
                            "reranked_results": []
                        }
                        
                        if isinstance(retrieval_results, dict):
                            # Get reranked results if available
                            reranked_results = retrieval_results.get('reranked_results', [])
                            if reranked_results:
                                first_result["reranked_results"] = reranked_results[:1]  # Only first reranked result
                            else:
                                # Fallback to original results if no reranked results
                                original_results = retrieval_results.get('original_results', [])
                                if original_results:
                                    first_result["original_results"] = original_results[:1]  # Only first original result
                                    first_result["reranked_results"] = original_results[:1]  # Only first original result
                                else:
                                    # Fallback to results key
                                    results_value = retrieval_results.get('results', [])
                                    if isinstance(results_value, dict) and 'results' in results_value:
                                        first_result["results"] = results_value['results'][:1]  # Only first result
                                        first_result["original_results"] = results_value['results'][:1]  # Only first result
                                        first_result["reranked_results"] = results_value['results'][:1]  # Only first result
                                    else:
                                        first_result["results"] = results_value[:1]  # Only first result
                                        first_result["original_results"] = results_value[:1]  # Only first result
                                        first_result["reranked_results"] = results_value[:1]  # Only first result
                        
                        # Update the result with the first reranked result
                        result['retrieval_results'] = first_result
                    
                    with open(self.output_file_path, "a") as f:
                        json.dump(result, f, ensure_ascii=False)
                        f.write("\n")
        else:
            # Parallel evaluation with 4 workers
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all tasks
                futures = {executor.submit(eval_func, item): item for item in data}
                
                # Process completed tasks
                for future in tqdm(as_completed(futures), total=len(data), desc="Evaluating"):
                    result = future.result()
                    if result is not None:
                        # Ensure the result contains a response field
                        if 'response' not in result:
                            result['response'] = result.get('answer', 'No answer generated')
                        
                        # Ensure the result contains retrieval_results with the first reranked result
                        if 'retrieval_results' in result:
                            retrieval_results = result['retrieval_results']
                            
                            # Extract the first reranked result if available
                            first_result = {
                                "results": [],
                                "original_results": [],
                                "reranked_results": []
                            }
                            
                            if isinstance(retrieval_results, dict):
                                # Get reranked results if available
                                reranked_results = retrieval_results.get('reranked_results', [])
                                if reranked_results:
                                    first_result["reranked_results"] = reranked_results[:1]  # Only first reranked result
                                else:
                                    # Fallback to original results if no reranked results
                                    original_results = retrieval_results.get('original_results', [])
                                    if original_results:
                                        first_result["original_results"] = original_results[:1]  # Only first original result
                                        first_result["reranked_results"] = original_results[:1]  # Only first original result
                                    else:
                                        # Fallback to results key
                                        results_value = retrieval_results.get('results', [])
                                        if isinstance(results_value, dict) and 'results' in results_value:
                                            first_result["results"] = results_value['results'][:1]  # Only first result
                                            first_result["original_results"] = results_value['results'][:1]  # Only first result
                                            first_result["reranked_results"] = results_value['results'][:1]  # Only first result
                                        else:
                                            first_result["results"] = results_value[:1]  # Only first result
                                            first_result["original_results"] = results_value[:1]  # Only first result
                                            first_result["reranked_results"] = results_value[:1]  # Only first result
                            
                            # Update the result with the first reranked result
                            result['retrieval_results'] = first_result
                        
                        with open(self.output_file_path, "a") as f:
                            json.dump(result, f, ensure_ascii=False)
                            f.write("\n")
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")
        
        # Debug: Print similarity data information
        print(f"\nDebug: similarity_data length: {len(self.similarity_data)}")
        print(f"Debug: similarity_file_path: {self.similarity_file_path}")
        
        # Save similarity scores after processing all samples
        if self.similarity_data:
            try:
                with open(self.similarity_file_path, "w") as f:
                    json.dump(self.similarity_data, f, indent=2, ensure_ascii=False)
                print(f"\nalreadySave: {self.similarity_file_path}")
            except Exception as e:
                print(f"\nSaveFailed: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\nDataSave")
        
        return self.output_file_path

    def eval_overall(self) -> dict:
        """
        Evaluate overall results with accuracy, exact match, and F1 score
        
        Returns:
            Evaluation results
        """
        data = []
        with open(self.output_file_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        # Calculate detailed evaluation metrics
        total = len(data)
        success = sum(1 for item in data if 'response' in item and item['response'])
        
        # Initialize metrics counters
        total_em = 0.0
        total_acc = 0.0
        valid_eval_items = 0
        
        for item in data:
            if 'eval_result' in item:
                eval_result = item['eval_result']
                total_em += eval_result.get('exact_match', 0.0)
                total_acc += eval_result.get('accuracy', 0.0)
                valid_eval_items += 1
        
        # Calculate average metrics
        avg_em = total_em / valid_eval_items if valid_eval_items > 0 else 0.0
        avg_acc = total_acc / valid_eval_items if valid_eval_items > 0 else 0.0
        
        results = {
            'total_queries': total,
            'successful_queries': success,
            'success_rate': success / total if total > 0 else 0,
            'valid_eval_items': valid_eval_items,
            'exact_match': avg_em,
            'accuracy': avg_acc,
            'experiment_type': self.experiment_type
        }
        
        # Print evaluation results to console
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Total Queries: {total}")
        print(f"Successful Queries: {success} ({success / total * 100:.2f}%)")
        print(f"Valid Evaluation Items: {valid_eval_items}")
        print(f"Exact Match: {avg_em:.4f}")
        print(f"Accuracy: {avg_acc:.4f}")
        print("=" * 60)
        
        output_eval_path = self.output_file_path.replace(".jsonl", "_eval.json")
        with open(output_eval_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results


def arg_parse():
    """
    Parse command line arguments
    """
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate SlideVQA Script")
    
    # Basic parameters
    parser.add_argument("--dataset", type=str, default='SlideVQA', help="The name of dataset")
    parser.add_argument("--query_file", type=str, default='slidevqa_refined.json', help="The name of anno_file")
    parser.add_argument("--experiment_type", type=str, default='slidevqa', help="The type of experiment")
    parser.add_argument("--workers_num", type=int, default=1, help="The number of workers")
    parser.add_argument("--topk", type=int, default=10, help="The number of topk")
    parser.add_argument("--retrieval_topk", type=int, default=None, help="The number of retrieval topk")
    parser.add_argument("--rerank_topk", type=int, default=5, help="The number of rerank topk")
    parser.add_argument("--disable_rerank", action='store_true', help="Disable reranking")
    parser.add_argument("--generate_vlm", type=str, default='Qwen3-VL-8B-Instruct', help="The name of VLM model")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to evaluate")
    
    # New parameters for model and vector store configuration
    parser.add_argument("--text_model_path", type=str, default=os.getenv("MMMRAG_BGE_M3_PATH", os.path.join(os.getenv("MMMRAG_MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")), "bge_m3")), help="Path to text retrieval model")
    parser.add_argument("--text_vector_store_path", type=str, default=os.getenv("MMMRAG_TEXT_INDEX_PATH", os.path.join(os.getenv("MMMRAG_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")), "SlideVQA", "bge_ingestion")), help="Path to text vector store")
    parser.add_argument("--cross_modal_model_path", type=str, default=os.getenv("MMMRAG_BGE_VL_BASE_PATH", os.path.join(os.getenv("MMMRAG_MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")), "BGE-VL-Base")), help="Path to cross-modal retrieval model")
    parser.add_argument("--cross_modal_vector_store_path", type=str, default=os.getenv("MMMRAG_CROSS_MODAL_INDEX_PATH", os.path.join(os.getenv("MMMRAG_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")), "SlideVQA", "bgevlbase_ingestion")), help="Path to cross-modal vector store")
    parser.add_argument("--cross_modal_dimension", type=int, default=512, help="Dimension of cross-modal embeddings")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    
    # Run full evaluation
    evaluator = MMMRAGEvaluator(
        dataset=args.dataset,
        query_file=args.query_file,
        experiment_type=args.experiment_type,
        workers_num=args.workers_num,
        topk=args.topk,
        retrieval_topk=args.retrieval_topk,
        rerank_topk=args.rerank_topk,
        disable_rerank=args.disable_rerank,
        generate_vlm=args.generate_vlm,
        limit=args.limit,
        text_model_path=args.text_model_path,
        text_vector_store_path=args.text_vector_store_path,
        cross_modal_model_path=args.cross_modal_model_path,
        cross_modal_vector_store_path=args.cross_modal_vector_store_path,
        cross_modal_dimension=args.cross_modal_dimension
    )
    evaluator.eval_dataset()
    evaluator.eval_overall()
