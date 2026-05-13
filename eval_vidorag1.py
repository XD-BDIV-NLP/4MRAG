#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate ViDoRAG Script
This script evaluates the ViDoRAG system using MMMRAG agent flow
"""

import os
import json
import re
import string
import logging
from typing import Optional, List, Dict, Any
from tqdm import tqdm

# Disable jieba logging
logging.getLogger('jieba').setLevel(logging.ERROR)
os.environ['JIEBA_DISABLE_INIT_LOG'] = '1'
import jieba

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MMMRAG modules
from mmmrag.config.config import Config
from mmmrag.utils.llm_interface import create_llm_interface, LLMInterface
from mmmrag.utils.device_manager import device_manager
from mmmrag.agents.retriever_agent import model_manager

# Try to import openai for evaluation
try:
    import openai
except ImportError:
    openai = None
    print("Warning: 'openai' module not found. LLM-based accuracy calculation will not work.")

# Import App
try:
    from app import MMMRAGApp
except ImportError:
    # Fallback if running from a different directory
    sys.path.append(os.getcwd())
    from app import MMMRAGApp

# Configuration (Hardcoded as requested)
DASHSCOPE_API_KEY = ""
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Categories
CATEGORIES = {
    "query_type": {
        "Single-hop": "single_hop",
        "Multi-hop": "multi_hop"
    },
    "source_type": {
        "Text": "text",
        "Table": "table",
        "Chart": "chart",
        "Layout": "2d_layout"
    }
}

def normalize_text(text):
    """Normalize text for evaluation (remove punctuation, lowercase, tokenize)"""
    if not text:
        return []
    
    # Remove English punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove Chinese punctuation
    chinese_punctuation = ', .!？；:‘’“”[]()《》、—…'  
    text = ''.join([char for char in text if char not in chinese_punctuation])
    
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        tokens = list(jieba.cut(text))
    else:
        tokens = text.split()
    
    return [token for token in tokens if token.strip()]


def exact_match_score(prediction, ground_truth):
    """Calculate exact match score"""
    pred_tokens = normalize_text(prediction)
    gt_tokens = normalize_text(ground_truth)
    return 1.0 if pred_tokens == gt_tokens else 0.0


def calculate_llm_accuracy(prediction, ground_truth):
    """
    Calculate Acc* using Qwen-Plus via DashScope API.
    Score 1-5, >=4 is correct (1.0).
    """
    if not openai:
        return 0.0
    
    client = openai.OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL
    )
    
    prompt = f"""
    Assess the semantic consistency between the model's response and the ground-truth answer.
    Score on a scale of 1 to 5, where:
    1: Completely incorrect or irrelevant.
    2: Mostly incorrect, misses key information.
    3: Partially correct, but contains errors or misses important details.
    4: Mostly correct, captures the main meaning with minor issues.
    5: Completely correct, semantically equivalent to the ground truth.

    Ground Truth: {ground_truth}
    Model Response: {prediction}

    Output ONLY the score (a single integer from 1 to 5).
    """
    
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        content = response.choices[0].message.content.strip()
        # Extract the number
        match = re.search(r'\d', content)
        if match:
            score = int(match.group())
            return 1.0 if score >= 4 else 0.0
        return 0.0
    except Exception as e:
        print(f"Error calling Qwen-Plus: {e}")
        return 0.0


def generate_table(averages):
    """Generate a formatted table for metrics"""
    category_order = ["Single-hop", "Multi-hop", "Text", "Table", "Chart", "Layout"]
    
    print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "", "Single-hop", "Multi-hop", "Text", "Table", "Chart", "Layout"))
    
    print("{:<10} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format(
        "", "Acc*", "EM", "Acc*", "EM", "Acc*", "EM", "Acc*", "EM", "Acc*", "EM", "Acc*", "EM"))
    
    values = []
    for category in category_order:
        metrics = averages.get(category, {"Acc*": 0.0, "EM": 0.0})
        values.append(metrics["Acc*"] * 100)
        values.append(metrics["EM"] * 100)
    
    print("{:<10} {:<5.2f} {:<5.2f} {:<5.2f} {:<5.2f} {:<5.2f} {:<5.2f} {:<5.2f} {:<5.2f} {:<5.2f} {:<5.2f} {:<5.2f} {:<5.2f}".format(
        "", *values))


class MMMRAGEvaluator:
    """MMMRAG Evaluator Class"""
    
    def __init__(self, 
                 dataset: str = 'ViDoSeek',
                 query_file: str = None,
                 experiment_type: str = 'vidorag',
                 generate_vlm: str = 'Qwen3-VL-8B-Instruct',
                 workers_num: int = 1,
                 retrieval_topk: int = 10,
                 rerank_topk: int = 5,
                 limit: Optional[int] = None,
                 disable_rerank: bool = False,
                 **kwargs):
        
        self.experiment_type = experiment_type
        self.workers_num = workers_num
        self.retrieval_topk = retrieval_topk
        self.rerank_topk = rerank_topk
        self.disable_rerank = disable_rerank
        self.dataset = dataset
        if query_file is None:
            _data_base = os.getenv("MMMRAG_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
            query_file = os.path.join(_data_base, 'vidoseek.json')
        self.limit = limit
        self.generate_vlm = generate_vlm
        
        # Initialize paths
        if self.dataset == 'ViDoSeek':
            self.dataset_dir = os.getenv("MMMRAG_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
            self.results_dir = './results/'
        else:
            self.dataset_dir = os.path.dirname(query_file) if os.path.isabs(query_file) else os.path.join('./data', dataset)
            self.results_dir = './results/'
        
        self.img_dir = os.path.join(self.dataset_dir, "img")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize App
        self._initialize_app()
        
        # Set evaluation function and output file
        if experiment_type == 'retrieval_infer':
            self.eval_func = self.retrieval_infer
            self.output_file_name = f'base_retrieval.jsonl'
        elif experiment_type == 'dynamic_hybird_retrieval_infer':
            self.eval_func = self.retrieval_infer
            self.output_file_name = f'dynamic_hybird_retrieval.jsonl'
        elif experiment_type == 'vidorag':
            self.eval_func = self.vidorag
            self.output_file_name = f'vidorag_{generate_vlm}.jsonl'
        
        self.output_file_path = os.path.join(self.results_dir, self.output_file_name.replace("/", "-"))
        self.similarity_file_path = os.path.join(self.results_dir, f'similarity_scores_{generate_vlm}.json')
        self.similarity_data = []

    def _initialize_app(self):
        """Initialize MMMRAG application and dependencies"""
        # Sync devices
        device_manager.sync_devices()
        model_manager.initialize()
        
        # Force Local Provider settings
        Config.LLM_PROVIDER = "local"
        
        # Update Config for ViDoSeek
        if self.dataset == 'ViDoSeek':
            Config.RETRIEVERS['local_text']['params']['index_path'] = os.getenv("MMMRAG_TEXT_INDEX_PATH", os.path.join(os.getenv("MMMRAG_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")), "ViDoSeek", "bge_ingestion"))
            Config.RETRIEVERS['cross_modal']['params']['index_path'] = os.getenv("MMMRAG_CROSS_MODAL_INDEX_PATH", os.path.join(os.getenv("MMMRAG_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")), "ViDoSeek", "bgevlbase_ingestion"))
        
        # Update retrieval parameters
        for retriever_name in Config.RETRIEVERS:
            Config.RETRIEVERS[retriever_name]['params']['top_k'] = self.retrieval_topk
            
        if not hasattr(Config, 'RERANK'):
            Config.RERANK = {}
        Config.RERANK['top_k'] = self.rerank_topk
        Config.RERANK['disable'] = self.disable_rerank
        
        # Initialize App
        self.mmmrag_app = MMMRAGApp()
        
        # Ensure the app uses the correct base_url for the VLM if specified
        # This is needed because MMMRAGApp initializes its own LLMInterface based on Config defaults
        if hasattr(self.mmmrag_app, 'llm_interface'):
            # If using Qwen3-VL, ensure base_url is set to multimodal port if needed
            if self.generate_vlm == "Qwen3-VL-8B-Instruct":
                # Check if we need to update base_url (default might be 8888, which is correct for multimodal)
                pass 
            else:
                # For text-only models, we might need a different port, e.g. 8889
                # But since we forced LLM_PROVIDER="local", we should update the existing interface
                if self.mmmrag_app.llm_interface.base_url is None:
                     self.mmmrag_app.llm_interface.base_url = os.getenv("LOCAL_QWEN_TEXT_API_URL", "http://localhost:8889")

    def retrieval_infer(self, sample: dict) -> dict:
        """Perform retrieval inference"""
        query = sample['query']
        query_data = {
            "question": query,
            "information": {
                "text": sample.get('context', ''),
                "images": sample.get('images', [])
            }
        }
        
        try:
            # Use unified retriever directly
            result = self.mmmrag_app.unified_retriever.process_query(query_data)
            
            # Extract results
            retrieval_results = {}
            if isinstance(result, dict) and "answer" in result:
                retrieval_results = result["answer"]
            else:
                retrieval_results = result
                
            sample['retrieval_results'] = retrieval_results
            return sample
            
        except Exception as e:
            print(f"Retrieval failed: {e}")
            sample['retrieval_results'] = {'results': [], 'error': str(e)}
            return sample

    def vidorag(self, sample: dict) -> dict:
        """Perform full ViDoRAG evaluation"""
        query = sample['query']
        ground_truth = sample.get('reference_answer') or sample.get('answer', '') or sample.get('ground_truth', {}).get('answer', '')
        
        input_data = {
            "question": query,
            "information": {
                "text": sample.get('context', ''),
                "images": sample.get('images', [])
            }
        }
        
        print(f"\n=== Processing Query: {query} ===")
        
        try:
            # Execute full flow
            results = self.mmmrag_app.process_query(input_data)
            
            # Extract answer
            response = results.get('answer', 'No answer generated')
            sample['response'] = response
            
            # Extract retrieval results
            # Note: app.py returns 'retrieval_results': {'results': [...]}
            retrieval_data = results.get('retrieval_results', {}).get('results', [])
            
            # Store in sample
            sample['retrieval_results'] = {
                'results': retrieval_data,
                'reranked_results': retrieval_data  # Assuming final results are reranked
            }
            
            # Calculate metrics
            em = exact_match_score(response, ground_truth)
            acc = calculate_llm_accuracy(response, ground_truth)
            
            sample['eval_result'] = {
                'exact_match': em,
                'accuracy': acc,
                'ground_truth': ground_truth
            }
           
            sample['complexity_scores'] = {
                'modality_complexity': 1,
                'multi_hop_complexity': 1
            }
            
            print(f"Answer: {response}")
            print(f"Metrics - EM: {em:.2f}, Acc: {acc:.2f}")
            
            return sample
            
        except Exception as e:
            print(f"Error processing {query}: {e}")
            sample['error'] = str(e)
            sample['eval_result'] = {'accuracy': 0.0, 'exact_match': 0.0}
            return sample

    def eval_dataset(self):
        """Evaluate full dataset"""
        rag_dataset_path = os.path.join(self.dataset_dir, "vidoseek.json")
        
        if not os.path.exists(rag_dataset_path):
             # Fallback logic from original script
             import glob
             files = glob.glob(os.path.join(self.dataset_dir, "*.json"))
             if files:
                 rag_dataset_path = files[0]
             else:
                 print("No dataset found, using dummy.")
                 data = [{"uid": "test", "query": "Test query", "answer": "Test answer"}]
                 self._run_eval(data)
                 return self.output_file_path

        with open(rag_dataset_path, "r") as f:
            data = json.load(f)
        data = data['examples'] if 'examples' in data else data
        
        if self.limit:
            data = data[:self.limit]
            
        self._run_eval(data)
        return self.output_file_path

    def _run_eval(self, data):
        """Internal evaluation loop"""
        # Resume logic
        processed_ids = set()
        if os.path.exists(self.output_file_path):
            with open(self.output_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        res = json.loads(line)
                        processed_ids.add(res.get('uid') or res.get('query'))
                    except: pass
        
        data_to_process = [d for d in data if (d.get('uid') or d.get('query')) not in processed_ids]
        
        if not data_to_process:
            print("All processed.")
            return

        print(f"Processing {len(data_to_process)} samples...")
        
        if self.workers_num > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
                futures = {executor.submit(self.eval_func, item): item for item in data_to_process}
                for future in tqdm(as_completed(futures), total=len(data_to_process)):
                    self._save_result(future.result())
        else:
            for item in tqdm(data_to_process):
                self._save_result(self.eval_func(item))

    def _save_result(self, result):
        if result:
            with open(self.output_file_path, "a", encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")

    def eval_overall(self):
        """Calculate overall metrics"""
        if not os.path.exists(self.output_file_path):
            return
            
        data = []
        with open(self.output_file_path, "r", encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        metrics = {k: {"Acc*": [], "EM": []} for k in ["Single-hop", "Multi-hop", "Text", "Table", "Chart", "Layout"]}
        
        for item in data:
            res = item.get('eval_result', {})
            acc, em = res.get('accuracy', 0), res.get('exact_match', 0)
            
            # Classify
            meta = item.get('meta_info', {})
            q_type = meta.get('query_type', 'single_hop').lower() # Default to single hop mapping
            s_type = meta.get('source_type', 'text').lower()
            
            # Map to categories
            for cat, key in CATEGORIES['query_type'].items():
                if key in q_type:
                    metrics[cat]["Acc*"].append(acc)
                    metrics[cat]["EM"].append(em)
            
            for cat, key in CATEGORIES['source_type'].items():
                if key in s_type:
                    metrics[cat]["Acc*"].append(acc)
                    metrics[cat]["EM"].append(em)

        # Averages
        averages = {}
        for cat, vals in metrics.items():
            averages[cat] = {
                "Acc*": sum(vals["Acc*"])/len(vals["Acc*"]) if vals["Acc*"] else 0,
                "EM": sum(vals["EM"])/len(vals["EM"]) if vals["EM"] else 0
            }
            
        generate_table(averages)


def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='ViDoSeek')
    parser.add_argument("--query_file", default='vidoseek.json')
    parser.add_argument("--experiment_type", default='vidorag')
    parser.add_argument("--workers_num", type=int, default=1)
    parser.add_argument("--retrieval_topk", type=int, default=10)
    parser.add_argument("--rerank_topk", type=int, default=5)
    parser.add_argument("--disable_rerank", action='store_true')
    parser.add_argument("--generate_vlm", default='Qwen3-VL-8B-Instruct')
    parser.add_argument("--limit", type=int, default=None)
    # Compat
    parser.add_argument("--topk", type=int, help="Deprecated") 
    
    args = parser.parse_args()
    if args.topk and not args.retrieval_topk:
        args.retrieval_topk = args.topk
    return args

if __name__ == "__main__":
    args = arg_parse()
    evaluator = MMMRAGEvaluator(**vars(args))
    evaluator.eval_dataset()
    evaluator.eval_overall()
