# -*- coding: utf-8 -*-

import os
import sys
import json
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from llama_index.core.schema import Document, ImageDocument
from transformers import AutoModel, AutoProcessor

import torch
from PIL import Image


class Ingestion:
    def __init__(self, dataset_dir, input_prefix='img', output_prefix='bgevlmllm_ingestion', 
                 model_path=None, embed_model_name='bge-vl-mllm-s1'):
        """
        Initialize Ingestion class.

        Args:
            dataset_dir: Dataset directory path
            input_prefix: Input folder prefix
            output_prefix: Output folder prefix
            model_path: Model path
            embed_model_name: Embedding model name
        """
        if model_path is None:
            _model_base = os.getenv("MMMRAG_MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
            model_path = os.getenv("MMMRAG_BGE_VL_MLLM_PATH", os.path.join(_model_base, 'bge-vl-mllm-s1'))
        
        self.dataset_dir = dataset_dir
        self.input_dir = os.path.join(dataset_dir, input_prefix)
        self.output_dir = os.path.join(dataset_dir, output_prefix)
        self.workers = 5
        self.model_path = model_path
        self.embed_model_name = embed_model_name
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        try:
            # Load model with float16
            self.model = AutoModel.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True
            ).eval()
            
            # Fix 1: Add missing image_newline parameter
            if not hasattr(self.model, 'image_newline'):
                if hasattr(self.model.config, 'hidden_size'):
                    hidden_size = self.model.config.hidden_size
                elif hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'hidden_size'):
                    hidden_size = self.model.config.text_config.hidden_size
                else:
                    hidden_size = 4096
                
                self.model.image_newline = torch.nn.Parameter(
                    torch.zeros(hidden_size, dtype=torch.float16, device=self.model.device)
                )
                print(f"Added image_newline parameter with shape: {self.model.image_newline.shape}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Fix 2: Ensure patch_size is set
            if hasattr(self.processor, 'patch_size') and self.processor.patch_size is not None:
                print(f"Patch size from processor: {self.processor.patch_size}")
            else:
                if hasattr(self.model.config, 'vision_config'):
                    vision_config = self.model.config.vision_config
                    patch_size = getattr(vision_config, 'patch_size', 14)
                    self.processor.patch_size = patch_size
                    print(f"Set patch_size from vision_config: {patch_size}")
                else:
                    self.processor.patch_size = 14
                    print("Warning: Using default patch_size=14")
            
            print(f"Model type: {getattr(self.model.config, 'model_type', 'unknown')}")
            print(f"Model dtype: {self.model.dtype}")
            print(f"Processor type: {type(self.processor)}")
            
        except Exception as e:
            print(f"Error initializing model components: {e}")
            import traceback
            traceback.print_exc()
            raise  

    def load_json_text(self, json_file):
        if not json_file.endswith('.json'):
            print(f"Skipping: {os.path.basename(json_file)} - Not a JSON file")
            return None
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                return data.get('text', '')
            elif isinstance(data, list):
                return ' '.join([item.get('text', '') if isinstance(item, dict) else str(item) for item in data])
            else:
                return str(data)
        except json.JSONDecodeError as e:
            print(f"Skipping: {os.path.basename(json_file)} - Invalid JSON: {e}")
            return None
        except Exception as e:
            print(f"Error: {os.path.basename(json_file)} - {str(e)}")
            return None

    def embed_text(self, text):
        """Embed text using model."""
        try:
            inputs = self.processor(text=text, images=None, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # CRITICAL FIX: Extract embeddings from hidden_states
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # Use the last hidden state and apply mean pooling
                hidden_states = outputs.hidden_states[-1]
                embeddings = hidden_states.mean(dim=1)
            elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                print(f"Available attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
                raise AttributeError(f"Cannot find hidden_states in output: {type(outputs)}")
            
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            print(f"Error embedding text: {e}")
            import traceback
            traceback.print_exc()
            return None

    def embed_image(self, image_path):
        """Embed image using model."""
        try:
            # Open image
            image = Image.open(image_path).convert("RGB")
            
            # Add <image> token
            text_with_image_token = "<image>"
            inputs = self.processor(text=text_with_image_token, images=image, return_tensors="pt", padding=True, truncation=True)
            
            # Move to device
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Forward pass with error handling
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
            except AttributeError as e:
                if "'list' object has no attribute 'to'" in str(e):
                    print(f"Model internal error for {image_path}, skipping")
                    return None
                else:
                    raise
            
            # CRITICAL FIX: Extract embeddings from hidden_states
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]
                embeddings = hidden_states.mean(dim=1)
            elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                print(f"Available attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
                raise AttributeError(f"Cannot find hidden_states in output: {type(outputs)}")
            
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            print(f"Error embedding image {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def ingestion_example(self, input_file, output_file):
        try:
            if not os.path.exists(input_file):
                print(f"Warning: File not found {input_file}")
                return False

            if input_file.endswith(('.jpg', '.jpeg', '.png')):
                documents = [ImageDocument(image_path=input_file, metadata={'file_path': input_file, 'data_type': 'image'})]
            else:
                text_content = self.load_json_text(input_file)
                if text_content is None:
                    print(f"Skipping: {os.path.basename(input_file)} - No valid content")
                    return False  
                if not text_content.strip():
                    print(f"Skipping: {os.path.basename(input_file)} - Empty content")
                    return False
                documents = [Document(text=text_content, metadata={'file_path': input_file, 'data_type': 'text'})]
            
            if not documents or len(documents) == 0:
                print(f"Warning: No valid documents created for {input_file}")
                return False
            
            nodes = []
            for doc in documents:
                embedding = None
                if doc.metadata['data_type'] == 'text':
                    embedding = self.embed_text(doc.text)
                elif doc.metadata['data_type'] == 'image':
                    embedding = self.embed_image(doc.image_path)
                
                if embedding:
                    doc.embedding = embedding[0]
                    nodes.append(doc)
                else:
                    print(f"Warning: No embedding generated for {doc.metadata['file_path']}")
            
            if not nodes:
                print(f"Warning: No nodes generated for {input_file}")
                return False
            
            nodes_json = [node.to_dict() for node in nodes]
            for node in nodes_json:
                node['metadata']['data_type'] = 'image' if input_file.endswith(('.jpg', '.jpeg', '.png')) else 'text'
            
            with open(output_file, 'w', encoding='utf-8') as json_file:
                json.dump(nodes_json, json_file, indent=2, ensure_ascii=False)
            
            print(f"Success: {os.path.basename(input_file)} -> {os.path.basename(output_file)} (generated {len(nodes)} nodes)")
            return True
            
        except Exception as e:
            print(f"Error: {input_file} - {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def ingestion_multi_session(self):
        file_to_process = []
        
        for file in os.listdir(self.input_dir):
            if not (file.endswith('.json') or file.endswith(('.jpg', '.jpeg', '.png'))):
                continue
                
            if file.startswith('.'):
                continue
                
            file_prefix, _ = os.path.splitext(file)
            input_file = os.path.join(self.input_dir, file)
            output_file = os.path.join(self.output_dir, file_prefix) + '.node'
            
            if not os.path.exists(output_file):
                file_to_process.append((input_file, output_file))
        
        if not file_to_process:
            print("No files to process. All outputs already exist.")
            return
        
        print(f"Found {len(file_to_process)} files to process")
        
        if self.workers == 1:
            for input_file, output_file in tqdm(file_to_process, desc="Processing files"):
                self.ingestion_example(input_file, output_file)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {
                    executor.submit(self.ingestion_example, input_file, output_file): (input_file, output_file)
                    for input_file, output_file in file_to_process
                }
                
                for future in tqdm(as_completed(future_to_file), total=len(file_to_process), desc='Processing files'):
                    future.result()


if __name__ == '__main__':
    root_path = os.getenv("MMMRAG_DATA_DIR", './data')
    datasets = ['ViDoSeek']
    
    for dataset in datasets:
        dataset_dir = os.path.join(root_path, dataset)
        
        _model_base = os.getenv("MMMRAG_MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
        model_path = os.getenv("MMMRAG_BGE_VL_MLLM_PATH", os.path.join(_model_base, 'bge-vl-mllm-s1'))
        
        ingestion = Ingestion(
            dataset_dir, 
            input_prefix='img', 
            output_prefix='bgevlmllm_ingestion', 
            model_path=model_path,
            embed_model_name='bge-vl-mllm-s1'
        )
        ingestion.ingestion_multi_session()
        
        print("\n" + "="*60)
        print("ALL TASKS COMPLETED")
        print("="*60)