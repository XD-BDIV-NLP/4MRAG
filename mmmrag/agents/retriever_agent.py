#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import json
import numpy as np
import time
import torch
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Optional

from PIL import Image

from ..config.config import Config
from ..utils.logger import get_logger
from ..utils.text_processor import TextProcessor
from ..utils.llm_interface import LLMInterface, create_llm_interface
from ..utils.device_manager import device_manager
import faiss

# Hardcoded device assignments for models
DEVICE_OVERRIDES = {
    "bge-vl-base": os.getenv("MMMRAG_BGE_VL_DEVICE", "cuda:0"),
    "bge-m3": os.getenv("MMMRAG_BGE_M3_DEVICE", "cuda:0"),
    "jina-reranker-m0": os.getenv("MMMRAG_JINA_DEVICE", "cuda:0"),
    "bge-reranker-v2-m3": os.getenv("MMMRAG_BGE_RERANKER_DEVICE", "cuda:0"),
}

# Global model manager to load models only once
class ModelManager:
    """Global model manager to load and cache models only once during initialization"""
    _instance = None
    _models = {}
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None: cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def initialize(self):
        if self._initialized: return
        import os
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoProcessor

        _cfg = Config()
        _model_base = getattr(_cfg, 'MODEL_BASE_DIR', os.path.join(Config.PROJECT_ROOT, 'models'))
        self._models = {}
        model_paths = {
            "bge-vl-base": Config.LOCAL_MODEL_PATHS.get("bge_vl_base", os.path.join(_model_base, 'BGE-VL-Base')),
            "bge-m3": Config.LOCAL_MODEL_PATHS.get("bge_m3", os.path.join(_model_base, 'bge_m3')),
            "jina-reranker-m0": Config.LOCAL_MODEL_PATHS.get("jina_reranker_m0", os.path.join(_model_base, 'jina-reranker-m0')),
            "bge-reranker-v2-m3": Config.LOCAL_MODEL_PATHS.get("bge_reranker_v2_m3", os.path.join(_model_base, 'bge_reranker_v2_m3'))
        }
        
        for name, path in model_paths.items():
            try:
                device = torch.device(DEVICE_OVERRIDES.get(name, "cuda:0"))
                print(f"Loading {name} on {device}...")
                
                if name == "bge-vl-base":
                    proc = AutoProcessor.from_pretrained(path, local_files_only=True)
                    model = AutoModel.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32, trust_remote_code=True)
                    if not hasattr(model, 'image_newline'): model.image_newline = "\n"
                    self._models[name] = {"model": model.to(device).eval(), "processor": proc, "device": device}
                elif name == "jina-reranker-m0":
                    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
                    model = AutoModel.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32, trust_remote_code=True)
                    self._models[name] = {"model": model.to(device).eval(), "tokenizer": tok, "device": device}
                else:
                    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
                    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32)
                    self._models[name] = {"model": model.to(device).eval(), "tokenizer": tok, "device": device}
                print(f"Loaded {name}")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                self._models[name] = None
        self._initialized = True
    
    def get_model(self, name):
        if not self._initialized: self.initialize()
        return self._models.get(name)

model_manager = ModelManager()


# Vector Store Class for efficient vector management
class VectorStore:
    def __init__(self, config: Optional[Config] = None, **kwargs):
        self.config, self.logger = config or Config(), get_logger("vector_store")
        self.metadata, self.index_path = [], kwargs.get("index_path", None)
        self.dimension, self.device = kwargs.get("dimension", 1024), kwargs.get("device", "cpu")
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.vectors, self._vectors_loaded = np.array([]), False
        self.logger.info(f"VectorStore init: dim={self.dimension}, device={self.device}")

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-8 else np.ones_like(vec) / np.sqrt(vec.shape[0])

    def add_vector(self, vector: List[float], metadata: Dict[str, Any]):
        try:
            vec_np = np.array(vector, dtype=np.float32)
            if vec_np.shape[0] != self.dimension: raise ValueError(f"Dim mismatch: {vec_np.shape[0]} != {self.dimension}")
            if abs(np.linalg.norm(vec_np) - 1.0) > 1e-4: vec_np = self._normalize(vec_np)
            
            self.faiss_index.add(vec_np.reshape(1, -1))
            self.metadata.append(metadata)
            self.vectors = vec_np.reshape(1, -1) if self.vectors.size == 0 else np.vstack([self.vectors, vec_np.reshape(1, -1)])
        except Exception as e:
            self.logger.error(f"Add vector failed: {e}")

    def add_vectors(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]):
        if not vectors or len(vectors) != len(metadatas): return
        try:
            vecs_np = np.array(vectors, dtype=np.float32).reshape(len(vectors), -1)
            if vecs_np.shape[1] != self.dimension: raise ValueError(f"Dim mismatch: {vecs_np.shape[1]} != {self.dimension}")
            
            norms = np.linalg.norm(vecs_np, axis=1, keepdims=True)
            if np.sum(np.abs(norms - 1.0) > 1e-4) > 0:
                vecs_np = vecs_np / np.where(norms < 1e-8, np.sqrt(vecs_np.shape[1]), norms)
            
            self.faiss_index.add(vecs_np)
            self.metadata.extend(metadatas)
            self.vectors = vecs_np if self.vectors.size == 0 else np.vstack([self.vectors, vecs_np])
            self.logger.info(f"Added {len(vectors)} vectors")
        except Exception as e:
            self.logger.error(f"Add batch vectors failed: {e}")

    def search(self, query_vector: List[float], top_k: int = 10, threshold: float = None) -> List[Dict[str, Any]]:
        if self.faiss_index.ntotal == 0: return []
        try:
            q_np = self._normalize(np.array(query_vector, dtype=np.float32))
            if np.isnan(q_np).any() or q_np.shape[0] != self.dimension: return []
            
            dists, idxs = self.faiss_index.search(q_np.reshape(1, -1), min(max(top_k * 2, 100), self.faiss_index.ntotal))
            results = []
            
            for score, idx in zip(dists[0], idxs[0]):
                if idx == -1 or idx >= len(self.metadata): continue
                if threshold is not None and score < threshold: continue
                
                meta = self.metadata[idx].copy()
                for k, v in meta.items():
                    if isinstance(v, (np.float32, np.float64)): meta[k] = float(v)
                    elif isinstance(v, (np.int32, np.int64)): meta[k] = int(v)
                
                results.append({"metadata": meta, "score": float(score), "index": int(idx)})
            
            return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def save(self, file_path: str):
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({"vectors": self.vectors.tolist(), "metadata": self.metadata, "dimension": self.dimension}, f)
            self.logger.info(f"Saved to {file_path}")
        except Exception as e: self.logger.error(f"Save failed: {e}")

    def load(self, file_path: str):
        if self._vectors_loaded: return
        import os, glob, pickle, concurrent.futures
        
        cache_file = os.path.join(file_path, "vector_store_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f: data = pickle.load(f)
                if data.get("dimension") == self.dimension:
                    self.vectors, self.metadata = data["vectors"], data["metadata"]
                    if self.vectors.size > 0: 
                        self.faiss_index = faiss.IndexFlatIP(self.dimension)
                        self.faiss_index.add(self.vectors)
                    self._vectors_loaded = True
                    self.logger.info(f"Loaded from cache: {len(self.metadata)} vectors")
                    return
            except Exception as e: self.logger.error(f"Cache load failed: {e}")

        if not Path(file_path).is_dir(): return
        node_files = glob.glob(os.path.join(file_path, "*.node"))
        if not node_files: return

        all_vecs, all_meta, all_texts = [], [], []
        
        def process_file(f_path):
            try:
                with open(f_path, "r", encoding="utf-8") as f: data = json.load(f)
                nodes = data if isinstance(data, list) else [data]
                res = []
                for node in nodes:
                    if not isinstance(node, dict): continue
                    meta = node.get("metadata", {})
                    meta.update({k: node[k] for k in ["image_path", "file_path"] if k in node})
                    for k in ["image_path", "file_path"]:
                        if k in meta and "ViDoSeek" in meta[k]:
                            data_base = getattr(Config, 'DATA_BASE_DIR', os.path.join(Config.PROJECT_ROOT, 'data'))
                            meta[k] = meta[k].replace("./data/", os.path.join(data_base, '/')).replace("data/ViDoSeek/", os.path.join(data_base, 'ViDoSeek/'))
                    
                    if "id" not in meta: meta["id"] = os.path.basename(f_path).replace('.node', '')
                    text = node.get("text", node.get("content", ""))
                    if text: meta["text"] = text
                    
                    vec = node.get("vector", node.get("embedding"))
                    res.append({"vector": vec, "text": text, "metadata": meta})
                return res
            except: return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(64, (os.cpu_count() or 1) * 4)) as exc:
            for file_res in exc.map(process_file, node_files, chunksize=100):
                for res in file_res:
                    if res["vector"]:
                        v = res["vector"]
                        if isinstance(v, list) and v and isinstance(v[0], list): v = [i for s in v for i in s]
                        all_vecs.append(v)
                        all_meta.append(res["metadata"])
                    elif res["text"]:
                        all_texts.append(res["text"])
                        all_meta.append(res["metadata"])

        if all_vecs:
            vecs_np = np.array(all_vecs, dtype=np.float32)
            self.dimension = vecs_np.shape[1]
            norms = np.linalg.norm(vecs_np, axis=1, keepdims=True)
            vecs_np = vecs_np / np.where(norms < 1e-8, np.sqrt(vecs_np.shape[1]), norms)
            
            if self.faiss_index.d != self.dimension: self.faiss_index = faiss.IndexFlatIP(self.dimension)
            self.faiss_index.add(vecs_np)
            self.metadata.extend(all_meta)
            self.vectors = vecs_np if self.vectors.size == 0 else np.vstack([self.vectors, vecs_np])
            self._vectors_loaded = True
            self.logger.info(f"Loaded {len(all_vecs)} vectors")
            
            try:
                with open(cache_file, "wb") as f: pickle.dump({"vectors": self.vectors, "metadata": self.metadata, "dimension": self.dimension}, f)
            except: pass

    def clear(self):
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.vectors, self.metadata, self._vectors_loaded = np.array([]), [], False

    def __len__(self): return min(self.faiss_index.ntotal, len(self.metadata))

# Helper functions for retrievers
def validate_query(query: str) -> bool:
    """Validate query effectiveness"""
    return bool(query and isinstance(query, str) and len(query.strip()) >= 2)

def format_results(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format retrieval results"""
    formatted = []
    for res in raw_results:
        meta = res.get("metadata", {})
        text = meta.get("text", res.get("text", ""))
        
        # Try to fix text if it looks like JSON
        if text.strip().startswith(("{", "[")):
            try:
                import json
                data = json.loads(text)
                if isinstance(data, dict): text = data.get("text", text)
                elif isinstance(data, list): text = " ".join([item.get("text", "") for item in data if isinstance(item, dict)])
            except: pass
        
        source = meta.get("file_name", meta.get("filename", meta.get("id", "unknown")))
        formatted.append({
            "text": text,
            "score": res.get("score", 0.0),
            "source": source,
            "metadata": meta
        })
    return formatted

# Local Text Retriever Agent
class LocalTextRetriever:
    """Local Text Retriever using bge-m3"""
    
    def __init__(self, config: Optional[Config] = None, **kwargs):
        self.config = config or Config()
        self.logger = get_logger(f"retriever_agent.{self.__class__.__name__}")
        self.kwargs = kwargs
        self.vector_store = None
        self._initialized = False
        self.initialize()

    def initialize(self):
        if self._initialized: return
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            import os
            
            self.model_path = self.kwargs.get("model_path", Config.LOCAL_MODEL_PATHS.get("bge_m3", os.path.join(Config.PROJECT_ROOT, 'models', 'bge_m3')))
            device = torch.device(DEVICE_OVERRIDES.get("bge-m3", "cuda:0"))
            self.vector_store = VectorStore(config=self.config, device=str(device))
            
            self.logger.info(f"Loading embedding model from {self.model_path}")
            self.embedding_model = SentenceTransformer(self.model_path, device=device, trust_remote_code=True)
            
            index_path = self.kwargs.get("index_path")
            if index_path and os.path.exists(index_path):
                self.logger.info(f"Loading vectors from {index_path}")
                self.vector_store.load(index_path)
                
                if torch.cuda.device_count() > 1:
                    device_alt = "cuda:1" if str(device) == "cuda:0" else "cuda:0"
                    self.vector_store_alt = VectorStore(config=self.config, device=device_alt)
                    if len(self.vector_store.vectors) > 0:
                        import faiss
                        self.vector_store_alt.vectors = self.vector_store.vectors.copy()
                        self.vector_store_alt.metadata = self.vector_store.metadata.copy()
                        self.vector_store_alt.dimension = self.vector_store.dimension
                        self.vector_store_alt.faiss_index = faiss.IndexFlatIP(self.vector_store_alt.dimension)
                        self.vector_store_alt.faiss_index.add(self.vector_store_alt.vectors)
                        self.vector_store_alt._vectors_loaded = True
            
            self._initialized = True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")

    def retrieve(self, query: str, **kwargs) -> Dict[str, Any]:
        if not validate_query(query): return {"results": [], "error": "Invalid query"}
        if not self._initialized: self.initialize()
        
        try:
            top_k = kwargs.get("top_k", self.kwargs.get("top_k", 5))
            query_vector = self.embedding_model.encode(query).tolist()
            
            if not self.vector_store:
                return {"results": [], "error": "Vector store not initialized"}
                
            raw_results = self.vector_store.search(query_vector, top_k=top_k)
            return {
                "query": query,
                "results": format_results(raw_results),
                "total": len(raw_results),
                "retriever": self.__class__.__name__
            }
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return {"results": [], "error": str(e)}
    
    def get_info(self) -> Dict[str, Any]:
        return {"name": self.__class__.__name__, "initialized": self._initialized}


    def add_document(self, document: Dict[str, Any]):
        """Add document to retriever"""
        text = document.get("text", "")
        if not text: return
        
        if not self._initialized: self.initialize()
        
        try:
            if self.embedding_model:
                vector = self.embedding_model.encode(text).tolist()
                self.vector_store.add_vector(vector, document)
            else:
                self.logger.error("Embedding model not initialized")
        except Exception as e:
            self.logger.error(f"Failed to add document: {e}")

    def save_index(self, file_path: str):
        """Save index to file"""
        if not self._initialized: self.initialize()
        if self.vector_store: self.vector_store.save(file_path)

    def load_index(self, file_path: str):
        """Load index from file"""
        if not self._initialized: self.initialize()
        if self.vector_store: self.vector_store.load(file_path)

    def clear_corpus(self):
        """Clear all documents and vectors"""
        if not self._initialized: self.initialize()
        if self.vector_store: self.vector_store.clear()

# Cross-modal Retriever Agent
class CrossModalRetrieverAgent:
    """
    Cross-Modal Retriever Agent
    Handles retrieval across multiple modalities (text, image, audio, etc.)
    Uses BGE-VL-Large model for cross-modal embedding and retrieval
    """
    
    def __init__(self, config: Optional[Config] = None, **kwargs):
        """
        Initialize cross-modal retriever agent
        
        Args:
            config: Configuration object
            **kwargs: Additional parameters
        """
        self.config = config or Config()
        self.logger = get_logger(f"retriever_agent.{self.__class__.__name__}")
        self.kwargs = kwargs
        
        self.text_corpus = kwargs.get("text_corpus", [])
        self.image_corpus = kwargs.get("image_corpus", [])
        self.audio_corpus = kwargs.get("audio_corpus", [])
        # BGE-VL-Base model for cross-modal retrieval
        self.model_path = kwargs.get("model_path", Config.LOCAL_MODEL_PATHS.get("bge_vl_base", os.path.join(Config.PROJECT_ROOT, 'models', 'BGE-VL-Base')))
        self.index_path = kwargs.get("index_path", os.getenv("MMMRAG_CROSS_MODAL_INDEX_PATH", os.path.join(Config.PROJECT_ROOT, 'data', 'ViDoSeek', 'bgevlbase_ingestion')))
        self.top_k = kwargs.get("top_k", 10)
        self.text_model = None
        self.visual_model = None
        self.tokenizer = None
        
        # Support shared vector store for hybrid retrieval
        model_name = "bge-m3"
        device = DEVICE_OVERRIDES.get(model_name, "cuda:0")
        self.vector_store = kwargs.get("shared_vector_store", VectorStore(config=config, device=device))
        # Create separate vector store for cross-modal retrieval with 512 dimensions
        cross_modal_dim = getattr(config, 'CROSS_MODAL_DIMENSION', 512)
        model_name_vl = "bge-vl-base"
        device_vl = DEVICE_OVERRIDES.get(model_name_vl, "cuda:0")
        self.crossmodal_vector_store = VectorStore(config=config, dimension=cross_modal_dim, device=device_vl)
        # Create vector store for same GPU (cuda:0)
        self.vector_store_alt = VectorStore(config=config, device=device_vl)
        self.crossmodal_vector_store_alt = VectorStore(config=config, dimension=cross_modal_dim, device=device_vl)
        
        # Support model cache to avoid reloading
        self._retriever_agent_ref = kwargs.get("retriever_agent_ref", None)
        self.use_cached_model = kwargs.get("use_cached_model", True)
        
        # Initialize reranker configuration
        self.reranker_config = kwargs.get("reranker", {})
        self.reranker = None
        self.reranker_model_path = self.reranker_config.get("model_path", Config.LOCAL_MODEL_PATHS.get("jina_reranker_m0", os.path.join(Config.PROJECT_ROOT, 'models', 'jina-reranker-m0')))
        self.reranker_top_k = self.reranker_config.get("top_k", self.top_k)
        
        # Device configuration
        self.device = None  # Will be set during initialization
        
        self._initialized = False
        self.initialize()
    
    def initialize(self):
        """
        Initialize the cross-modal retriever with BGE-VL-Large model
        Ensures idempotency - multiple calls won't reset state
        """
        # Check if already initialized - if yes, skip to ensure idempotency
        if self._initialized:
            self.logger.info("Cross-modal retriever already initialized, skipping")
            return
            
        try:
            from PIL import Image

            import torch
            import os
            import json
            
            # Use hardcoded device assignment from DEVICE_OVERRIDES
            model_name = "bge-vl-base"
            self.device = torch.device(DEVICE_OVERRIDES.get(model_name, "cuda:0"))
            self.logger.info(f"Using hardcoded device {self.device} for CrossModalRetrieverAgent to load {model_name} model")
            
            # Load knowledge base index if index_path is provided
            if hasattr(self, 'index_path') and self.index_path:

                # Check if we're using a shared vector store (for hybrid retrieval)
                using_shared_store = hasattr(self, '_using_shared_vector_store') and self._using_shared_vector_store
                
                # Only load vectors if not using shared store
                if not using_shared_store:
                    # Directly check if it's a string first
                    if not isinstance(self.index_path, str):
                        self.logger.info(f"index_path is not a string, skipping index loading: {type(self.index_path)}")
                    elif not self.index_path.strip():
                        self.logger.info("index_path is empty, skipping index loading")
                    else:
                        # Now we know it's a non-empty string
                        index_path_str = self.index_path.strip()
                        try:
                            if os.path.isfile(index_path_str):
                                with open(index_path_str, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    # Load text corpus and image corpus from the index
                                    # Use 'text_corpus' key if available, fallback to 'corpus'
                                    # Only set if not already populated (preserve existing content)
                                    if not self.text_corpus:
                                        self.text_corpus = data.get("text_corpus", data.get("corpus", []))
                                    if not self.image_corpus:
                                        self.image_corpus = data.get("image_corpus", [])
                                    if not self.audio_corpus:
                                        self.audio_corpus = data.get("audio_corpus", [])
                                    self.logger.info(f"Loaded knowledge base from {index_path_str} with {len(self.text_corpus)} text, {len(self.image_corpus)} image, and {len(self.audio_corpus)} audio documents")
                            elif os.path.isdir(index_path_str):
                                # If it's a directory, process all .node files using VectorStore
                                self.logger.info(f"Processing directory {index_path_str} for .node files")
                                # Load index on primary GPU
                                self.vector_store.load(index_path_str)
                                self.logger.info(f"Loaded {len(self.vector_store)} vectors from .node files on primary device")
                                
                                # Optimize: Copy data to secondary vector store instead of reloading from disk
                                if len(self.vector_store.vectors) > 0:
                                    self.logger.info("Copying vectors to secondary device store (skipping disk reload)")
                                    import faiss
                                    # Copy vectors (numpy array)
                                    self.vector_store_alt.vectors = self.vector_store.vectors.copy()
                                    # Copy metadata (list)
                                    self.vector_store_alt.metadata = self.vector_store.metadata.copy()
                                    # Set dimension
                                    self.vector_store_alt.dimension = self.vector_store.dimension
                                    # Rebuild FAISS index on secondary store
                                    self.vector_store_alt.faiss_index = faiss.IndexFlatIP(self.vector_store_alt.dimension)
                                    self.vector_store_alt.faiss_index.add(self.vector_store_alt.vectors)
                                    self.vector_store_alt._vectors_loaded = True
                                    
                                    self.logger.info(f"Loaded {len(self.vector_store_alt)} vectors to secondary device from memory")
                                else:
                                    # Fallback if primary load failed or was empty
                                    self.vector_store_alt.load(index_path_str)
                                    
                                # Add metadata to corpus for compatibility with existing code
                                for metadata in self.vector_store.metadata:
                                    # Check if it's an image document based on various image path fields
                                    image_path = None
                                    if "image_path" in metadata:
                                        image_path = metadata["image_path"]
                                    elif "file_path" in metadata:
                                        image_path = metadata["file_path"]
                                    elif "image_resource" in metadata and isinstance(metadata["image_resource"], dict):
                                        image_path = metadata["image_resource"].get("path")
                                    
                                    # Fix relative paths for image files
                                    if image_path:
                                        _data_base = getattr(Config, 'DATA_BASE_DIR', os.path.join(Config.PROJECT_ROOT, 'data'))
                                        if image_path.startswith("./data/ViDoSeek/"):
                                            image_path = image_path.replace("./data/", _data_base + "/")
                                        elif image_path.startswith("data/ViDoSeek/"):
                                            image_path = os.path.join(_data_base, image_path)
                                        elif image_path.startswith("data/SlideVQA/"):
                                            image_path = os.path.join(_data_base, image_path)
                                        elif image_path.startswith("./data/SlideVQA/"):
                                            image_path = image_path.replace("./data/", _data_base + "/")
                                        
                                    if image_path:
                                        # Create ImageDocument for image documents
                                        doc = {
                                            "id": metadata.get("id", f"img_{len(self.image_corpus)}"),
                                            "text": metadata.get("text", ""),  # Use empty string instead of default
                                            "image_path": image_path,
                                            "metadata": metadata
                                        }
                                        self.image_corpus.append(doc)
                                    else:
                                        # Create text document for non-image documents
                                        doc = {
                                            "id": metadata.get("id", f"doc_{len(self.text_corpus)}"),
                                            "text": metadata.get("text", ""),  # Use empty string instead of default
                                            "metadata": metadata
                                        }
                                        if doc["text"]:
                                            self.text_corpus.append(doc)
                                self.logger.info(f"Loaded knowledge base from {index_path_str} with {len(self.text_corpus)} text documents and {len(self.image_corpus)} image documents from .node files")
                            else:
                                # It's neither a file nor a directory
                                self.logger.info(f"index_path is neither a file nor a directory, skipping index loading: {index_path_str}")
                        except Exception as e:
                            self.logger.error(f"Failed to load knowledge base index: {e}")
                else:
                    self.logger.info("Using shared vector store, skipping index loading")
            
            # Use ModelManager to get pre-loaded model
            self.logger.info(f"Loading BGE-VL-Base model from ModelManager")
            
            # Get model from ModelManager
            model_info = model_manager.get_model("bge-vl-base")
            if model_info and "model" in model_info and "processor" in model_info:
                self.logger.info(f"Using pre-loaded model from ModelManager")
                self.model = model_info["model"]
                self.processor = model_info["processor"]
                self.logger.info(f"Successfully loaded model of type: {type(self.model).__name__}")
                
                # Set image_newline attribute if it doesn't exist (fix for LLaVANextForEmbedding)
                if not hasattr(self.model, 'image_newline'):
                    self.logger.info("Setting image_newline attribute on model")
                    self.model.image_newline = "\n"
                
                # Set model to evaluation mode
                self.model.eval()
                self.logger.info("Model set to evaluation mode")
                
                # Initialize with fallback for text and visual models for backward compatibility
                self.text_model = None
                self.visual_model = None
                self.tokenizer = None  # Use processor instead
                
                # Initialize multimodal reranker
                self._initialize_reranker()
                
                self._initialized = True
                self.logger.info(f"Cross-modal retriever initialized with BGE-VL-Base model (pre-loaded)")
                self.logger.info(f"Cross-modal retriever initialized with {len(self.text_corpus)} text, {len(self.image_corpus)} image, {len(self.audio_corpus)} audio documents")
                return
            
            # If ModelManager fails, use fallback to direct loading
            self.logger.warning("Model not found in ModelManager, falling back to direct loading")
            
            # Check if the model path is valid
            if self.model_path is None:
                self.logger.error("Model path is None")
                self._initialized = True  # Still allow initialization for fallback
                return
            
            # Convert to string if it's not already
            try:
                model_path_str = str(self.model_path)
            except Exception as e:
                self.logger.error(f"Failed to convert model path to string: {e}")
                self._initialized = True  # Still allow initialization for fallback
                return
            
            # Check if the model path exists
            try:
                if not os.path.exists(model_path_str):
                    self.logger.warning(f"Model path does not exist: {model_path_str}")
                    self._initialized = True  # Still allow initialization for fallback
                    return
            except Exception as e:
                self.logger.error(f"Failed to check if model path exists: {e}")
                self._initialized = True  # Still allow initialization for fallback
                return
            
            # Load BGE-VL-Large model using transformers
            self.logger.info(f"Loading BGE-VL-Large model from local path: {model_path_str}")
            
            # Check if we should use cached model
            if self.use_cached_model and self._retriever_agent_ref and hasattr(self._retriever_agent_ref, '_model_cache') and isinstance(self._retriever_agent_ref._model_cache, dict):
                # Check if model is already cached
                cache_key = model_path_str
                if cache_key in self._retriever_agent_ref._model_cache:
                    # Always use cached model without expiration check
                    self.logger.info(f"Using cached model from {cache_key} (permanent cache)")
                    self.model = self._retriever_agent_ref._model_cache[cache_key]["model"]
                    self.processor = self._retriever_agent_ref._model_cache[cache_key]["processor"]
                    self.logger.info(f"Successfully loaded cached model of type: {type(self.model).__name__}")
                    
                    # Set image_newline attribute if it doesn't exist (fix for LLaVANextForEmbedding)
                    if not hasattr(self.model, 'image_newline'):
                        self.logger.info("Setting image_newline attribute on model")
                        self.model.image_newline = "\n"
                    
                    # Move model to the correct device
                    self.logger.info(f"Attempting to move cached model to device: {self.device}")
                    # Check if model is already on the correct device
                    model_device = next(self.model.parameters()).device
                    if model_device != self.device:
                        # Use the correct device index based on the selected device
                        device_index = int(str(self.device).split(':')[-1]) if ':' in str(self.device) else 0
                        # Move model to the correct device
                        self.model = device_manager.distribute_model(self.model, device_index)
                        self.logger.info(f"Moved cached model to device: {self.device}")
                        # Update the cache with the model on the current device to avoid duplicate copies
                        self._retriever_agent_ref._model_cache[cache_key]["model"] = self.model
                        self._retriever_agent_ref._model_cache[cache_key]["device"] = str(self.device)
                        self.logger.info(f"Updated cache with model on device: {self.device}")
                    else:
                        self.logger.info(f"Cached model is already on device: {self.device}, skipping move")
                    
                    # Set model to evaluation mode
                    self.model.eval()
                    self.logger.info("Model set to evaluation mode")
                    
                    # Initialize with fallback for text and visual models for backward compatibility
                    self.text_model = None
                    self.visual_model = None
                    self.tokenizer = None  # Use processor instead
                    
                    # Initialize multimodal reranker
                    self._initialize_reranker()
                    
                    self._initialized = True
                    self.logger.info(f"Cross-modal retriever initialized with BGE-VL-Large model (cached)")
                    self.logger.info(f"Cross-modal retriever initialized with {len(self.text_corpus)} text, {len(self.image_corpus)} image, {len(self.audio_corpus)} audio documents")
                    return
            
            try:
                # Import transformers for BGE-VL-Large
                from transformers import AutoModel, AutoProcessor
                import os
                
                # Load processor
                self.logger.info("Loading BGE-VL-Large processor...")
                self.processor = AutoProcessor.from_pretrained(model_path_str, local_files_only=True)
                self.logger.info(f"Successfully loaded processor of type: {type(self.processor).__name__}")
                
            # Load model with float32 for better CUDA compatibility
                self.logger.info(f"Loading BGE-VL-Large model on device: {self.device}")
                self.logger.info(f"Device type: {type(self.device)}, Device string: {str(self.device)}")
                
                # Explicitly set the device for model loading
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device).split(':')[-1]
                self.logger.info(f"Set CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")
                
                self.model = AutoModel.from_pretrained(
                    model_path_str,
                    torch_dtype=torch.float32,  # Use float32 for CUDA compatibility
                    local_files_only=True,
                    low_cpu_mem_usage=False,  # Disable low memory usage for stability
                    device_map=None,  # Disable device map for direct CUDA usage
                    trust_remote_code=True
                )
                
                # Set image_newline attribute if it doesn't exist (fix for LLaVANextForEmbedding)
                if not hasattr(self.model, 'image_newline'):
                    self.logger.info("Setting image_newline attribute on model")
                    self.model.image_newline = "\n"
                
                # Move model to the correct device using device manager
                device_index = int(str(self.device).split(':')[-1]) if ':' in str(self.device) else 0
                self.model = device_manager.distribute_model(self.model, device_index)
                self.logger.info(f"Successfully loaded model of type: {type(self.model).__name__} and moved to device: {self.device}")
                
                # Set model to evaluation mode
                self.model.eval()
                self.logger.info("Model set to evaluation mode")
                
                # Cache the loaded model and processor
                if self._retriever_agent_ref and hasattr(self._retriever_agent_ref, '_model_cache'):
                    cache_key = model_path_str
                    import time
                    # Store the model already on the correct device to avoid duplicate copies
                    self._retriever_agent_ref._model_cache[cache_key] = {
                        "model": self.model,
                        "processor": self.processor,
                        "timestamp": time.time(),
                        "device": str(self.device)  # Store the device where the model is located
                    }
                    # No expiration time needed for permanent cache
                    self.logger.info(f"Cached model and processor for {cache_key} (permanent cache) on device {self.device}")
                
                # Initialize with fallback for text and visual models for backward compatibility
                self.text_model = None
                self.visual_model = None
                self.tokenizer = None  # Use processor instead
                
                # Initialize multimodal reranker
                self._initialize_reranker()
                
                self._initialized = True
                self.logger.info(f"Cross-modal retriever initialized with BGE-VL-Large model")
                self.logger.info(f"Cross-modal retriever initialized with {len(self.text_corpus)} text, {len(self.image_corpus)} image, {len(self.audio_corpus)} audio documents")
            except Exception as e:
                self.logger.error(f"Failed to load BGE-VL-Large model components: {e}")
                self.logger.warning("Cross-modal retriever initialization failed, will use fallback implementation")
                self.logger.info("Please check if transformers is properly installed and updated")
                self.logger.info("Try: pip install transformers")
                self._initialized = False  # Initialization failed
                return
        except ImportError as e:
            self.logger.error(f"Failed to import required libraries for cross-modal retriever: {e}")
            self.logger.warning("Cross-modal retriever initialization failed, will use fallback implementation")
            self.logger.info("Please install colpali-engine: pip install colpali-engine==0.3.0")
            self._initialized = False  # Initialization failed
        except Exception as e:
            self.logger.error(f"Failed to initialize cross-modal retriever: {e}")
            self.logger.warning("Cross-modal retriever initialization failed, will use fallback implementation")
            self._initialized = False  # Initialization failed
    
    def _initialize_reranker(self):
        """
        Initialize the multimodal reranker
        """
        try:
            self.logger.info(f"Initializing multimodal reranker with model: {self.reranker_model_path}")
            self.reranker = MultimodalReranker(self.config, model_path=self.reranker_model_path, top_k=self.reranker_top_k)
            self.reranker.initialize()
            
            if self.reranker._initialized:
                self.logger.info("Multimodal reranker successfully initialized")
            else:
                self.logger.warning("Multimodal reranker initialization failed, will skip reranking")
                self.reranker = None
        except Exception as e:
            self.reranker = None
    
    def _text_to_image(self, text: str) -> Image.Image:
        """
        Convert text query to image with blank background
        
        Args:
            text: Input text query
            
        Returns:
            PIL Image object with text rendered on white background
        """
        from PIL import Image, ImageDraw, ImageFont
        
        # Create blank white image
        width, height = 512, 256
        image = Image.new('RGB', (width, height), color='white')
        
        # Draw text
        draw = ImageDraw.Draw(image)
        
        # Try different fonts to find available one
        fonts_to_try = ["arial", "calibri", "times", "courier"]
        font = None
        
        for font_name in fonts_to_try:
            try:
                font = ImageFont.truetype(font_name, 24)
                break
            except:
                continue
        
        # Fallback to default font if no truetype font available
        if font is None:
            font = ImageFont.load_default()
        
        # Calculate text position to center it
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (width - text_width) // 2
            y = (height - text_height) // 2
        except:
            # Fallback to simple positioning
            x, y = 50, 100
        
        # Draw text
        draw.text((x, y), text, fill='black', font=font)
        
        self.logger.info(f"Converted text query to image: '{text[:30]}...'")
        return image

    def _generate_image_vector(self, image_path: str) -> List[float]:
        """
        Generate vector for image from path using BGE-VL-Large model
        
        Args:
            image_path: Path to image file
            
        Returns:
            Vector representation
        """
        from PIL import Image
        
        _data_base = getattr(Config, 'DATA_BASE_DIR', os.path.join(Config.PROJECT_ROOT, 'data'))
        if image_path.startswith("./data/ViDoSeek/img/"):
            image_path = os.path.join(_data_base, image_path.lstrip('./'))
            self.logger.debug(f"Converted ViDoSeek image path: {image_path}")
            
        if image_path.startswith("data/SlideVQA/"):
             if "/img/" not in image_path:
                 image_path = image_path.replace("data/SlideVQA/", "data/SlideVQA/img/")
             image_path = os.path.join(_data_base, image_path)
             self.logger.debug(f"Converted SlideVQA image path: {image_path}")
        
        # Open image
        try:
            image = Image.open(image_path)
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            self.logger.error(f"Failed to open image {image_path}: {e}")
            raise
        
        # Use the same text as the query for consistency
        # For image vectors, we'll use an empty string as the text prompt
        # since we want to represent the image itself, not the query
        return self._generate_image_vector_from_pil(image, text="")

    def _generate_image_vector_from_pil(self, image: Image.Image, text: str) -> List[float]:
        """
        Generate vector for PIL image using BGE-VL-Large model
        
        Args:
            image: PIL Image object
            text: Text description to use with the image (the original query)
            
        Returns:
            Vector representation
        """
        import torch
        import numpy as np
        
        self.logger.info(f"Starting _generate_image_vector_from_pil with text: '{text[:30]}...'")
        self.logger.info(f"Image type: {type(image).__name__}, mode: {image.mode}")
        
        # Process image input with the provided text (required for vision-language models)
        try:
            self.logger.info(f"Processing inputs with processor of type: {type(self.processor).__name__}")
            inputs = self.processor(
                text=text,  # Use the original query text
                images=image,
                return_tensors="pt"
            )
            self.logger.info(f"Successfully processed inputs, keys: {list(inputs.keys())}")
        except Exception as e:
            self.logger.error(f"Error processing inputs: {e}")
            import traceback
            self.logger.error(f"Error details: {traceback.format_exc()}")
            raise
        
        # Move inputs to the same device as the model
        try:
            self.logger.info(f"Moving inputs to device: {self.device}")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            self.logger.info("Successfully moved inputs to device")
        except Exception as e:
            self.logger.error(f"Error moving inputs to device: {e}")
            import traceback
            self.logger.error(f"Error details: {traceback.format_exc()}")
            raise
        
        # Forward pass through the model
        try:
            self.logger.info("Performing model forward pass")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            self.logger.info(f"Successfully performed forward pass, type: {type(outputs).__name__}")
                
        except Exception as e:
            self.logger.error(f"Error in model forward pass: {e}")
            import traceback
            self.logger.error(f"Error details: {traceback.format_exc()}")
            raise
        
        # Extract image embeddings from outputs
        try:
            # Check for common embedding attributes
            if hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
                image_features = outputs.image_embeds
            elif hasattr(outputs, 'image_embeddings'):
                image_features = outputs.image_embeddings
            elif hasattr(outputs, 'embeddings'):
                image_features = outputs.embeddings
            elif hasattr(outputs, 'last_hidden_state'):
                # Take mean of last hidden state
                image_features = outputs.last_hidden_state
                image_features = image_features.mean(dim=1)
            elif hasattr(outputs, 'pooler_output'):
                image_features = outputs.pooler_output
            elif isinstance(outputs, torch.Tensor):
                # If output is already a tensor, use it directly
                image_features = outputs
            else:
                # Try to get vision_model_output if available
                if hasattr(outputs, 'vision_model_output'):
                    vision_output = outputs.vision_model_output
                    if hasattr(vision_output, 'last_hidden_state'):
                         image_features = vision_output.last_hidden_state.mean(dim=1)
                    elif hasattr(vision_output, 'pooler_output'):
                         image_features = vision_output.pooler_output
                    else:
                         # Fallback to text_embeds if image_embeds is missing (e.g. text-only query)
                         if hasattr(outputs, 'text_embeds') and outputs.text_embeds is not None:
                             image_features = outputs.text_embeds
                         else:
                             raise AttributeError(f"Could not find embeddings in {type(outputs).__name__}")
                # Fallback to text_embeds if image_embeds is missing (e.g. text-only query)
                elif hasattr(outputs, 'text_embeds') and outputs.text_embeds is not None:
                     image_features = outputs.text_embeds
                else:
                    # Final fallback: try to get any tensor attribute
                    if hasattr(outputs, 'last_hidden_state'):
                        image_features = outputs.last_hidden_state[:, 0, :]
                    else:
                        raise AttributeError(f"Could not find embeddings in {type(outputs).__name__}")
                
        except Exception as e:
            self.logger.error(f"Error extracting embeddings: {e}")
            # Final fallback
            if hasattr(outputs, 'last_hidden_state'):
                 image_features = outputs.last_hidden_state[:, 0, :]
            else:
                 raise
        
        # Convert to float32 to handle BFloat16 type
        try:
            if hasattr(image_features, 'dtype') and image_features.dtype == torch.bfloat16:
                image_features = image_features.to(dtype=torch.float32)
        except Exception as e:
            pass
        
        # Ensure proper tensor shape
        try:
            image_features = image_features.squeeze()
            if image_features.dim() == 0:
                image_features = image_features.unsqueeze(0)
            elif image_features.dim() == 2 and image_features.shape[0] == 1:
                image_features = image_features.squeeze(0)
        except Exception as e:
            self.logger.error(f"Error reshaping image features: {e}")
            raise
        
        # Convert to numpy array
        try:
            embedding_np = image_features.cpu().numpy()
        except Exception as e:
            self.logger.error(f"Error converting to numpy: {e}")
            raise
        
        return embedding_np.tolist()

    def _generate_text_vector(self, text: str, for_crossmodal: bool = False) -> List[float]:
        """
        Generate vector for text using BGE-VL-Large model
        
        Args:
            text: Input text
            for_crossmodal: Whether to generate vector for cross-modal retrieval (default: False)
                           When True, returns original vector for cross-modal similarity
                           When False, returns 1024-dimensional vector for text retrieval
            
        Returns:
            Vector representation
        """
        import torch
        import numpy as np
        
        # Validate input text for research-grade quality
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty for model-based vector generation")
        
        # Ensure proper initialization
        if not self._initialized:
            raise ValueError("Cross-modal retriever not initialized")
        
        if not hasattr(self, 'model') or self.model is None or not hasattr(self, 'processor') or self.processor is None:
            raise ValueError("Model or processor not available")
        
        # For cross-modal retrieval, convert text to image and use image retrieval
        if for_crossmodal:
            self.logger.info("Using text-to-image conversion for cross-modal retrieval")
            # Convert text to image
            text_image = self._text_to_image(text)
            # Generate vector using image processing pipeline with the original text
            return self._generate_image_vector_from_pil(text_image, text=text)
        
        # For pure text retrieval, raise error since we've removed the old method
        raise ValueError("Pure text retrieval is no longer supported, use cross-modal retrieval instead")
    
    def score(self, image_embeddings, text_embeddings):
      
        # For BGE-VL-Large, use cosine similarity directly instead of score_multi_vector
        try:
            import numpy as np
            import torch
            
            # Convert to numpy arrays if they are tensors
            if isinstance(image_embeddings, torch.Tensor):
                image_embeddings = image_embeddings.cpu().numpy()
            if isinstance(text_embeddings, torch.Tensor):
                text_embeddings = text_embeddings.cpu().numpy()
            
            # Ensure both are 1D vectors
            if image_embeddings.ndim > 1:
                # Handle different shapes
                if image_embeddings.shape[0] == 1 and image_embeddings.ndim == 2:
                    image_embeddings = image_embeddings[0]
                else:
                    image_embeddings = image_embeddings.mean(axis=tuple(range(1, image_embeddings.ndim)))
            if text_embeddings.ndim > 1:
                # Handle different shapes
                if text_embeddings.shape[0] == 1 and text_embeddings.ndim == 2:
                    text_embeddings = text_embeddings[0]
                else:
                    text_embeddings = text_embeddings.mean(axis=tuple(range(1, text_embeddings.ndim)))
            
            # Calculate cosine similarity
            dot_product = np.dot(image_embeddings, text_embeddings)
            norm_image = np.linalg.norm(image_embeddings)
            norm_text = np.linalg.norm(text_embeddings)
            
            # Avoid division by zero
            similarity = dot_product / (norm_image * norm_text + 1e-10)
            
            # Clamp to valid range
            similarity = max(-1.0, min(1.0, similarity))
            
            return similarity
        except Exception as e:
            self.logger.error(f"Failed to calculate similarity: {e}")
            return 0.0

    def _calculate_similarity(self, query_vector: List[float], doc_vector: List[float]) -> float:
     
        try:
            import numpy as np
            
            # Convert to numpy arrays
            query_np = np.array(query_vector)
            doc_np = np.array(doc_vector)
            
            # Handle different dimensions
            self.logger.debug(f"Calculating similarity between vectors of shapes: query={query_np.shape}, doc={doc_np.shape}")
            
            # If either vector is 2D (e.g., [batch, sequence, embedding]), take the mean
            if query_np.ndim > 1:
                query_np = query_np.mean(axis=tuple(range(1, query_np.ndim)))
            if doc_np.ndim > 1:
                doc_np = doc_np.mean(axis=tuple(range(1, doc_np.ndim)))
            
            # Ensure both vectors are 1D
            if query_np.ndim != 1 or doc_np.ndim != 1:
                self.logger.error(f"Vectors must be 1D after flattening, got query={query_np.shape}, doc={doc_np.shape}")
                return 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(query_np, doc_np)
            norm_query = np.linalg.norm(query_np)
            norm_doc = np.linalg.norm(doc_np)
            
            # Avoid division by zero
            similarity = dot_product / (norm_query * norm_doc + 1e-10)
            
            # Clamp to valid range
            similarity = max(-1.0, min(1.0, similarity))
            
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Failed to calculate similarity: {e}")
            return 0.0

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 format
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image data with data URI prefix
        """
        try:
            from PIL import Image
            from io import BytesIO
            import base64
            import os
            
            _data_base = getattr(Config, 'DATA_BASE_DIR', os.path.join(Config.PROJECT_ROOT, 'data'))
            if image_path.startswith("./data/ViDoSeek/img/"):
                image_path = os.path.join(_data_base, image_path.lstrip('./'))
                self.logger.debug(f"Converted ViDoSeek image path: {image_path}")

            if "SlideVQA" in image_path:
                if "data/SlideVQA/img/" in image_path:
                    image_path = image_path.replace("data/SlideVQA/img/", os.path.join(_data_base, 'SlideVQA', 'img') + '/')
                elif image_path.startswith("data/SlideVQA/"):
                    image_path = os.path.join(_data_base, image_path)
                self.logger.debug(f"Converted SlideVQA image path: {image_path}")
            
            # Check if image path exists
            if not os.path.exists(image_path):
                self.logger.warning(f"Image file not found: {image_path}")
                return None
            
            # Open and encode image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image to reasonable size for LLM
                max_size = 1024
                img.thumbnail((max_size, max_size))
                
                # Save to buffer
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                buffer.seek(0)
                
                # Encode to base64
                base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Return data URI
                return f"data:image/jpeg;base64,{base64_data}"
        except Exception as e:
            self.logger.error(f"Failed to encode image to base64: {e}")
    
    def retrieve(self, query: str, image_data: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute cross-modal retrieval with image data support

        Args:
            query: Query text
            image_data: Base64 encoded image data (optional)
            **kwargs: Other parameters

        Returns:
            Cross-modal retrieval results
        """
        if not validate_query(query):
            return {"results": [], "error": "Invalid query"}

        try:
            # Check if we have valid initialization
            if not self._initialized:
                self.initialize()
            
            # Synchronize devices before retrieval
            device_manager.sync_devices()
            self.logger.info("Synchronized devices before cross-modal retrieval")
            
            # Get query images if any
            query_images = kwargs.get("images", [])
            
            # If image_data is provided, add it to query_images
            if image_data:
                query_images.append(image_data)
                self.logger.info("Added provided image_data to query_images for cross-modal retrieval")
            
            # Determine dominant modality for query
            query_modality = "image" if query_images else "text"
            self.logger.info(f"Modality analysis: {query_modality}")
            
            # Execute retrieval based on dominant modality
            if query_modality == "image":
                # If we have query images, use them for retrieval
                results = self._retrieve_by_images(query_images, query, **kwargs)
            else:
                # For cross-modal retrieval (hybrid), use for_crossmodal=True
                is_hybrid = kwargs.get("hybrid", False)
                
                # Check if we should use top_k from kwargs or default to self.top_k
                retrieval_top_k = kwargs.pop("top_k", self.top_k)
                
                self.logger.info(f"Calling _retrieve_by_text with for_crossmodal={is_hybrid}, top_k={retrieval_top_k}")
                results = self._retrieve_by_text(query, for_crossmodal=is_hybrid, top_k=retrieval_top_k, **kwargs)
            
            # Check if reranking is disabled
            disable_rerank = kwargs.get('disable_rerank', False) or getattr(self.config, 'RERANK', {}).get('disable', False)

            # Apply reranking if reranker is available and initialized and not disabled
            if self.reranker and self.reranker._initialized and results and not disable_rerank:
                self.logger.info(f"Applying multimodal reranking to {len(results)} results")
                try:
                    # Get rerank_top_k from kwargs or use default value
                    rerank_top_k = kwargs.get('rerank_top_k', 5)
                    reranked_results = self.reranker.rerank(query, results, top_k=rerank_top_k)
                    # Use reranked results if successful
                    if reranked_results:
                        results = reranked_results  # Already limited to top_k by rerank method
                        self.logger.info(f"Successfully reranked results, returning top {len(results)}")
                except Exception as e:
                    self.logger.error(f"Reranking failed: {e}")
            elif results:
                # No reranking, just limit to rerank_top_k results
                rerank_top_k = kwargs.get('rerank_top_k', 5)
                results = results[:rerank_top_k]
            
            # Encode images to base64 for LLM understanding
            encoded_results = []
            for result in results:
                encoded_result = result.copy()
                
                # Check if this is an image document
                if result.get("modality") == "image":
                    # Get image path from metadata or image_path field
                    image_path = result.get("image_path") or result.get("metadata", {}).get("image_path") or result.get("metadata", {}).get("path")
                    
                    if image_path:
                        # Encode image to base64
                        base64_image = self._encode_image_to_base64(image_path)
                        if base64_image:
                            encoded_result["base64_image"] = base64_image
                            encoded_result["image_encoding_processed"] = True
                        else:
                            encoded_result["image_encoding_processed"] = False
                
                encoded_results.append(encoded_result)
            
            # Determine which retrieval method was used
            retrieval_method = "text-to-image cross-modal" if query_modality == "text" else "image cross-modal"
            
            # Prepare final result with retrieved_results for LLM
            final_result = {
                "query": query,
                "query_modality": query_modality,
                "image_data_provided": True if image_data else False,
                "results": format_results(encoded_results), # Use standard format for results field
                "retrieved_results": encoded_results,  # For LLM input (includes base64)
                "total": len(encoded_results),
                "retriever": "CrossModalRetrieverAgent",
                "retrieval_method": retrieval_method
            }
            
            return final_result            
        except Exception as e:
            self.logger.error(f"Cross-modal retrieval failed: {e}")
            return {"results": [], "error": f"Cross-modal retrieval failed: {str(e)}", "retriever": "CrossModalRetrieverAgent"}


    def _retrieve_by_text(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents by text query
        
        Args:
            query: Query text
            **kwargs: Parameters
            
        Returns:
            Retrieval results
        """
        top_k = kwargs.get("top_k", self.top_k)
        # Check if this is for cross-modal retrieval
        for_crossmodal = kwargs.get("for_crossmodal", False)
        
        # Generate query vector
        try:
            # Always force cross-modal for CrossModalRetrieverAgent
            for_crossmodal = True
            query_vector = self._generate_text_vector(query, for_crossmodal=for_crossmodal)
        except Exception as e:
            self.logger.error(f"Error generating query vector: {e}")
            raise
        
        # Check if we have vectors in the vector store
        if for_crossmodal:
            # For cross-modal retrieval, check if we have pre-computed vectors in crossmodal_vector_store
            
            # 1. Try to use existing vectors in crossmodal_vector_store
            if self.crossmodal_vector_store.faiss_index.ntotal > 0:
                search_results = self.crossmodal_vector_store.search(query_vector, top_k=top_k)
                # Map results to expected format
                results = []
                for res in search_results:
                    results.append({
                        "text": res["metadata"].get("text", ""),
                        "score": res["score"],
                        "source": res["metadata"].get("source", res["metadata"].get("id", "unknown")),
                        "modality": res["metadata"].get("modality", "image"),
                        "metadata": res["metadata"]
                    })
                return results

            # 2. Try to load from cache
            import os
            import pickle
            import time
            
            # Determine cache path
            cache_dir = self.index_path if isinstance(self.index_path, str) and os.path.isdir(self.index_path) else os.path.dirname(self.index_path) if isinstance(self.index_path, str) else "."
            cache_path = os.path.join(cache_dir, "crossmodal_vectors_cache.pkl")
            
            if os.path.exists(cache_path):
                self.logger.info(f"Found cross-modal vector cache: {cache_path}, loading...")
                try:
                    with open(cache_path, "rb") as f:
                        cache_data = pickle.load(f)
                    
                    if cache_data.get("dimension") == self.crossmodal_vector_store.dimension:
                        vectors = cache_data["vectors"]
                        metadatas = cache_data["metadata"]
                        
                        # Add to store
                        self.crossmodal_vector_store.add_vectors(vectors, metadatas)
                        
                        # Search
                        search_results = self.crossmodal_vector_store.search(query_vector, top_k=top_k)
                        results = []
                        for res in search_results:
                            results.append({
                                "text": res["metadata"].get("text", ""),
                                "score": res["score"],
                                "source": res["metadata"].get("source", res["metadata"].get("id", "unknown")),
                                "modality": res["metadata"].get("modality", "image"),
                                "metadata": res["metadata"]
                            })
                        return results
                except Exception as e:
                    self.logger.error(f"Failed to load cross-modal cache: {e}")

            # 3. Slow path: Generate vectors and cache them
            self.logger.info("No cache found, generating vectors for all documents (this may take a while)...")
            
            # Combine all documents for cross-modal retrieval
            all_docs = []
            
            # Add text documents
            for doc in self.text_corpus:
                all_docs.append({
                    "id": doc.get("id", str(len(all_docs))),
                    "text": doc.get("text", ""),
                    "modality": "text",
                    "metadata": doc.get("metadata", {})
                })
            
            # Add image documents with captions
            for doc in self.image_corpus:
                # For image documents, use the actual description or caption if available
                image_text = doc.get("description", "")
                if not image_text:
                    image_text = doc.get("text", "")
                all_docs.append({
                    "id": doc.get("id", str(len(all_docs))),
                    "text": image_text,  # Use actual description or empty string, not default
                    "modality": "image",
                    "metadata": doc.get("metadata", {})
                })
            
            # Calculate cross-modal similarities using direct cosine similarity
            results = []
            
            # Collect vectors for caching
            generated_vectors = []
            generated_metadatas = []
            
            # Use batch processing for efficiency if possible, otherwise sequential
            for i, doc in enumerate(all_docs):
                # Generate vector for document based on its modality
                try:
                    if doc["modality"] == "text":
                        doc_vector = self._generate_text_vector(doc["text"], for_crossmodal=True)
                    else:  # image
                        # For image documents, always try to use image path if available
                        image_path = doc["metadata"].get("image_path")
                        if image_path:
                            try:
                                # First try to find pre-computed vector in vector store
                                doc_vector = None
                                for i, metadata in enumerate(self.vector_store.metadata):
                                    if metadata.get("image_path") == image_path:
                                        doc_vector = self.vector_store.vectors[i].tolist()
                                        break
                                
                                # If no pre-computed vector found, generate from image
                                if not doc_vector:
                                    doc_vector = self._generate_image_vector(image_path)
                            except Exception as e:
                                # If image fails, only then use text as fallback
                                if doc.get("text"):
                                    doc_vector = self._generate_text_vector(doc["text"], for_crossmodal=True)
                                else:
                                    continue
                        else:
                            # If no image path, only use text if available
                            if doc.get("text"):
                                doc_vector = self._generate_text_vector(doc["text"], for_crossmodal=True)
                            else:
                                continue
                except Exception as e:
                    continue
                
                # Add to collections for caching
                if doc_vector is not None:
                    # Ensure doc_vector is a list
                    if hasattr(doc_vector, "tolist"):
                        doc_vector = doc_vector.tolist()
                    generated_vectors.append(doc_vector)
                    
                    # Create metadata object
                    meta = doc.get("metadata", {}).copy()
                    if "text" not in meta and doc.get("text"):
                        meta["text"] = doc.get("text")
                    if "modality" not in meta:
                        meta["modality"] = doc["modality"]
                    if "id" not in meta:
                        meta["id"] = doc["id"]
                    generated_metadatas.append(meta)
                
                # Calculate similarity using direct cosine similarity
                try:
                    similarity = self._calculate_similarity(query_vector, doc_vector)
                except Exception as e:
                    continue
                
                results.append({
                    "text": doc["text"],
                    "score": similarity,
                    "source": doc["metadata"].get("source", doc["metadata"].get("id", "unknown")),
                    "modality": doc["modality"],
                    "metadata": doc["metadata"]
                })
            
            # --- Save to Cache Logic ---
            if generated_vectors:
                try:
                    self.logger.info(f"Saving {len(generated_vectors)} vectors to cross-modal cache: {cache_path}")
                    with open(cache_path, "wb") as f:
                        pickle.dump({
                            "vectors": generated_vectors,
                            "metadata": generated_metadatas,
                            "dimension": self.crossmodal_vector_store.dimension
                        }, f)
                    
                    # Also populate the store for future use in this session
                    self.crossmodal_vector_store.add_vectors(generated_vectors, generated_metadatas)
                except Exception as e:
                    self.logger.error(f"Failed to save cross-modal cache: {e}")
            
            # Sort and limit results
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        else:
            # Use regular vector store with 1024 dimensions
            if len(self.vector_store.metadata) > 0:
                search_results = self.vector_store.search(query_vector, top_k=top_k)
                
                # Format results
                results = []
                for result in search_results:
                    metadata = result["metadata"]
                    results.append({
                        "text": metadata.get("text", ""),
                        "score": result["score"],
                        "source": metadata.get("source", metadata.get("id", "unknown")),  # Set source from metadata
                        "modality": metadata.get("modality", "text"),
                        "metadata": metadata,
                        "image_path": metadata.get("path", "")  # Add image path for cross-modal results
                    })
                
                return results
        return []

    def _retrieve_by_images(self, query_images: List[str], query_text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents by query images
        
        Args:
            query_images: List of query image paths
            query_text: Query text
            **kwargs: Parameters
            
        Returns:
            Retrieval results
        """
        top_k = kwargs.get("top_k", self.top_k)
        
        # Generate vectors for query images
        query_vectors = []
        for image_path in query_images:
            try:
                image_vector = self._generate_image_vector(image_path)
                query_vectors.append(image_vector)
            except Exception as e:
                self.logger.error(f"Failed to generate vector for query image {image_path}: {e}")
        
        if not query_vectors:
            # Fallback to text-based retrieval if no image vectors
            return self._retrieve_by_text(query_text, **kwargs)
        
        # Check if we have vectors in the vector store
        if len(self.vector_store) > 0:
            # Use vector_store.search for efficient retrieval
            
            # Perform retrieval for each query vector and combine results
            all_results = {}
            for i, query_vector in enumerate(query_vectors):
                search_results = self.vector_store.search(query_vector, top_k=top_k * 2)  # Get more results for combination
                
                for result in search_results:
                    metadata = result["metadata"]
                    text = metadata.get("text", "")
                    
                    if text not in all_results:
                        all_results[text] = {
                            "text": text,
                            "score": 0,
                            "count": 0,
                            "modality": metadata.get("modality", "image"),
                            "metadata": metadata,
                            "source": metadata.get("source", metadata.get("id", "unknown"))  # Set source from metadata
                        }
                    
                    all_results[text]["score"] += result["score"]
                    all_results[text]["count"] += 1
            
            # Calculate average scores and convert to list
            results = []
            for text, item in all_results.items():
                item["score"] = item["score"] / item["count"]
                del item["count"]
                # Add image_path field for cross-modal results
                item["image_path"] = item["metadata"].get("path", "")
                results.append(item)
            
            # Sort and limit results
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        else:
            # Fallback to original retrieval method if no vectors in store
            # Combine all documents for retrieval
            all_docs = []
            
            # Add text documents
            for doc in self.text_corpus:
                all_docs.append({
                    "id": doc.get("id", str(len(all_docs))),
                    "text": doc.get("text", ""),
                    "modality": "text",
                    "metadata": doc.get("metadata", {})
                })
            
            # Add image documents with captions
            for doc in self.image_corpus:
                all_docs.append({
                    "id": doc.get("id", str(len(all_docs))),
                    "text": doc.get("description", ""),
                    "modality": "image",
                    "metadata": doc.get("metadata", {})
                })
            
            # Calculate similarities
            results = []
            for doc in all_docs:
                # Generate vector for document based on its modality
                if doc["modality"] == "text":
                    doc_vector = self._generate_text_vector(doc["text"], for_crossmodal=True)
                else:  # image
                    # If we have an image path in metadata, generate image vector
                    image_path = doc["metadata"].get("image_path")
                    if image_path:
                        try:
                            doc_vector = self._generate_image_vector(image_path)
                        except Exception as e:
                            continue
                    else:
                        # Fallback to text vector if no image path
                        doc_vector = self._generate_text_vector(doc["text"], for_crossmodal=True)
                
                # Calculate average similarity across all query images
                similarities = []
                for query_vector in query_vectors:
                    # Calculate similarity using the new score method if available
                    similarity = None
                    if hasattr(self, 'score') and hasattr(self, 'processor') and self.processor is not None:
                        try:
                            # Try to use the new score method for cross-modal retrieval
                            import torch
                            # Generate text vector for cross-modal retrieval
                            query_crossmodal_vector = self._generate_text_vector(query_text, for_crossmodal=True)
                            doc_crossmodal_vector = self._generate_text_vector(doc["text"], for_crossmodal=True)
                            
                            # Convert to tensors
                            query_tensor = torch.tensor(query_crossmodal_vector)
                            doc_tensor = torch.tensor(doc_crossmodal_vector)
                            
                            # Calculate similarity
                            similarity = self.score(doc_tensor, query_tensor)
                            # Convert to scalar
                            if isinstance(similarity, torch.Tensor):
                                similarity = similarity.item()
                        except Exception as e:
                            # Fallback to traditional similarity calculation
                            similarity = self._calculate_similarity(query_vector, doc_vector)
                    else:
                        # Fallback to traditional similarity calculation
                        similarity = self._calculate_similarity(query_vector, doc_vector)
                    
                    similarities.append(similarity)
                
                # Use average similarity
                avg_similarity = sum(similarities) / len(similarities)
                
                results.append({
                    "text": doc["text"],
                    "score": avg_similarity,
                    "source": doc["metadata"].get("source", doc["metadata"].get("id", "unknown")),  # Set source from metadata
                    "modality": doc["modality"],
                    "metadata": doc["metadata"],
                    "image_path": doc["metadata"].get("path", "")  # Add image path for cross-modal results
                })
            
            # Sort and limit results
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
    
    def save_index(self, file_path: str):
        """
        Save cross-modal index with text and image corpus
        
        Args:
            file_path: File path to save the index
        """
        import json
        
        data = {
            "text_corpus": self.text_corpus,
            "image_corpus": self.image_corpus,
            "audio_corpus": self.audio_corpus
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Cross-modal index saved to {file_path}")

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.__class__.__name__, "initialized": self._initialized}

    def add_document(self, document: Dict[str, Any]):
        """Add document (text or image) to retriever"""
        if not self._initialized: self.initialize()
        
        try:
            import torch
            from PIL import Image
            import os
            
            text = document.get("text", "")
            image_path = document.get("image_path", "")
            
            vector = None
            
            if image_path and os.path.exists(image_path):
                vector = self._generate_image_vector(image_path)
            elif text:
                vector = self._generate_text_vector(text, for_crossmodal=True)
            
            if vector:
                self.vector_store.add_vector(vector, document)
                
        except Exception as e:
            self.logger.error(f"Failed to add document: {e}")

    def load_index(self, file_path: str):
        """Load index from file"""
        if not self._initialized: self.initialize()
        if self.vector_store: self.vector_store.load(file_path)

    def clear_corpus(self):
        """Clear all documents and vectors"""
        if not self._initialized: self.initialize()
        if self.vector_store: self.vector_store.clear()

# Text Reranker
class TextReranker:
    """Text Reranker using bge-reranker-v2-m3"""
    
    def __init__(self, config: Optional[Config] = None, **kwargs):
        self.config = config or Config()
        self.logger = get_logger(f"reranker.{self.__class__.__name__}")
        self.kwargs = kwargs
        self.model = None
        self._initialized = False
        self.initialize()

    def initialize(self):
        if self._initialized: return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            self.model_path = self.kwargs.get("model_path", Config.LOCAL_MODEL_PATHS.get("bge_reranker_v2_m3", os.path.join(Config.PROJECT_ROOT, 'models', 'bge_reranker_v2_m3')))
            device = torch.device(DEVICE_OVERRIDES.get("bge-reranker-v2-m3", "cuda:0"))
            
            self.logger.info(f"Loading reranker from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path, local_files_only=True, torch_dtype=torch.float32
            ).to(device).eval()
            
            self._initialized = True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._initialized or not candidates: return candidates
        try:
            import torch
            top_k = self.kwargs.get("top_k", 5)
            pairs = []
            valid_indices = []
            for i, doc in enumerate(candidates):
                text = doc.get("text", "")
                if text:
                    pairs.append([query, text])
                    valid_indices.append(i)
            
            if not pairs: return candidates[:top_k]
            
            with torch.no_grad():
                inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                scores = self.model(**inputs, return_dict=True).logits.view(-1,).float().cpu().tolist()
            
            for idx, score in zip(valid_indices, scores):
                candidates[idx]["score"] = score
                
            candidates.sort(key=lambda x: x.get("score", -float("inf")), reverse=True)
            return candidates[:top_k]
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            return candidates

# Multimodal Reranker
class MultimodalReranker:
    
    def __init__(self, config: Optional[Config] = None, **kwargs):
        self.config = config or Config()
        self.logger = get_logger(f"reranker.{self.__class__.__name__}")
        self.kwargs = kwargs
        self.model = None
        self._initialized = False
        self.initialize()

    def initialize(self):
        if self._initialized: return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            import os
            
            self.model_path = self.kwargs.get("model_path", Config.LOCAL_MODEL_PATHS.get("jina_reranker_m0", os.path.join(Config.PROJECT_ROOT, 'models', 'jina-reranker-m0')))
            device = torch.device(DEVICE_OVERRIDES.get("jina-reranker-m0", "cuda:0"))
            
            # Use ModelManager to load model to avoid duplicates and save memory
            model_info = model_manager.get_model("jina-reranker-m0")
            if model_info:
                self.logger.info(f"Using pre-loaded reranker from ModelManager")
                self.model = model_info["model"]
                self.tokenizer = model_info["tokenizer"]
                self._initialized = True
                return

            self.logger.info(f"Loading reranker from {self.model_path}")
            
            # Clear CUDA cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=True,
                dtype=torch.float32, low_cpu_mem_usage=False
            ).to(device).eval()
            
            self._initialized = True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._initialized or not candidates: return candidates
        try:
            import torch
            # Use top_k argument if provided, otherwise use self.kwargs default, or default to 5
            limit_k = top_k if top_k is not None else self.kwargs.get("top_k", 5)
            
            pairs = []
            valid_indices = []
            for i, doc in enumerate(candidates):
                text = doc.get("text", "")
                if text:
                    pairs.append([query, text])
                    valid_indices.append(i)
            
            if not pairs: return candidates[:limit_k]
            
            # Batch processing to save memory
            batch_size = 4  # Reduced batch size to avoid OOM
            scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                with torch.no_grad():
                    inputs = self.tokenizer(batch_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs, return_dict=True)
                    if hasattr(outputs, 'logits'):
                        batch_scores = outputs.logits.view(-1,).float().cpu().tolist()
                        scores.extend(batch_scores)
                    else:
                        scores.extend([0.0] * len(batch_pairs))
                
                # Clear cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for idx, score in zip(valid_indices, scores):
                candidates[idx]["score"] = score
                
            candidates.sort(key=lambda x: x.get("score", -float("inf")), reverse=True)
            return candidates[:limit_k]
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            # If OOM, try to return un-reranked results but limited
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.warning("OOM during reranking, returning original top-k candidates")
            return candidates[:top_k] if top_k else candidates

class RetrieverAgent:
    """
    Retrieval Agent - Standardized Hot-plug Interface
    Role Definition: A standardized hot-plug interface that supports dynamic loading of different retrievers 
    to handle tasks with varying complexity. The module can be triggered by other agents during task execution, 
    integrating multiple retrieval modes through unified API: TextExactRetriever, SemanticRetriever, CrossModalRetriever.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize Retrieval Agent
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = get_logger("retriever_agent")
        
        # Initialize LLM interface for prompt-based decision making
        self.llm_interface = create_llm_interface({
            "provider": "local",  # Use local provider instead of OpenAI
            "model": "Qwen3-VL-8B-Instruct",  # Use local model
            "max_tokens": 100,
            "temperature": 0.0
        })
        
        # Available retrievers for dynamic loading
        self.retrievers = {
            "local_text": "LocalTextRetriever",
            "cross_modal": "CrossModalRetrieverAgent"
        }
        
        # Available rerankers for dynamic loading - use string references to avoid import order issues
        self.rerankers = {
            "text": "TextReranker",
            "multimodal": "MultimodalReranker"
        }
        
        # Default retriever configurations
        self.default_configs = {
            "local_text": {
                "model_path": Config.LOCAL_MODEL_PATHS.get("bge_m3", os.path.join(Config.PROJECT_ROOT, 'models', 'bge_m3')),
                "index_path": os.getenv("MMMRAG_TEXT_INDEX_PATH", os.path.join(Config.PROJECT_ROOT, 'data', 'ViDoSeek', 'bge_ingestion')),
                "top_k": 10
            },
            "cross_modal": {
                "model_path": Config.LOCAL_MODEL_PATHS.get("bge_vl_base", os.path.join(Config.PROJECT_ROOT, 'models', 'BGE-VL-Base')),
                "index_path": os.getenv("MMMRAG_CROSS_MODAL_INDEX_PATH", os.path.join(Config.PROJECT_ROOT, 'data', 'ViDoSeek', 'bgevlbase_ingestion')),
                "top_k": 10
            }
        }
        
        self.default_reranker_configs = {
            "text": {"model_path": Config.LOCAL_MODEL_PATHS.get("bge_reranker_v2_m3", os.path.join(Config.PROJECT_ROOT, 'models', 'bge_reranker_v2_m3')), "top_k": 5},
            "multimodal": {"model_path": Config.LOCAL_MODEL_PATHS.get("jina_reranker_m0", os.path.join(Config.PROJECT_ROOT, 'models', 'jina-reranker-m0')), "top_k": 5}
        }
        
        # Shared vector stores for hybrid retrieval to avoid reloading
        self.shared_vector_stores = {}
        
        # Model cache to avoid reloading
        self._model_cache = {}
        self._processor_cache = {}
      
        self.logger.info("Retriever Agent initialized with standardized hot-plug interface")
    
    def _score_question_unified(self, question: str, has_images: bool = False, known_information: str = "") -> float:
        """
        Unified question scoring on 0-1 scale, consistent with ScorePlanningAgent
        
        Args:
            question: Question or subquestion text
            has_images: Whether the question contains images
            known_information: Known information relevant to the question
            
        Returns:
            Score between 0-1 indicating retrieval complexity and necessity
        """
        # Create scoring prompt with aligned criteria
        scoring_prompt = f"""
        Estimate the probability that the model can answer the question correctly WITHOUT retrieval.
        
        Definition:
        - 1.0 → almost certain (>95% chance correct)
        - 0.8 → high confidence (likely correct, minor risk)
        - 0.5 → uncertain (roughly 50/50)
        - 0.2 → low confidence (likely incorrect)
        - 0.0 → almost impossible
        
        Question: {question}
        
        Return ONLY a number between 0.0 and 1.0.
        """
        
        try:
            # Call LLM for scoring
            response = self.llm_interface.generate(
                scoring_prompt,
                system_prompt="You are a specialized Score Planning Agent that provides precise scores.",
                temperature=0.1,
                max_tokens=50
            )
            
            # Parse scoring result
            score_text = response.get("text", "0.0").strip()
            # Extract number
            import re
            score_match = re.search(r"\d+\.\d+", score_text)
            if score_match:
                score = float(score_match.group())
                # Ensure score is between 0-1
                return max(0.0, min(1.0, score))
            else:
                # Parse failed, return default value
                return 0.0
        except Exception as e:
            self.logger.warning(f"Failed to score question: {e}")
            # Return default value in exception case
            return 0.0

    def process_query(self, query_data: Dict[str, Any], retrieval_top_k: int = 10, rerank_top_k: int = 5) -> Dict[str, Any]:
        """
        This method implements the standardized hot-plug interface that dynamically loads 
        appropriate retrievers based on complexity scores to handle tasks with different complexity.
        
        Input Format (JSON - Standard):
        {
            "question": "Q",
            "information": {
                "text": "I",
                "images": []
            }, 
            "score": {
                "modality_complexity": "modality complexity score",
                "multi_hop_complexity": "multi-hop complexity score"
            }
        }
        
        Input Format (JSON - Dataset):
        {
            "id": "",
            "question": "?",
            "information": {"text": "", "images": []},
            "metadata": {"source_dataset": "", "domain": ""},
            "ground_truth": {"answer": ""}
        }
        
        Output Format (JSON):
        {
            "question": "Q",
            "information": {"text": "Information text", "images": []},
            "score": {
                "modality_complexity": "modality complexity score", 
                "multi_hop_complexity": "multi-hop complexity score"
            },
            "answer": "A"  // Raw retrieval results from retrieval module
        }
        
        Args:
            query_data: Query data containing question, information, and score objects
            retrieval_top_k: Number of results to retrieve before reranking (default: 10)
            rerank_top_k: Number of results to return after reranking (default: 1)
            
        Returns:
            Results with answer field containing raw retrieval results
        """
        try:
            # Extract input components
            question = query_data.get("question", "")
            information = query_data.get("information", {"text": "", "images": []})
            score = query_data.get("score", {})
            
            # Check if input contains images
            has_images = False
            if isinstance(information, dict):
                images = information.get("images", [])
                has_images = len(images) > 0 and all(img is not None for img in images)
            
            # Handle dataset format where information is a dictionary
            if isinstance(information, dict):
                # Extract text from information dictionary for retrieval
                information_text = information.get("text", "")
            else:
                information_text = information

            # Use unified question scoring method (consistent with ScorePlanningAgent)
            question_score = self._score_question_unified(question, has_images, information_text)
            self.logger.info(f"Calculated question_score: {question_score} for question: {question[:50]}...")
            
            # Use LLM to decide retrieval strategy instead of hardcoded logic
            retrieval_strategy = self._decide_retrieval_strategy(question, information_text, has_images, question_score)
            
            if retrieval_strategy["need_retrieval"] is False:
                # Return direct answer result without retrieval
                response = {
                    "question": question,
                    "information": information,
                    "score": score,
                    "answer": {
                        "retrieval_method": "direct_answer",
                        "query": question,
                        "context": information_text,
                        "results": {
                            "results": [],
                            "total": 0,
                            "reason": retrieval_strategy["reason"]
                        },
                        "processing_summary": "Direct answer provided - no retrieval needed"
                    }
                }
                
                # Preserve dataset-specific fields if present
                if "id" in query_data:
                    response["id"] = query_data["id"]
                if "metadata" in query_data:
                    response["metadata"] = query_data["metadata"]
                if "ground_truth" in query_data:
                    response["ground_truth"] = query_data["ground_truth"]
                
                self.logger.info(f"Direct answer provided for question - no retrieval needed")
                return response
            
            # Execute retrieval with the selected strategy
            selected_retriever = retrieval_strategy["retrieval_type"]
            # For hybrid retrieval, we can optionally pass specific methods
            retrieval_results = self._execute_retrieval(question, information_text, selected_retriever, retrieval_top_k=retrieval_top_k, rerank_top_k=rerank_top_k)
            
            # Format response - preserve original information format
            response = {
                "question": question,
                "information": information,
                "score": {
                    **score
                },
                "answer": retrieval_results
            }
            
            # Preserve dataset-specific fields if present
            if "id" in query_data:
                response["id"] = query_data["id"]
            if "metadata" in query_data:
                response["metadata"] = query_data["metadata"]
            if "ground_truth" in query_data:
                response["ground_truth"] = query_data["ground_truth"]
            
            self.logger.info(f"Successfully processed query using {retrieval_results.get('retrieval_method', selected_retriever)} retriever")
            return response
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            # Return default response with error
            response = {
                "question": query_data.get("question", ""),
                "information": query_data.get("information", {"text": "", "images": []}),
                "score": query_data.get("score", {}),
                "answer": {"error": f"Retrieval failed: {str(e)}"}
            }
            
            # Preserve dataset-specific fields in error response
            if "id" in query_data:
                response["id"] = query_data["id"]
            if "metadata" in query_data:
                response["metadata"] = query_data["metadata"]
            if "ground_truth" in query_data:
                response["ground_truth"] = query_data["ground_truth"]
            
            return response
    
    
    def _execute_retrieval(self, question: str, information: str, retriever_type: str, hybrid_methods: Optional[List[str]] = None, retrieval_top_k: int = 10, rerank_top_k: int = 5) -> Dict[str, Any]:
        """
        Execute retrieval
        
        Args:
            question: Original question
            information: Additional information/context
            retriever_type: Type of retriever to use
            hybrid_methods: List of retrieval methods to use for hybrid retrieval (optional)
            retrieval_top_k: Number of results to retrieve per method before reranking (default: 10)
            rerank_top_k: Number of top results to keep per method after reranking (default: 1)
            
        Returns:
            Retrieval results
        """
        try:
            # Handle hybrid retrieval separately
            if retriever_type == "hybrid":
                self.logger.info("Executing hybrid retrieval strategy")
                
                # Get available retrieval methods from retrievers dictionary
                available_methods = list(self.retrievers.keys())
                
                # Use provided hybrid_methods if available, otherwise use a dynamic selection
                if hybrid_methods:
                    # Validate that all provided methods are available
                    valid_hybrid_methods = [method for method in hybrid_methods if method in available_methods]
                else:
                    # Dynamic selection based on query characteristics
                    # Default to all available methods for hybrid retrieval
                    valid_hybrid_methods = list(available_methods)
                
                # Ensure we have at least 2 methods for hybrid retrieval
                if len(valid_hybrid_methods) < 2:
                    # Fallback to single method retrieval if not enough methods available
                    fallback_method = available_methods[0] if available_methods else "local_text"
                    self.logger.warning(f"Not enough methods for hybrid retrieval, falling back to {fallback_method}")
                    retriever_type = fallback_method
                else:
                    # For hybrid retrieval, execute multiple retrieval methods
                    # Store results by modality to ensure we get top1 from each
                    results_by_modality = {"text": [], "image": []}
                    
                    # Pre-load and share vector stores for cross_modal retrieval
                    # This ensures cross_modal retriever can access loaded vectors
                    if "cross_modal" in valid_hybrid_methods:
                        # Check if vector store is already shared and valid
                        if "cross_modal" in self.shared_vector_stores:
                            existing_store = self.shared_vector_stores["cross_modal"]
                            if hasattr(existing_store, 'vectors') and len(existing_store.vectors) > 0:
                                self.logger.info(f"Using existing shared vector store for cross_modal: {len(existing_store.vectors)} vectors")
                            else:
                                self.logger.info("Existing shared vector store is empty, reloading")
                                self.shared_vector_stores.pop("cross_modal", None)
                        
                        if "cross_modal" not in self.shared_vector_stores:
                            self.logger.info("Pre-loading vector store for cross_modal retrieval in hybrid strategy")
                            try:
                                # Create a temporary CrossModalRetrieverAgent to load vectors
                                temp_retriever_config = self.default_configs.get("cross_modal", {})
                                temp_cross_modal = CrossModalRetrieverAgent(config=self.config, retriever_agent_ref=self, use_cached_model=True, **temp_retriever_config)
                                temp_cross_modal.initialize()
                                
                                # Store: loaded vector store for sharing
                                if hasattr(temp_cross_modal, 'vector_store') and temp_cross_modal.vector_store:
                                    self.shared_vector_stores["cross_modal"] = temp_cross_modal.vector_store
                                    self.logger.info(f"Shared vector store for cross_modal: {len(temp_cross_modal.vector_store)} vectors")
                                else:
                                    self.logger.warning("Failed to load vector store: no vectors found")
                            except Exception as e:
                                self.logger.error(f"Failed to pre-load vector store for cross_modal: {e}")
                    
                    # Execute each retrieval method
                    for method in valid_hybrid_methods:
                        self.logger.info(f"Executing {method} retrieval as part of hybrid strategy")
                        try:
                           
                            # Get retriever configuration
                            retriever_config = self.default_configs.get(method, {})
                            
                            # Initialize retriever
                            if method in self.retrievers:
                                # Get the class name from the dictionary
                                class_name = self.retrievers[method]
                                # Get the actual class object from the current module
                                import sys
                                current_module = sys.modules[__name__]
                                retriever_class = getattr(current_module, class_name)
                                
                                # Pass shared vector store for cross_modal retrieval
                                if method == "cross_modal" and "cross_modal" in self.shared_vector_stores:
                                    retriever_config["shared_vector_store"] = self.shared_vector_stores["cross_modal"]
                                    self.logger.info(f"Using shared vector store for {method} retrieval")
                                
                                # Pass retriever_agent_ref and use_cached_model for model caching
                                retriever = retriever_class(config=self.config, retriever_agent_ref=self, use_cached_model=True, **retriever_config)
                            else:
                                continue
                            
                            # Determine modality
                            modality = "cross" if method == "cross_modal" else "text"
                            
                            # Execute retrieval with retrieval_top_k results per method
                            # For hybrid retrieval, pass hybrid=True to ensure cross-modal uses 128 dimensions
                            if retriever_type == "hybrid":
                                results = retriever.retrieve(question, modality=modality, top_k=retrieval_top_k, hybrid=True)  # Get retrieval_top_k results per method
                            else:
                                results = retriever.retrieve(question, modality=modality, top_k=retrieval_top_k)  # Get retrieval_top_k results per method
                            
                            # Refine results - using original results since _refine_retrieval_results was removed
                            refined_results = results
                            method_results = refined_results.get("results", [])
                            
                            # Apply reranking to this method's results individually
                            disable_rerank = getattr(self.config, 'RERANK', {}).get('disable', False)
                            if method_results and not disable_rerank:
                                # Load and initialize reranker for this method
                                method_reranker = None
                                reranker_type = "text" if method == "local_text" else "multimodal"
                                reranker_config = self.default_reranker_configs.get(reranker_type, {})
                                
                                try:
                                    # Get reranker class
                                    reranker_class_name = self.rerankers.get(reranker_type)
                                    if reranker_class_name:
                                        import sys
                                        reranker_class = getattr(sys.modules[__name__], reranker_class_name)
                                        method_reranker = reranker_class(self.config, **reranker_config)
                                        self.logger.info(f"Loaded {reranker_type} reranker for {method} retrieval")
                                except Exception as e:
                                    self.logger.error(f"Failed to load reranker for {method} retrieval: {e}")
                                
                                # Apply reranking if reranker is initialized
                                if method_reranker and method_reranker._initialized:
                                    self.logger.info(f"Applying {reranker_type} reranking to {len(method_results)} results from {method} retrieval")
                                    try:
                                        # Rerank the method's results
                                        reranked_candidates = method_reranker.rerank(question, method_results)
                                        # Keep top results per method
                                        method_results = reranked_candidates[:rerank_top_k]
                                    except Exception as e:
                                        self.logger.error(f"Reranking failed for {method} retrieval: {e}")
                                        # Fall back to original results, keep only top rerank_top_k
                                        method_results = method_results[:rerank_top_k]
                                else:
                                    # No reranking, just keep top rerank_top_k results
                                    self.logger.info(f"No reranking applied to {method} results, returning top {rerank_top_k} results")
                                    method_results = method_results[:rerank_top_k]
                            elif method_results:
                                config_rerank_top_k = getattr(self.config, 'RERANK', {}).get('top_k', rerank_top_k)
                                method_results = method_results[:config_rerank_top_k]
                            
                            # Store results by modality (text or image)
                            # For local_text method, store as text modality
                            # For cross_modal method, store as image modality
                            target_modality = "text" if method == "local_text" else "image"
                            for result in method_results:
                                result["retrieval_source"] = method
                                result["modality"] = target_modality
                                results_by_modality[target_modality].append(result)
                        except Exception as e:
                            self.logger.warning(f"Hybrid retrieval method {method} failed: {e}")
                            continue
                    
                    # Get top results from each modality (not just top1)
                    text_results = results_by_modality["text"][:rerank_top_k]  # Get top k text results
                    image_results = results_by_modality["image"][:rerank_top_k]  # Get top k image results
                    
                    # Log results for debugging
                    self.logger.info(f"Hybrid retrieval results: text_results={len(text_results)}, image_results={len(image_results)}")
                    
                    # Build final results list with all results from both modalities
                    final_results = []
                    if text_results:
                        final_results.extend(text_results)
                    if image_results:
                        final_results.extend(image_results)
                    
                    # Format final hybrid results - preserve both retrieval method results separately
                    hybrid_results = {
                        "query": question,
                        "text_results": text_results,  # local_text top results
                        "image_results": image_results,  # cross_modal top results
                        "results": final_results,  # Contains all results from both modalities
                        "total": len(final_results),
                        "retriever": "hybrid",
                        "hybrid_methods": valid_hybrid_methods,
                        "refined": True,
                        "reranked": True,  # Each modality was reranked individually
                        "reranker": "per_modality"  # Indicates per-modality reranking
                    }
                    
                    return {
                        "retrieval_method": "hybrid",
                        "query": question,
                        "context": information,
                        "results": hybrid_results,
                        "processing_summary": f"Hybrid retrieval completed with {len(final_results)} results from {len(valid_hybrid_methods)} methods",
                        "modality_separated": True  # Indicate that modalities are kept separate
                    }
            
            # Regular retrieval logic
       
            # Get retriever configuration
            retriever_config = self.default_configs.get(retriever_type, {})
            
            # Dynamic retriever selection and initialization
            if retriever_type in self.retrievers:
                # Get the class name from the dictionary
                class_name = self.retrievers[retriever_type]
                # Get the actual class object from the current module
                import sys
                current_module = sys.modules[__name__]
                retriever_class = getattr(current_module, class_name)
                retriever = retriever_class(config=self.config, retriever_agent_ref=self, use_cached_model=True, **retriever_config)
            else:
                # Fallback to local text retriever if specified retriever is not available
                retriever_class = LocalTextRetriever
                retriever = retriever_class(config=self.config, retriever_agent_ref=self, use_cached_model=True, **self.default_configs.get("local_text", {}))
                retriever_type = "local_text"
            
            # Determine modality based on input
            # Always use "cross" for cross-modal retrieval, "text" for pure text
            modality = "cross" if retriever_type == "cross_modal" else "text"
            
            # Execute retrieval with the question and modality
            # Use the original question as the query to avoid prompt pollution
            results = retriever.retrieve(question, modality=modality, top_k=retrieval_top_k)  # Retrieve retrieval_top_k results for reranking
            
            # Refine results to ensure strong relevance to the subquestion
            # Using original results since _refine_retrieval_results was removed
            refined_results = results
            
            # Apply reranking if we have results
            reranked_results = refined_results
            disable_rerank = getattr(self.config, 'RERANK', {}).get('disable', False)
            if refined_results.get("results") and not disable_rerank:
                # Get reranker configuration based on the modality
                reranker_type = "text" if retriever_type == "local_text" else "multimodal"
                reranker_config = self.default_reranker_configs.get(reranker_type, {})
                
                # Load and initialize reranker dynamically
                reranker = None
                try:
                    # Get reranker class
                    reranker_class_name = self.rerankers.get(reranker_type)
                    if reranker_class_name:
                        # Dynamic import using sys.modules
                        import sys
                        reranker_class = getattr(sys.modules[__name__], reranker_class_name)
                        reranker = reranker_class(self.config, **reranker_config)
                        self.logger.info(f"Loaded reranker: {reranker_class_name} with config: {reranker_config}")
                except Exception as e:
                    self.logger.error(f"Failed to load reranker: {e}")
                    self.logger.warning("Reranker loading failed, will return original candidates")
                
                # Apply reranking if reranker is initialized
                if reranker and reranker._initialized:
                    self.logger.info(f"Applying {reranker_type} reranking to {len(refined_results['results'])} results")
                    try:
                        # Extract just the results list for reranking
                        candidates = refined_results["results"]
                        # Rerank the candidates
                        reranked_candidates = reranker.rerank(question, candidates)
                        # Keep top k results after reranking
                        top_k_results = reranked_candidates[:rerank_top_k]
                        reranked_flag = True
                        reranker_name = reranker_class_name
                    except Exception as e:
                        self.logger.error(f"Reranking failed: {e}")
                        # Fall back to original results
                        top_k_results = refined_results["results"][:rerank_top_k]
                        reranked_flag = False
                        reranker_name = None
                else:
                    # No reranking, just keep top k results
                    self.logger.info(f"No reranking applied, returning top {rerank_top_k} results")
                    # Keep top k results after reranking
                    top_k_results = refined_results["results"][:rerank_top_k]
                    reranked_flag = False
                    reranker_name = None
                
                # Update results with top k results
                reranked_results = refined_results.copy()
                reranked_results["results"] = top_k_results
                reranked_results["total"] = len(top_k_results)
                reranked_results["reranked"] = reranked_flag  # Set to True if reranking was applied
                reranked_results["reranker"] = reranker_name  # Name of reranker used
            else:
                 config_rerank_top_k = getattr(self.config, 'RERANK', {}).get('top_k', rerank_top_k)
                 if refined_results.get("results"):
                     refined_results["results"] = refined_results["results"][:config_rerank_top_k]
                     refined_results["total"] = len(refined_results["results"])
                 reranked_results = refined_results
            
            # Format results with unified structure
            return {
                "retrieval_method": retriever_type,
                "query": question,
                "context": information,
                "results": reranked_results,
                "processing_summary": f"Retrieved {reranked_results.get('total', 0)} results using {retriever_type} retriever"
            }
            
        except Exception as e:
            self.logger.error(f"Unified retrieval failed: {e}")
            return {"error": f"Unified retrieval failed: {str(e)}"}
    
    
    def _decide_retrieval_strategy(self, question: str, known_information: str, has_images: bool, question_score: float = 0.0) -> Dict[str, Any]:
        """
        Decide retrieval strategy based on confidence score and threshold
        
        Args:
            question: Question or subquestion text
            known_information: Known information relevant to the question
            has_images: Whether the question contains images
            question_score: Confidence score (0-1) - higher means more confident in answering without retrieval
            
        Returns:
            Dictionary containing retrieval decision
        """
        # Get threshold from injected config or use default 0.9
        threshold = 0.9
        if hasattr(self, '_threshold_config'):
            threshold = self._threshold_config.confidence_threshold
        
        # Decide if retrieval is needed based on score and threshold
        need_retrieval = question_score < threshold
        
        self.logger.info(f"Retrieval decision: score={question_score}, threshold={threshold}, need_retrieval={need_retrieval}")
        
        if not need_retrieval:
            return {
                "need_retrieval": False,
                "retrieval_type": "direct_answer",
                "reason": f"Score {question_score} >= threshold {threshold}, confident in answering without retrieval"
            }
        
        # Determine retrieval type based on question characteristics
        prompt = f"""
Analyze the question and determine the best retrieval method.

Question: {question}
Has Images: {has_images}
Known Info: {known_information[:200] if known_information else "None"}

Choose the retrieval type:
- "local_text": For factual questions where the answer is likely in a text document
- "cross_modal": For questions asking about visual details, scene description, or object counting in an image
- "hybrid": For complex tasks involving Charts, Tables, Infographics, or multiple modalities

Output ONLY the retrieval type (one word): local_text, cross_modal, or hybrid
"""
        
        # Call LLM to determine retrieval type
        try:
            response = self.llm_interface.call_llm(prompt)
            if isinstance(response, dict) and 'text' in response:
                response_text = response['text'].strip().lower()
                
                # Parse retrieval type
                retrieval_type = "local_text"  # default
                if "cross_modal" in response_text:
                    retrieval_type = "cross_modal"
                elif "hybrid" in response_text:
                    retrieval_type = "hybrid"
                
                return {
                    "need_retrieval": True,
                    "retrieval_type": retrieval_type,
                    "reason": f"Score {question_score} < threshold {threshold}, retrieval needed. Question requires: {retrieval_type}"
                }
        except Exception as e:
            self.logger.error(f"Failed to determine retrieval type: {e}")
        
        # Default fallback
        return {
            "need_retrieval": True,
            "retrieval_type": "local_text",
            "reason": f"Default: Score {question_score} < threshold {threshold}, using local_text retrieval"
        }
  
    def register_retriever(self, name: str, retriever_class: type):
    
        self.retrievers[name] = retriever_class
        self.logger.info(f"Registered new retriever: {name}")
    
    def get_available_retrievers(self) -> List[str]:

        return list(self.retrievers.keys())
    
    def get_retriever_info(self, retriever_type: str) -> Dict[str, Any]:
       
        if retriever_type not in self.retrievers:
            return {"error": f"Retriever type '{retriever_type}' not found"}
        
        return {
            "name": retriever_type,
            "class": self.retrievers[retriever_type].__name__,
            "default_config": self.default_configs.get(retriever_type, {}),
        }
    
