#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MMMRAG Framework Main Application Entry Point
"""

import os
import sys
import json
import argparse
import datetime
import re
import base64
from typing import Dict, List, Any, Optional

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mmmrag import (
    Config,
    ScorePlanningAgent,
    QuestionDecomposer,
    SubqueryReviewer,
    AnswerFuser,
    LLMInterface,
    TextProcessor,
    get_logger
)
from mmmrag.agents.retriever_agent import RetrieverAgent, LocalTextRetriever, CrossModalRetrieverAgent


class PlaceholderResolver:
    """Placeholder Resolver - Core Component"""
    
    @staticmethod
    def resolve(query: str, dependencies: List[str], answer_cache: Dict[str, str], logger) -> str:
        """Force resolve all placeholders, including natural language references"""
        resolved = query
        
        # Create ID mapping to handle different formats (sub_1 <-> Q1)
        id_mapping = {}
        for cache_id in answer_cache.keys():
            if cache_id.startswith('sub_'):
                num = cache_id.split('_')[1]
                q_id = f"Q{num}"
                id_mapping[cache_id] = q_id
                id_mapping[q_id] = cache_id
            elif cache_id.startswith('Q'):
                num = cache_id[1:]
                sub_id = f"sub_{num}"
                id_mapping[cache_id] = sub_id
                id_mapping[sub_id] = cache_id
        
        # 1. Replace standard placeholders [Answer from QX] or [Answer from sub_X]
        # First find all possible placeholder formats
        placeholder_pattern = re.compile(r'\[Answer from ([^\]]+)\]')
        placeholders = placeholder_pattern.findall(resolved)
        
        for placeholder_id in placeholders:
            placeholder = f"[Answer from {placeholder_id}]"
            
            # Check direct match
            if placeholder_id in answer_cache:
                answer = answer_cache[placeholder_id]
                resolved = resolved.replace(placeholder, f'"{answer}"')
                logger.info(f"Resolved placeholder {placeholder} -> '{answer}'")
            # Check mapping match
            elif placeholder_id in id_mapping:
                mapped_id = id_mapping[placeholder_id]
                if mapped_id in answer_cache:
                    answer = answer_cache[mapped_id]
                    resolved = resolved.replace(placeholder, f'"{answer}"')
                    logger.info(f"Resolved placeholder {placeholder} -> '{answer}' (via mapping {placeholder_id} -> {mapped_id})")
            else:
                logger.warning(f"Dependency {placeholder_id} not found in cache, cannot resolve placeholder")
        
        # 2. Replace natural language references (sub_1, sub_2, etc.)
        # Match references like "sub_1", "sub_2"
        for dep_id in dependencies:
            if dep_id in answer_cache:
                # Handle "country identified in sub_1" type text
                resolved = re.sub(
                    rf'country identified in {dep_id}',  
                    f'"{answer_cache[dep_id]}"',  
                    resolved,
                    flags=re.IGNORECASE
                )
                # Handle "from sub_1" type text
                resolved = re.sub(
                    rf'from {dep_id}',  
                    f'from "{answer_cache[dep_id]}"',  
                    resolved,
                    flags=re.IGNORECASE
                )
        
        # 3. Verify all placeholders are resolved
        remaining_placeholders = placeholder_pattern.findall(resolved)
        if remaining_placeholders:
            logger.warning(f"Unresolved placeholders remaining in query: {resolved}")
        
        return resolved


class MMMRAGApp:
    """
    MMMRAG Application Main Class
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MMMRAG Application
        
        Args:
            config_path: Path to configuration file
        """
        # Initialize configuration
        self.config = Config(config_path)
        
        # Initialize logger
        self.logger = get_logger(
            "mmrag_app",
            log_level=self.config.log_level,
            log_file=self.config.log_file,
            file_output=self.config.file_logging
        )
        self.logger.info(f"MMMRAG Application starting, version: {self.config.version}")
        
        # Initialize text processor
        self.text_processor = TextProcessor(language="en")
        
        # Initialize LLM interface
        self.llm_interface = LLMInterface(**self.config.llm_config)
        
        # Initialize retrievers
        self.retrievers = {}
        self._initialize_retrievers()

        # Initialize unified retriever agent FIRST, so the planner reuses the exact same
        # scoring instance (single source of truth; also the one the threshold eval script
        # monkey-patches). This removes the old "two independent scoring instances" hazard.
        self.unified_retriever = RetrieverAgent(config=self.config)

        # Initialize agents
        self.score_planning_agent = ScorePlanningAgent(
            llm_interface=self.llm_interface,
            config=self.config,
            unified_retriever=self.unified_retriever
        )
        self.question_decomposer = QuestionDecomposer(
            llm_interface=self.llm_interface,
            config=self.config
        )
        self.subquery_reviewer = SubqueryReviewer(
            llm_interface=self.llm_interface,
            config=self.config
        )
        self.answer_fuser = AnswerFuser(
            llm_interface=self.llm_interface,
            config=self.config
        )
        
        self.logger.info("MMMRAG Application initialization complete")
    
    def _initialize_retrievers(self):
        """
        Initialize retriever agents
        """
        for retriever_name, retriever_config in self.config.RETRIEVERS.items():
            try:
                # Get retriever type from config, default to retriever_name
                retriever_type = retriever_config.get("type", retriever_name)
                # Extract params, default to empty dict
                retriever_params = retriever_config.get("params", {})
                
                # Instantiate retriever directly
                retriever = None
                if retriever_type in ["local_text", "LocalTextRetriever"]:
                    retriever = LocalTextRetriever(config=self.config, **retriever_params)
                elif retriever_type in ["cross_modal", "CrossModalRetrieverAgent"]:
                    retriever = CrossModalRetrieverAgent(config=self.config, **retriever_params)
                else:
                    self.logger.warning(f"Unknown retriever type: {retriever_type}, trying to load as LocalTextRetriever")
                    retriever = LocalTextRetriever(config=self.config, **retriever_params)
                
                if retriever:
                    self.retrievers[retriever_name] = retriever
                    self.logger.info(f"Retriever agent {retriever_name} initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize retriever agent {retriever_name}: {e}")
    
    def _image_to_base64(self, image_path: str) -> Optional[str]:
        """
        Convert image to base64 encoded string with data URI prefix
        """
        try:
            valid_extensions = ['', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
            actual_path = None
            
            if os.path.splitext(image_path)[1]:
                if os.path.exists(image_path):
                    actual_path = image_path
            
            if not actual_path:
                base_path = os.path.splitext(image_path)[0]
                for ext in valid_extensions:
                    test_path = base_path + ext
                    if os.path.exists(test_path):
                        actual_path = test_path
                        break
            
            if not actual_path:
                self.logger.warning(f"Image file not found: {image_path}")
                return None
                
            with open(actual_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                mime_type = "image/jpeg"
                if actual_path.lower().endswith('.png'):
                    mime_type = "image/png"
                elif actual_path.lower().endswith('.gif'):
                    mime_type = "image/gif"
                elif actual_path.lower().endswith('.webp'):
                    mime_type = "image/webp"
                return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            self.logger.error(f"Image conversion failed: {e}")
            return None
    
    def _is_valid_image_path(self, path: str) -> bool:
        """Check if path is a valid image file"""
        if not os.path.exists(path):
            return False
        
        valid_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
        file_extension = os.path.splitext(path)[1].lower()
        return file_extension in valid_extensions
    
    def _parse_image_path(self, input_str: str) -> Optional[str]:
        """Parse input string to extract image path"""
        if input_str.lower().startswith("image:"):
            image_path = input_str[6:].strip()
            if self._is_valid_image_path(image_path):
                return image_path
        
        if self._is_valid_image_path(input_str):
            return input_str
        
        return None
    
    def process_query(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user query (supports new format)
        """
        # Handle input data format compatibility
        if "query" in input_data:
            # Old format compatibility
            query = input_data["query"]
            image_path = input_data.get("image_path")
            question = query
            information = {
                "text": "",
                "images": [image_path] if image_path else []
            }
        else:
            # New format processing
            question = input_data.get("question", "")
            information = input_data.get("information", {"text": "", "images": []})
        
        if not question or not question.strip():
            return {"error": "Query cannot be empty"}
        
        try:
            self.logger.info(f"Start processing query: {question}")
            
            # Process images
            image_base64_list = []
            image_paths = information.get("images", [])
            should_preload_images = len(image_paths) <= 5
            
            if image_paths and should_preload_images:
                self.logger.info(f"Processing input images: {image_paths}")
                for image_path in image_paths:
                    if image_path:
                        image_base64 = self._image_to_base64(image_path)
                        if image_base64:
                            image_base64_list.append(image_base64)
                        else:
                            self.logger.warning(f"Image {image_path} conversion failed, skipping")
            
            image_base64 = image_base64_list[0] if image_base64_list else None
            
            # Step 1: Scoring and Planning
            planning_input = {
                "question": question,
                "information": information
            }
            planning_result = self.score_planning_agent.process_query(planning_input)
            self.logger.info(f"Scoring and planning complete: {planning_result}")
            
            modality_complexity = planning_result.get("score", {}).get("modality_complexity", 1)
            multi_hop_complexity = planning_result.get("score", {}).get("multi_hop_complexity", 1)
            
            if multi_hop_complexity == 1:
                self.logger.info("Detected single-hop question")
                direct_result = self._execute_direct_query(question, image_base64, information)
                final_result = self._fuse_direct_answer(direct_result, question, planning_result, image_base64, information)
            else:
                # Step 2: Question Decomposition
                MAX_DECOMPOSITION_ATTEMPTS = 2
                decomposition_attempts = 0
                review_approved = False
                
                while decomposition_attempts < MAX_DECOMPOSITION_ATTEMPTS and not review_approved:
                    decomposition_attempts += 1
                    
                    decomposition_input = {
                        "question": question,
                        "information": information,
                        "score": planning_result.get("score", {}),
                        "decomposition_depth": decomposition_attempts - 1
                    }
                    
                    decomposition_result = self.question_decomposer.process_question(decomposition_input)
                    subqueries = decomposition_result.get('subqueries', [])
                    self.logger.info(f"Question decomposition complete (Attempt {decomposition_attempts}/{MAX_DECOMPOSITION_ATTEMPTS})")
                    
                    # Step 3: Sub-question Review
                    decomposition_result["decomposition_depth"] = decomposition_attempts - 1
                    review_result = self.subquery_reviewer.review(decomposition_result)
                    
                    if review_result.get("status") == "approved":
                        review_approved = True
                        self.logger.info("Sub-question review approved")
                    else:
                        if decomposition_attempts < MAX_DECOMPOSITION_ATTEMPTS:
                            reprocessing_suggestions = review_result.get("reprocessing_suggestions", [])
                            decomposition_input["reasoning"] = f"Review failed, suggestions: {reprocessing_suggestions}"
                            decomposition_input["decomposition_depth"] = decomposition_attempts
                            self.logger.info("Review failed, retrying decomposition")
                        else:
                            self.logger.info("Max attempts reached")
                            if 2 <= len(subqueries) <= 5:
                                self.logger.info("Sub-question count reasonable, accepting result")
                                review_approved = True
                                review_result = {
                                    "status": "approved",
                                    "question": question,
                                    "information": information,
                                    "score": planning_result.get("score", {}),
                                    "reviewed_subqueries": subqueries,
                                    "timestamp": self._get_timestamp(),
                                    "review_version": "2.5"
                                }
                            else:
                                self.logger.info("Sub-question count unreasonable, breaking")
                                break
                
                # Step 4: Execute Retrieval and Answering
                subquery_results = self._execute_subqueries(review_result, image_base64, information, question)
                
                # Step 5: Answer Fusion
                final_result = self._fuse_answers(subquery_results, question, planning_result, image_base64, information)
            
            # Format output
            image_paths = [image_base64] if image_base64 else []
            if information and "images" in information:
                image_paths = information["images"]
            
            final_answer = final_result.get("final_answer", "")
            if isinstance(final_answer, dict):
                final_answer = final_answer.get("answer", "")
            
            retrieval_results = {"results": []}
            if "subquery_performance" in final_result:
                all_search_results = []
                for subquery_perf in final_result.get("subquery_performance", []):
                    if "search_results" in subquery_perf:
                        all_search_results.extend(subquery_perf["search_results"])
                retrieval_results["results"] = all_search_results
            
            return {
                "question": question,
                "information": {
                    "text": information.get("text", "") if information else "",
                    "images": image_paths
                },
                "answer": final_answer,
                "retrieval_results": retrieval_results
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "error": f"Query processing failed: {str(e)}",
                "original_query": question
            }

    def _execute_subqueries(self, review_result: Dict[str, Any], image_base64: Optional[str] = None, information: Optional[Dict[str, Any]] = None, original_question: str = "") -> Dict[str, Any]:
        """
        Execute sub-questions retrieval and answering
        """
        # Get subqueries
        subqueries = review_result.get("subqueries", [])
        if not subqueries:
            subqueries = review_result.get("reviewed_subqueries", [])
        
        # Filter and validate
        valid_subqueries = []
        for i, subquery_info in enumerate(subqueries):
            subquery = subquery_info.get("query", "").strip() or subquery_info.get("question", "").strip()
            if subquery:
                subquery_info["query"] = subquery
                subquery_info["question"] = subquery
                subquery_info.setdefault("id", f"subq_{i+1}")
                subquery_info.setdefault("retriever", subquery_info.get("retriever_type", "text"))
                subquery_info.setdefault("top_k", 10)
                valid_subqueries.append(subquery_info)
        
        if not valid_subqueries and original_question:
            self.logger.info("No valid sub-questions, using original question")
            valid_subqueries = [{"id": "subq_1", "query": original_question, "question": original_question, "retriever": "text", "top_k": 10}]
        
        # Sort subqueries by ID to ensure correct execution order (Q1, Q2, ...)
        # QuestionDecomposer generates sequential IDs, so sorting by ID handles dependencies
        try:
            valid_subqueries.sort(key=lambda x: int(re.search(r'\d+', x.get("id", "0")).group()) if re.search(r'\d+', x.get("id", "0")) else 0)
        except Exception:
            pass # Keep original order if sorting fails
            
        execution_results = {
            "original_review": review_result,
            "subquery_results": []
        }
        
        for i, subquery_info in enumerate(valid_subqueries):
            try:
                subquery = subquery_info.get("query", "").strip()
                retriever_name = subquery_info.get("retriever", "text")
                
                self.logger.info(f"Executing sub-question {i+1}: {subquery}")
                
                enhanced_query = subquery
                # Image analysis logic can be added here if needed, or rely on RetrieverAgent
                
                # Resolve placeholders
                if i > 0:
                    previous_results = execution_results.get('subquery_results', [])
                    answer_map = {}
                    for result_item in previous_results:
                        result_id = result_item.get('id', '')
                        if result_id:
                            answer_map[result_id] = result_item.get('answer', '')
                            # Map ID formats
                            if result_id.startswith('sub_'):
                                answer_map[f"Q{result_id.split('_')[1]}"] = result_item.get('answer', '')
                            elif result_id.startswith('Q'):
                                answer_map[f"sub_{result_id[1:]}"] = result_item.get('answer', '')
                    
                    enhanced_query = PlaceholderResolver.resolve(
                        query=enhanced_query,
                        dependencies=subquery_info.get('depends_on', []),
                        answer_cache=answer_map,
                        logger=self.logger
                    )
                
                # Execute retrieval
                subquery_data = {
                    "question": enhanced_query,
                    "information": {
                        "text": information.get("text", "") if information else "",
                        "images": [image_base64] if image_base64 else []
                    }
                }
                
                retrieval_result = self.unified_retriever.process_query(subquery_data)
                
                # Extract results
                answer_data = retrieval_result.get("answer", {})
                if isinstance(answer_data, dict):
                    if answer_data.get("retrieval_method") == "hybrid":
                        results_list = answer_data.get("results", {}).get("results", [])
                    else:
                        results_field = answer_data.get("results", [])
                        results_list = results_field.get("results", []) if isinstance(results_field, dict) else results_field
                else:
                    results_list = []

                # Clean results (remove vectors)
                search_results = []
                for res in results_list:
                    if not res: continue
                    safe_res = res.copy()
                    if "metadata" in safe_res:
                        safe_res["metadata"] = {k: v for k, v in safe_res["metadata"].items() if k not in ["vector", "embedding"]}
                    search_results.append(safe_res)
                
                # Generate answer using AnswerFuser (Reuse logic!)
                fuser_input = [{
                    "question": enhanced_query,
                    "information": {
                        "text": information.get("text", "") if information else "",
                        "images": [image_base64] if image_base64 else [],
                        "retrieved_results": search_results
                    },
                    "score": {"modality_complexity": 1, "multi_hop_complexity": 1},
                    "answer": "" 
                }]
                
                answer = self.answer_fuser.fuse_answers(fuser_input, original_question=enhanced_query, return_full_json=False)
                
                subquery_result = {
                    "id": subquery_info.get("id", f"subq_{i+1}"),
                    "query": subquery,
                    "enhanced_query": enhanced_query,
                    "retriever": retriever_name,
                    "search_results": search_results,
                    "answer": answer,
                    "source": "retrieval"
                }
                
                execution_results["subquery_results"].append(subquery_result)
                
            except Exception as e:
                self.logger.error(f"Failed to execute sub-question {i+1}: {e}")
                execution_results["subquery_results"].append({
                    "id": subquery_info.get("id", f"subq_{i+1}"),
                    "query": subquery_info.get("query", ""),
                    "error": str(e),
                    "answer": "Error executing sub-question"
                })
        
        self.logger.info(f"Sub-question execution complete, total {len(execution_results['subquery_results'])} results")
        return execution_results
    
    def _execute_direct_query(self, query: str, image_base64: Optional[str] = None, information: Optional[Dict[str, Any]] = None, use_retrieval: bool = True) -> Dict[str, Any]:
        """Execute direct single-hop query.

        `use_retrieval=False` (added for the retrieval-gating experiment) skips the
        vector/cross-modal retriever entirely and answers from the LLM alone. This is
        what makes a "gate says no retrieval" decision actually suppress retrieval —
        otherwise `_execute_direct_query` always retrieves (line below), so the gating
        threshold could never change answer quality. Default True preserves the
        production behavior (the real pipeline always retrieves for single-hop).
        """
        self.logger.info(f"Executing direct query: {query} (use_retrieval={use_retrieval})")

        query_data = {
            "question": query,
            "information": {
                "text": information.get("text", "") if information else "",
                "images": [image_base64] if image_base64 else []
            }
        }

        if use_retrieval:
            retrieval_result = self.unified_retriever.process_query(query_data)

            # Extract and clean results
            answer_data = retrieval_result.get("answer", {})
            if isinstance(answer_data, dict):
                if answer_data.get("retrieval_method") == "hybrid":
                    results_list = answer_data.get("results", {}).get("results", [])
                else:
                    results_field = answer_data.get("results", [])
                    results_list = results_field.get("results", []) if isinstance(results_field, dict) else results_field
            else:
                results_list = []

            search_results = []
            for res in results_list:
                if not res: continue
                safe_res = res.copy()
                if "metadata" in safe_res:
                    safe_res["metadata"] = {k: v for k, v in safe_res["metadata"].items() if k not in ["vector", "embedding"]}
                search_results.append(safe_res)
        else:
            self.logger.info(f"[no_retrieval] Gated skip of retrieval for: {query[:60]}...")
            search_results = []
            
        # Generate answer using AnswerFuser
        fuser_input = [{
            "question": query,
            "information": {
                "text": information.get("text", "") if information else "",
                "images": [image_base64] if image_base64 else [],
                "retrieved_results": search_results
            },
            "score": {"modality_complexity": 1, "multi_hop_complexity": 1},
            "answer": ""
        }]
        
        answer = self.answer_fuser.fuse_answers(fuser_input, original_question=query, return_full_json=False)
        
        return {
            "query": query,
            "enhanced_query": query,
            "search_results": search_results,
            "answer": answer
        }
    
    def _fuse_direct_answer(self, direct_result: Dict[str, Any], query: str, planning_result: Dict[str, Any], image_base64: Optional[str] = None, information: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fuse direct query answer"""
        try:
            enhanced_information = information.copy() if information else {}
            enhanced_information["retrieved_results"] = direct_result.get("search_results", [])
            
            sub_results = [{
                "question": query,
                "information": enhanced_information,
                "score": planning_result.get("score", {"modality_complexity": 1, "multi_hop_complexity": 1}),
                "answer": direct_result.get("answer", "")
            }]
            
            # Call fuse_answers directly (same pattern as _execute_direct_query:581, which
            # works on the server) instead of the fuse() forwarder, whose signature mismatch
            # ("missing fuser_input") makes single-hop final_answer an error string.
            # NOTE: do NOT pass routing_strategy= -- the server's fuse_answers does not
            # accept it; mirror the working call at line 581 exactly.
            fusion_result = self.answer_fuser.fuse_answers(
                sub_results,
                original_question=query,
                return_full_json=True
            )
            
            final_result = {
                "original_query": query,
                "planning_info": planning_result,
                "final_answer": fusion_result.get("answer", ""),
                "subquery_performance": [direct_result],
                "sources": fusion_result.get("fusion_metadata", {}).get("sources", []),
                "timestamp": self._get_timestamp()
            }
        except Exception as e:
            final_result = {
                "original_query": query,
                "planning_info": planning_result,
                "final_answer": f"Error fusing answer: {str(e)}",
                "subquery_performance": [direct_result],
                "sources": [],
                "timestamp": self._get_timestamp()
            }
        
        return final_result
    
    def _fuse_answers(self, subquery_results: Dict[str, Any], query: str, planning_result: Dict[str, Any], image_base64: Optional[str] = None, information: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fuse sub-question answers"""
        routing_strategy = planning_result.get("routing_strategy", "simple_direct")
        all_results = subquery_results.get("subquery_results", [])
        
        if not all_results:
            return {
                "original_query": query,
                "planning_info": planning_result,
                "final_answer": "No sub-question results to fuse",
                "subquery_performance": [],
                "timestamp": self._get_timestamp()
            }
        
        sub_results = []
        for sr in all_results:
            if "error" not in sr:
                sub_info = information.copy() if information else {}
                if "search_results" in sr:
                    sub_info["retrieved_results"] = sr["search_results"]
                
                sub_results.append({
                    "question": sr["query"],
                    "information": sub_info,
                    "score": {"modality_complexity": 1, "multi_hop_complexity": 1},
                    "answer": sr["answer"]
                })
        
        if not sub_results:
            sub_results.append({
                "question": query,
                "information": information if information else {},
                "score": {"modality_complexity": 1, "multi_hop_complexity": 1},
                "answer": "Analysis completed, consolidating results."
            })
            
        try:
            # Call fuse_answers directly (same pattern as _execute_direct_query:581) instead
            # of the fuse() forwarder, which has a signature mismatch on the server.
            fusion_result = self.answer_fuser.fuse_answers(
                sub_results,
                original_question=query,
                return_full_json=True
            )
            
            fusion_answer = fusion_result.get("answer", "") if isinstance(fusion_result, dict) else fusion_result
            sources = fusion_result.get("fusion_metadata", {}).get("sources", []) if isinstance(fusion_result, dict) else []
            
            final_result = {
                "original_query": query,
                "planning_info": planning_result,
                "final_answer": fusion_answer,
                "subquery_performance": all_results,
                "sources": sources,
                "timestamp": self._get_timestamp()
            }
        except Exception as e:
            self.logger.error(f"Error fusing answers: {e}")
            final_result = {
                "original_query": query,
                "planning_info": planning_result,
                "final_answer": f"Error fusing answers: {str(e)}",
                "subquery_performance": all_results,
                "timestamp": self._get_timestamp()
            }
            
        return final_result

    def _get_timestamp(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def run_cli(self):
        """Run command line interface"""
        print("=== MMMRAG Interactive QA System ===")
        print("Type 'exit' or 'quit' to exit")
        print("Type 'help' for help")
        
        while True:
            try:
                question = input("\nPlease enter your question: ")
                if question.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                if not question.strip():
                    continue
                    
                information = {"text": "", "images": []}
                if input("Any additional info? (y/n): ").strip().lower() == 'y':
                    information["text"] = input("Enter text info (optional): ").strip()
                    print("Enter image path (type 'done' to finish):")
                    while True:
                        path = input("Image path: ").strip()
                        if path == 'done' or not path: break
                        parsed = self._parse_image_path(path)
                        if parsed:
                            information["images"].append(parsed)
                            print(f"Added: {parsed}")
                        else:
                            print("Invalid path")
                
                print("\nProcessing...")
                result = self.process_query({"question": question, "information": information})
                print(json.dumps(result, ensure_ascii=False, indent=2))
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="MMMRAG: Multimodal Multi-hop Retrieval-Augmented Generation")
    parser.add_argument("--query", help="Question to query")
    parser.add_argument("--text", help="Related text info")
    parser.add_argument("--image", action="append", help="Related image path")
    args = parser.parse_args()
    
    app = MMMRAGApp()
    
    if args.query:
        result = app.process_query({
            "question": args.query,
            "information": {
                "text": args.text or "",
                "images": args.image or []
            }
        })
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        app.run_cli()

if __name__ == "__main__":
    main()
