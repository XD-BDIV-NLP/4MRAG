#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer Fuser Agent -
Collects intermediate results from parallel or bridge paths, executes multi-dimensional 
evaluation (reliability, relevance) and generates the final unified answer.
"""

from typing import List, Dict, Any
from ..config.config import Config
from ..utils.logger import get_logger
from ..utils.llm_interface import create_llm_interface

class AnswerFuser:
    """
    Answer Fuser Agent
    
    Role Definition: Collects intermediate results from parallel or bridge paths, executes multi-dimensional 
evaluation (reliability, relevance) and generates the final unified answer. Follows standardized 
MMMRAG workflow with unified input/output format.
    """
    
    def __init__(self, llm_interface=None, config=None):
        # Process input configuration object
        if hasattr(config, 'AGENT_CONFIGS'):  # If it's a Config instance
            self.config_obj = config
            self.config = config.AGENT_LLM_CONFIGS["answer_fuser"]
        elif isinstance(config, dict):  # If it's a dictionary configuration
            self.config_obj = None
            self.config = config
        else:  # Default configuration
            self.config_obj = Config()
            self.config = self.config_obj.AGENT_LLM_CONFIGS["answer_fuser"]
        self.logger = get_logger("answer_fuser")
        
        # Use the provided llm_interface or create a default one
        if llm_interface:
            self.llm_interface = llm_interface
        else:
            self.llm_interface = create_llm_interface({
                "provider": self.config["provider"],
                "model": self.config["model"],
                "max_tokens": self.config["max_tokens"],
                "temperature": self.config["temperature"],
                "timeout": self.config["timeout"]
            })
        self.logger.info("AnswerFuser initialized with 2.5 standard interface")
    
    def fuse(self, original_query: str, subquery_results: List[Dict[str, Any]], routing_strategy: str, return_full_json: bool = True) -> Dict[str, Any]:
        """Compatibility layer method: Forward fuse calls to fuse_answers method
        
        Args:
            original_query: Original query
            subquery_results: List of sub-query results
            routing_strategy: Routing strategy
            return_full_json: Whether to return full JSON structure or only answer content
            
        Returns:
            Fused result
        """
        # Forward call to fuse_answers method
        return self.fuse_answers(
            subquery_results=subquery_results,
            original_question=original_query,
            routing_strategy=routing_strategy,
            return_full_json=return_full_json
        )
    
    def fuse_answers(self, subquery_results: List[Dict[str, Any]], 
                    original_question: str = None,
                    routing_strategy: str = None,
                    return_full_json: bool = True) -> Dict[str, Any]:
        
        try:
            # Detect input format and process accordingly
            if isinstance(subquery_results, list):
                # New 2.5 format - use unified prompt workflow
                return self._fuse_with_unified_workflow(subquery_results, original_question, return_full_json)
            elif isinstance(subquery_results, dict):
                # Legacy format - maintain backward compatibility
                return self._fuse_with_legacy_format(subquery_results, original_question, routing_strategy)
            else:
                raise ValueError("Invalid subquery_results format. Must be list (2.5) or dict (legacy)")
                
        except Exception as e:
            self.logger.error(f"Answer fusion failed: {e}")
            return self._format_error_response(subquery_results, str(e))
    
    def _fuse_with_unified_workflow(self, subquery_results: List[Dict[str, Any]], original_question: str = None, return_full_json: bool = True) -> Dict[str, Any]:
     
        # Validate input format
        if not subquery_results or len(subquery_results) == 0:
            raise ValueError("Subquery results cannot be empty")
        
        # Validate each sub-query result has required fields
        for i, result in enumerate(subquery_results):
            required_fields = ["question", "information", "score", "answer"]
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                raise ValueError(f"Sub-query {i+1} missing required fields: {missing_fields}")
            
            # Validate information field format
            information = result.get("information", {})
            if not isinstance(information, dict):
                result["information"] = {"text": "", "images": [], "retrieved_results": []}
                continue
            
            # Validate retrieved_results field format
            retrieved_results = information.get("retrieved_results", [])
            if retrieved_results and not isinstance(retrieved_results, list):
                result["information"]["retrieved_results"] = []
            else:
                # Filter retrieved results to remove vectors and invalid data
                filtered_results = []
                for retrieved in retrieved_results:
                    if isinstance(retrieved, dict):
                        # Check if retrieved result contains text content
                        text = retrieved.get("text", "")
                        if text and isinstance(text, str):
                            # Only filter out actual vector data, not text with square brackets
                            is_vector = False
                            if text.strip().startswith('[') and text.strip().endswith(']'):
                                # Check if content is a numeric vector
                                import re
                                if re.match(r'^\s*\[\s*([0-9]+(\.[0-9]+)?\s*,\s*)*[0-9]+(\.[0-9]+)?\s*\]\s*$', text):
                                    is_vector = True
                            if not is_vector:
                                filtered_results.append(retrieved)
                result["information"]["retrieved_results"] = filtered_results
        
        # Use the provided original_question if available, otherwise use first sub-query's question
        if not original_question:
            original_question = subquery_results[0].get("question", "")
            
        self.logger.info(f"Starting answer fusion for question: {original_question}")
        
        # Combine all subquery information instead of just using the first one
        original_information = {"text": "", "images": [], "retrieved_results": []}
        original_score = {}
        
        # Combine all subquery information
        for result in subquery_results:
            # Combine text, images, and retrieved results
            sub_info = result.get("information", {})
            if isinstance(sub_info, dict):
                # Combine text information
                if "text" in sub_info:
                    original_information["text"] += sub_info["text"] + " "
                # Combine image information
                if "images" in sub_info and isinstance(sub_info["images"], list):
                    original_information["images"].extend(sub_info["images"])
                # Combine retrieved results
                if "retrieved_results" in sub_info and isinstance(sub_info["retrieved_results"], list):
                    original_information["retrieved_results"].extend(sub_info["retrieved_results"])
            
            # Combine score information
            sub_score = result.get("score", {})
            for key, value in sub_score.items():
                if key not in original_score:
                    original_score[key] = value
        
        # Deduplicate and clean combined information
        original_information["text"] = original_information["text"].strip()
        # Deduplicate image information
        original_information["images"] = list(set(original_information["images"]))
        # Deduplicate retrieved results (based on text content)
        seen_texts = set()
        unique_retrieved = []
        for item in original_information["retrieved_results"]:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_retrieved.append(item)
        original_information["retrieved_results"] = unique_retrieved
        
        # Check if we have image data in the information field
        image_data = None
        # Support multiple image data formats
        if isinstance(original_information, dict):
            # Check images field
            images = original_information.get("images", [])
            if images and isinstance(images, list) and len(images) > 0:
                # Assume images field contains base64-encoded image data
                for img in images:
                    if img and isinstance(img, str) and img.startswith("data:image/"):
                        image_data = img
                        break
        elif isinstance(original_information, str) and original_information.startswith("data:image/"):
            image_data = original_information
        
        # Check if retrieved results contain direct answer
        direct_answer = self._extract_direct_answer(original_question, subquery_results)
        if direct_answer:
            self.logger.info("Found direct answer in retrieved results, returning directly")
            
            # If return_full_json is False, only return the answer content
            if not return_full_json:
                return direct_answer
            
            # Build fusion metadata
            fusion_metadata = {
                "sources": [f"Q{i+1}" for i in range(len(subquery_results))],
                "subquery_count": len(subquery_results),
                "direct_answer": True
            }
            
            # Return formatted response according to 2.5 standard
            return {
                "question": original_question,
                "information": original_information,
                "score": original_score,
                "answer": direct_answer,
                "fusion_metadata": fusion_metadata,
                "timestamp": self._get_timestamp(),
                "version": "2.5"
            }
        
        # Build unified fusion prompt
        unified_prompt = self._build_fusion_prompt(subquery_results, original_question)
        
        self.logger.info("Calling LLM for answer fusion")
        
        # Call LLM with unified prompt, passing image data if available
        if image_data:
            response = self.llm_interface.generate_multimodal(unified_prompt, image_data=image_data)
        else:
            response = self.llm_interface.call_llm(unified_prompt)
        
        # Generate final fused answer
        final_answer = self._process_fusion_response(response, original_question, subquery_results, return_full_json)
        
        self.logger.info(f"Answer fusion complete. Final answer length: {len(final_answer)}")
        
        # If return_full_json is False, only return the answer content
        if not return_full_json:
            # Try to parse the answer as JSON to extract the actual answer content
            try:
                import json
                parsed_answer = json.loads(final_answer)
                if isinstance(parsed_answer, dict) and "answer" in parsed_answer:
                    return parsed_answer["answer"]  #  :returncharacter
            except json.JSONDecodeError:
                # If it's not JSON, return it as is
                return final_answer  # :returncharacter
        
        # Build fusion metadata
        fusion_metadata = {
            "sources": [f"Q{i+1}" for i in range(len(subquery_results))],
            "subquery_count": len(subquery_results),
            "direct_answer": False
        }
        
        # Return formatted response according to 2.5 standard
        return {
            "question": original_question,
            "information": original_information,
            "score": original_score,
            "answer": final_answer,
            "fusion_metadata": fusion_metadata,
            "timestamp": self._get_timestamp(),
            "version": "2.5"
        }
    
    def _extract_direct_answer(self, original_question: str, subquery_results: List[Dict[str, Any]]) -> str:
        """
        Extract direct answer from retrieved results if available
        
        Args:
            original_question: Original question to be answered
            subquery_results: List of sub-query result dictionaries in 2.5 format
            
        Returns:
            Empty string to ensure all answers are passed to LLM
        """
        # Collect all retrieved knowledge from sub-query results
        all_retrieved_knowledge = []
        
        for i, result in enumerate(subquery_results, 1):
            # Extract retrieved results from each sub-query
            retrieved_results = result.get('information', {}).get('retrieved_results', [])
            for j, retrieved in enumerate(retrieved_results):
                if isinstance(retrieved, dict):
                    content = retrieved.get('text', '')
                    if content and isinstance(content, str):
                        # Skip vector data
                        is_vector = False
                        if content.strip().startswith('[') and content.strip().endswith(']'):
                            # Check if content is a numeric vector
                            import re
                            if re.match(r'^\s*\[\s*([0-9]+(\.[0-9]+)?\s*,\s*)*[0-9]+(\.[0-9]+)?\s*\]\s*$', content):
                                is_vector = True
                        if not is_vector:
                            all_retrieved_knowledge.append(content)
        
        # Return empty string to ensure all answers are passed to LLM
        # This removes hardcoded browser-specific logic and lets the LLM handle it
        return ""
    
    def _fuse_with_legacy_format(self, subquery_results: Dict[str, Dict[str, Any]], 
                               original_question: str, routing_strategy: str) -> Dict[str, Any]:
        """
        Fuse answers using legacy format (for backward compatibility)
        
        Args:
            subquery_results: Legacy dictionary format
            original_question: Original question
            routing_strategy: Routing strategy
            
        Returns:
            Fused answer in legacy format converted to 2.5 standard
        """
        if not original_question:
            raise ValueError("Original question is required for legacy format")
        if not routing_strategy:
            routing_strategy = "parallel_processing"  # Default fallback
        
        # Convert legacy format to 2.5 format
        converted_results = []
        for sub_id, result in subquery_results.items():
            converted_result = {
                "question": f"Sub-question {sub_id}",
                "information": {"text": "", "images": []},
                "score": {
                    "modality_complexity": "Legacy",
                    "multi_hop_complexity": "Legacy"
                },
                "answer": result.get("answer", "")
            }
            converted_results.append(converted_result)
        
        # Use unified workflow with converted results
        fused_result = self._fuse_with_unified_workflow(converted_results)
        
        # Convert back to legacy format but maintain 2.5 structure
        return {
            "question": original_question,
            "information": {"text": "", "images": []},
            "score": {
                "modality_complexity": "Legacy",
                "multi_hop_complexity": "Legacy"
            },
            "answer": fused_result.get("answer", ""),
            "fusion_metadata": {
                "sources": list(subquery_results.keys()),
                "legacy_routing_strategy": routing_strategy,
                "conversion_note": "Converted from legacy format"
            },
            "timestamp": self._get_timestamp(),
            "version": "2.5_legacy_compatible"
        }
    
    def _format_error_response(self, subquery_results: Any, error_message: str) -> Dict[str, Any]:
        """
        Format error response according to 2.5 standard
        
        Args:
            subquery_results: Original input data
            error_message: Error message
            
        Returns:
            Formatted error response
        """
        # Extract basic info from first sub-query if available
        if subquery_results and isinstance(subquery_results, list) and subquery_results:
            original_question = subquery_results[0].get("question", "")
            original_information = subquery_results[0].get("information", {"text": "", "images": []})
            original_score = subquery_results[0].get("score", {})
        else:
            original_question = ""
            original_information = {"text": "", "images": []}
            original_score = {}
        
        return {
            "question": original_question,
            "information": original_information,
            "score": original_score,
            "answer": f"Error occurred during answer fusion: {error_message}",
            "fusion_metadata": {
                "sources": [],
                "subquery_count": len(subquery_results) if isinstance(subquery_results, list) else 0,
                "error": error_message
            },
            "timestamp": self._get_timestamp(),
            "version": "2.5"
        }
    
    def _build_fusion_prompt(self, subquery_results: List[Dict[str, Any]], original_question: str) -> str:
        """
        Build unified fusion prompt template in English
        
        Args:
            subquery_results: List of sub-query result dictionaries
            original_question: Original question to be answered
            
        Returns:
            Formatted English prompt template
        """
        # Collect all retrieved knowledge from sub-query results
        all_retrieved_knowledge = []
        
        for i, result in enumerate(subquery_results, 1):
            # Extract retrieved results from each sub-query
            retrieved_results = result.get('information', {}).get('retrieved_results', [])
            # Use ALL retrieved results instead of just top 1
            for j, retrieved in enumerate(retrieved_results):
                # Ensure retrieved is a dictionary type
                if not isinstance(retrieved, dict):
                    continue
                
                # Get text content, filter out vector data
                content = retrieved.get('text', '')
                # Check if content is valid text, not vector or other non-text data
                if content and isinstance(content, str):
                    # Only filter out content that looks like vector data: starts with [, ends with ], and contains only numbers, commas, spaces, and newlines
                    is_vector = False
                    if content.strip().startswith('[') and content.strip().endswith(']'):
                        # Check if content contains only numbers, commas, spaces, and newlines
                        import re
                        if re.match(r'^\s*\[\s*([0-9]+(\.[0-9]+)?\s*,\s*)*[0-9]+(\.[0-9]+)?\s*\]\s*$', content):
                            is_vector = True
                    # If it's not a vector, add it to retrieved knowledge
                    if not is_vector:
                        retrieved_info = {
                            'source_id': f"Q{i}_R{j+1}",
                            'content': content,
                            'score': retrieved.get('score', 0.0),
                            'source': retrieved.get('source', ''),
                            'modality': retrieved.get('modality', 'text'),
                            'retrieval_source': retrieved.get('retrieval_source', 'unknown'),
                            'image_path': retrieved.get('image_path', ''),
                            'has_base64': bool(retrieved.get('base64_image', ''))
                        }
                        all_retrieved_knowledge.append(retrieved_info)
        
        # Add key information extracted from images (if any)
        image_clues = []
        for i, result in enumerate(subquery_results):
            # Check if there is image-related information
            if 'enhanced_query' in result:
                # If there's an enhanced query, it may contain information extracted from images
                enhanced_query = result.get('enhanced_query', '')
                if enhanced_query and enhanced_query != result.get('question', ''):
                    image_clues.append({
                        'source_id': f"Image_Clues_{i+1}",
                        'content': f"Enhanced query (may include image clues): {enhanced_query}",
                        'score': 0.9,  # Give higher score to image-extracted information
                        'source': 'image_extraction'
                    })
        
        # Format retrieved knowledge into a unified string
        retrieved_knowledge_text = ""
        
        # Combine all retrieved knowledge and image clues
        combined_knowledge = []
        if all_retrieved_knowledge:
            combined_knowledge.extend(all_retrieved_knowledge)
        if image_clues:
            combined_knowledge.extend(image_clues)
        
        if combined_knowledge:
            retrieved_knowledge_text += "Retrieved Knowledge:\n"
            retrieved_knowledge_text += "==================\n"
            for item in combined_knowledge:
                # Handle different score types (float/int/string)
                score = item['score']
                if isinstance(score, (int, float)):
                    score_str = f"{score:.2f}"
                else:
                    score_str = str(score)
                
                retrieved_knowledge_text += f"[{item['source_id']}] Score: {score_str}, Modality: {item.get('modality', 'text')}\n"
                retrieved_knowledge_text += f"Content: {item['content']}\n"
                if item['source']:
                    retrieved_knowledge_text += f"Source: {item['source']}\n"
                if item.get('image_path'):
                    retrieved_knowledge_text += f"Image Path: {item['image_path']}\n"
                if item.get('has_base64'):
                    retrieved_knowledge_text += f"Base64 Image: Available\n"
                retrieved_knowledge_text += "---\n"
        else:
            retrieved_knowledge_text = "No retrieved knowledge available."
        
        # Check if this is a direct answer mode (no retrieved knowledge)
        is_direct_answer_mode = not retrieved_knowledge_text or retrieved_knowledge_text == "No retrieved knowledge available."
        
        if is_direct_answer_mode:
            # Direct answer mode - use knowledge-based prompt
            unified_prompt = f"""You are a Knowledge-Based Answer Agent.
 
 **TASK**: Answer the question directly using your own knowledge.
 
 ---
 
 ## 🔴 Mandatory Answer Procedure (DO NOT SKIP)
 
 1. Identify question type (numeric / entity / yes-no)
 2. Generate the minimal correct answer
 3. Remove all extra words, formatting, and explanations
 4. Ensure final output is a single plain-text answer
 
 ---
 
 ## CRITICAL OUTPUT RULES
 
 - Output ONLY the final answer
 - No explanation, reasoning, or extra text
 - Plain text only (no markdown, no bullets, no line breaks)
 - No phrases like "the answer is", "I think", etc.
 - Match the question language exactly
 - Answer must be specific and non-empty
 
 ---
 
 ## ANSWER VALIDATION
 
 - Numeric → number + unit only
 - Yes/No → "Yes" or "No" only
 - Entity → exact name only
 - Must directly answer the question
 
 ---
 
 ## QUESTION
 {original_question}
 
 ---
 
 ## EXAMPLE
 Q: What is the capital of France?
 A: Paris
 
 ---
 
 ## FINAL INSTRUCTION
 Output ONLY the answer.
 """
        else:
            # Normal fusion mode - use knowledge fusion prompt
            unified_prompt = f"""You are an Answer Fusion Agent.
 
 **TASK**: Output the direct answer using RETRIEVED KNOWLEDGE as the primary source.
 
 ---
 
 ## 🔴 Mandatory Answer Procedure (DO NOT SKIP)
 
 1. Identify the exact entity asked in the question
 2. Extract relevant information ONLY about that entity from RETRIEVED KNOWLEDGE
 3. Use SUB-RESULTS only if needed
 4. Generate the minimal correct answer
 5. Remove all extra words and formatting
 
 ---
 
 ## CRITICAL OUTPUT RULES
 
 - Output ONLY the final answer
 - No explanation, reasoning, or meta language
 - Plain text only (no markdown, no formatting)
 - Match the question language exactly
 - Answer must be specific and non-empty
 - Do NOT include unrelated entities or general info
 
 ---
 
 ## ANSWER VALIDATION
 
 - Numeric → number + unit only
 - Yes/No → "Yes" or "No" only
 - Entity → exact name only
 - Must directly answer the question
 
 ---
 
 ## SOURCE PRIORITY
 
 1. RETRIEVED KNOWLEDGE (primary)
 2. SUB-RESULTS (supplementary)
 
 ---
 
 ## QUESTION
 {original_question}
 
 ## RETRIEVED KNOWLEDGE
 {retrieved_knowledge_text}
 
 ## SUB-RESULTS
 {chr(10).join([f"Q{i+1}: {result.get('answer', '')}" for i, result in enumerate(subquery_results)])}
 
 ---
 
 ## EXAMPLES
 
 Q: What percentage of the population uses public transportation?
 A: 28%
 
 Q: Which etiology caused the most liver transplants in 2020?
 A: Alcohol
 
 ---
 
 ## FINAL INSTRUCTION
 Output ONLY the answer.
 """
        return unified_prompt
    
    def _validate_answer(self, answer: str, question: str) -> str:
        """
        Validate and correct the answer to meet all the requirements
        
        Args:
            answer: The answer to validate
            question: The original question for reference
            
        Returns:
            Validated and corrected answer
        """
        if not answer:
            return answer
        
        # Basic formatting cleanup (final check after LLM processing)
        import re
        answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)  # Remove bold
        answer = re.sub(r'\*(.*?)\*', r'\1', answer)  # Remove italic
        answer = re.sub(r'^\s*[-*+]\s+', '', answer, flags=re.MULTILINE)  # Remove list markers
        answer = re.sub(r'\n+', ' ', answer)  # Remove newlines
        answer = re.sub(r'\s+', ' ', answer)  # Remove extra whitespace
        
        # Check for unhelpful responses
        unhelpful_responses = [
            "No specific information available",
            "No information available",
            "I don't know",
            "I'm not sure",
            "It's not specified",
            "Not mentioned",
            "No answer available",
            "No answer generated"
        ]
        
        # Convert to lowercase for comparison
        normalized_answer = answer.lower().strip()
        if normalized_answer in [resp.lower() for resp in unhelpful_responses]:
            return ""
        
        return answer.strip()
    

    def _process_fusion_response(self, response: Any, original_question: str, subquery_results: List[Dict[str, Any]], return_full_json: bool = True) -> str:
        try:
            # Process response, supporting both string and dictionary types
            answer_content = ""
            if isinstance(response, dict):
                # If it's a dictionary, check if it contains the required fields
                if 'text' in response:
                    answer_content = response['text'].strip()
                elif 'content' in response:
                    answer_content = response['content'].strip()
                elif all(key in response for key in ['question', 'information', 'answer']):
                    answer_content = response['answer']
            elif isinstance(response, str):
                # Direct string response, use as answer content
                answer_content = response.strip()
            
            # If no answer content from LLM, use sub-query answers directly
            if not answer_content:
                # Collect all sub-query answers
                sub_answers = [result.get('answer', '') for result in subquery_results if result.get('answer', '').strip()]
                if sub_answers:
                    # Join sub-query answers
                    answer_content = " ".join(sub_answers).strip()
                else:
                    # If no sub-query answers, try to generate a knowledge-based answer
                    self.logger.info(f"No response from LLM and no sub-query answers, generating knowledge-based answer")
                    # Generate a direct answer using LLM knowledge
                    knowledge_prompt = f"""Answer the question directly based on your knowledge. Be concise and accurate. Output only the answer.

Question: {original_question}
"""
                    try:
                        knowledge_response = self.llm_interface.generate(knowledge_prompt, temperature=0.1, max_tokens=100)
                        if isinstance(knowledge_response, dict) and 'text' in knowledge_response:
                            answer_content = knowledge_response['text'].strip()
                    except Exception as e:
                        self.logger.error(f"Error generating knowledge-based answer: {e}")
            
            # Validate and correct the answer
            validated_answer = self._validate_answer(answer_content, original_question)
            
            # Ensure answer is not empty
            if not validated_answer or validated_answer.strip() == "":
                self.logger.info(f"Generated answer is empty, trying knowledge-based fallback")
                # Try one more time with a simple knowledge-based prompt
                try:
                    simple_prompt = f"Answer: {original_question}"
                    fallback_response = self.llm_interface.generate(simple_prompt, temperature=0.1, max_tokens=50)
                    if isinstance(fallback_response, dict) and 'text' in fallback_response:
                        fallback_answer = fallback_response['text'].strip()
                        if fallback_answer and fallback_answer.lower() not in ["no specific information available", "i don't know", "no answer"]:
                            return fallback_answer
                except Exception as e:
                    self.logger.error(f"Error generating fallback answer: {e}")
                return ""
            
            return validated_answer
        except Exception as e:
            # When error occurs, log and return error
            self.logger.error(f"Error processing fusion response: {e}")
            # Try knowledge-based fallback on error
            try:
                fallback_prompt = f"Answer the question directly based on your knowledge: {original_question}"
                fallback_response = self.llm_interface.generate(fallback_prompt, temperature=0.1, max_tokens=50)
                if isinstance(fallback_response, dict) and 'text' in fallback_response:
                    return fallback_response['text'].strip()
            except:
                pass
            return ""
    
    def _get_timestamp(self) -> str:
        import datetime
        return datetime.datetime.now().isoformat()
