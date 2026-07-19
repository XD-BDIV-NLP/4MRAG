#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Score Planning Agent 
As the core decision-making module of the system, responsible for receiving user query questions, performing multi-dimensional scoring, and dynamically formulating processing plans
"""

from typing import Dict, List, Tuple, Any
from ..config.config import Config
from ..utils.logger import get_logger
from ..utils.llm_interface import create_llm_interface

class ScorePlanningAgent:
    def __init__(self, llm_interface=None, config=None, unified_retriever=None):
        # ProcessConfig
        if hasattr(config, 'AGENT_LLM_CONFIGS'):  # If it's a Config instance
            self.config_obj = config
            self.config = config.AGENT_LLM_CONFIGS["score_planning"]
            # Get LLM configuration
            self.llm_config = config.AGENT_LLM_CONFIGS["score_planning"]
        elif isinstance(config, dict):  # If it's a dictionary configuration
            self.config_obj = None
            self.config = config
            # Use default LLM configuration
            self.llm_config = Config.AGENT_LLM_CONFIGS["score_planning"]
        else:  # Default configuration
            self.config_obj = Config()
            self.config = self.config_obj.AGENT_LLM_CONFIGS["score_planning"]
            self.llm_config = self.config_obj.AGENT_LLM_CONFIGS["score_planning"]
        
        # Initializelogger
        self.logger = get_logger("score_planning_agent")
        
        # llm_interfaceDefault
        if llm_interface:
            self.llm_interface = llm_interface
        else:
            self.llm_interface = create_llm_interface({
                "provider": self.llm_config["provider"],
                "model": self.llm_config["model"],
                "max_tokens": self.llm_config["max_tokens"],
                "temperature": self.llm_config["temperature"],
                "timeout": self.llm_config["timeout"]
            })
        self.logger.info("ScorePlanningAgent initialized with 2.5 standard interface")

        # Shared retriever instance used for unified confidence scoring. When injected
        # (app.py passes app.unified_retriever), the planner scores through the exact
        # same instance the RetrieverAgent uses -- single source of truth, and the one
        # the threshold eval script monkey-patches. Falls back to a self-built
        # RetrieverAgent() only when none is injected (standalone usage).
        self.unified_retriever = unified_retriever

    def _unified_score(self, question: str, has_images: bool, known_information: str) -> float:
        """Score a question via the shared retriever instance when available."""
        if self.unified_retriever is not None and hasattr(self.unified_retriever, "_score_question_unified"):
            return self.unified_retriever._score_question_unified(question, has_images, known_information)
        from .retriever_agent import RetrieverAgent
        return RetrieverAgent()._score_question_unified(question, has_images, known_information)
    
    def score_and_plan(self, query: str) -> Dict[str, Any]:
        """
        , convert toprocess_query
        
        Args:
            query: 
            
        Returns:
            Result
        """
        # Parameterprocess_query
        input_data = {
            "question": query,
            "information": {"text": "", "images": []}  # DefaultEmptyInfo
        }
        return self.process_query(input_data)
    
    def process_query(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query according to 2.5 standard format
        
        Input Format (JSON):
        {
          "question": "Q",
          "information": {
            "text": "I",
            "images": []
          }
        }
        
        Output Format (JSON):
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
        
        Args:
            input_data: Input data containing question and information
            
        Returns:
            Dictionary containing scoring results in standard format
        """
        try:
            question = input_data.get("question", "")
            information = input_data.get("information", {"text": "", "images": []})
            
            # Create enhanced prompt with detailed scoring rules
            # Use string concatenation instead of format to avoid conflict with LaTeX {R}
            unified_prompt = """
 You are a specialized Score Planning Agent .
 
 **System Role:** 
 As the core decision-making module of the system, responsible for receiving user query questions, performing multi-dimensional scoring, and dynamically formulating processing plans
 
 ---
 
 ## Few-shot Examples (Follow strictly)
 
 ### Example 1
 Question: What is the capital of France?
 
 Output:
 {
   "question": "What is the capital of France?",
   "information": "",
   "score": {
     "modality_complexity": 1,
     "multi_hop_complexity": 1,
     "retrieval_necessity": 0
   },
   "routing_decision": "Direct_Gen",
   "reasoning": "Common knowledge, single-hop question, no retrieval required."
 }
 
 ---
 
 ### Example 2
 Question: Who are the founders of Tesla and BYD respectively?
 
 Output:
 {
   "question": "Who are the founders of Tesla and BYD respectively?",
   "information": "",
   "score": {
     "modality_complexity": 1,
     "multi_hop_complexity": 2,
     "retrieval_necessity": 1
   },
   "routing_decision": "Multi_Agent",
   "reasoning": "Two independent sub-questions (Tesla, BYD), parallel structure, requires factual retrieval."
 }
 
 ---
 
 ### Example 3
 Question: How does deforestation increase global warming, and can afforestation mitigate this effect?
 
 Output:
 {
   "question": "How does deforestation increase global warming, and can afforestation mitigate this effect?",
   "information": "",
   "score": {
     "modality_complexity": 1,
     "multi_hop_complexity": 3,
     "retrieval_necessity": 1
   },
   "routing_decision": "Multi_Agent",
   "reasoning": "Two sub-questions with dependency: the second refers to 'this effect', forming a bridged structure."
 }
 
 ---
 
 **Input Processing:**
 - Original Question: """
            unified_prompt += question
            unified_prompt += """
 - Additional Context: """
            unified_prompt += str(information)
            unified_prompt += """

 **Processing Requirements:**
 - Determine modality complexity score (1-3) based on question type and information
 - Determine multi-hop complexity score (1-5) based on question structure
 - Determine retrieval necessity ($S_r$) based on whether the question requires external knowledge
 - Generate routing strategy according to unified MMMRAG workflow
 - Follow the established scoring standards and output format

 **Retrieval Necessity ($S_r$) - Binary Decision (0 or 1):**

 $S_r = 0$:
 - Universal, stable knowledge
 - Answerable without external retrieval

 $S_r = 1$:
 - Requires specific, recent (2024+) or precise information
 - Needs external verification, citation, or factual lookup

 **Modality Complexity Scoring (1-3 points):**
 - 1: Pure text
 - 2: Text + one modality
 - 3: Multiple modalities

 **Multi-hop Complexity Scoring (1-5 points):**

 1 point:
 - Single-step answer

 2–5 points:
 - Decompose into sub-questions
 - Independent → Parallel
 - Dependent (reference like "this effect") → Bridged

 Scoring:
 - 2 sub-questions → Parallel: 2 | Bridged: 3
 - ≥3 sub-questions → Parallel: 4 | Bridged: 5

 **Routing Decision Formula:**
 Based on the paper's unified routing function:
 $$\mathcal{R}(Q) = \begin{cases} 
 \text{Direct\_Gen}, & \text{if } S_r=0 \\ 
 \text{Multi\_Agent}, & \text{otherwise} 
 \end{cases}$$

 **If $S_r=0$ (no retrieval needed) → Direct Generation**
 **If $S_r=1$ (retrieval needed) → Multi-Agent pipeline**

 **Expected Output Format:**
 Return a JSON object with the following fields:
 - question: The original question
 - information: The original information
 - score: Object containing:
   - modality_complexity: Modality complexity score (1-3)
   - multi_hop_complexity: Multi-hop complexity score (1-5)
   - retrieval_necessity: Retrieval necessity score ($S_r$, 0 or 1)
 - routing_decision: Final routing decision based on the formula above ("Direct_Gen" or "Multi_Agent")
 - reasoning: Detailed explanation of the analysis

 Please analyze the input and return structured results.
            """
            
            # Call LLM with enhanced prompt
            response = self.llm_interface.call_llm(unified_prompt)
            
            # Parse response and return scores
            if isinstance(response, dict) and 'text' in response:
                response_text = response['text']
                try:
                    import json
                    import re
                    
                    # Step 1: Clean response text by removing common artifacts
                    cleaned_response = response_text.strip()
                    
                    # Remove markdown code blocks if present
                    if cleaned_response.startswith('```json'):
                        cleaned_response = cleaned_response[7:]
                    if cleaned_response.startswith('```'):
                        cleaned_response = cleaned_response[3:]
                    if cleaned_response.endswith('```'):
                        cleaned_response = cleaned_response[:-3]
                    
                    cleaned_response = cleaned_response.strip()
                    
                    # Step 2: Enhanced JSON cleaning for LLM-generated responses
                    def clean_llm_json(json_str):
                        """Clean LLM-generated JSON to fix common formatting issues"""
                        import re
                        
                        # 1. Remove comments (/* ... */ and // ...)
                        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
                        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
                        
                        # 2. Replace single quotes with double quotes, but preserve inside strings
                        json_str = re.sub(r"(?<!\\)'", '"', json_str)
                        
                        # 3. Remove trailing commas before closing brackets
                        json_str = re.sub(r',\s*(\}|\])', r'\1', json_str)
                        
                        # 4. Fix missing commas between key-value pairs (more conservative approach)
                        # Only add commas where clearly needed: after quoted values before next key
                        json_str = re.sub(r'"\s+}"\s*"', '"}', json_str)
                        
                        # 5. Remove any non-JSON content at the beginning or end
                        json_str = re.sub(r'^[^\{\[]*', '', json_str)
                        json_str = re.sub(r'[^\}\]]*$', '', json_str)
                        
                        # 6. Validate and fix JSON structure
                        # Try to parse and fix common issues
                        try:
                            json.loads(json_str)
                            # If parsing succeeds, return the cleaned string
                            return json_str
                        except json.JSONDecodeError:
                            # If parsing fails, try more aggressive cleanup
                            pass
                        
                        # More aggressive cleanup for malformed JSON
                        # Fix missing commas in arrays and objects
                        json_str = re.sub(r'"\s*}"\s*"', '"}', json_str)
                        json_str = re.sub(r'"\s*]"', ']', json_str)
                        
                        # Fix extra commas
                        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                        
                        # Fix missing commas in objects
                        json_str = re.sub(r'"\s*"\s*([a-zA-Z_])\s*:', r'"\1,":', json_str)
                        
                        # Remove duplicate commas
                        json_str = re.sub(r',\s*,', ',', json_str)
                        
                        # Final validation
                        try:
                            json.loads(json_str)
                            return json_str
                        except json.JSONDecodeError:
                            # If still fails, return original string to avoid breaking
                            self.logger.warning(f"JSON cleanup failed, returning original string")
                            return json_str
                    
                    # Step 3: Extract and clean JSON objects
                    def extract_json_objects(text):
                        stack = []
                        json_objects = []
                        start_idx = None
                        
                        for i, char in enumerate(text):
                            if char == '{':
                                if not stack:
                                    start_idx = i
                                stack.append(char)
                            elif char == '}':
                                if stack:
                                    stack.pop()
                                    if not stack and start_idx is not None:
                                        json_objects.append(text[start_idx:i+1])
                                        start_idx = None
                        
                        return json_objects
                    
                    # Try to extract JSON objects first
                    json_objects = extract_json_objects(cleaned_response)
                    
                    if json_objects:
                        # Try the largest JSON object first (most likely to be complete)
                        largest_json = max(json_objects, key=len)
                        
                        # Apply enhanced cleaning
                        cleaned_json = clean_llm_json(largest_json)
                        
                        try:
                            result = json.loads(cleaned_json)
                            
                            # Unified retrieval scoring: derive retrieval_necessity from a
                            # continuous confidence score (shared source with RetrieverAgent)
                            # instead of trusting the model's direct 0/1 guess. This keeps the
                            # planner and retriever consistent and makes the 0.9 threshold the
                            # single decision knob (score >= 0.9 -> no retrieval needed).
                            has_images = len(information.get("images", [])) > 0
                            confidence_score = self._unified_score(
                                question, has_images, information.get("text", "")
                            )
                            retrieval_necessity = 0 if confidence_score >= 0.9 else 1

                            # Apply routing formula: R(Q) = Direct_Gen if Sr=0, else Multi_Agent
                            routing_decision = "Direct_Gen" if retrieval_necessity == 0 else "Multi_Agent"

                            return {
                                "question": result.get("question", question),
                                "information": result.get("information", information),
                                "score": {
                                    "modality_complexity": result.get("score", {}).get("modality_complexity", 1),
                                    "multi_hop_complexity": result.get("score", {}).get("multi_hop_complexity", 1),
                                    "retrieval_necessity": retrieval_necessity
                                },
                                "routing_decision": routing_decision,
                                "reasoning": result.get("reasoning", "Analysis completed")
                            }
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Cleaned JSON parsing failed: {e}")
                            # Try with the raw JSON object as fallback
                            try:
                                result = json.loads(largest_json)
                                
                                # Extract retrieval_necessity (default to 1 if not present)
                                retrieval_necessity = result.get("score", {}).get("retrieval_necessity", 1)
                                
                                # Apply routing formula
                                routing_decision = "Direct_Gen" if retrieval_necessity == 0 else "Multi_Agent"
                                
                                return {
                                    "question": result.get("question", question),
                                    "information": result.get("information", information),
                                    "score": {
                                        "modality_complexity": result.get("score", {}).get("modality_complexity", 1),
                                        "multi_hop_complexity": result.get("score", {}).get("multi_hop_complexity", 1),
                                        "retrieval_necessity": retrieval_necessity
                                    },
                                    "routing_decision": routing_decision,
                                    "reasoning": result.get("reasoning", "Analysis completed")
                                }
                            except json.JSONDecodeError as e2:
                                self.logger.warning(f"Raw JSON parsing failed: {e2}")
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse LLM response: {e}")
                    # Try a second parsing attempt with more aggressive cleanup
                    try:
                        # Apply enhanced cleaning to the entire response
                        compact_response = clean_llm_json(cleaned_response)
                        
                        # Remove newlines and extra spaces for more compact format
                        compact_response = re.sub(r'\s+', ' ', compact_response)
                        
                        # Additional fixes for tricky cases
                        compact_response = compact_response.replace('\n', '').replace('\r', '')
                        compact_response = re.sub(r'\}\s*\{', '}, {', compact_response)  # Fix missing commas between objects
                        compact_response = re.sub(r'\]\s*\[', '], [', compact_response)  # Fix missing commas between arrays
                        
                        result = json.loads(compact_response)
                        # Extract retrieval_necessity (default to 1 if not present)
                        retrieval_necessity = result.get("score", {}).get("retrieval_necessity", 1)
                        
                        # Apply routing formula
                        routing_decision = "Direct_Gen" if retrieval_necessity == 0 else "Multi_Agent"
                        
                        return {
                            "question": result.get("question", question),
                            "information": result.get("information", information),
                            "score": {
                                "modality_complexity": result.get("score", {}).get("modality_complexity", 1),
                                "multi_hop_complexity": result.get("score", {}).get("multi_hop_complexity", 1),
                                "retrieval_necessity": retrieval_necessity
                            },
                            "routing_decision": routing_decision,
                            "reasoning": result.get("reasoning", "Analysis completed")
                        }
                    except Exception as e2:
                        self.logger.warning(f"Second parsing attempt failed: {e2}")
                        self.logger.debug(f"Response text causing error: {compact_response[:500]}...")
                except Exception as e:
                    self.logger.warning(f"Unexpected error parsing response: {e}")
             
            # Last resort: Basic modality detection with enhanced fallback logic
            modality_type, modality_score = self._determine_modality_complexity(question, information.get("images", []))
            has_images = len(information.get("images", [])) > 0
            
            # Enhanced fallback: Use the unified parallel question detection method
            hop_score = self._detect_parallel_question(question)
            hop_type = self._get_hop_type_from_score(hop_score)
            
            # Calculate retrieval_necessity using unified scoring method
            score = self._unified_score(question, has_images, information.get("text", ""))
            # Lower score means higher retrieval necessity
            retrieval_necessity = 1 if score < 0.9 else 0
            
            # Apply routing formula for fallback
            routing_decision = "Direct_Gen" if retrieval_necessity == 0 else "Multi_Agent"
            
            return {
                "question": question,
                "information": information,
                "score": {
                    "modality_complexity": modality_score,
                    "multi_hop_complexity": hop_score,
                    "retrieval_necessity": retrieval_necessity
                },
                "routing_decision": routing_decision,
                "reasoning": f"Final fallback analysis: {modality_type}, {hop_type}"
            }
        except Exception as e:
            # Fallback to default analysis due to unexpected errors
            question = input_data.get("question", "")
            question_lower = question.lower()
            information = input_data.get("information", {"text": "", "images": []})
            modality_type, modality_score = self._determine_modality_complexity(question, information.get("images", []))
            has_images = len(information.get("images", [])) > 0
            
            # Enhanced fallback: Use the unified parallel question detection method
            hop_score = self._detect_parallel_question(question)
            hop_type = self._get_hop_type_from_score(hop_score)
            
            # Calculate retrieval_necessity using unified scoring method
            score = self._unified_score(question, has_images, information.get("text", ""))
            # Lower score means higher retrieval necessity
            retrieval_necessity = 1 if score < 0.9 else 0
            
            # Apply routing formula for fallback
            routing_decision = "Direct_Gen" if retrieval_necessity == 0 else "Multi_Agent"
            
            return {
                "question": question,
                "information": information,
                "score": {
                    "modality_complexity": modality_score,
                    "multi_hop_complexity": hop_score,
                    "retrieval_necessity": retrieval_necessity
                },
                "routing_decision": routing_decision,
                "reasoning": f"Enhanced fallback analysis due to error: {str(e)}. {modality_type}, {hop_type}"
            }
    
    def analyze_question(self, question: str, images: List[str] = None) -> Dict[str, Any]:
        """
        Analyze user questions to determine modality complexity and multi-hop complexity
        
        Args:
            question: User question text
            images: List of relevant image paths
            
        Returns:
            Dictionary containing scoring and routing decisions
        """
        # Use the new process_query method for consistency
        input_data = {
            "question": question,
            "images": images or []
        }
        return self.process_query(input_data)
    
    def _determine_modality_complexity(self, question: str, images: List[str] = None) -> Tuple[str, int]:
        """
        Determine the modality complexity of the question
        
        Returns:
            (modality type, complexity score)
        """
        if images is None or len(images) == 0:
            return "text", 1
        elif len(images) == 1:
            return "single_multimodal", 2
        else:
            # Multiple images or mixed modalities
            return "mixed_multimodal", 3
    
    def _get_hop_type_from_score(self, hop_score: int) -> str:
        """
        Get multi-hop type based on score
        
        Args:
            hop_score: Multi-hop score
            
        Returns:
            Multi-hop type description
        """
        hop_types = {
            1: "single_hop",
            2: "parallel_2hop",
            3: "bridged_2hop",
            4: "parallel_3plus",
            5: "bridged_3plus"
        }
        return hop_types.get(hop_score, "Unknown type")
    

    def _determine_routing_strategy(self, score: Tuple[int, int]) -> str:
        """
        Determine routing strategy based on scores
        
        Args:
            score: (modality complexity score, multi-hop complexity score)
            
        Returns:
            Routing strategy name
        """
        modality_score, hop_score = score
        
        if hop_score == 1:
            return "simple_direct"
        elif hop_score in [2, 4]:
            return "parallel_processing"
        elif hop_score in [3, 5]:
            return "bridge_processing"

        return "bridge_processing"    
    def _detect_parallel_question(self, question: str) -> int:
        """
        Detect parallel questions based on delimiters and structure
        
        Args:
            question: User question text
            
        Returns:
            Multi-hop complexity score (1, 2, or 4)
        """
        question_lower = question.lower()
        
        # Check for comparison questions with 3+ items
        if "compare" in question_lower or "comparison" in question_lower:
            # Count the number of items being compared
            # Look for common delimiters: and, or, comma-separated lists
            items = []
            # Simple split approach for common comparison patterns
            if "of" in question_lower:
                # "Compare the X of A, B, and C"
                compare_part = question_lower.split("compare the")[-1].strip()
                if " of " in compare_part:
                    # Extract items after "of"
                    items_part = compare_part.split(" of ")[-1]
                    # Split by commas and "and"
                    items = [item.strip() for item in items_part.replace(" and ", ",").split(",") if item.strip()]
            
            # If we found 3 or more items, it's a parallel_3plus question
            if len(items) >= 3:
                return 4
            # If we found 2 items, it's a parallel_2hop question
            elif len(items) == 2:
                return 2
        
        # Check for parallel questions with Chinese delimiters (、)
        elif "、" in question:
            # Split by Chinese delimiter and count items
            items = [item.strip() for item in question.split("、") if item.strip()]
            if len(items) >= 3:
                # More than 2 items with Chinese delimiter, likely parallel_3plus
                return 4
            elif len(items) == 2:
                # Exactly 2 items with Chinese delimiter, likely parallel_2hop
                return 2
        
        # Check for parallel questions with English commas
        elif "," in question:
            # Split by English comma and count items
            items = [item.strip() for item in question.split(",") if item.strip()]
            if len(items) >= 3:
                # More than 2 items with English comma, likely parallel_3plus
                return 4
            elif len(items) == 2:
                # Exactly 2 items with English comma, likely parallel_2hop
                return 2
        
        # Default to single-hop if no parallel structure detected
        return 1
    
    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()