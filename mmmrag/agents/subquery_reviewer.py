#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Subquery Reviewer Module - 2.5 Standard
Implements sub-question review agent for validating and optimizing decomposed sub-questions
"""

from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import json
from ..config.config import Config
from ..utils.llm_interface import create_llm_interface
from ..utils.logger import get_logger


class SubqueryReviewer:
   
    
    def __init__(self, llm_interface=None, config=None):
        """Initialize sub-question reviewer"""
        # Handle the passed configuration object
        if hasattr(config, 'AGENT_LLM_CONFIGS'):  # If it's a Config instance
            self.config_obj = config
            self.config = config.AGENT_LLM_CONFIGS.get("subquery_reviewer", {})
        elif isinstance(config, dict):  # If it's a dictionary configuration
            self.config_obj = None
            self.config = config
        else:  # Default configuration
            self.config_obj = Config()
            self.config = self.config_obj.AGENT_LLM_CONFIGS.get("subquery_reviewer", {})
        
        # Use the passed llm_interface or create a default one
        if llm_interface:
            self.llm_interface = llm_interface
        else:
            # Use create_llm_interface function to create interface, not hardcoding provider and model
            # This will use global configuration, including local model configuration
            self.llm_interface = create_llm_interface({
                "max_tokens": self.config.get("max_tokens", 2000),
                "temperature": self.config.get("temperature", 0.1),
                "timeout": self.config.get("timeout", 30)
            })
        self.logger = get_logger("subquery_reviewer")
        self.logger.info("SubqueryReviewer initialized with 2.5 standard interface")
    
    def review(self, decomposition_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility layer method: Forward old interface calls to the new review_with_unified_prompt method
        
        Args:
            decomposition_result: Decomposition result containing sub-question list and other information
            
        Returns:
            Review result
        """
        # Use the default MMMRAG system prompt
        system_prompt = "You are the MMMRAG system, a multimodal retrieval augmented generation system."
        
        # Call the new review_with_unified_prompt method
        return self.review_with_unified_prompt(
            system_prompt=system_prompt,
            agent_config=self.config,
            input_data=decomposition_result
        )
    
    def review_with_unified_prompt(self, system_prompt: str, agent_config: Dict[str, Any], 
                                  input_data: Dict[str, Any]) -> Dict[str, Any]:
     
        self.logger.info(f"Starting sub-question review, original question: '{input_data.get('question', '')}'")
        
        try:
            # Validate input format (only basic validation)
            if "question" not in input_data:
                raise ValueError("Missing required field: question")
            
            subqueries = input_data.get("subqueries", [])
            self.logger.info(f"Received sub-question list, total {len(subqueries)} sub-questions")
            
            if not subqueries:
                return self._format_rejection_response(
                    input_data, "Sub-question list is empty, cannot perform review",
                    failed_subquery_id="N/A",
                    failed_rule="subquery_count",
                    is_fatal=True,
                    suggestion="Please ensure at least 2 sub-questions are generated"
                )
            
            original_question = input_data["question"]
            
            # Record sub-question details
            for i, sq in enumerate(subqueries):
                sq_id = sq.get('id', f'Q{i+1}')
                dependencies = sq.get('dependencies', [])
                question_text = sq.get('question', sq.get('text', ''))
                self.logger.debug(f"Sub-question {sq_id}: '{question_text}', dependencies: {dependencies}")
            
            # 1. Basic validation: Only check for circular dependencies
            self.logger.info("Performing circular dependency check")
            if self._has_circular_dependencies(subqueries):
                return self._format_rejection_response(
                    input_data, "There are circular dependencies between sub-questions",
                    failed_subquery_id="N/A",
                    failed_rule="circular_dependency",
                    is_fatal=True,
                    suggestion="Please check the dependencies between sub-questions to ensure no circular dependencies exist"
                )
            self.logger.info("Circular dependency check passed")
            
            # 2. Comprehensive Review (Redundancy, Conflict, Structure)
            self.logger.info("Performing comprehensive review (Redundancy, Conflict, Structure)")
            review_result = self._perform_comprehensive_review(subqueries, original_question)
            
            # Update subqueries with refined version
            subqueries = review_result.get("refined_subqueries", subqueries)
            
            # Log improvements
            if review_result.get("modifications"):
                self.logger.info(f"Review modifications: {json.dumps(review_result.get('modifications'), ensure_ascii=False)}")
            
            # 3. Final Approval
            approval_response = self._format_approval_response(
                input_data, subqueries
            )
            
            approval_response["reviewed_subqueries"] = subqueries
            approval_response["review_reason"] = "Standard comprehensive review passed"
            approval_response["review_details"] = review_result.get("analysis", [])
            
            self.logger.info(f"Sub-question review passed: Total {len(subqueries)} sub-questions")
            return approval_response
            
        except Exception as e:
            self.logger.error(f"Sub-question review failed: {str(e)}", exc_info=True)
            return self._format_rejection_response(
                input_data, f"Error occurred during review: {str(e)}",
                failed_subquery_id="N/A",
                failed_rule="internal_error",
                is_fatal=True,
                suggestion="Please check system logs for detailed error information"
            )

    def _perform_comprehensive_review(self, subqueries: List[Dict[str, Any]], original_question: str) -> Dict[str, Any]:
        """
        Perform comprehensive review (Redundancy, Conflict, Structure) in a single LLM call
        
        Args:
            subqueries: List of sub-questions
            original_question: Original question
            
        Returns:
            Review result containing refined subqueries and analysis
        """
        try:
            prompt = self._build_comprehensive_review_prompt(subqueries, original_question)
            response = self.llm_interface.call_llm(prompt)
            
            try:
                result_text = response["text"]
                # Handle potential markdown code blocks
                if "```json" in result_text:
                    import re
                    match = re.search(r'```json\s*([\s\S]*?)\s*```', result_text)
                    if match:
                        result_text = match.group(1)
                elif "```" in result_text:
                    import re
                    match = re.search(r'```\s*([\s\S]*?)\s*```', result_text)
                    if match:
                        result_text = match.group(1)
                        
                result = json.loads(result_text)
                
                refined_subqueries = result.get("refined_subqueries", [])
                
                # Validation
                if not isinstance(refined_subqueries, list) or len(refined_subqueries) == 0:
                    self.logger.warning("Comprehensive review returned invalid subqueries, using original")
                    return {"refined_subqueries": subqueries, "modifications": [], "analysis": ["Review output invalid"]}
                
                # Check consistency
                if len(refined_subqueries) != len(subqueries):
                     self.logger.warning(f"Sub-question count changed from {len(subqueries)} to {len(refined_subqueries)}, ensuring alignment")
                
                # Enforce placeholder preservation for Bridge questions
                import re
                placeholder_pattern = re.compile(r'\[Answer from Q\d+\]')
                
                for i, sq in enumerate(refined_subqueries):
                    # Check if it should be a bridge question (based on original or type)
                    # We check if the original had dependencies or if the new one is marked as Bridge/Bridging
                    orig_sq = subqueries[i] if i < len(subqueries) else {}
                    orig_deps = orig_sq.get("dependencies", []) or orig_sq.get("depends_on", [])
                    
                    is_bridge = (
                        sq.get("type") in ["Bridge", "Bridging"] or 
                        len(sq.get("dependencies", [])) > 0 or
                        len(orig_deps) > 0
                    )
                    
                    if is_bridge and i > 0: # First question is usually independent
                        # Check for placeholder
                        current_q = sq.get("question", "")
                        has_placeholder = placeholder_pattern.search(current_q)
                        
                        if not has_placeholder:
                            self.logger.warning(f"Bridge question {sq.get('id')} missing placeholder, attempting to restore")
                            
                            # Identify dependency ID
                            deps = sq.get("dependencies", [])
                            if not deps and orig_deps:
                                deps = orig_deps
                                sq["dependencies"] = deps # Restore dependencies
                            
                            dep_id = deps[0] if deps else f"Q{i}" # Default to previous
                            # Ensure dep_id format Qx
                            if dep_id.startswith("sub_"):
                                dep_id = f"Q{dep_id.split('_')[1]}"
                            
                            placeholder = f"[Answer from {dep_id}]"
                            
                            # Strategy 1: If original had it, maybe revert to original if rewrite was too aggressive?
                            # But we want the "refined" quality. So let's inject.
                            if current_q.lower().startswith("based on"):
                                # "Based on ..., what is..." -> "Based on [Answer from Qx], what is..."
                                # Too risky to parse.
                                pass
                            
                            # Simple Injection: Prepend
                            sq["question"] = f"Based on {placeholder}, {current_q}"
                            self.logger.info(f"Restored placeholder: {sq['question']}")

                return {
                    "refined_subqueries": refined_subqueries,
                    "modifications": result.get("modifications", []),
                    "analysis": result.get("analysis", [])
                }
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Comprehensive review JSON parse failed: {e}")
                return {"refined_subqueries": subqueries, "modifications": [], "analysis": [f"JSON parse error: {e}"]}
                
        except Exception as e:
            self.logger.error(f"Comprehensive review failed: {e}")
            return {"refined_subqueries": subqueries, "modifications": [], "analysis": [f"Exception: {e}"]}

    def _build_comprehensive_review_prompt(self, subqueries: List[Dict[str, Any]], original_question: str) -> str:
        """Build unified prompt for all review tasks"""
        subqueries_text = "\n".join([
            f"ID: {sq.get('id', 'N/A')}, Question: {sq.get('question', 'N/A')}, Dependencies: {sq.get('dependencies', [])}, Type: {sq.get('type', 'N/A')}"
            for sq in subqueries
        ])
        
        return f"""
You are an expert Subquery Reviewer Agent. Your goal is to optimize a list of decomposed sub-questions for the question: "{original_question}".

**Input Sub-questions:**
{subqueries_text}

**Review Tasks (Perform all):**
1. **Redundancy**: Merge or refine duplicate/overlapping questions.
2. **Conflict**: Fix logical contradictions between questions.
3. **Structure**: Ensure dependencies (Bridging vs Parallel) are logical and valid.
4. **Clarity**: Ensure pronouns are resolved and questions are self-contained.
5. **PLACEHOLDER PRESERVATION**: CRITICAL! For 'Bridging' questions (Type: Bridging), you MUST PRESERVE the `[Answer from QX]` placeholders exactly. Do NOT replace them with natural language or descriptions.

**Requirements:**
- Return the OPTIMIZED list of sub-questions.
- Maintain the same number of sub-questions UNLESS redundancy is severe.
- Ensure 'dependencies' array is correct for Bridging types (Q2 depends on Q1, etc.).
- Ensure 'type' is correct (Independent vs Bridging).
- **NEVER REMOVE PLACEHOLDERS** like `[Answer from Q1]` from Bridging questions.

**Output Format (JSON Only):**
{{
    "refined_subqueries": [
        {{
            "id": "Q1",
            "question": "Optimized text...",
            "dependencies": [],
            "type": "Independent"
        }}
    ],
    "modifications": ["Fixed redundancy in Q2", "Resolved pronoun in Q3"],
    "analysis": ["Structure is valid", "No conflicts found"]
}}
"""
    
    def _format_approval_response(self, input_data: Dict[str, Any], 
                                reviewed_subqueries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format approval response
        
        Args:
            input_data: Original input data
            reviewed_subqueries: Reviewed sub-question list
            
        Returns:
            Approval response
        """
        response = {
            "status": "approved",
            "question": input_data.get("question", ""),
            "information": input_data.get("information", {"text": "", "images": []}),
            "score": input_data.get("score", {}),
            "reviewed_subqueries": reviewed_subqueries,
            "timestamp": self._get_timestamp(),
            "review_version": "2.5"
        }
        
        self.logger.info(f"Generated approval response, sub-question count: {len(reviewed_subqueries)}")
        return response
    
    def _format_rejection_response(self, input_data: Dict[str, Any], 
                                 feedback: str, 
                                 failed_subquery_id: str = "N/A",
                                 failed_rule: str = "unknown",
                                 is_fatal: bool = True,
                                 suggestion: str = "",
                                 suggestions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Format rejection response with detailed failure information
        
        Args:
            input_data: Original input data
            feedback: Issue description
            failed_subquery_id: ID of the failed sub-question
            failed_rule: Specific rule that failed
            is_fatal: Whether it's a fatal error
            suggestion: Specific fix suggestion
            suggestions: Reprocessing suggestions
            
        Returns:
            Rejection response with detailed failure information
        """
        # Build detailed failure information
        detailed_feedback = {
            "main_feedback": feedback,
            "failed_subquery_id": failed_subquery_id,
            "failed_rule": failed_rule,
            "is_fatal": is_fatal,
            "suggestion": suggestion
        }
        
        # Enhance log recording
        self.logger.error(f"Review failure details: {json.dumps(detailed_feedback, ensure_ascii=False)}")
        
        response = {
            "status": "rejected",
            "redirect_to": "Question Decomposer",
            "feedback": feedback,
            "detailed_feedback": detailed_feedback,
            "question": input_data.get("question", ""),
            "information": input_data.get("information", {"text": "", "images": []}),
            "score": input_data.get("score", {}),
            "timestamp": self._get_timestamp(),
            "review_version": "2.5"
        }
        
        if suggestions:
            response["reprocessing_suggestions"] = suggestions
        elif suggestion:
            response["reprocessing_suggestions"] = [suggestion]
            
        return response
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp
        
        Returns:
            ISO format timestamp
        """
        return datetime.now().isoformat()
    
    def _has_circular_dependencies(self, subqueries: List[Dict[str, Any]]) -> bool:
        """
        Check if there are circular dependencies between sub-questions
        
        Args:
            subqueries: List of sub-questions
            
        Returns:
            Whether there are circular dependencies
        """
        # Build dependency graph
        dependency_graph = {}
        for sq in subqueries:
            sq_id = sq.get('id', '')
            dependencies = sq.get('dependencies', [])
            dependency_graph[sq_id] = dependencies
        
        # Detect circular dependencies
        visited = set()
        recursion_stack = set()
        
        def dfs(node):
            visited.add(node)
            recursion_stack.add(node)
            
            for neighbor in dependency_graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    return True
            
            recursion_stack.remove(node)
            return False
        
        for node in dependency_graph:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
