#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Question Decomposer Agent - 2.5 Standard Implementation
Processes high-complexity multi-hop questions using unified MMMRAG workflow
Supports two decomposition modes: parallel routing and bridge routing
"""

from typing import List, Dict, Any
from datetime import datetime
import json
from ..config.config import Config
from ..utils.logger import get_logger
from ..utils.llm_interface import create_llm_interface


class QuestionDecomposer:
    """
    Question Decomposer Agent - 2.5 Standard Interface
    
    Role Definition: Handles high-complexity multi-hop questions, selecting execution paths 
    based on routing strategies from the Score Planning Agent. Supports two decomposition modes:
    - Parallel Routing: Decomposes questions into independent sub-problem sequences for concurrent processing
    - Bridge Routing: Invokes Question Relation Analyzer to handle logical dependencies between sub-questions
    """

    def __init__(self, llm_interface=None, config=None):
        """Initialize Question Decomposer Agent with LLM interface and configuration
        
        Args:
            llm_interface: LLM interface instance
            config: Configuration object or dictionary
        """
        # Handle the configuration object passed in, consistent with other agents
        if hasattr(config, 'AGENT_LLM_CONFIGS'):  # If it's a Config instance
            self.config_obj = config
            self.config = config.AGENT_LLM_CONFIGS["question_decomposer"]
            # Get LLM configuration
            self.llm_config = config.AGENT_LLM_CONFIGS["question_decomposer"]
        elif isinstance(config, dict):  # If it's a dictionary configuration
            self.config_obj = None
            self.config = config
            # Use default LLM configuration
            self.llm_config = Config.AGENT_LLM_CONFIGS["question_decomposer"]
        else:  # Default configuration
            self.config_obj = Config()
            self.config = self.config_obj.AGENT_LLM_CONFIGS["question_decomposer"]
            self.llm_config = self.config_obj.AGENT_LLM_CONFIGS["question_decomposer"]
        
        # Initialize logging
        self.logger = get_logger("question_decomposer")
        
        # Initialize LLM interface
        if llm_interface:
            self.llm_interface = llm_interface
        else:
            # Initialize LLM interface in the same way as other agents
            self.llm_interface = create_llm_interface({
                "provider": self.llm_config["provider"],
                "model": self.llm_config["model"],
                "max_tokens": self.llm_config["max_tokens"],
                "temperature": self.llm_config["temperature"],
                "timeout": self.llm_config["timeout"]
            })
        
        # Initialize 2.5 standard compatible default values
        self.version = "2.5"
        self.supported_modes = ["parallel_processing", "bridge_processing"]
        
        self.logger.info(f"Question Decomposer initialized with version: {self.version}")
        self.logger.info(f"Supported modes: {', '.join(self.supported_modes)}")

    def decompose(self, query: str, planning_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compatibility layer method, forwarding calls to the process_question method
        
        Args:
            query: User query
            planning_result: Complete planning result containing scoring information and reasoning field
            
        Returns:
            Decomposition result
        """
        self.logger.info(f"Received decompose request with query: {query[:50]}...")
        
        # Adjust parameter format to match process_question method requirements
        if planning_result and isinstance(planning_result, dict):
            # Use actual scoring information from planning result
            input_data = {
                "question": planning_result.get("question", query),
                "information": planning_result.get("information", {"text": "", "images": []}),
                "score": planning_result.get("score", {}),
                "reasoning": planning_result.get("reasoning", "")
            }
        else:
            # If complete planning result is not provided, use default values (backward compatibility)
            routing_strategy = planning_result  # Support old call method, planning_result as routing_strategy
            if isinstance(routing_strategy, str):
                if routing_strategy == "simple_direct":
                    multi_hop_complexity = 1
                elif routing_strategy == "parallel_processing":
                    multi_hop_complexity = 4  # Use 4 to ensure 3 sub-questions are generated
                else:  # bridge_processing
                    multi_hop_complexity = 3
            else:
                multi_hop_complexity = 1
            
            input_data = {
                "question": query,
                "information": {"text": "", "images": []},
                "score": {
                    "modality_complexity": 1,
                    "multi_hop_complexity": multi_hop_complexity
                },
                "reasoning": ""
            }
        
        # Call process_question method for actual processing
        return self.process_question(input_data)

    def process_question(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process question using unified MMMRAG workflow (2.5 Standard)
        
        Input Format (JSON):
        {
            "question": "Q",
            "information": {"text": "Information text", "images": []}, 
            "score": {
                "modality_complexity": "Score1_modality_complexity",
                "multi_hop_complexity": "Score1_multi_hop_complexity"
            }
        }
        
        Output Format (JSON):
        {
            "decomposition_type": "parallel|bridge|direct",
            "question": "Q",
            "original_question": "Q",
            "information": {"text": "Information text", "images": []},
            "score": {"modality_complexity": 1, "multi_hop_complexity": 2},
            "subquestions": [
                {
                    "id": "sub_1",
                    "question": "Sub-question text",
                    "type": "Independent|Bridging",
                    "depends_on": [],
                    "retriever": "default"
                }
            ],
            "subqueries": [
                {
                    "id": "Q1",
                    "question": "Sub-question text",
                    "dependencies": [],
                    "retriever_type": "default"
                }
            ],
            "routing_strategy": "parallel_processing|bridge_processing|no_decomposition",
            "processing_type": "parallel|bridge|direct",
            "timestamp": "ISO timestamp",
            "decomposition_version": "2.5"
        }
        
        Args:
            input_data: Input data containing question, information, and score objects
            
        Returns:
            Dictionary containing question decomposition results according to 2.5 standard
        """
        try:
            self.logger.info(f"Processing question with input_data: {json.dumps(input_data, ensure_ascii=False, indent=2)}")
            
            # Validate input format
            required_fields = ["question", "information", "score"]
            for field in required_fields:
                if field not in input_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Extract input components
            question = input_data.get("question", "")
            information = input_data.get("information", {"text": "", "images": []})
            score = input_data.get("score", {})
            reasoning = input_data.get("reasoning", "")
            
            # Extract complexity scores
            modality_complexity = score.get("modality_complexity", 1)
            multi_hop_complexity = score.get("multi_hop_complexity", 1)
            
            # Determine routing strategy based on multi-hop complexity and question semantics
            routing_strategy = self._determine_routing_strategy(multi_hop_complexity, question, reasoning)
            self.logger.info(f"Determined routing strategy: {routing_strategy} based on multi_hop_complexity: {multi_hop_complexity}")
            
            # Handle cases that don't require decomposition
            if routing_strategy == "no_decomposition":
                self.logger.info(f"No decomposition needed for complexity: {multi_hop_complexity}")
                return self._format_direct_response(input_data)
            
            # Build decomposition prompt
            decomposition_prompt = self._build_decomposition_prompt(
                question=question,
                information=information,
                modality_complexity=modality_complexity,
                multi_hop_complexity=multi_hop_complexity,
                routing_strategy=routing_strategy,
                reasoning=reasoning
            )
            
            self.logger.info(f"Built decomposition prompt, calling LLM for {routing_strategy}")
            
            # Call LLM, using vLLM-compatible format
            self.logger.info(f"Calling LLM for decomposition with {len(decomposition_prompt)} characters prompt")
            self.logger.debug(f"Full prompt: {decomposition_prompt}")
            
            response = self.llm_interface.call_llm(decomposition_prompt)
            
            # Extract response text
            response_text = response["text"]
            self.logger.info(f"Received LLM response: {response_text[:1000]}...")
            self.logger.debug(f"Full LLM response: {response_text}")
            
            # Parse response
            self.logger.info("Starting decomposition response parsing")
            decomposition_result = self._parse_decomposition_response(
                response_text=response_text,
                routing_strategy=routing_strategy,
                question=question,
                information=information,
                score=score
            )
            self.logger.info(f"Parsing completed, result type: {type(decomposition_result).__name__}")
            
            # Post-process decomposition results to ensure they meet 2.5 standard
            processed_result = self._post_process_decomposition(
                decomposition_result=decomposition_result,
                routing_strategy=routing_strategy,
                input_data=input_data
            )
            
            # Add metadata
            processed_result.update({
                "routing_strategy": routing_strategy,
                "processing_type": self._determine_processing_type(routing_strategy),
                "timestamp": self._get_current_timestamp(),
                "decomposition_version": "2.5"
            })
            
            self.logger.info(f"Decomposition processing completed for {routing_strategy}")
            return processed_result
            
        except Exception as e:
            self.logger.error(f"Decomposition error: {str(e)}")
            # Raise error instead of fallback, to avoid silent failures
            raise e

    def _format_direct_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate direct response without decomposition
        
        Args:
            input_data: Input data
            
        Returns:
            Direct response result
        """
        question = input_data.get("question", "")
        information = input_data.get("information", {"text": "", "images": []})
        score = input_data.get("score", {})
        
        return {
            "decomposition_type": "direct",
            "question": question,
            "original_question": question,
            "information": information,
            "score": score,
            "subquestions": [
                {
                    "id": "sub_1",
                    "question": question,
                    "type": "Independent",
                    "depends_on": [],
                    "retriever": "default"
                }
            ],
            "subqueries": [
                {
                    "id": "Q1",
                    "question": question,
                    "dependencies": [],
                    "retriever_type": "default"
                }
            ],
            "routing_strategy": "no_decomposition",
            "processing_type": "direct",
            "timestamp": self._get_current_timestamp(),
            "decomposition_version": "2.5"
        }

    def _determine_routing_strategy(self, multi_hop_complexity: int, question: str = "", reasoning: str = "") -> str:
        """
        Determine routing strategy based on multi-hop complexity score and reasoning semantic
        
        Args:
            multi_hop_complexity: Multi-hop complexity score (1-5)
            question: Original question text
            reasoning: Reasoning from planning, used for semantic analysis
            
        Returns:
            Routing strategy: "parallel_processing", "bridge_processing", or "no_decomposition"
        """
        self.logger.info(f"Starting routing strategy determination, multi-hop complexity: {multi_hop_complexity}, question: '{question[:100]}...'")
        
        # Ensure multi_hop_complexity is integer type
        try:
            multi_hop_complexity = int(multi_hop_complexity)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid multi_hop_complexity value: {multi_hop_complexity}, defaulting to 3 (bridge-type)")
            multi_hop_complexity = 3

        if multi_hop_complexity == 1:
            self.logger.info(f"Multi-hop complexity is 1, no decomposition needed, direct retrieval")
            return "no_decomposition"  # Direct retrieval, no decomposition needed
        if multi_hop_complexity in [2, 4]:
            strategy = "parallel_processing"           
            self.logger.info(f"Selected routing strategy: {strategy} based on multi-hop complexity {multi_hop_complexity} (parallel-type)")
            return strategy
        elif multi_hop_complexity in [3, 5]:
            strategy = "bridge_processing"      # Complexity 3/5 are bridge-type multi-hop
            self.logger.info(f"Selected routing strategy: {strategy} based on multi-hop complexity {multi_hop_complexity} (bridge-type)")
            return strategy
        
        # Default fallback
        self.logger.info(f"No clear complexity identified, using bridge_processing as default")
        return "bridge_processing"

    def _determine_processing_type(self, routing_strategy: str) -> str:
        """Determine processing type based on routing strategy
        
        Args:
            routing_strategy: Routing strategy
            
        Returns:
            Processing type: "direct", "parallel", or "bridge"
        """
        if routing_strategy == "no_decomposition":
            return "direct"
        elif routing_strategy == "parallel_processing":
            return "parallel"
        elif routing_strategy == "bridge_processing":
            return "bridge"
        else:
            return "direct"

    def _build_decomposition_prompt(self, question: str, information: Dict[str, Any],
                                   modality_complexity: int, multi_hop_complexity: int,
                                   routing_strategy: str, reasoning: str = "",
                                   previous_results: List[Dict[str, Any]] = None) -> str:
        """Build decomposition prompt based on routing strategy and complexity"""

    
        structure = "Parallel" if routing_strategy == "parallel_processing" else "Bridged"


        prompt_template = """**ROLE**: You are a strict Question Decomposition Agent following the MMMRAG 2.5 Standard. Your output MUST be machine-validatable.

Your task is to decompose the following question into sub-questions based on its semantic structure.

## Question
{question}

## Core Decomposition Rules
- **Complexity 1**: Process directly with 1 sub-question (no decomposition needed)
- **Complexity 2 or 3**: Generate EXACTLY 2 sub-questions. This is a STRICT constraint.
- **Complexity 4 or 5**: Generate 3 or more sub-questions. This is a STRICT constraint.

- First identify semantic sub-questions, then map them to Q1, Q2, Q3

---

### Parallel Structure Decomposition Rules (Complexity 2, 4)
- All sub-questions must be independent
- depends_on MUST be []
- No placeholders allowed
- type MUST be "Independent"
- Each question must be self-contained and retrievable
- Use IDs: Q1, Q2, Q3...

**Anti-Fragmentation Rule:**
- If items are listed (A, B, C...), group them unless complexity ≥4
- Complexity 2 → avoid over-splitting

---

### Bridged Structure Decomposition Rules (Complexity 3, 5)
- Q1 must be "Independent"
- Q2+ must be "Bridged"
- Each dependent question MUST include [Answer from QX]
- depends_on must match placeholders exactly
- Dependencies must form a chain (Q1 → Q2 → Q3...), not a tree
- No natural language references (only placeholders)
- Each sub-question must be complete and retrievable
- Use IDs: Q1, Q2, Q3...

---

### Placeholder Rules
- Format: [Answer from QX]
- Required ONLY for Bridged questions (Q2+)
- Q1 MUST NOT contain placeholders
- Each dependency MUST have a matching placeholder

---

### General Format Rules
- Valid JSON (parsable by json.loads)
- No extra text
- All required fields must be present
- Ensure correct types

"""

        # Insert question
        prompt = prompt_template.format(question=question)

        prompt += "## Output Format Template\n\n"

        # Select example based on structure
        if structure == "Parallel":
            example = """### Parallel Structure Example (Complexity 2)

**Original Question**: "What is the difference between a tornado and a hurricane?"

**Correct Decomposition**:
{
  "structure": "Parallel",
  "sub_questions": [
    {
      "id": "Q1",
      "text": "What is a tornado?",
      "type": "Independent",
      "depends_on": []
    },
    {
      "id": "Q2",
      "text": "What is a hurricane?",
      "type": "Independent",
      "depends_on": []
    }
  ]
}
"""
            output_format = """## Parallel Structure Output Format
{
  "structure": "Parallel",
  "sub_questions": [
    {
      "id": "Q1",
      "text": "FIRST COMPLETE QUESTION WITHOUT PLACEHOLDERS",
      "type": "Independent",
      "depends_on": []
    },
    {
      "id": "Q2",
      "text": "SECOND COMPLETE QUESTION WITHOUT PLACEHOLDERS",
      "type": "Independent",
      "depends_on": []
    }
  ]
}
"""
        else:  # Bridged
            example = """### Bridged Structure Example (Complexity 3)

**Original Question**: "What is the capital of the country where the 2024 Olympics will be held?"

**Correct Decomposition**:
{
  "structure": "Bridged",
  "sub_questions": [
    {
      "id": "Q1",
      "text": "Which country will host the 2024 Olympics?",
      "type": "Independent",
      "depends_on": []
    },
    {
      "id": "Q2",
      "text": "Based on [Answer from Q1], what is the capital of that country?",
      "type": "Bridged",
      "depends_on": ["Q1"]
    }
  ]
}
"""
            output_format = """## Bridged Structure Output Format
{
  "structure": "Bridged",
  "sub_questions": [
    {
      "id": "Q1",
      "text": "FIRST COMPLETE QUESTION WITHOUT PLACEHOLDERS",
      "type": "Independent",
      "depends_on": []
    },
    {
      "id": "Q2",
      "text": "SECOND COMPLETE QUESTION WITH [Answer from Q1]",
      "type": "Bridged",
      "depends_on": ["Q1"]
    },
    {
      "id": "Q3",
      "text": "THIRD COMPLETE QUESTION WITH [Answer from Q2]",
      "type": "Bridged",
      "depends_on": ["Q2"]
    }
  ]
}
"""

        prompt += output_format
        prompt += "\n"
        prompt += example
        prompt += "\n"

        return prompt

    def _parse_decomposition_response(self, response_text: str, routing_strategy: str, question: str, information: Dict[str, Any], score: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse decomposition response from LLM
        
        Args:
            response_text: LLM response text
            routing_strategy: Routing strategy
            question: Original question
            information: Information dictionary
            score: Score dictionary
            
        Returns:
            Parsed decomposition result
        """
        try:
            self.logger.debug(f"Raw LLM response: {response_text[:500]}...")
            
            # Clean response text, remove potential quotes and newlines
            cleaned_response = response_text.strip()
            if cleaned_response.startswith('"') and cleaned_response.endswith('"'):
                cleaned_response = cleaned_response[1:-1]
            
            # Remove markdown code blocks if present
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()

            import re
            
            # Enhanced JSON repair function
            def repair_json_string(json_str):
                # 1. Fix escaped backslashes first
                # Replace \\" with \" (unescape escaped quotes)
                json_str = json_str.replace('\\\\"', '\\"')
                
                def fix_text_field(match):
                    full_match = match.group(0)
                    key = match.group(1) # "text":
                    value = match.group(2) # "..."

                    content = value[1:-1] # strip surrounding quotes
                    
                    # Escape unescaped quotes
                    # Look for " that is not preceded by \
                    content_fixed = re.sub(r'(?<!\\)"', r'\\"', content)
                    
                    return f'{key}"{content_fixed}"'

                # Apply fix to "text" and "question" fields
                json_str = re.sub(r'("text"\s*:\s*)("[^"]*")', fix_text_field, json_str)
                json_str = re.sub(r'("question"\s*:\s*)("[^"]*")', fix_text_field, json_str)
                
                return json_str

            # Attempt 1: Direct Parse
            try:
                parsed_result = json.loads(cleaned_response)
                return parsed_result
            except json.JSONDecodeError:
                pass

            # Attempt 2: Apply Repairs
            repaired_response = repair_json_string(cleaned_response)
            try:
                parsed_result = json.loads(repaired_response)
                self.logger.info("Successfully parsed JSON after repairs")
                return parsed_result
            except json.JSONDecodeError:
                pass
            
            # Attempt 3: Extract JSON object with Regex
            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, cleaned_response)
            if match:
                extracted_json = match.group(0)
                try:
                    parsed_result = json.loads(extracted_json)
                    return parsed_result
                except json.JSONDecodeError:
                    # Try repairing the extracted JSON
                    repaired_extracted = repair_json_string(extracted_json)
                    try:
                        parsed_result = json.loads(repaired_extracted)
                        return parsed_result
                    except json.JSONDecodeError:
                        pass

            raise ValueError(f"Failed to parse any JSON from response: {cleaned_response[:200]}...")
            
        except Exception as e:
            self.logger.error(f"Failed to parse decomposition response: {str(e)}")
            self.logger.debug(f"Full response text: {response_text}")
            return {"error": f"Parsing error: {str(e)}"}

    def _post_process_decomposition(self, decomposition_result: Dict[str, Any], 
                                  routing_strategy: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process decomposition result to ensure it meets 2.5 standard
        
        Args:
            decomposition_result: Original decomposition result
            routing_strategy: Routing strategy
            input_data: Original input data
            
        Returns:
            Processed result that meets 2.5 standard
        """
        question = input_data.get("question", "")
        information = input_data.get("information", {"text": "", "images": []})
        score = input_data.get("score", {})
        
        # 1. Base result object with necessary metadata
        base_result = {
                "decomposition_type": "parallel" if routing_strategy == "parallel_processing" else "bridge",
                "question": question,
                "original_question": question,
                "information": information,
                "score": score,
                "routing_strategy": routing_strategy,
                "processing_type": self._determine_processing_type(routing_strategy),
                "timestamp": self._get_current_timestamp(),
                "decomposition_version": "2.5"
        }
        
        if "error" in decomposition_result:
            self.logger.error(f"Decomposition result contains error: {decomposition_result['error']}")
            return self._generate_fallback_decomposition(input_data, error=decomposition_result['error'])
        
        # 2. Extract raw sub-questions
        raw_subquestions = []
        # Try various fields
        candidates = [
            decomposition_result.get("sub_questions"),
            decomposition_result.get("subquestions"),
            decomposition_result.get("subqueries"),
            decomposition_result.get("questions"),
            decomposition_result.get("decomposition_result")
        ]
        
        for candidate in candidates:
            if isinstance(candidate, list) and len(candidate) > 0:
                raw_subquestions = candidate
                break
        
        if not raw_subquestions:
            self.logger.warning(f"No valid sub-questions found in {routing_strategy} result, generating fallback")
            return self._generate_fallback_decomposition(input_data)
            
        self.logger.info(f"Found {len(raw_subquestions)} raw sub-questions")

        # 3. Standardize and Fix (The Core Logic)
        processed_subquestions = []
        import re

        for i, raw_sq in enumerate(raw_subquestions):
            # Extract basic info
            sq_text = raw_sq.get("text", "") or raw_sq.get("question", "")
            if not sq_text:
                continue
                
            sq_id = f"Q{i+1}" # Force standard ID
            
            # Default values
            sq_type = "Independent"
            sq_dependencies = []
            
            # Apply logic based on routing strategy
            if routing_strategy == "parallel_processing":
                # Parallel: Always Independent, No Dependencies, No Placeholders
                sq_type = "Independent"
                sq_dependencies = []
                
                # Clean placeholders
                if "[Answer from" in sq_text:
                    sq_text = re.sub(r'\s*\[Answer from Q\d+\]\s*', '', sq_text)
                    sq_text = re.sub(r'\s*based on \[Answer from Q\d+\]\s*', '', sq_text, flags=re.IGNORECASE)
                    
            elif routing_strategy == "bridge_processing":
                # Bridge: Q1 Independent, Q>1 Bridging
                if i == 0:
                    sq_type = "Independent"
                    sq_dependencies = []
                else:
                    sq_type = "Bridging"
                    # Default dependency to previous question
                    prev_id = f"Q{i}"
                    sq_dependencies = [prev_id]
                    
                    # Check/Inject Placeholder
                    expected_placeholder = f"[Answer from {prev_id}]"
                    if expected_placeholder not in sq_text:
                        # Try generic check
                        if not re.search(r'\[Answer from Q\d+\]', sq_text):
                            self.logger.info(f"Injecting placeholder {expected_placeholder} into {sq_id}")
                            if sq_text.strip().endswith("?"):
                                sq_text = sq_text.strip()[:-1] + f" based on {expected_placeholder}?"
                            else:
                                sq_text = f"{sq_text} based on {expected_placeholder}"
            
            processed_subquestions.append({
                "id": f"sub_{i+1}", # Internal ID format
                "question": sq_text,
                "type": sq_type,
                "depends_on": sq_dependencies,
                "information": information,
                "score": {
                    "modality_complexity": score.get("modality_complexity", 1),
                    "multi_hop_complexity": 1
                }
            })

        # 4. Populate output fields
        base_result["subquestions"] = processed_subquestions
        
        # Generate subqueries for the next agent
        base_result["subqueries"] = []
        for i, sq in enumerate(processed_subquestions):
            base_result["subqueries"].append({
                "id": f"Q{i+1}",
                "question": sq["question"],
                "dependencies": sq["depends_on"],
                "retriever_type": "default"
            })
            
        # Set structure field
        base_result["structure"] = "Parallel" if routing_strategy == "parallel_processing" else "Sequential"
        
        self.logger.info(f"Post-processing complete. Strategy: {routing_strategy}, Questions: {len(processed_subquestions)}")
        return base_result

    def _generate_fallback_decomposition(self, input_data: Dict[str, Any], error: str = "") -> Dict[str, Any]:
        """Generate fallback decomposition result by calling LLM with intelligent prompt
        
        Args:
            input_data: Original input data
            error: Error message
            
        Returns:
            Fallback decomposition result
        """
        question = input_data.get("question", "")
        information = input_data.get("information", {"text": "", "images": []})
        score = input_data.get("score", {})
        modality_complexity = score.get("modality_complexity", 1)
        multi_hop_complexity = score.get("multi_hop_complexity", 1)
        
        self.logger.warning(f"=== GENERATING FALLBACK DECOMPOSITION ===")
        self.logger.warning(f"Reason: {error}")
        self.logger.warning(f"Input question: {question[:200]}...")
        self.logger.warning(f"Multi-hop complexity: {multi_hop_complexity}")
        
        # Determine question type for better prompt engineering
        question_lower = question.lower()
        if "" in question or "" in question:
            question_type = "conditional"
        elif any(keyword in question_lower for keyword in ["", "", "", ""]):
            question_type = "origin"
        else:
            question_type = "general"
        
        # Build intelligent fallback prompt
        fallback_prompt = self._build_fallback_prompt(
            question=question,
            information=information,
            modality_complexity=modality_complexity,
            multi_hop_complexity=multi_hop_complexity,
            question_type=question_type
        )
        
        self.logger.info(f"Built fallback prompt with {len(fallback_prompt)} characters")
        self.logger.debug(f"Fallback prompt: {fallback_prompt}")
        
        # Call LLM with fallback prompt
        try:
            self.logger.info("Calling LLM for fallback decomposition")
            response = self.llm_interface.call_llm(fallback_prompt)
            response_text = response["text"]
            
            self.logger.info(f"Received fallback LLM response: {response_text[:1000]}...")
            self.logger.debug(f"Full fallback LLM response: {response_text}")
            
            # Parse the response
            parsed_result = self._parse_decomposition_response(
                response_text=response_text,
                routing_strategy="bridge_processing",  # Default to bridge for fallback
                question=question,
                information=information,
                score=score
            )
            
            # Post-process the result to ensure it meets 2.5 standard
            processed_result = self._post_process_decomposition(
                decomposition_result=parsed_result,
                routing_strategy="bridge_processing",
                input_data=input_data
            )
            
            # Add error information
            processed_result["error"] = error if error else "Fallback decomposition generated due to invalid LLM response"
            
            self.logger.info("Successfully generated fallback decomposition using LLM")
            return processed_result
            
        except Exception as e:
            self.logger.error(f"Failed to generate fallback decomposition using LLM: {str(e)}")
            # If LLM fallback also fails, use minimal hard-coded fallback
            self.logger.info("Using minimal hard-coded fallback as last resort")
            
            # Minimal fallback - simple sequential structure
            subquestions = [
                {
                    "id": "sub_1",
                    "question": f"What is the key aspect of {question} that needs to be understood first?",
                    "type": "Independent",
                    "depends_on": [],
                    "information": information,
                    "score": {
                        "modality_complexity": 1,
                        "multi_hop_complexity": 1
                    }
                },
                {
                    "id": "sub_2",
                    "question": f"Based on [Answer from Q1], what is the answer to the original question: {question}?",
                    "type": "Bridging",
                    "depends_on": ["Q1"],
                    "information": information,
                    "score": {
                        "modality_complexity": 1,
                        "multi_hop_complexity": 1
                    }
                }
            ]
            
            # Generate subqueries field
            subqueries = []
            for i, sq in enumerate(subquestions):
                subqueries.append({
                    "id": f"Q{i+1}",
                    "question": sq["question"],
                    "dependencies": sq.get("depends_on", []),
                    "retriever_type": "default"
                })
            
            return {
                "decomposition_type": "bridge",
                "question": question,
                "original_question": question,
                "information": information,
                "score": score,
                "subquestions": subquestions,
                "subqueries": subqueries,
                "routing_strategy": "bridge_processing",
                "processing_type": "bridge",
                "timestamp": self._get_current_timestamp(),
                "decomposition_version": "2.5",
                "error": f"LLM fallback failed: {str(e)}. Using minimal hard-coded fallback."
            }

    def _get_current_timestamp(self) -> str:
        """Get current timestamp
        
        Returns:
            ISO formatted timestamp
        """
        return datetime.now().isoformat()



    def _topological_sort(self, sub_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform topological sorting on dependent sub-questions
        
        Args:
            sub_questions: List of sub-questions
            
        Returns:
            Topologically sorted sub-questions list
        """
        # Build dependency graph
        graph = {}
        in_degree = {}
        
        for sq in sub_questions:
            sq_id = sq["id"]
            graph[sq_id] = []
            in_degree[sq_id] = 0
        
        for sq in sub_questions:
            sq_id = sq["id"]
            dependencies = sq.get("depends_on", [])
            for dep in dependencies:
                graph[dep].append(sq_id)
                in_degree[sq_id] += 1
        
        # Perform topological sorting
        result = []
        queue = [sq_id for sq_id in in_degree if in_degree[sq_id] == 0]
        
        while queue:
            current = queue.pop(0)
            # Find corresponding sub-question
            current_sq = next((sq for sq in sub_questions if sq["id"] == current), None)
            if current_sq:
                result.append(current_sq)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cyclic dependencies
        if len(result) != len(sub_questions):
            self.logger.warning("Cycle detected in sub-question dependencies, returning original order")
            return sub_questions
        
        return result
