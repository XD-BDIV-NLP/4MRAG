#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import requests
from typing import Dict, List, Any, Optional, Union
from ..config.api_keys_config import APIKeysConfig as APIKeys
from ..config.config import Config


class LLMInterface:
    def __init__(self, **kwargs):
        self.provider = kwargs.get("provider", "openai")
        self.model_name = kwargs.get("model_name", kwargs.get("model", "gpt-3.5-turbo"))
        self.api_key = kwargs.get("api_key", None)
        self.base_url = kwargs.get("base_url", None)
        self.timeout = kwargs.get("timeout", 60)
        self.max_retries = kwargs.get("max_retries", 3)
        self.kwargs = kwargs

        from ..utils.logger import get_logger
        self.logger = get_logger(f"llm_interface.{self.__class__.__name__}")

        self.client = self._initialize_client()

    def _initialize_client(self):
        if self.provider.lower() == "openai":
            return self._initialize_openai_client()
        elif self.provider.lower() == "anthropic":
            return self._initialize_anthropic_client()
        elif self.provider.lower() == "local":
            return self._initialize_local_client()
        else:
            print(f"Warning: Unsupported provider '{self.provider}', using mock client")
            return self._initialize_mock_client()

    def _initialize_openai_client(self):
        try:
            import openai

            if self.api_key:
                openai.api_key = self.api_key
            if self.base_url:
                openai.api_base = self.base_url

            self.logger.info(f"OpenAI client initialized, model: {self.model_name}")
            return openai
        except ImportError:
            self.logger.warning("OpenAI library not found, using mock client")
            return self._initialize_mock_client()
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            return self._initialize_mock_client()

    def _initialize_anthropic_client(self):
        try:
            import anthropic

            client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=self.timeout
            )

            self.logger.info(f"Anthropic client initialized, model: {self.model_name}")
            return client
        except ImportError:
            self.logger.warning("Anthropic library not found, using mock client")
            return self._initialize_mock_client()
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic client: {e}")
            return self._initialize_mock_client()

    def _initialize_local_client(self):
        try:
            self.logger.info(f"Local model client initializing, model: {self.model_name}")

            api_config = APIKeys.get_api_config()
            local_config = api_config.get("local", {}).get("qwen3vl", {})

            local_client = {
                "model_name": self.model_name,
                "api_url": self.base_url or local_config.get("api_url", os.getenv("LOCAL_QWEN3VL_API_URL", "http://localhost:8888")),
                "api_key": self.api_key or local_config.get("api_key", ""),
                "proxy": local_config.get("proxy", None)
            }

            self.logger.info(f"Local model client configured: API URL={local_client['api_url']}")
            return local_client
        except Exception as e:
            self.logger.error(f"Failed to initialize local model client: {e}")
            raise e

    def _initialize_mock_client(self):
        self.logger.info(f"Mock client initialized, model: {self.model_name}")
        return MockClient(model_name=self.model_name)

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        top_p = kwargs.get("top_p", 1.0)

        kwargs_copy = kwargs.copy()
        kwargs_copy.pop("temperature", None)
        kwargs_copy.pop("max_tokens", None)
        kwargs_copy.pop("top_p", None)

        for attempt in range(self.max_retries):
            try:
                if self.provider.lower() == "openai":
                    return self._generate_openai(prompt, temperature, max_tokens, top_p, **kwargs_copy)
                elif self.provider.lower() == "anthropic":
                    return self._generate_anthropic(prompt, temperature, max_tokens, top_p, **kwargs_copy)
                elif self.provider.lower() == "local":
                    return self._generate_local(prompt, temperature, max_tokens, top_p, **kwargs_copy)
                else:
                    return self._generate_mock(prompt, temperature, max_tokens, top_p, **kwargs_copy)
            except Exception as e:
                self.logger.error(f"Generation failed (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.warning("Max retries reached, returning empty result")
                    return {"text": "", "error": str(e)}

    def _generate_openai(self, prompt: str, temperature: float, max_tokens: int,
                         top_p: float, **kwargs) -> Dict[str, Any]:
        try:
            messages = [
                {"role": "system", "content": kwargs.get("system_prompt", "You are a helpful assistant")},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                timeout=self.timeout,
                **kwargs
            )

            text = response.choices[0].message.content.strip()
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

            return {
                "text": text,
                "token_usage": token_usage,
                "model": self.model_name
            }
        except Exception as e:
            raise Exception(f"OpenAI generation error: {str(e)}")

    def _generate_anthropic(self, prompt: str, temperature: float, max_tokens: int,
                           top_p: float, **kwargs) -> Dict[str, Any]:
        try:
            system_prompt = kwargs.get("system_prompt", "You are a helpful assistant")

            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs
            )

            text = response.content[0].text.strip()
            token_usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }

            return {
                "text": text,
                "token_usage": token_usage,
                "model": self.model_name
            }
        except Exception as e:
            raise Exception(f"Anthropic generation error: {str(e)}")

    def _generate_local(self, prompt: str, temperature: float, max_tokens: int,
                       top_p: float, **kwargs) -> Dict[str, Any]:
        try:
            self.logger.info(f"Generating text with local model: {self.model_name}")

            client_config = self.client
            if isinstance(client_config, dict):
                api_url = client_config.get("api_url", os.getenv("LOCAL_QWEN3VL_API_URL", "http://localhost:8888"))
                api_key = client_config.get("api_key", "")
                proxy = client_config.get("proxy", None)
            else:
                api_url = os.getenv("LOCAL_QWEN3VL_API_URL", "http://localhost:8888")
                api_key = ""
                proxy = None

            proxies = None
            if proxy:
                proxies = {
                    "http": proxy,
                    "https": proxy
                }

            headers = {
                "Content-Type": "application/json"
            }

            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            normalized_url = api_url.rstrip('/')

            if normalized_url.endswith('/v1'):
                base_endpoint = normalized_url
            else:
                base_endpoint = f"{normalized_url}/v1"

            model_name_lower = self.model_name.lower()

            is_qwen3vl = False
            if "qwen3" in model_name_lower:
                if "vl" in model_name_lower or "-vl" in model_name_lower or " vl" in model_name_lower:
                    is_qwen3vl = True
                elif "qwen3-vl-8b-instruct" in model_name_lower:
                    is_qwen3vl = True

            if is_qwen3vl:
                endpoint = f"{base_endpoint}/chat/completions"

                messages = [
                    {"role": "system", "content": kwargs.get("system_prompt", "You are a helpful assistant")},
                    {"role": "user", "content": prompt}
                ]

                image_data = kwargs.get("image_data", None)
                if image_data:
                    messages[1]["content"] = [
                        {"type": "text", "text": prompt}
                    ]

                    images_to_process = image_data if isinstance(image_data, list) else [image_data]
                    for img in images_to_process:
                        if img.startswith("data:image/"):
                            messages[1]["content"].append({
                                "type": "image_url",
                                "image_url": {
                                    "url": img
                                }
                            })
                        else:
                            messages[1]["content"].append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img}"
                                }
                            })

                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p
                }

                print(f"Sending request to local Qwen3VL API: {endpoint}")
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    proxies=proxies,
                    timeout=self.timeout
                )

                response.raise_for_status()

                response_data = response.json()

                if "choices" in response_data and len(response_data["choices"]) > 0:
                    text = response_data["choices"][0]["message"]["content"].strip()
                else:
                    raise Exception("No generated text in response")

                token_usage = {
                    "prompt_tokens": response_data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": response_data.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": response_data.get("usage", {}).get("total_tokens", 0)
                }

                return {
                    "text": text,
                    "token_usage": token_usage,
                    "model": self.model_name,
                    "local": True
                }
            else:
                endpoint = f"{base_endpoint}/completions"

                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p
                }

                print(f"Sending request to local model API: {endpoint}")
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    proxies=proxies,
                    timeout=self.timeout
                )

                response.raise_for_status()

                response_data = response.json()

                if "choices" in response_data and len(response_data["choices"]) > 0:
                    text = response_data["choices"][0]["text"].strip()
                else:
                    raise Exception("No generated text in response")

                token_usage = {
                    "prompt_tokens": response_data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": response_data.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": response_data.get("usage", {}).get("total_tokens", 0)
                }

                return {
                    "text": text,
                    "token_usage": token_usage,
                    "model": self.model_name,
                    "local": True
                }
        except requests.exceptions.RequestException as e:
            print(f"Local model generation error: {str(e)}")
            raise Exception(f"Local model API call failed: {str(e)}")
        except Exception as e:
            print(f"Local model generation error: {str(e)}")
            raise Exception(f"Local model processing failed: {str(e)}")

    def _generate_mock(self, prompt: str, temperature: float, max_tokens: int,
                       top_p: float, **kwargs) -> Dict[str, Any]:
        mock_responses = {
            "question_decomposition": "I have decomposed the question into the following sub-questions:\n1. Sub-question 1: ...\n2. Sub-question 2: ...\n3. Sub-question 3: ...",
            "score_planning": "Modality complexity score: 0.6\nMulti-hop complexity score: 0.7\nRouting strategy: parallel_processing",
            "answer_fusion": "Based on the results of each sub-question, the final answer is: ...\n\nAnswer confidence: 0.85\nSources: Integration of sub-question 1, 2, 3 results",
            "subquery_review": "Subquery review results:\n1. Sub-question 1: Valid, can be solved independently\n2. Sub-question 2: Valid, requires result from sub-question 1\n3. Sub-question 3: Valid, no dependencies\n\nRetriever assignment:\nSub-question 1: Semantic retrieval\nSub-question 2: Exact text retrieval\nSub-question 3: Hybrid retrieval"
        }

        for key, response in mock_responses.items():
            if key in prompt:
                return {
                    "text": response,
                    "token_usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 100, "total_tokens": 150},
                    "model": "mock-model",
                    "mock": True
                }

        return {
            "text": f"Mock response: This is an answer to the prompt.\nPrompt length: {len(prompt)} characters",
            "token_usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 50, "total_tokens": 100},
            "model": "mock-model",
            "mock": True
        }

    def batch_generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results

    def embed(self, text: str) -> List[float]:
        try:
            if self.provider.lower() == "openai":
                return self._embed_openai(text)
            else:
                return self._embed_mock(text)
        except Exception as e:
            print(f"Failed to generate embedding: {e}")
            return []

    def _embed_openai(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"OpenAI embedding error: {str(e)}")

    def _embed_mock(self, text: str) -> List[float]:
        return [0.01 * i for i in range(768)]

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        self.client = self._initialize_client()

    def set_model(self, model_name: str):
        self.model_name = model_name
        print(f"Model changed to: {model_name}")

    def call_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        return self.generate(prompt, **kwargs)

    def generate_multimodal(self, prompt: str, image_data: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        top_p = kwargs.get("top_p", 1.0)

        kwargs_copy = kwargs.copy()
        kwargs_copy.pop("temperature", None)
        kwargs_copy.pop("max_tokens", None)
        kwargs_copy.pop("top_p", None)

        kwargs_copy["image_data"] = image_data

        for attempt in range(self.max_retries):
            try:
                if self.provider.lower() == "local":
                    return self._generate_local(prompt, temperature, max_tokens, top_p, **kwargs_copy)
                else:
                    print(f"Warning: {self.provider} provider multimodal support not implemented, using text fallback")
                    return self.generate(prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs_copy)
            except Exception as e:
                print(f"Multimodal generation failed (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached, returning empty result")
                    return {"text": "", "error": str(e)}

    def _generate_openai_multimodal(self, prompt: str, image_data: Union[str, List[str]], temperature: float,
                                  max_tokens: int, top_p: float, **kwargs) -> Dict[str, Any]:
        try:
            print(f"Generating with OpenAI multimodal: {self.model_name}")

            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}
            ]

            if isinstance(image_data, list):
                for img in image_data:
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": img}
                    })
            else:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": image_data}
                })

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs
            )

            return {
                "text": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": response.model
            }
        except Exception as e:
            print(f"OpenAI multimodal generation failed: {e}")
            raise Exception(f"OpenAI multimodal generation error: {str(e)}")


class MockClient:
    def __init__(self, model_name="mock-model"):
        self.model_name = model_name

    def __getattr__(self, name):
        return MockSubModule(self.model_name, name)


class MockSubModule:
    def __init__(self, model_name, module_name):
        self.model_name = model_name
        self.module_name = module_name

    def __getattr__(self, name):
        def mock_method(**kwargs):
            if name == "create":
                if self.module_name == "chat.completions":
                    return MockCompletionResponse()
                elif self.module_name == "embeddings":
                    return MockEmbeddingResponse()
            return MockResponse()
        return mock_method


class MockCompletionResponse:
    def __init__(self):
        self.choices = [MockChoice()]
        self.usage = MockUsage()


class MockChoice:
    def __init__(self):
        self.message = MockMessage()


class MockMessage:
    def __init__(self):
        self.content = "This is a mock response message."


class MockUsage:
    def __init__(self):
        self.prompt_tokens = 50
        self.completion_tokens = 100
        self.total_tokens = 150


class MockEmbeddingResponse:
    def __init__(self):
        self.data = [MockEmbeddingData()]


class MockEmbeddingData:
    def __init__(self):
        self.embedding = [0.01 * i for i in range(768)]


class MockResponse:
    def __init__(self):
        pass


def create_llm_interface(config: Optional[Dict[str, Any]] = None) -> LLMInterface:
    if config is None:
        config = {}

    global_config = Config()
    if hasattr(global_config, "llm_config"):
        config = {**global_config.llm_config, **config}

    config["model"] = "Qwen3-VL-8B-Instruct"
    config["model_name"] = "Qwen3-VL-8B-Instruct"

    return LLMInterface(**config)


def create_qwen3vl_interface(api_url: Optional[str] = None, api_key: Optional[str] = None) -> LLMInterface:
    api_config = APIKeys.get_api_config()
    default_url = api_config.get("local", {}).get("qwen3vl", {}).get("api_url", os.getenv("LOCAL_QWEN3VL_API_URL", "http://localhost:8888"))
    default_key = api_config.get("local", {}).get("qwen3vl", {}).get("api_key", "")

    config = {
        "provider": "local",
        "model_name": "Qwen3-VL-8B-Instruct",
        "base_url": api_url or default_url,
        "api_key": api_key or default_key,
        "timeout": 60,
        "max_retries": 3
    }

    return LLMInterface(**config)


__all__ = ["LLMInterface", "create_llm_interface"]
