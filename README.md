# MMMRAG: Multimodal Multi-hop Retrieval-Augmented Generation

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/your-username/mmmrag.svg)](https://github.com/your-username/mmmrag/issues)
[![GitHub Stars](https://img.shields.io/github/stars/your-username/mmmrag.svg)](https://github.com/your-username/mmmrag/stargazers)

MMMRAG is an advanced **multimodal multi-hop retrieval-augmented generation (RAG)** framework that leverages agent collaboration to efficiently handle complex queries involving multiple modalities and multi-step reasoning.

This work is described in our research paper submitted to **Information Fusion** journal:

> **4MRAG: A Modular Multi-Agent Framework for Adaptive Multimodal Multi-Hop Retrieval**

## 🚀 Overview

Built on cutting-edge multimodal models, MMMRAG provides a robust architecture for:
- **Multimodal Understanding**: Process text, images, and other modalities simultaneously
- **Multi-hop Reasoning**: Tackle complex questions requiring multiple reasoning steps
- **Intelligent Routing**: Automatically select optimal processing strategies based on query complexity
- **Hot-pluggable Components**: Easily integrate and swap different retrievers and models

### 📝 Naming Convention Note

While this framework is referred to as **4MRAG** in academic publications and papers, the codebase uses **MMMRAG** as the project name. This naming difference stems from architectural considerations and consistency with the codebase naming conventions. Both names refer to the same system - a **Multimodal Multi-hop Retrieval-Augmented Generation** framework. The core functionality and architecture remain identical regardless of the naming convention used.

## 🧠 Core Components

| Component | Description |
|-----------|-------------|
| **Scoring-based Planning Agent (SPA)** | Dynamically formulates processing pipelines via a three-dimensional scoring mechanism and adaptive routing |
| **Question Decomposer Agent (QDA)** | Implements a logic-aware decomposition strategy where questions are categorized as bridging, parallel comparison, etc. |
| **Sub-question Reviewer Agent (SRA)** | Verifies the validity, completeness, and logical consistency of decomposed sub-tasks |
| **Plug-and-play Retrieval Agent (PRA)** | Supports on-demand invocation of composite-modality retrieval modules |
| **Answer Fusion Agent (AFA)** | Integrates intermediate results to generate coherent final answers |

## ✨ Key Features

- **Intelligent Routing Strategy**: Automatically selects optimal processing path (simple / parallel / bridging)
- **Multimodal Support**: Handles text, images for input and retrieval
- **Multi-hop Reasoning**: Supports complex multi-step reasoning questions
- **Hot-pluggable Retrievers**: Easy integration of different retrieval implementations
- **Flexible Configuration**: All paths and parameters configurable via environment variables
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 📁 Directory Structure

```
mmmrag/
├── mmmrag/                        # Core package
│   ├── agents/                    # Agent modules
│   │   ├── score_planning_agent.py    # SPA: Scoring-based Planning Agent
│   │   ├── question_decomposer.py     # QDA: Question Decomposer Agent
│   │   ├── subquery_reviewer.py       # SRA: Sub-question Reviewer Agent
│   │   ├── answer_fuser.py            # AFA: Answer Fusion Agent
│   │   └── retriever_agent.py         # PRA: Plug-and-play Retrieval Agent
│   ├── config/                    # Configuration
│   │   ├── config.py              # Main config
│   │   └── api_keys_config.py     # API key management
│   ├── utils/                     # Utilities
│   │   ├── llm_interface.py       # LLM provider abstraction
│   │   ├── llm_utils.py
│   │   ├── model_manager.py       # Multi-model routing
│   │   ├── device_manager.py
│   │   ├── text_processor.py
│   │   └── logger.py
│   └── __init__.py
├── app.py                         # Main application entry point
├── ingestion.py                   # Knowledge base builder
├── eval_*.py                      # Evaluation scripts
├── visualize_results.py           # Result visualization
├── threshold.py                   # Threshold tuning
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU (recommended)
- Local LLM service (e.g., vLLM serving Qwen3-VL-8B-Instruct and Qwen3-8B, see `eval/qwen3vl.sh` for startup script)

### Steps

```bash
# Clone the repository
git clone https://github.com/your-username/mmmrag.git
cd mmmrag

# Install dependencies
pip install -r requirements.txt

# Configure environment variables (create .env file)
cat > .env << EOF
MMMRAG_MODEL_DIR=/path/to/your/models
MMMRAG_DATA_DIR=/path/to/your/data
LOCAL_QWEN3VL_API_URL=http://localhost:8888
EOF

# Prepare models and build knowledge base
python ingestion.py
```

## 🚀 Usage

### Interactive CLI Mode
```bash
python app.py
```

### Single Query Mode
```bash
python app.py --query "Describe this image" --image /path/to/image.jpg
```

### Programmatic Usage
```python
from mmmrag import MMMRAGApp

app = MMMRAGApp()
result = app.process_query({
    "question": "Your question",
    "information": {"text": "", "images": []}
})
print(result["answer"])
```

## ⚙️ Configuration

All configuration is managed through environment variables:

| Variable | Description |
|----------|-------------|
| `MMMRAG_MODEL_DIR` | Base directory for all models |
| `MMMRAG_DATA_DIR` | Base directory for data and indexes |
| `LOCAL_QWEN3VL_API_URL` | Multimodal LLM API URL |
| `LOCAL_QWEN_TEXT_API_URL` | Text-only LLM API URL |
| `MMMRAG_GPU_DEVICES` | Available GPU devices |

## 🧪 Evaluation

```bash
# ViDoSeek dataset
python eval_vidorag1.py

# SlideVQA dataset
python eval_slidevqa.py

# MMQA dataset
python eval_mmmrag_mmqa.py
```

## 📊 Visualization

python visualize_results.py

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📧 Contact

For questions or support, please open an issue on GitHub.
