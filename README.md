# 4MRAG: A Modular Multi-Agent Framework for Adaptive Multimodal Multi-Hop Retrieval-Augmented Generation

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Official implementation of the paper **"4MRAG: A Modular Multi-Agent Framework for Adaptive Multimodal Multi-Hop Retrieval-Augmented Generation"**, submitted to *Expert Systems With Applications*.

> **Naming note.** The framework is called **4MRAG** in the paper and **MMMRAG** in the codebase (Python package name and environment-variable prefixes). Both names refer to exactly the same system.

---

## Overview

4MRAG is a multimodal multi-hop Retrieval-Augmented Generation (RAG) framework built around five collaborating LLM agents. It supports:

- **Adaptive routing** — question complexity is scored along three dimensions (modality complexity, multi-hop complexity, retrieval necessity) and the pipeline is planned accordingly;
- **Dual-mode decomposition** — questions are decomposed in a bridging or parallel-comparison style, with a reviewer agent validating each sub-question and sending invalid ones back for re-decomposition;
- **Plug-and-play retrieval** — retrievers over different modalities are invoked on demand through a uniform interface;
- **Confidence-based retrieval gating** — a verbalized confidence score decides whether external retrieval is needed at all;
- **Multimodal inputs** — text and images, with hot-swappable retrieval backends.

## Core Components (Paper to Code)

| Component (paper) | Role | Implementation |
|---|---|---|
| Scoring-based Planning Agent (SPA) | Three-dimensional scoring and adaptive routing | `mmmrag/agents/score_planning_agent.py` |
| Question Decomposer Agent (QDA) | Logic-aware decomposition (bridging / parallel) | `mmmrag/agents/question_decomposer.py` |
| Sub-question Reviewer Agent (SRA) | Validity, completeness and consistency checking | `mmmrag/agents/subquery_reviewer.py` |
| Plug-and-play Retrieval Agent (PRA) | On-demand multimodal retrieval | `mmmrag/agents/retriever_agent.py` |
| Answer Fusion Agent (AFA) | Evidence fusion and final answer generation | `mmmrag/agents/answer_fuser.py` |

## Repository Layout

```
.
├── mmmrag/                          # Core Python package
│   ├── agents/                      # The five collaborating agents
│   │   ├── score_planning_agent.py      # SPA: scoring + adaptive routing
│   │   ├── question_decomposer.py       # QDA: bridging / parallel decomposition
│   │   ├── subquery_reviewer.py         # SRA: sub-question validation
│   │   ├── retriever_agent.py           # PRA: plug-and-play retrieval
│   │   └── answer_fuser.py              # AFA: answer fusion
│   ├── config/
│   │   ├── config.py                    # Model paths, thresholds, per-agent LLM configs
│   │   └── api_keys_config.py           # API key management
│   └── utils/
│       ├── llm_interface.py             # LLM provider abstraction
│       ├── llm_utils.py
│       ├── model_manager.py             # Multi-model routing
│       ├── device_manager.py
│       ├── text_processor.py
│       └── logger.py
├── app.py                           # Entry point (interactive CLI / single query)
├── ingestion.py                     # Knowledge-base builder (FAISS index)
├── eval_vidorag1.py                 # Evaluation on ViDoSeek
├── eval_slidevqa.py                 # Evaluation on SlideVQA
├── qwen3vl.sh                       # Example vLLM startup script (Qwen3-VL-8B-Instruct)
├── requirements.txt                 # Pinned, tested environment
├── LICENSE
└── README.md
```

## Requirements

- Python 3.10+
- A CUDA-capable GPU (an NVIDIA GPU is strongly recommended for the local retrieval stack)
- A local LLM service. We use **vLLM** to serve two backbones:
  - `Qwen3-VL-8B-Instruct` (multimodal reasoning), and
  - `Qwen3-8B` (text-only utilities).
  See `qwen3vl.sh` at the repository root for an example startup script.

The pinned environment is provided in `requirements.txt` (key versions: `torch==2.9.0+cu128`, `vllm==0.13.0`, `transformers==4.57.1`, `faiss-cpu==1.13.0`, `sentence-transformers==5.1.2`, `colpali_engine`).

## Installation

Download the snapshot of this repository (use the **Download** button on the repository page) and extract it locally, then install the dependencies.

```bash
# extract the downloaded archive, then:
cd 4MRAG-3AC3
pip install -r requirements.txt
```

## Configuration

All paths and endpoints are configured through environment variables (e.g., via a `.env` file or exported in your shell):

| Variable | Description |
|---|---|
| `MMMRAG_MODEL_DIR` | Base directory of all local models |
| `MMMRAG_DATA_DIR` | Base directory of datasets and indexes |
| `LOCAL_QWEN3VL_API_URL` | OpenAI-compatible endpoint of the multimodal LLM |
| `LOCAL_QWEN_TEXT_API_URL` | OpenAI-compatible endpoint of the text-only LLM |
| `MMMRAG_GPU_DEVICES` | Visible GPU devices (e.g., `0,1`) |

Example:

```bash
cat > .env << 'EOF'
MMMRAG_MODEL_DIR=/path/to/your/models
MMMRAG_DATA_DIR=/path/to/your/data
LOCAL_QWEN3VL_API_URL=http://localhost:8888
LOCAL_QWEN_TEXT_API_URL=http://localhost:8889
MMMRAG_GPU_DEVICES=0,1
EOF
```

## Building the Knowledge Base

```bash
python ingestion.py
```

This parses the corpus under `MMMRAG_DATA_DIR`, computes text and visual embeddings, and builds the FAISS indexes used by the retrieval agent.

## Usage

Interactive CLI:

```bash
python app.py
```

Single query (text + image):

```bash
python app.py --query "Describe the trend shown in this figure" --image /path/to/image.jpg
```

Programmatic use:

```python
from mmmrag import MMMRAGApp

app = MMMRAGApp()
result = app.process_query({
    "question": "Your question",
    "information": {"text": "", "images": []}
})
print(result["answer"])
```

## Evaluation

```bash
# ViDoSeek
python eval_vidorag1.py

# SlideVQA
python eval_slidevqa.py
```

**Settings used in the paper:** per-modality candidate pool `k = 5`, cross-encoder reranking down to the final evidence set `N = 3`, and random seed `42`. The reported accuracy is the Acc* metric described in the paper.

## Datasets

Experiments are conducted on public benchmarks (**ViDoSeek** and **SlideVQA**). Please obtain each dataset from its official source and place it under `MMMRAG_DATA_DIR` following the layout expected by the evaluation scripts. Datasets are **not** redistributed in this repository, in accordance with their licenses.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Questions

For questions about the code, please raise them through the journal's review system during review, or contact the authors afterwards.
