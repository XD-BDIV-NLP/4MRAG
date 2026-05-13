# 多模态RAG系统设置指南

## 系统概述

这个多模态RAG系统集成了多个智能体和检索器，提供强大的信息检索和问答能力：

### 智能体配置

| 智能体 | 模型 | 提供商 | 配置 |
|--------|------|--------|------|
| **评分规划智能体** | GPT-5 | OpenAI | `score_planning_agent.py` |
| **问题分解智能体** | Claude 3.5 Sonnet | Anthropic | `question_decomposer.py` |
| **子问题审查智能体** | GPT-5 nano | OpenAI | `subquery_reviewer.py` |
| **答案融合智能体** | Gemini 2.5 Pro | Google | `answer_fuser.py` |

### 检索器配置

| 检索器类型 | 模型/API | 提供商 | 配置 |
|------------|----------|--------|------|
| **文本检索** | Google DeepMind Gemini Pro | Google | `RetrieverAgentAPI` |
| **图像检索** | Google Cloud Vision API | Google | `RetrieverAgentAPI` |
| **多模态检索** | OpenAI CLIP | OpenAI | `RetrieverAgentAPI` |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

#### 方法一：设置环境变量 (推荐)

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-your-openai-api-key-here"
$env:ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key-here"
$env:GOOGLE_API_KEY="your-google-api-key-here"
$env:GOOGLE_CLOUD_API_KEY="your-google-cloud-api-key-here"
$env:GOOGLE_CLOUD_PROJECT_ID="your-project-id-here"
```

**Linux/Mac (bash):**
```bash
export OPENAI_API_KEY="sk-your-openai-api-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key-here"
export GOOGLE_API_KEY="your-google-api-key-here"
export GOOGLE_CLOUD_API_KEY="your-google-cloud-api-key-here"
export GOOGLE_CLOUD_PROJECT_ID="your-project-id-here"
```

#### 方法二：创建 .env 文件

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
GOOGLE_CLOUD_API_KEY=your-google-cloud-api-key-here
GOOGLE_CLOUD_PROJECT_ID=your-project-id-here
```

### 3. 获取API密钥

- **OpenAI API Key**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic API Key**: [https://console.anthropic.com/](https://console.anthropic.com/)
- **Google API Key**: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
- **Google Cloud API Key**: [https://console.cloud.google.com/](https://console.cloud.google.com/)

### 4. 验证配置

运行以下代码来验证API密钥配置：

```python
from mmmrag.config.api_keys_config import APIKeysConfig

# 检查配置状态
APIKeysConfig.print_status()

# 查看详细设置说明
print(APIKeysConfig.get_setup_instructions())
```

### 5. 运行系统

```python
# 基本使用示例
from mmmrag.agents.score_planning_agent import ScorePlanningAgent
from mmmrag.agents.question_decomposer import QuestionDecomposer
from mmmrag.agents.subquery_reviewer import SubqueryReviewer
from mmmrag.agents.answer_fuser import AnswerFuser

# 初始化智能体
score_planner = ScorePlanningAgent()
question_decomposer = QuestionDecomposer()
subquery_reviewer = SubqueryReviewer()
answer_fuser = AnswerFuser()

# 处理问题示例
def process_question(question):
    # 1. 评分规划
    planning_result = score_planner.analyze_question(question)
    
    # 2. 问题分解
    subqueries = question_decomposer.decompose_question(question, planning_result)
    
    # 3. 审查子问题
    reviewed_subqueries = subquery_reviewer.review_subqueries(
        subqueries, question, planning_result.get('routing_strategy', 'parallel_processing')
    )
    
    # 4. 检索和融合答案
    # ... 后续处理逻辑
    
    return {
        "planning": planning_result,
        "subqueries": reviewed_subqueries
    }

# 示例使用
result = process_question("比较人工智能在医疗诊断领域的应用情况")
print(result)
```

## 📁 文件结构

```
mmmrag/
├── config/
│   ├── config.py              # 系统配置 (已更新)
│   ├── api_keys.py            # API密钥管理 (原有)
│   └── api_keys_config.py     # API密钥配置 (新增)
├── agents/
│   ├── agents_api.py          # 智能体API接口 (已更新)
│   ├── score_planning_agent.py # 评分规划智能体 (已更新)
│   ├── question_decomposer.py  # 问题分解智能体 (已更新)
│   ├── subquery_reviewer.py    # 子问题审查智能体 (已更新)
│   ├── answer_fuser.py         # 答案融合智能体 (已更新)
│   └── retriever_agent.py      # 检索器智能体 (已更新)
└── SETUP_GUIDE.md             # 本设置指南
```

## 🔧 配置详解

### 智能体LLM配置 (`config.py`)

```python
AGENT_LLM_CONFIGS = {
    "score_planning": {
        "provider": "openai",
        "model": "gpt-5",  # 评分规划专用
        "max_tokens": 2000,
        "temperature": 0.3,
        "timeout": 30
    },
    "question_decomposer": {
        "provider": "anthropic", 
        "model": "claude-3.5-sonnet",  # 问题分解专用
        "max_tokens": 1500,
        "temperature": 0.7,
        "timeout": 25
    },
    "subquery_reviewer": {
        "provider": "openai",
        "model": "gpt-5-nano",  # 子问题审查专用
        "max_tokens": 1000,
        "temperature": 0.2,
        "timeout": 20
    },
    "answer_fuser": {
        "provider": "google",
        "model": "gemini-2.5-pro",  # 答案融合专用
        "max_tokens": 3000,
        "temperature": 0.5,
        "timeout": 35
    }
}
```

### 检索器API配置 (`config.py`)

```python
RETRIEVAL_AGENT_APIS = {
    "text_retrieval": {
        "provider": "google",
        "api_type": "gemini_pro",  # Google DeepMind Gemini Pro
        "description": "文本语义检索",
        "max_tokens": 1000,
        "temperature": 0.1,
        "timeout": 15
    },
    "image_retrieval": {
        "provider": "google",
        "api_type": "vision_api",  # Google Cloud Vision API
        "description": "图像内容识别",
        "timeout": 20
    },
    "multimodal_retrieval": {
        "provider": "openai",
        "model": "clip",  # OpenAI CLIP
        "description": "跨模态检索",
        "timeout": 25
    }
}
```

## 🧪 测试系统

### 1. 测试API密钥配置

```bash
python mmmrag/config/api_keys_config.py
```

### 2. 测试智能体初始化

```python
from mmmrag.agents.score_planning_agent import ScorePlanningAgent
from mmmrag.config.api_keys_config import APIKeysConfig

# 检查API配置
APIKeysConfig.print_status()

# 测试智能体
try:
    agent = ScorePlanningAgent()
    print("✓ 评分规划智能体初始化成功")
except Exception as e:
    print(f"✗ 智能体初始化失败: {e}")
```

### 3. 完整系统测试

```python
from mmmrag.config.api_keys_config import APIKeysConfig

# 确保所有密钥都已配置
if APIKeysConfig.is_configured():
    print("✓ 所有API密钥都已配置，开始测试系统...")
    
    # 运行你的测试代码
    test_question = "分析人工智能在教育领域的应用前景"
    # ... 你的测试逻辑
    
else:
    missing = APIKeysConfig.get_missing_keys()
    print(f"✗ 缺少API密钥: {missing}")
    print("请参考设置指南配置API密钥")
```

## 🔍 故障排除

### 常见问题

1. **API密钥未配置**
   - 确保设置了所有必需的环境变量
   - 检查API密钥是否有效

2. **模型访问权限**
   - 确保你的API账户有访问指定模型的权限
   - 检查账户余额和配额

3. **网络连接问题**
   - 确保网络连接正常
   - 检查防火墙设置

4. **依赖包问题**
   - 重新安装依赖: `pip install -r requirements.txt`
   - 检查Python版本兼容性

### 调试模式

启用详细日志输出：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 现在运行你的代码将显示详细的调试信息
```

## 📞 支持

如果遇到问题，请检查：

1. API密钥配置是否正确
2. 网络连接是否正常
3. 依赖包是否正确安装
4. 模型访问权限是否充足

## 🎯 后续开发

系统已经配置完成，你可以：

1. 添加自定义的检索策略
2. 扩展新的智能体类型
3. 优化检索和融合算法
4. 添加更多的模型支持

---

**注意**: 请妥善保管你的API密钥，不要将其提交到版本控制系统中。建议使用环境变量或安全的密钥管理服务。