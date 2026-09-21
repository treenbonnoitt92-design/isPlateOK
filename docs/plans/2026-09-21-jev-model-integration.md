# Jev 模型集成 Implementation Plan

> **For Claude / Antigravity:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 增加 Jev 模型（通过 OpenRouter Decisions API），替代/切换本地 Laya 模型，默认使用 Jev，并完成 Web 前端、API 接口和命令行工具的统一支持。

**Architecture:** 抽象决策模型客户端（`DecisionBackend`），实现 `JevOpenRouterBackend` 封装 OpenRouter 的 `/api/alpha/decisions` 非自回归端点，并在 `app.py` 中支持 `jev`（默认）与 `laya`（本地）双模型切换，前端提供选择开关并展示调用元数据。

**Tech Stack:** Python 3.10+, FastAPI, httpx, python-dotenv, TailwindCSS, OpenRouter Decisions API.

---

### Task 1: 建立 Jev 决策客户端模块 (`jev_client.py`)

**Files:**
- Create: `D:\code\jevlaya\jev_client.py`
- Test: `D:\code\jevlaya\test_jev_client.py`

**Step 1: Write test for Jev client**

编写 `test_jev_client.py` 验证客户端初始化、状态与问题发送，以及输出格式解析。

```python
import os
import pytest
from dotenv import load_dotenv
from jev_client import JevClient

load_dotenv()

def test_jev_client_predict():
    client = JevClient()
    state = {
        "entrance_plate": "京NC6545",
        "exit_plate": "京NC0545",
        "description": "6 vs 0 OCR confusion test"
    }
    questions = {
        "is_same_vehicle": {
            "type": "noul",
            "instructions": "Are these plates from the same vehicle?"
        }
    }
    res = client.predict(state, questions)
    assert "answers" in res
    assert "is_same_vehicle" in res["answers"]
    assert "noul" in res["answers"]["is_same_vehicle"]
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_jev_client.py`
Expected: FAIL (No module named 'jev_client')

**Step 3: Implement minimal code for `jev_client.py`**

实现 `JevClient`：
- 读取环境变量 `OPENROUTER_API_KEY` 与 `OPENROUTER_API_MODEL`（默认 `~typesafe/jev-latest`）
- 提供同步 `predict(state, questions)` 与异步 `apredict(state, questions)`
- 请求 `https://openrouter.ai/api/alpha/decisions`

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_jev_client.py`
Expected: PASS

**Step 5: Commit**

```bash
git add jev_client.py test_jev_client.py
git commit -m "feat: implement JevClient for OpenRouter decisions API"
```

---

### Task 2: 改造 `app.py` 支持 Jev（默认）与模型切换

**Files:**
- Modify: `D:\code\jevlaya\app.py`
- Test: `D:\code\jevlaya\test_api.py`

**Step 1: Write API tests for `app.py`**

编写 `test_api.py` 测试：
- `/api/match` 默认使用 `jev`，返回包含 `model_used: "jev"`
- `/api/match` 指定 `model: "jev"`
- 门禁规则短路判定（无需调用模型，0.1ms）

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_api.py`
Expected: FAIL

**Step 3: Update `app.py`**

- 引入 `JevClient`，默认实例化 Jev 客户端
- `laya` 采用懒加载方式（仅在请求 `model == "laya"` 时加载）
- `MatchRequest` 增加 `model: str = "jev"`
- `/api/match` 处理逻辑分流，返回 `model_used`、`model_name`
- 更新 Web 前端界面：
  - 增加模型选择 Radio / Segmented Button（`Jev (OpenRouter 云端决策)` ⭐️ 默认 vs `Laya (本地 GPU 模型)`）
  - 前端结果面板展示当前生效模型及推断统计

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_api.py`
Expected: PASS

**Step 5: Commit**

```bash
git add app.py test_api.py
git commit -m "feat: add Jev model support and UI switcher in app.py"
```

---

### Task 3: 更新命令行测试脚本 `plate_matching.py`

**Files:**
- Modify: `D:\code\jevlaya\plate_matching.py`

**Step 1: Update `plate_matching.py`**

- 默认调用 `JevClient` 执行决策
- 支持 `--model laya` 切换为本地 Laya 测试
- 美化控制台打印输出，包含模型信息与消耗指标

**Step 2: Run script to verify execution**

Run: `python plate_matching.py`
Expected: 成功输出 Jev 模型的决策结果、同车概率与评分

**Step 3: Commit**

```bash
git add plate_matching.py
git commit -m "feat: update plate_matching.py to use Jev model by default"
```

---

### Task 4: 更新系统文档 `README.md`

**Files:**
- Modify: `D:\code\jevlaya\README.md`

**Step 1: Update README.md**

- 说明系统支持的双模型架构：Jev（OpenRouter 云端决策，默认）+ Laya（本地决策模型）
- 增加 `.env` 配置说明（`OPENROUTER_API_KEY`, `OPENROUTER_API_MODEL`）
- 更新安装与快速启动指令

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with Jev model and OpenRouter configuration"
```
