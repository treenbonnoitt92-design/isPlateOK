# Jev 模型集成设计文档 (OpenRouter Decisions API)

- **日期**：2026-09-21
- **状态**：已评审通过 (Approved)
- **目标**：在现有车牌对齐与一致性决策系统中增加 Jev 模型，使用 OpenRouter 的 Decisions API 替代/切换 Laya，默认使用 Jev。

---

## 1. 背景与目标

当前系统依赖本地 PyTorch 与 `laya` 库进行决策推断，需要 GPU 显存且冷启动开销较大。
通过接入 OpenRouter 的非自回归决策 API (`/api/alpha/decisions`) 与 Jev 模型 (`~typesafe/jev-latest`)：
1. 实现无需本地 GPU 算力的高可用轻量化云端决策；
2. 保持与 Laya 一致的结构化决策问题协议（`noul`、`choice`、`score`）；
3. 系统支持 Jev 与 Laya 双模型无缝切换，**默认采用 Jev**。

---

## 2. 核心架构与模型适配

### 2.1 决策模型适配层 (`DecisionBackend`)

- **`JevOpenRouterBackend`** (默认)：
  - 读取 `.env` 中的 `OPENROUTER_API_KEY` 与 `OPENROUTER_API_MODEL`；
  - 请求端点：`POST https://openrouter.ai/api/alpha/decisions`；
  - 请求体：
    ```json
    {
      "model": "OPENROUTER_API_MODEL",
      "state": { ... },
      "questions": { ... }
    }
    ```
  - 响应解析：提取 `answers.is_same_vehicle.noul`、`answers.plate_match_decision.choice`、`answers.match_confidence_score.score`，以及 token 用量与费用信息。
- **`LayaLocalBackend`**：
  - 保留原有本地模型加载方式，采用懒加载（Lazy Loading），未请求时避免占用 GPU 显存。

### 2.2 后端 API 接口设计

- **接口**：`POST /api/match`
- **请求参数**：
  ```json
  {
    "plate_in": "京NC6545",
    "plate_out": "京NC0545",
    "model": "jev" // 可选: "jev" (默认) | "laya"
  }
  ```
- **响应参数**：
  ```json
  {
    "conclusion": "极大概率为同一辆车...",
    "badge_color": "blue",
    "probability_same": 84.0,
    "confidence": 0.97,
    "decision": "same_vehicle_ocr_mismatch",
    "decision_label": "同一车辆（OCR识别误差/漏读）",
    "decision_probabilities": { ... },
    "score": 2.98,
    "latency_ms": 120.5,
    "model_used": "jev",
    "model_name": "typesafe/jev-1.13-20260917",
    "analysis": { ... },
    "raw_response": { ... }
  }
  ```

---

## 3. 前端控制台设计

1. **模型切换选择器**：
   - 顶部提供模型切换选项（`Jev (OpenRouter 云端决策)` ⭐️ 默认 与 `Laya (本地 GPU 模型)`）。
   - 切换模型后自动进行实时对比推断。
2. **状态与指标看板**：
   - 动态显示当前调用模型名称、推断延迟（ms）及决策置信度。
3. **快速预设与对齐流水线视图**：
   - 保留原有六大快速测试场景及逐位 Levenshtein 对齐图表。

---

## 4. 命令行工具升级 (`plate_matching.py`)

- 支持直接读取 `.env` 并在终端运行 Jev 决策比对；
- 提供命令行参数或选项以方便对比 Laya 与 Jev 输出。

---

## 5. 实施与验证步骤

1. **模型客户端模块编写**：实现 Jev OpenRouter 决策客户端与统一接口封装；
2. **`app.py` 改造**：集成双模型后端，默认配置为 Jev，添加模型路由与前端模型切换 UI；
3. **`plate_matching.py` 升级**：默认切换为 Jev 模型调用；
4. **功能测试与对齐验证**：运行多场景测试用例，验证 Jev 判定逻辑与响应正确性；
5. **文档完善**：更新 `README.md` 与使用说明。
