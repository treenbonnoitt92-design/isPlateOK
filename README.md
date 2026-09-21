# 🚗 车牌进出一致性智能比对系统 (Jev / Laya 双引擎架构)

> 基于非自回归决策模型 **Jev**（OpenRouter Decisions API，默认推荐）与 **Laya**（Convai Innovations 本地模型），结合**加权编辑距离对齐算法（Weighted Levenshtein Alignment）**构建的智能车牌比对与 OCR 错漏识别分析系统。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Jev](https://img.shields.io/badge/Jev-OpenRouter%20Decisions%20API-blueviolet.svg)](https://openrouter.ai/docs/decisions)
[![Laya](https://img.shields.io/badge/Laya-convaiinnovations%2Flaya-purple.svg)](https://huggingface.co/convaiinnovations/laya)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2Bcu124%20(Optional)-EE4C2C.svg)](https://pytorch.org/)

---

## 📌 项目背景

在智慧停车场、高速公路卡口、ETC 收费站等实际业务场景中，由于光照反光、车速动态模糊、车牌污损、遮挡或相机角度偏转，出入口车牌识别（OCR/ANPR）相机经常出现识别偏差：
- **字形混淆替换**：如 `6` 误识为 `0`（`京NC6545` ⇄ `京NC0545`）、`8` 误识为 `B`（`粤B88888` ⇄ `粤B8888B`）；
- **字符漏读/增读错位**：相机漏读一位字符（如 `京NC6545` ⇄ `京C6545`），导致后续字符全部错位；
- **长短牌型不一**：新能源 8 位绿牌漏读能源标识字母变为 7 位蓝牌。

本项目结合**领域特征工程（智能动态规划对齐）**与**非自回归决策模型（Jev / Laya）**，实现毫秒级的同车可能性校准裁决与交互式比对。

---

## ✨ 核心特性

### 1. ⚡ 双决策引擎架构（Dual-Engine Architecture）
系统支持云端轻量与本地加速两种非自回归决策引擎，兼顾极简部署与离线私有化需求：

- **Jev 云端决策引擎（默认推荐，Enabled by Default）**：
  - 基于 **OpenRouter Decisions API**（`POST https://openrouter.ai/api/alpha/decisions`，默认模型 `~typesafe/jev-latest`）；
  - **零 GPU 门槛**：极轻量部署，无需安装庞大的 PyTorch 或 CUDA 环境，单台轻量云服务器即可秒级上线；
  - **毫秒级高速决策**：输出数学上校准的置信概率（`noul`）、多分类判定（`choice`）、分级评分（`score`），并提供完整的 Token 用量与成本跟踪。
- **Laya 本地决策模型（可选本地引擎，Optional Fallback / Comparison）**：
  - 基于本地 PyTorch 权重模型（`convaiinnovations/laya`）；
  - **GPU 硬件加速**：支持 CUDA `bfloat16` 硬件加速，在本地离线环境下实现 25ms ~ 35ms 的单次前向推断；
  - **按需懒加载（Lazy Loading）**：仅在显式请求 Laya 引擎时动态加载模型，避免启动时常驻显存。

### 2. 🧠 智能加权编辑距离对齐（Weighted Levenshtein Alignment）
- **自适应错位修复（Anti-Shift）**：自动识别字符漏读或增读，防止传统静态逐位遍历造成的整串误判；
- **OCR 混淆对降低惩罚（Weighted Cost）**：内置已知高发物理混淆对知识库（`6 ⇄ 0`、`8 ⇄ B`、`0 ⇄ D`、`1 ⇄ I`、`2 ⇄ Z`、`5 ⇄ S`、`3 ⇄ 8`、`E ⇄ F`、`C ⇄ G`），匹配此类混淆时自适应降低编辑距离成本；
- **四维对齐流回溯（Alignment Flow）**：
  - 🟩 **匹配（Match）**：字符完全一致
  - 🟥 **替换（Replace）**：字形混淆替换
  - 🟨 **漏读（Delete）**：出口相机缺失/漏识字符
  - 🟪 **增读（Insert）**：卡口反光伪增字符

### 3. 🛡️ 双层自适应裁决架构（Two-Tier Architecture）
- **Layer 1 - 规则短路门禁（0.1ms）**：
  - 车牌完全一致：直接 100% 放行通过（耗时 0.1ms）；
  - 差异过大（编辑距离 $\ge 3$ 或匹配率 $< 50\%$）：直接短路拦截拒绝（耗时 0.1ms），节省 99% 的无效模型计算与 API 调用开销；
- **Layer 2 - Jev / Laya 深度推断（~30ms - 150ms）**：
  - 对存在 1~2 处微小差异或漏字错位的车牌，调用非自回归模型输出权威数学概率。

### 4. 🎨 现代化交互式 Web 控制台
- **引擎实时一键切换**：支持在界面顶栏自由选择 Jev（云端）或 Laya（本地）引擎；
- **车牌视觉仿真**：自适应呈现中国蓝牌（燃油车）与渐变绿牌（新能源）；
- **对齐流水线视图**：逐位直观呈现字符比对、错漏成因说明及 Token/耗时指标；
- **一键快捷案例**：内置各类典型工业场景一键对比测试。

---

## 📊 典型场景测试基准

| 测试场景 | 进口车牌 ⇄ 出口车牌 | 加权编辑距离 | 匹配率 | 模型判定可能性 | 业务裁决结果 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **经典字符替换** | `京NC6545` ⇄ `京NC0545` | **0.6** | 85.7% | **`93.67%`** | 同一车辆（6/0 替换混淆） |
| **相机漏读错位** | `京NC6545` ⇄ `京C6545` | **1.0** | 85.7% | **`91.97%`** | **同一车辆（自适应对齐，漏读 N）** |
| **新能源车牌漏读** | `粤BD12345` ⇄ `粤B12345` | **1.0** | 87.5% | **`90.92%`** | **同一车辆（绿牌 8 位与 7 位自适应对齐）** |
| **夜间红外混淆** | `粤B88888` ⇄ `粤B8888B` | **0.6** | 85.7% | **`90.69%`** | 同一车辆（8/B 替换混淆） |
| **完全相同车辆** | `苏E12345` ⇄ `苏E12345` | **0.0** | 100% | **`100.0%`** | 同一车辆（规则短路秒级放行） |
| **完全随机不同车** | `沪A12345` ⇄ `浙B67890` | **6.6** | 0.0% | **`0.5%`** | 不同车辆（规则短路拦截拒绝） |

---

## ⚙️ 环境变量配置 (.env)

系统使用 `.env` 文件或环境变量管理 OpenRouter API 配置：

```bash
# OpenRouter API Key（必需，用于 Jev 引擎调用）
OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# OpenRouter 决策模型名称（可选，默认值为 "~typesafe/jev-latest"）
OPENROUTER_API_MODEL="~typesafe/jev-latest"
```

> [!TIP]
> 如果只使用短路规则测试或仅在本地加载 Laya 模型，可无需配置 API Key；使用默认 Jev 引擎时必须提供有效的 `OPENROUTER_API_KEY`。

---

## 🛠️ 安装与快速上手

### 1. 环境要求
- 操作系统：Windows / Linux / macOS
- Python：3.10 ~ 3.13
- 显卡（仅本地 Laya 模型需要）：NVIDIA 独立显卡（具备 CUDA 12+ 支持）

### 2. 安装依赖库

#### 推荐：轻量模式（Jev 云端决策，无需 GPU）
只需安装轻量 Web 与网络依赖，无需安装庞大的 PyTorch 或 CUDA：
```bash
pip install fastapi uvicorn httpx python-dotenv
```

#### 可选：本地模式（启用 Laya 本地模型）
若需在本地离线运行或对比 Laya 模型，请额外安装 `laya` 与 `torch`：
```bash
pip install laya
# 根据系统 CUDA 版本安装 PyTorch（例如 cu124）
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 3. 启动 Web 交互服务
```bash
python app.py
```
启动成功后，打开浏览器访问：
👉 **http://127.0.0.1:8000**

在右上角即可切换比对引擎（Jev 或 Laya）。

### 4. 运行命令行快速测试

#### 默认模式（使用 Jev 云端决策）
```bash
python plate_matching.py
```

#### 切换至本地 Laya 模型
```bash
python plate_matching.py --model laya
```

#### 自定义出入车牌测试
```bash
# 使用 Jev 测试自定义车牌
python plate_matching.py --plate-in 京NC6545 --plate-out 京NC0545

# 使用 Laya 测试新能源车牌漏读
python plate_matching.py --model laya --plate-in 粤BD12345 --plate-out 粤B12345
```

### 5. 🐳 Docker 容器化一键部署 (`deploy.sh`)

项目提供了标准 `Dockerfile`、`docker-compose.yml` 及自动化部署脚本 `deploy.sh`：
- **基础镜像**：`python:3.12-slim`
- **运行方式**：容器内默认采用 `uv run` 启动服务
- **外部端口映射**：默认映射为 **`13580:8000`**
- **Compose 规范**：`docker-compose.yml` 严格使用镜像 `image: jevlaya:latest`（不包含 `build`）

#### 快速启动（默认轻量模式，无 CUDA）
```bash
chmod +x deploy.sh
./deploy.sh
```
部署成功后直接访问：👉 **http://<服务器IP>:13580**

#### 可选：启用 CUDA / 本地 Laya 编译开关
```bash
./deploy.sh --cuda
```

#### 管理容器
```bash
# 查看实时日志
docker compose logs -f

# 停止服务
docker compose down
```

---

## 🔌 API 接口文档

### 车牌一致性比对接口
- **路径**：`POST /api/match`
- **请求格式**：`application/json`
- **请求体 (JSON)**：
```json
{
  "plate_in": "京NC6545",
  "plate_out": "京NC0545",
  "model": "jev"
}
```

| 请求字段 | 类型 | 必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `plate_in` | `string` | 是 | - | 入口相机识别车牌字符串 |
| `plate_out` | `string` | 是 | - | 出口相机识别车牌字符串 |
| `model` | `string` | 否 | `"jev"` | 决策引擎类型：`"jev"`（云端模型，默认）或 `"laya"`（本地模型） |

- **响应体 (JSON)**：
```json
{
  "conclusion": "极大概率为同一辆车（判定为相机OCR识别或漏读误差）",
  "badge_color": "blue",
  "probability_same": 93.67,
  "confidence": 0.9367,
  "decision": "same_vehicle_ocr_mismatch",
  "decision_label": "同一车辆（OCR识别误差/漏读）",
  "decision_probabilities": {
    "same_vehicle_ocr_mismatch": 0.9367,
    "distinct_different_vehicles": 0.0633
  },
  "score": 2.85,
  "latency_ms": 112.50,
  "model_used": "jev",
  "model_name": "~typesafe/jev-latest",
  "analysis": {
    "p_in": "京NC6545",
    "p_out": "京NC0545",
    "weighted_distance": 0.6,
    "diff_count": 1,
    "matched_chars": 6,
    "total_chars": 7,
    "matched_ratio": 0.8571,
    "has_known_ocr_confusion": true,
    "confusion_notes": [
      "数字 '6' 与 '0' 属于典型高发混淆对（强光或污损下字圈闭合易混淆）"
    ],
    "diff_details": [
      {"op": "match", "c_in": "京", "c_out": "京", "match": true, "note": "字符完全匹配"},
      {"op": "match", "c_in": "N", "c_out": "N", "match": true, "note": "字符完全匹配"},
      {"op": "match", "c_in": "C", "c_out": "C", "match": true, "note": "字符完全匹配"},
      {"op": "replace", "c_in": "6", "c_out": "0", "match": false, "note": "数字 '6' 与 '0' 属于典型高发混淆对..."},
      {"op": "match", "c_in": "5", "c_out": "5", "match": true, "note": "字符完全匹配"},
      {"op": "match", "c_in": "4", "c_out": "4", "match": true, "note": "字符完全匹配"},
      {"op": "match", "c_in": "5", "c_out": "5", "match": true, "note": "字符完全匹配"}
    ]
  },
  "raw_response": {
    "id": "dec-xxx",
    "model": "~typesafe/jev-latest",
    "answers": {
      "is_same_vehicle": {"noul": 0.9367, "confidence": 0.9367},
      "plate_match_decision": {"choice": "same_vehicle_ocr_mismatch"},
      "match_confidence_score": {"score": 2.85}
    },
    "usage": {
      "input_tokens": 128,
      "output_tokens": 12,
      "total_tokens": 140,
      "cost": 0.000042
    }
  }
}
```

| 响应字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `conclusion` | `string` | 业务综合结论描述（极大概率同一车 / 疑似同一车 / 大概率不同车） |
| `badge_color` | `string` | UI 状态徽章配色（`blue` / `amber` / `red` / `emerald`） |
| `probability_same` | `float` | 同车概率百分比值（0 ~ 100） |
| `confidence` | `float` | 模型校准置信度（0 ~ 1.0） |
| `decision` | `string` | 分类裁决标识符（`same_vehicle_ocr_mismatch` / `distinct_different_vehicles`） |
| `decision_label` | `string` | 分类裁决中文说明 |
| `decision_probabilities` | `object` | 各决策类别的概率分布 |
| `score` | `float` | 匹配相似度量化评分（0.0 ~ 3.0） |
| `latency_ms` | `float` | 推断或裁决执行耗时（毫秒） |
| `model_used` | `string` | 实际生效的比对机制：`"jev"`、`"laya"` 或短路门禁规则 `"rule"` |
| `model_name` | `string` | 生效的具体模型标识或规则名（如 `"~typesafe/jev-latest"`、`"convaiinnovations/laya"`、`"exact_match_rule"`） |
| `analysis` | `object` | 动态规划加权编辑距离对齐分析详情及逐字符对齐流 |
| `raw_response` | `object` | 决策模型或规则返回的底层原始数据体（包含 token / cost 用量等） |

---

## 📂 项目结构

```text
D:\code\jevlaya\
├── app.py              # Web 应用程序主入口（FastAPI + UI + 双引擎决策服务）
├── jev_client.py       # Jev 云端决策模型客户端（OpenRouter Decisions API）
├── plate_matching.py   # 命令行基础评测工具（支持 --model jev/laya 与自定义车牌）
├── test_align.py       # 智能加权编辑距离对齐算法独立测试脚本
├── test_api.py         # Web API 端点单元与集成测试（覆盖双引擎与短路规则）
├── test_jev_client.py  # JevClient 单元与集成测试
├── .env                # 环境变量配置（OpenRouter API Key 与模型配置）
└── README.md           # 项目完整开发与使用文档
```

---

## 📜 开源协议与鸣谢
- Jev 决策模型与 API 服务：[OpenRouter Decisions API](https://openrouter.ai/docs/decisions)
- Laya 决策模型：[Convai Innovations / Laya](https://huggingface.co/convaiinnovations/laya)
- 开源模型架构体系：ModernBERT & mmBERT
