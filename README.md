# 🚗 Laya 车牌进出一致性智能比对系统

> 基于最新开源 **Laya 决策模型**（Convai Innovations）与**加权编辑距离对齐算法（Weighted Levenshtein Alignment）**构建的智能车牌比对与 OCR 错漏识别分析系统。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Laya-convaiinnovations%2Flaya-purple.svg)](https://huggingface.co/convaiinnovations/laya)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)

---

## 📌 项目背景

在智慧停车场、高速公路卡口、ETC 收费站等实际业务场景中，由于光照反光、车速动态模糊、车牌污损、遮挡或相机角度偏转，出入口车牌识别（OCR/ANPR）相机经常出现识别偏差：
- **字形混淆替换**：如 `6` 误识为 `0`（`京NC6545` ⇄ `京NC0545`）、`8` 误识为 `B`（`粤B88888` ⇄ `粤B8888B`）；
- **字符漏读/增读错位**：相机漏读一位字符（如 `京NC6545` ⇄ `京C6545`），导致后续字符全部错位；
- **长短牌型不一**：新能源 8 位绿牌漏读能源标识字母变为 7 位蓝牌。

本项目结合**领域特征工程（智能动态规划对齐）**与**非自回归决策模型 Laya**，实现毫秒级的同车可能性校准裁决与交互式比对。

---

## ✨ 核心特性

### 1. ⚡ 毫秒级 GPU 极速决策（Laya System 1 Model）
- **非自回归单步前向推断**：Laya 不生成自由文本，避免了自回归大模型的解码延迟与幻觉问题；
- **数学上校准的置信概率**：通过 `noul`（布尔概率）、`choice`（多选分类）、`score`（分级评分）输出精确概率；
- **GPU 硬件加速**：在 RTX 3060 Ti GPU（`bfloat16`）上，单次模型推断仅需 **25ms ~ 35ms**（较 CPU 提升 50 倍以上），满足闸机毫秒级秒抬杆需求。

### 2. 🧠 智能加权编辑距离对齐（Weighted Levenshtein Alignment）
- **自适应错位修复（Anti-Shift）**：自动识别字符漏读或增读，防止传统静态循环遍历造成的整串误判；
- **OCR 混淆对降低惩罚（Weighted Cost）**：内置已知高发物理混淆对知识库（`6 ⇄ 0`、`8 ⇄ B`、`0 ⇄ D`、`1 ⇄ I`、`2 ⇄ Z`、`5 ⇄ S`、`3 ⇄ 8`、`E ⇄ F`、`C ⇄ G`），匹配此类混淆时自适应降低编辑距离成本；
- **四维对齐流回溯（Alignment Flow）**：
  - 🟩 **匹配（Match）**：字符完全一致
  - 🟥 **替换（Replace）**：字形混淆替换
  - 🟨 **漏读（Delete）**：出口相机缺失/漏识字符
  - 🟪 **增读（Insert）**：卡口反光伪增字符

### 3. 🛡️ 双层自适应裁决架构（Two-Tier Architecture）
- **Layer 1 - 规则短路门禁（0.1ms）**：
  - 车牌完全一致：直接 100% 通过（耗时 0.1ms）；
  - 差异过大（编辑距离 $\ge 3$ 或匹配率 $< 50\%$）：直接短路拦截拒绝（耗时 0.1ms），节省 99% 的无效 GPU 算力；
- **Layer 2 - Laya 深度推断（~30ms）**：
  - 对存在 1~2 处微小差异或漏字错位的车牌，调用 Laya 输出权威数学概率。

### 4. 🎨 现代化交互式 Web 控制台
- **车牌视觉仿真**：自适应呈现中国蓝牌（燃油车）与渐变绿牌（新能源）；
- **对齐流水线视图**：逐位直观呈现字符比对与错漏成因说明；
- **一键快捷案例**：内置各类典型工业场景一键对比测试。

---

## 📊 典型场景测试基准

| 测试场景 | 进口车牌 ⇄ 出口车牌 | 加权编辑距离 | 匹配率 | Laya 模型判定可能性 | 业务裁决结果 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **经典字符替换** | `京NC6545` ⇄ `京NC0545` | **0.6** | 85.7% | **`93.67%`** | 同一车辆（6/0 替换混淆） |
| **相机漏读错位** | `京NC6545` ⇄ `京C6545` | **1.0** | 85.7% | **`91.97%`** | **同一车辆（自适应对齐，漏读 N）** |
| **新能源车牌漏读** | `粤BD12345` ⇄ `粤B12345` | **1.0** | 87.5% | **`90.92%`** | **同一车辆（绿牌 8 位与 7 位自适应对齐）** |
| **夜间红外混淆** | `粤B88888` ⇄ `粤B8888B` | **0.6** | 85.7% | **`90.69%`** | 同一车辆（8/B 替换混淆） |
| **完全相同车辆** | `苏E12345` ⇄ `苏E12345` | **0.0** | 100% | **`100.0%`** | 同一车辆（0 差异秒级放行） |
| **完全随机不同车** | `沪A12345` ⇄ `浙B67890` | **6.6** | 0.0% | **`0.5%`** | 不同车辆（短路拦截拒绝） |

---

## 🛠️ 安装与快速上手

### 1. 环境要求
- 操作系统：Windows / Linux / macOS
- Python：3.10 ~ 3.13
- 显卡（可选，推荐）：NVIDIA 独立显卡（具备 CUDA 12+ 支持）

### 2. 安装依赖库
```bash
# 安装 Laya 决策模型与核心依赖
pip install laya fastapi uvicorn httpx

# 启用 GPU 加速（根据系统 CUDA 版本安装 PyTorch，例如 cu124）
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 3. 启动 Web 交互服务
```bash
python app.py
```
启动成功后，打开浏览器访问：
👉 **http://127.0.0.1:8000**

### 4. 运行命令行快速测试
```bash
python plate_matching.py
```

---

## 🔌 API 接口文档

### 车牌一致性比对接口
- **路径**：`POST /api/match`
- **请求体 (JSON)**：
```json
{
  "plate_in": "京NC6545",
  "plate_out": "京NC0545"
}
```

- **响应体 (JSON)**：
```json
{
  "conclusion": "极大概率为同一辆车（判定为相机OCR识别或漏读误差）",
  "badge_color": "blue",
  "probability_same": 93.67,
  "confidence": 0.9367,
  "decision": "same_vehicle_ocr_mismatch",
  "decision_label": "同一车辆（OCR识别误差/漏读）",
  "score": 1.14,
  "latency_ms": 55.84,
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
  }
}
```

---

## 📂 项目结构

```text
D:\code\jevlaya\
├── app.py              # Web 应用程序主入口（FastAPI + UI + Laya GPU 服务）
├── plate_matching.py   # 命令行基础评测脚本
├── test_align.py       # 智能加权编辑距离对齐算法独立测试脚本
└── README.md           # 项目完整开发与使用文档
```

---

## 📜 开源协议与鸣谢
- 模型来源：[Convai Innovations / Laya](https://huggingface.co/convaiinnovations/laya)
- 开源开源模型体系：ModernBERT & mmBERT
