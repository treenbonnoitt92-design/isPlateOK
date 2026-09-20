import time
import json
from typing import Dict, Any, List
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import laya

# 常见车牌 OCR 字符形近混淆知识库
OCR_CONFUSION_PAIRS = {
    frozenset(['6', '0']): "数字 '6' 与 '0' 属于典型高发混淆对（强光或污损下字圈闭合易混淆）",
    frozenset(['8', 'B']): "数字 '8' 与字母 'B' 结构极似（红外夜拍常混淆）",
    frozenset(['0', 'D']): "数字 '0' 与字母 'D' 边缘弧度相似（角度偏转易混淆）",
    frozenset(['1', 'I']): "数字 '1' 与字母 'I' 细线结构几乎一致",
    frozenset(['2', 'Z']): "数字 '2' 与字母 'Z' 折角在动态模糊时易混淆",
    frozenset(['5', 'S']): "数字 '5' 与字母 'S' 曲线轮廓在低分辨率下易混淆",
    frozenset(['3', '8']): "数字 '3' 与 '8' 左侧局部被遮挡或螺丝遮挡时易混淆",
    frozenset(['E', 'F']): "字母 'E' 与 'F' 下横笔划缺失或污损易混淆",
    frozenset(['C', 'G']): "字母 'C' 与 'G' 弧线相似"
}

agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    print("[INFO] 正在预加载 Laya 决策模型 (convaiinnovations/laya)...")
    agent = laya.load("convaiinnovations/laya")
    print("[INFO] Laya 决策模型预加载就绪！")
    yield

app = FastAPI(title="Laya 车牌智能对齐与一致性决策系统", lifespan=lifespan)

class MatchRequest(BaseModel):
    plate_in: str
    plate_out: str

def analyze_plate_diff_smart(p_in: str, p_out: str) -> Dict[str, Any]:
    """
    智能动态规划编辑距离对齐算法（Weighted Levenshtein Alignment）
    自适应解决：
    1. 字符替换与 OCR 形近字识别混淆（如 6 ⇄ 0, 8 ⇄ B）
    2. 字符漏读或多读导致的序列错位偏移（Shift，如 京NC6545 ⇄ 京C6545）
    3. 蓝牌 7 位与绿牌 8 位长短不一的结构对齐
    """
    s1 = p_in.strip().upper()
    s2 = p_out.strip().upper()
    m, n = len(s1), len(s2)

    # DP 矩阵与回溯指针
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    backtrack = [[(0, 0, 'none')] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i * 1.0
        backtrack[i][0] = (i - 1, 0, 'delete')
    for j in range(1, n + 1):
        dp[0][j] = j * 1.0
        backtrack[0][j] = (0, j - 1, 'insert')

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c1, c2 = s1[i - 1], s2[j - 1]
            if c1 == c2:
                cost_sub = 0.0
                op_sub = 'match'
            else:
                pair = frozenset([c1, c2])
                # 典型 OCR 混淆对成本降低，提升容错先验
                cost_sub = 0.6 if pair in OCR_CONFUSION_PAIRS else 1.0
                op_sub = 'replace'

            sub_score = dp[i - 1][j - 1] + cost_sub
            del_score = dp[i - 1][j] + 1.0
            ins_score = dp[i][j - 1] + 1.0

            min_score = min(sub_score, del_score, ins_score)
            dp[i][j] = min_score

            if min_score == sub_score:
                backtrack[i][j] = (i - 1, j - 1, op_sub)
            elif min_score == del_score:
                backtrack[i][j] = (i - 1, j, 'delete')
            else:
                backtrack[i][j] = (i, j - 1, 'insert')

    # 回溯生成对齐详情
    alignment = []
    curr_i, curr_j = m, n
    diff_count = 0
    match_count = 0
    confusion_notes = []

    while curr_i > 0 or curr_j > 0:
        prev_i, prev_j, op = backtrack[curr_i][curr_j]
        c_in = s1[prev_i] if curr_i > prev_i else ''
        c_out = s2[prev_j] if curr_j > prev_j else ''

        if op == 'match':
            note = '字符完全匹配'
            match_count += 1
        elif op == 'replace':
            diff_count += 1
            pair = frozenset([c_in, c_out])
            if pair in OCR_CONFUSION_PAIRS:
                note = OCR_CONFUSION_PAIRS[pair]
                confusion_notes.append(note)
            else:
                note = f"字符替换差异 ({c_in} ⇄ {c_out})"
        elif op == 'delete':
            diff_count += 1
            note = f"相机位移漏读字符 [{c_in}]"
            confusion_notes.append(f"位移漏读：出口相机漏读字符 '{c_in}'")
        elif op == 'insert':
            diff_count += 1
            note = f"相机位移多识别字符 [{c_out}]"
            confusion_notes.append(f"位移增读：出口相机多识别字符 '{c_out}'")

        alignment.append({
            'op': op,
            'c_in': c_in,
            'c_out': c_out,
            'match': (op == 'match'),
            'note': note
        })
        curr_i, curr_j = prev_i, prev_j

    alignment.reverse()
    total_aligned_len = len(alignment)
    matched_ratio = round(match_count / total_aligned_len, 4) if total_aligned_len > 0 else 0.0

    return {
        "p_in": s1,
        "p_out": s2,
        "weighted_distance": round(dp[m][n], 2),
        "diff_count": diff_count,
        "matched_chars": match_count,
        "total_chars": total_aligned_len,
        "matched_ratio": matched_ratio,
        "has_known_ocr_confusion": len(confusion_notes) > 0,
        "confusion_notes": list(set(confusion_notes)),
        "diff_details": alignment
    }

@app.post("/api/match")
async def match_plates(req: MatchRequest):
    global agent
    if agent is None:
        agent = laya.load("convaiinnovations/laya")

    analysis = analyze_plate_diff_smart(req.plate_in, req.plate_out)

    # 1. 完全一致
    if analysis["diff_count"] == 0:
        return {
            "conclusion": "车牌完全一致",
            "badge_color": "emerald",
            "probability_same": 100.0,
            "confidence": 1.0,
            "decision": "same_vehicle",
            "decision_label": "同一车辆（车牌无差异）",
            "score": 3.0,
            "latency_ms": 0.1,
            "analysis": analysis,
            "raw_laya": {"notice": "100% exact match across all aligned characters"}
        }

    # 2. 差异过大（编辑距离 >= 3 或匹配率低于 50%）直接短路拦截
    if analysis["diff_count"] >= 3 or analysis["matched_ratio"] < 0.5:
        return {
            "conclusion": "完全不同车辆（多处字符不匹配）",
            "badge_color": "red",
            "probability_same": 0.5,
            "confidence": 0.99,
            "decision": "different_vehicles",
            "decision_label": "不同车辆（多字符相异）",
            "score": 0.0,
            "latency_ms": 0.1,
            "analysis": analysis,
            "raw_laya": {"notice": f"Significant mismatch: {analysis['diff_count']} differing characters after alignment"}
        }

    # 3. 存在 1-2 处微小差异或漏字错位，交给 Laya 进行非自回归概率裁决
    state = {
        "entrance_plate": analysis["p_in"],
        "exit_plate": analysis["p_out"],
        "alignment_summary": (
            f"After dynamic Levenshtein alignment: {analysis['matched_chars']} out of {analysis['total_chars']} "
            f"characters match ({analysis['matched_ratio'] * 100:.1f}% match ratio). "
            f"Difference operations count: {analysis['diff_count']}. "
            f"Weighted edit distance: {analysis['weighted_distance']}. "
            f"Known OCR noise/shift pattern: {analysis['has_known_ocr_confusion']} "
            f"({'; '.join(analysis['confusion_notes'])})."
        ),
        "difference_count": analysis["diff_count"],
        "matched_ratio": f"{analysis['matched_ratio'] * 100:.1f}%"
    }

    questions = {
        "is_same_vehicle": {
            "type": "noul",
            "instructions": "Are these entrance and exit camera records likely from the same vehicle despite minor OCR noise or single-character omission?"
        },
        "plate_match_decision": {
            "type": "choice",
            "instructions": "Determine if these records belong to the same vehicle or different vehicles.",
            "criteria": {
                "same_vehicle_ocr_mismatch": "Same vehicle with minor OCR camera recognition error or single omission",
                "distinct_different_vehicles": "Different vehicles entirely with distinct plates"
            }
        },
        "match_confidence_score": {
            "type": "score",
            "instructions": "Rate the probability that this is the same car on a 4-level scale (0 to 3).",
            "criteria": [
                "level_0: definitely different cars",
                "level_1: unlikely to be the same car",
                "level_2: plausible / possible same car",
                "level_3: highly probable same car with single-character OCR error"
            ]
        }
    }

    t0 = time.perf_counter()
    laya_result = agent.predict(state, questions)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    answers = laya_result.get("answers", {})
    is_same = answers.get("is_same_vehicle", {})
    decision = answers.get("plate_match_decision", {})
    conf_score = answers.get("match_confidence_score", {})

    prob_same = is_same.get("noul", 0.0)

    if prob_same >= 0.80:
        conclusion = "极大概率为同一辆车（判定为相机OCR识别或漏读误差）"
        badge_color = "blue"
    elif prob_same >= 0.50:
        conclusion = "疑似同一辆车（建议人工核验）"
        badge_color = "amber"
    else:
        conclusion = "大概率为不同车辆"
        badge_color = "red"

    choice_labels = {
        "same_vehicle_ocr_mismatch": "同一车辆（OCR识别误差/漏读）",
        "distinct_different_vehicles": "不同车辆"
    }

    return {
        "conclusion": conclusion,
        "badge_color": badge_color,
        "probability_same": round(prob_same * 100, 2),
        "confidence": is_same.get("confidence", 0.0),
        "decision": decision.get("choice", ""),
        "decision_label": choice_labels.get(decision.get("choice", ""), decision.get("choice", "")),
        "decision_probabilities": decision.get("probabilities", {}),
        "score": conf_score.get("score", 0.0),
        "latency_ms": latency_ms,
        "analysis": analysis,
        "raw_laya": laya_result
    }

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Laya AI 智能车牌对齐与一致性决策系统</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />
  <style>
    .plate-badge-blue {
      background: linear-gradient(180deg, #1b5bb5 0%, #0d3f82 100%);
      color: #ffffff;
      border: 2px solid #ffffff;
      box-shadow: 0 4px 10px rgba(13, 63, 130, 0.4);
      font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
      font-weight: bold;
      letter-spacing: 2px;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
    }
    .plate-badge-green {
      background: linear-gradient(180deg, #ffffff 0%, #48bb78 50%, #2f855a 100%);
      color: #1a202c;
      border: 2px solid #2f855a;
      box-shadow: 0 4px 10px rgba(47, 133, 90, 0.4);
      font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
      font-weight: bold;
      letter-spacing: 2px;
    }
    .char-match {
      color: #10b981;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .char-replace {
      color: #ef4444;
      background: rgba(239, 68, 68, 0.2);
      border: 1px dashed #ef4444;
      animation: pulse 2s infinite;
    }
    .char-delete {
      color: #f59e0b;
      background: rgba(245, 158, 11, 0.2);
      border: 1px dashed #f59e0b;
    }
    .char-insert {
      color: #a855f7;
      background: rgba(168, 85, 247, 0.2);
      border: 1px dashed #a855f7;
    }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen">
  <div class="max-w-5xl mx-auto px-4 py-8">
    <header class="text-center mb-8">
      <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-semibold mb-3">
        <i class="fa-solid fa-brain"></i> 升级版：智能加权编辑距离对齐 + Laya 决策模型
      </div>
      <h1 class="text-3xl sm:text-4xl font-extrabold tracking-tight bg-gradient-to-r from-blue-400 via-indigo-300 to-purple-400 bg-clip-text text-transparent">
        车牌智能对齐与一致性决策系统
      </h1>
      <p class="text-slate-400 text-sm mt-2 max-w-2xl mx-auto">
        针对任意随机车牌输入，自适应解决<b>字符替换</b>（如 6/0 混淆）、<b>相机漏读错位</b>（Shift）、<b>位数不一致</b>等卡口复杂场景。
      </p>
    </header>

    <div class="bg-slate-900/90 border border-slate-800 rounded-2xl shadow-2xl p-6 sm:p-8 backdrop-blur-sm">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="space-y-2">
          <label class="block text-sm font-medium text-slate-300">
            <i class="fa-solid fa-arrow-right-to-bracket text-emerald-400 mr-1.5"></i>进口识别车牌
          </label>
          <input
            id="plateIn"
            type="text"
            value="京NC6545"
            placeholder="例如：京NC6545"
            class="w-full bg-slate-800/80 border border-slate-700 rounded-xl px-4 py-3 text-lg font-mono tracking-wider focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent uppercase transition"
          />
        </div>

        <div class="space-y-2">
          <label class="block text-sm font-medium text-slate-300">
            <i class="fa-solid fa-arrow-right-from-bracket text-blue-400 mr-1.5"></i>出口识别车牌
          </label>
          <input
            id="plateOut"
            type="text"
            value="京NC0545"
            placeholder="例如：京NC0545"
            class="w-full bg-slate-800/80 border border-slate-700 rounded-xl px-4 py-3 text-lg font-mono tracking-wider focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent uppercase transition"
          />
        </div>
      </div>

      <!-- 快速预设 -->
      <div class="mt-4 flex flex-wrap items-center gap-2 text-xs">
        <span class="text-slate-400 font-medium"><i class="fa-solid fa-wand-magic-sparkles mr-1"></i>快捷测试场景:</span>
        <button onclick="setPreset('京NC6545', '京NC0545')" class="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-slate-300 transition">
          京NC6545 ⇄ 京NC0545 (6 vs 0 替换混淆)
        </button>
        <button onclick="setPreset('京NC6545', '京C6545')" class="px-2.5 py-1 bg-amber-950/40 hover:bg-amber-900/40 border border-amber-500/30 rounded-lg text-amber-300 transition">
          <i class="fa-solid fa-arrows-split-up-and-left mr-1"></i>京NC6545 ⇄ 京C6545 (漏读N 错位对齐)
        </button>
        <button onclick="setPreset('粤BD12345', '粤B12345')" class="px-2.5 py-1 bg-emerald-950/40 hover:bg-emerald-900/40 border border-emerald-500/30 rounded-lg text-emerald-300 transition">
          <i class="fa-solid fa-leaf mr-1"></i>粤BD12345 ⇄ 粤B12345 (绿牌8位 ⇄ 蓝牌7位)
        </button>
        <button onclick="setPreset('粤B88888', '粤B8888B')" class="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-slate-300 transition">
          粤B88888 ⇄ 粤B8888B (8 vs B 混淆)
        </button>
        <button onclick="setPreset('苏E12345', '苏E12345')" class="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-slate-300 transition">
          苏E12345 ⇄ 苏E12345 (完全一致)
        </button>
        <button onclick="setPreset('沪A12345', '浙B67890')" class="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-slate-300 transition">
          沪A12345 ⇄ 浙B67890 (完全不同)
        </button>
      </div>

      <div class="mt-6">
        <button
          id="btnEvaluate"
          onclick="evaluatePlates()"
          class="w-full py-3.5 px-6 rounded-xl bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-500 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold shadow-lg shadow-blue-500/25 transition duration-200 flex items-center justify-center gap-2"
        >
          <i class="fa-solid fa-magnifying-glass-chart"></i>
          <span>智能对齐并评估同车可能性</span>
        </button>
      </div>
    </div>

    <!-- 结果卡片 -->
    <div id="resultCard" class="hidden mt-8 space-y-6">
      <div id="statusBanner" class="rounded-2xl p-6 border flex flex-col md:flex-row items-center justify-between gap-6 bg-slate-900 border-slate-800">
        <div class="flex items-center gap-4">
          <div id="statusIcon" class="w-14 h-14 rounded-2xl flex items-center justify-center text-2xl bg-blue-500/20 text-blue-400">
            <i class="fa-solid fa-car-side"></i>
          </div>
          <div>
            <div class="text-xs font-semibold uppercase tracking-wider text-slate-400">Laya 综合裁决</div>
            <div id="conclusionText" class="text-xl sm:text-2xl font-bold text-white mt-1">--</div>
            <div id="decisionDesc" class="text-xs text-slate-400 mt-1">--</div>
          </div>
        </div>
        
        <div class="flex items-center gap-4 bg-slate-800/80 px-6 py-4 rounded-xl border border-slate-700/60">
          <div class="text-right">
            <div class="text-xs text-slate-400">同一车辆可能性 (P)</div>
            <div id="probValue" class="text-3xl font-black text-blue-400 font-mono">0.0%</div>
          </div>
          <div class="text-xs text-slate-500 border-l border-slate-700 pl-4 space-y-1">
            <div>置信度: <span id="confValue" class="font-semibold text-slate-300">--</span></div>
            <div>推断耗时: <span id="latencyValue" class="font-semibold text-emerald-400">-- ms</span></div>
          </div>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-4">
          <h3 class="text-sm font-semibold text-slate-300 flex items-center gap-2">
            <i class="fa-solid fa-camera text-blue-400"></i> 卡口相机图像仿真预览
          </h3>
          <div class="flex flex-col sm:flex-row gap-4 items-center justify-around py-2">
            <div class="text-center">
              <div class="text-xs text-slate-400 mb-1">进口相机拍摄</div>
              <div id="previewIn" class="plate-badge-blue px-4 py-2 rounded-lg text-lg">京NC6545</div>
            </div>
            <div class="text-slate-500 text-xl"><i class="fa-solid fa-arrows-left-right"></i></div>
            <div class="text-center">
              <div class="text-xs text-slate-400 mb-1">出口相机拍摄</div>
              <div id="previewOut" class="plate-badge-blue px-4 py-2 rounded-lg text-lg">京NC0545</div>
            </div>
          </div>
        </div>

        <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-3">
          <h3 class="text-sm font-semibold text-slate-300 flex items-center gap-2">
            <i class="fa-solid fa-sliders text-indigo-400"></i> 动态规划对齐指标
          </h3>
          <div class="grid grid-cols-3 gap-2 text-center">
            <div class="bg-slate-800/60 p-2.5 rounded-lg">
              <div class="text-xs text-slate-400">自适应对齐匹配率</div>
              <div id="matchRatio" class="text-lg font-bold text-slate-200 font-mono">--</div>
            </div>
            <div class="bg-slate-800/60 p-2.5 rounded-lg">
              <div class="text-xs text-slate-400">加权编辑距离</div>
              <div id="editDist" class="text-lg font-bold text-slate-200 font-mono">--</div>
            </div>
            <div class="bg-slate-800/60 p-2.5 rounded-lg">
              <div class="text-xs text-slate-400">OCR错漏混淆特征</div>
              <div id="ocrConfusionTag" class="text-xs font-semibold text-slate-200 mt-1">--</div>
            </div>
          </div>
          <div id="confusionNoteBox" class="text-xs text-amber-300/90 bg-amber-500/10 border border-amber-500/20 p-2.5 rounded-lg hidden">
          </div>
        </div>
      </div>

      <!-- 智能对齐逐位展示 -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-5">
        <div class="flex items-center justify-between mb-3">
          <h3 class="text-sm font-semibold text-slate-300 flex items-center gap-2">
            <i class="fa-solid fa-diagram-project text-emerald-400"></i> 智能自适应对齐流（Levenshtein Alignment）
          </h3>
          <div class="flex items-center gap-3 text-[11px] text-slate-400">
            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded bg-emerald-500"></span>匹配</span>
            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded bg-red-500"></span>替换</span>
            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded bg-amber-500"></span>漏读(缺失)</span>
            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded bg-purple-500"></span>增读(误识)</span>
          </div>
        </div>
        <div id="charDiffTable" class="flex flex-wrap gap-2"></div>
      </div>

      <details class="bg-slate-900/60 border border-slate-800 rounded-xl p-4 text-xs">
        <summary class="cursor-pointer text-slate-400 hover:text-slate-200 font-mono flex items-center justify-between">
          <span><i class="fa-solid fa-code mr-1"></i>查看完整对齐细节与 Laya 原始决策结构 (JSON)</span>
          <i class="fa-solid fa-chevron-down text-xs"></i>
        </summary>
        <pre id="rawJson" class="mt-3 p-3 bg-slate-950 rounded-lg overflow-x-auto text-emerald-400 font-mono text-[11px] leading-relaxed"></pre>
      </details>
    </div>
  </div>

  <script>
    function setPreset(pIn, pOut) {
      document.getElementById('plateIn').value = pIn;
      document.getElementById('plateOut').value = pOut;
      evaluatePlates();
    }

    async function evaluatePlates() {
      const pIn = document.getElementById('plateIn').value.trim();
      const pOut = document.getElementById('plateOut').value.trim();
      const btn = document.getElementById('btnEvaluate');
      const resultCard = document.getElementById('resultCard');

      if (!pIn || !pOut) {
        alert('请输入完整的进出车牌！');
        return;
      }

      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>智能对齐与 Laya 推断中...</span>';

      try {
        const resp = await fetch('/api/match', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ plate_in: pIn, plate_out: pOut })
        });

        if (!resp.ok) throw new Error('请求错误: ' + resp.statusText);
        const data = await resp.json();

        renderResult(data);
        resultCard.classList.remove('hidden');
      } catch (err) {
        alert('比对推断失败: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-magnifying-glass-chart"></i> <span>智能对齐并评估同车可能性</span>';
      }
    }

    function renderResult(data) {
      document.getElementById('conclusionText').innerText = data.conclusion;
      document.getElementById('probValue').innerText = data.probability_same + '%';
      document.getElementById('confValue').innerText = data.confidence;
      document.getElementById('latencyValue').innerText = data.latency_ms + ' ms';
      document.getElementById('decisionDesc').innerText = 'Laya判定: ' + (data.decision_label || data.decision);

      const banner = document.getElementById('statusBanner');
      const icon = document.getElementById('statusIcon');
      banner.className = 'rounded-2xl p-6 border flex flex-col md:flex-row items-center justify-between gap-6 bg-slate-900 ';
      if (data.badge_color === 'emerald') {
        banner.classList.add('border-emerald-500/40', 'bg-emerald-950/20');
        icon.className = 'w-14 h-14 rounded-2xl flex items-center justify-center text-2xl bg-emerald-500/20 text-emerald-400';
      } else if (data.badge_color === 'blue') {
        banner.classList.add('border-blue-500/40', 'bg-blue-950/20');
        icon.className = 'w-14 h-14 rounded-2xl flex items-center justify-center text-2xl bg-blue-500/20 text-blue-400';
      } else if (data.badge_color === 'amber') {
        banner.classList.add('border-amber-500/40', 'bg-amber-950/20');
        icon.className = 'w-14 h-14 rounded-2xl flex items-center justify-center text-2xl bg-amber-500/20 text-amber-400';
      } else {
        banner.classList.add('border-red-500/40', 'bg-red-950/20');
        icon.className = 'w-14 h-14 rounded-2xl flex items-center justify-center text-2xl bg-red-500/20 text-red-400';
      }

      const isGreen = data.analysis.p_in.length === 8;
      const plateClass = isGreen ? 'plate-badge-green' : 'plate-badge-blue';
      document.getElementById('previewIn').className = `${plateClass} px-4 py-2 rounded-lg text-lg`;
      document.getElementById('previewOut').className = `${plateClass} px-4 py-2 rounded-lg text-lg`;
      document.getElementById('previewIn').innerText = data.analysis.p_in;
      document.getElementById('previewOut').innerText = data.analysis.p_out;

      document.getElementById('matchRatio').innerText = (data.analysis.matched_ratio * 100).toFixed(1) + '%';
      document.getElementById('editDist').innerText = (data.analysis.weighted_distance !== undefined ? data.analysis.weighted_distance : data.analysis.diff_count);
      
      const ocrTag = document.getElementById('ocrConfusionTag');
      const noteBox = document.getElementById('confusionNoteBox');
      if (data.analysis.has_known_ocr_confusion) {
        ocrTag.innerText = '检出特征混淆/位移';
        ocrTag.className = 'text-xs font-semibold text-amber-400 mt-1';
        noteBox.classList.remove('hidden');
        noteBox.innerHTML = '<i class="fa-solid fa-triangle-exclamation mr-1"></i> ' + data.analysis.confusion_notes.join('<br>');
      } else {
        ocrTag.innerText = data.analysis.diff_count === 0 ? '完全一致' : '非典型特征';
        ocrTag.className = 'text-xs font-semibold text-slate-400 mt-1';
        noteBox.classList.add('hidden');
      }

      // 对齐流渲染
      const table = document.getElementById('charDiffTable');
      table.innerHTML = '';
      data.analysis.diff_details.forEach((item, idx) => {
        const div = document.createElement('div');
        div.className = 'flex flex-col items-center p-2.5 rounded-xl bg-slate-800/80 border border-slate-700 min-w-[78px] transition hover:bg-slate-800';
        
        let opClass = 'char-match';
        let opLabel = '匹配';
        let badgeColor = 'text-emerald-400';

        if (item.op === 'replace') {
          opClass = 'char-replace';
          opLabel = '替换';
          badgeColor = 'text-red-400';
        } else if (item.op === 'delete') {
          opClass = 'char-delete';
          opLabel = '漏读(缺失)';
          badgeColor = 'text-amber-400';
        } else if (item.op === 'insert') {
          opClass = 'char-insert';
          opLabel = '增读(多识)';
          badgeColor = 'text-purple-400';
        }

        const cInDisplay = item.c_in ? item.c_in : '<span class="opacity-40">-</span>';
        const cOutDisplay = item.c_out ? item.c_out : '<span class="opacity-40">-</span>';

        div.innerHTML = `
          <div class="text-[10px] text-slate-400 mb-1.5">位 #${idx + 1}</div>
          <div class="font-mono text-sm font-bold px-2 py-1 rounded-lg ${opClass} flex items-center gap-1">
            <span>${cInDisplay}</span>
            <span class="text-[10px] opacity-60">⇄</span>
            <span>${cOutDisplay}</span>
          </div>
          <div class="text-[10px] mt-1.5 ${badgeColor} font-semibold text-center">${opLabel}</div>
        `;
        table.appendChild(div);
      });

      document.getElementById('rawJson').innerText = JSON.stringify({
        alignment_analysis: data.analysis,
        laya_output: data.raw_laya
      }, null, 2);
    }

    window.addEventListener('DOMContentLoaded', () => {
      evaluatePlates();
    });
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
