import json
import laya

def evaluate_plates():
    # 1. 加载 Laya Agent (或使用 Router)
    agent = laya.load("convaiinnovations/laya")

    # 2. 构造输入状态 (State)
    # 进口：京NC6545，出口：京NC0545
    # 场景分析：两车牌前3位（京NC）和后3位（545）完全一致，仅第4位 '6' 与 '0' 存在差异。
    # 在车牌相机OCR识别中，'6' 与 '0' 属于典型的形状混淆字符对。
    state = {
        "entrance_plate": "京NC6545",
        "exit_plate": "京NC0545",
        "match_analysis": {
            "total_length": 7,
            "matched_chars": 6,
            "differing_index": 3,
            "differing_pair": ("6", "0"),
            "ocr_confusion_risk": "high (digits '6' and '0' have high visual similarity in optical character recognition under occlusion or glare)",
        },
        "description": "A vehicle entered a parking facility identified as '京NC6545' and exited identified as '京NC0545'. 6 out of 7 characters match exactly. The only difference is digit '6' vs digit '0', which is a classic camera OCR misidentification."
    }

    # 3. 定义结构化决策问题 (Questions)
    # Laya 支持 choice (多选分类), noul (布尔判断 P(true)), score (分级评分)
    questions = {
        # 布尔判断：是否极有可能是同一辆车 (noul 返回 True 的校准概率)
        "is_same_vehicle": {
            "type": "noul",
            "instructions": "Is it likely that the entrance plate '京NC6545' and exit plate '京NC0545' belong to the same vehicle due to camera OCR misrecognition between '6' and '0'?"
        },
        # 决策分类：同一辆车(OCR识别误差) vs 不同车辆
        "plate_match_decision": {
            "type": "choice",
            "instructions": "Classify the relationship between the entrance and exit license plate records.",
            "criteria": {
                "same_vehicle_ocr_mismatch": "Same vehicle, difference caused by OCR confusion on similar characters (6 vs 0)",
                "distinct_different_vehicles": "Different vehicles entirely, plate difference is genuine"
            }
        },
        # 匹配置信度打分 (0-3分)
        "match_confidence_score": {
            "type": "score",
            "instructions": "Rate the likelihood that these two records represent the exact same car on a 4-level scale.",
            "criteria": [
                "level_0: definitely different cars",
                "level_1: unlikely to be the same car",
                "level_2: plausible / possible same car",
                "level_3: highly probable same car with single-character OCR error"
            ]
        }
    }

    # 4. 执行单次前向推理 (Single forward pass, 非自回归极速决策)
    result = agent.predict(state, questions)

    print("=== Laya 模型决策结果 ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    answers = result.get("answers", {})
    is_same = answers.get("is_same_vehicle", {})
    decision = answers.get("plate_match_decision", {})
    conf_score = answers.get("match_confidence_score", {})

    print("\n=== 结论汇总 ===")
    print(f"1. 同车概率 (noul P(True)): {is_same.get('noul', 0) * 100:.2f}% (置信度: {is_same.get('confidence', 0)})")
    print(f"2. 分类判定: {decision.get('choice')} (各类别概率: {decision.get('probabilities')})")
    print(f"3. 相似度评分: {conf_score.get('score', 0):.2f} / 3.0")

if __name__ == "__main__":
    evaluate_plates()
