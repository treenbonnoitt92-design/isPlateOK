import argparse
import json
import sys
import time
from typing import Any, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")



def build_state_and_questions(plate_in: str, plate_out: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    # Specific preset for classic 京NC6545 vs 京NC0545
    if plate_in == "京NC6545" and plate_out == "京NC0545":
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
        q_instructions = "Is it likely that the entrance plate '京NC6545' and exit plate '京NC0545' belong to the same vehicle due to camera OCR misrecognition between '6' and '0'?"
        choice_criteria = {
            "same_vehicle_ocr_mismatch": "Same vehicle, difference caused by OCR confusion on similar characters (6 vs 0)",
            "distinct_different_vehicles": "Different vehicles entirely, plate difference is genuine"
        }
    else:
        total_len = max(len(plate_in), len(plate_out))
        matched_chars = sum(1 for a, b in zip(plate_in, plate_out) if a == b)
        differing = [f"index {i}: '{a}' vs '{b}'" for i, (a, b) in enumerate(zip(plate_in, plate_out)) if a != b]
        diff_str = ", ".join(differing) if differing else "Exact match"

        state = {
            "entrance_plate": plate_in,
            "exit_plate": plate_out,
            "match_analysis": {
                "total_length": total_len,
                "matched_chars": matched_chars,
                "differing_pairs": differing,
            },
            "description": f"A vehicle entered identified as '{plate_in}' and exited identified as '{plate_out}'. {matched_chars}/{total_len} characters match. Differences: {diff_str}."
        }
        q_instructions = f"Is it likely that the entrance plate '{plate_in}' and exit plate '{plate_out}' belong to the same vehicle despite character differences?"
        choice_criteria = {
            "same_vehicle_ocr_mismatch": "Same vehicle, difference caused by OCR confusion on similar characters or noise",
            "distinct_different_vehicles": "Different vehicles entirely, plate difference is genuine"
        }

    questions = {
        # 布尔判断：是否极有可能是同一辆车 (noul 返回 True 的校准概率)
        "is_same_vehicle": {
            "type": "noul",
            "instructions": q_instructions
        },
        # 决策分类：同一辆车(OCR识别误差) vs 不同车辆
        "plate_match_decision": {
            "type": "choice",
            "instructions": "Classify the relationship between the entrance and exit license plate records.",
            "criteria": choice_criteria
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
    return state, questions


def print_decision_summary(result: Dict[str, Any], latency_ms: float):
    answers = result.get("answers", {})
    is_same = answers.get("is_same_vehicle", {})
    decision = answers.get("plate_match_decision", {})
    conf_score = answers.get("match_confidence_score", {})

    print("\n=== 结论汇总 ===")
    conf_str = f" (置信度: {is_same['confidence']})" if "confidence" in is_same else ""
    print(f"1. 同车概率 (noul P(True)): {is_same.get('noul', 0) * 100:.2f}%{conf_str}")
    print(f"2. 分类判定: {decision.get('choice')} (各类别概率: {decision.get('probabilities')})")
    print(f"3. 相似度评分: {conf_score.get('score', 0):.2f} / 3.0")
    print(f"4. 决策耗时: {latency_ms:.2f} ms")

    usage = result.get("usage")
    if usage:
        print("\n=== OpenRouter Usage / Cost ===")
        if "input_tokens" in usage:
            print(f"- 输入 Tokens: {usage.get('input_tokens')}")
        if "output_tokens" in usage:
            print(f"- 输出 Tokens: {usage.get('output_tokens')}")
        if "total_tokens" in usage:
            print(f"- 总计 Tokens: {usage.get('total_tokens')}")
        if "cost" in usage:
            print(f"- 推理花费: ${usage.get('cost'):.6f}")


def evaluate_plates(model: str = "jev", plate_in: str = "京NC6545", plate_out: str = "京NC0545"):
    state, questions = build_state_and_questions(plate_in, plate_out)

    print("[*] 正在执行车牌匹配决策...")
    print(f"[*] 模型引擎: {model.upper()}")
    print(f"[*] 进场车牌: {plate_in} | 出场车牌: {plate_out}\n")

    t0 = time.perf_counter()
    if model == "jev":
        try:
            from jev_client import JevClient
            client = JevClient()
        except RuntimeError as e:
            print(f"[!] 初始化 JevClient 失败: {e}", file=sys.stderr)
            print("[!] 请确保设置了 OPENROUTER_API_KEY 环境变量或在 .env 文件中配置。", file=sys.stderr)
            raise

        result = client.predict(state, questions)
        latency_ms = (time.perf_counter() - t0) * 1000

        print("=== Jev 模型决策结果 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print_decision_summary(result, latency_ms)

    elif model == "laya":
        try:
            import laya
        except ImportError as e:
            print("[!] 本地 Laya 库未安装。请运行 `pip install laya torch` 进行安装，或使用默认的 Jev 云端引擎 (`--model jev`)。", file=sys.stderr)
            raise

        print("[*] 正在加载 Laya 模型 (convaiinnovations/laya)...")
        agent = laya.load("convaiinnovations/laya")
        result = agent.predict(state, questions)
        latency_ms = (time.perf_counter() - t0) * 1000

        print("=== Laya 模型决策结果 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print_decision_summary(result, latency_ms)
    else:
        raise ValueError(f"Unknown model: {model}")


def main():
    parser = argparse.ArgumentParser(
        description="License plate matching using Jev or Laya non-autoregressive decision models."
    )
    parser.add_argument(
        "--model",
        choices=["jev", "laya"],
        default="jev",
        help="Decision engine to use: 'jev' (OpenRouter cloud model, default) or 'laya' (local model)."
    )
    parser.add_argument(
        "--plate-in",
        default="京NC6545",
        help="Entrance license plate (default: '京NC6545')."
    )
    parser.add_argument(
        "--plate-out",
        default="京NC0545",
        help="Exit license plate (default: '京NC0545')."
    )
    args = parser.parse_args()

    try:
        evaluate_plates(
            model=args.model,
            plate_in=args.plate_in,
            plate_out=args.plate_out
        )
    except KeyboardInterrupt:
        print("\n[!] 操作已取消")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] 决策执行失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
