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

def align_plates(s1: str, s2: str):
    s1 = s1.strip().upper()
    s2 = s2.strip().upper()
    m, n = len(s1), len(s2)
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
            note = '字符完全一致'
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
            note = f"出口相机漏读字符 [{c_in}]"
            confusion_notes.append(f"位移漏读：出口相机漏识别 '{c_in}'")
        elif op == 'insert':
            diff_count += 1
            note = f"出口相机多识别字符 [{c_out}]"
            confusion_notes.append(f"位移增读：出口相机误增识别 '{c_out}'")

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

if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    for p1, p2 in [('京NC6545', '京NC0545'), ('京NC6545', '京C6545'), ('粤BD12345', '粤B12345'), ('沪A12345', '浙B67890')]:
        res = align_plates(p1, p2)
        print(f"\n{p1} vs {p2} -> diff_count={res['diff_count']}, match_ratio={res['matched_ratio']}, weighted_dist={res['weighted_distance']}")
        for d in res['diff_details']:
            print(f"  [{d['op'].upper():7}] in: '{d['c_in']}' <-> out: '{d['c_out']}' | {d['note']}")
