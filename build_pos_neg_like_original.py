#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build positive/negative pairs like the original code:
- Positive (train_pos): (Code1, Code2) from the same Task.
- Negative (eval_pairs): for each anchor, pick the first example with a different Task (1 per anchor),
  and emit both positives and negatives as labeled pairs.

Input: pairs_test.jsonl with fields:
  Task, ID1, Category1, Code1, ID2, Category2, Code2

Outputs (under --out_dir):
  - train_pos.jsonl : positives only (same Task)
  - eval_pairs.jsonl: positives + negatives (1 neg/anchor by default)

  # 正例（train_pos）と評価用の正負（eval_pairs）を作成
python build_pos_neg_like_original.py \
  --pairs_jsonl /path/to/pairs_test.jsonl \
  --out_dir out_pairs \
  --neg-per-anchor 1
"""

import json, argparse, random, os
from pathlib import Path
from typing import List, Dict, Tuple, Any

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            rows.append(json.loads(ln))
    return rows

def normalize_code(s: str) -> str:
    # original code normalizes by ' '.join(code.replace('\n',' ').strip().split())
    return ' '.join((s or "").replace('\n', ' ').strip().split())

def build_pos_list(items: List[dict]) -> List[dict]:
    """同一 Task の Code1–Code2 を正例として列挙（学習の正例相当）。"""
    out = []
    for o in items:
        pos = {
            "task": o["Task"],
            "lang_a": o.get("Category1",""),
            "lang_b": o.get("Category2",""),
            "id_a":   o.get("ID1",""),
            "id_b":   o.get("ID2",""),
            "code_a": normalize_code(o.get("Code1","")),
            "code_b": normalize_code(o.get("Code2","")),
            "label":  1
        }
        out.append(pos)
    return out

def build_eval_pairs_like_original(items: List[dict], neg_per_anchor: int = 1, seed: int = 42) -> List[dict]:
    """
    元コードの評価ロジックに合わせて、
    - cos_right:  同一行の (Code1, Code2) を正例として1本
    - cos_wrong:  異 Task の最初に見つかった1本だけ（break）を負例として追加
    を各アンカー i について作る。neg_per_anchor > 1 の場合は複数サンプル。
    """
    rng = random.Random(seed)
    # 事前に正規化
    exs = []
    for o in items:
        exs.append({
            "task": o["Task"],
            "lang1": o.get("Category1",""),
            "lang2": o.get("Category2",""),
            "id1":   o.get("ID1",""),
            "id2":   o.get("ID2",""),
            "code1": normalize_code(o.get("Code1","")),
            "code2": normalize_code(o.get("Code2","")),
        })
    n = len(exs)
    out = []

    # 正例（cos_right相当）
    for i in range(n):
        e = exs[i]
        out.append({
            "task_a": e["task"], "task_b": e["task"],
            "lang_a": e["lang1"], "lang_b": e["lang2"],
            "id_a":   e["id1"],   "id_b":   e["id2"],
            "code_a": e["code1"], "code_b": e["code2"],
            "label":  1
        })

    # 負例（cos_wrong相当）— 異 Task を1本（または K 本）だけ
    for i in range(n):
        e = exs[i]
        # 候補インデックスを作る（異 Task のみ）
        cand = [j for j in range(n) if j != i and exs[j]["task"] != e["task"]]
        if not cand:
            continue
        if neg_per_anchor <= 1:
            # 元コードは「最初に見つかった1本」で break する
            j = cand[0]
            e2 = exs[j]
            out.append({
                "task_a": e["task"], "task_b": e2["task"],
                "lang_a": e["lang1"], "lang_b": e2["lang1"],  # Code1 vs Code1 として1本
                "id_a":   e["id1"],   "id_b":   e2["id1"],
                "code_a": e["code1"], "code_b": e2["code1"],
                "label":  0
            })
        else:
            # 追加で K 本サンプリング（繰り返しなし）
            rng.shuffle(cand)
            for j in cand[:neg_per_anchor]:
                e2 = exs[j]
                out.append({
                    "task_a": e["task"], "task_b": e2["task"],
                    "lang_a": e["lang1"], "lang_b": e2["lang1"],
                    "id_a":   e["id1"],   "id_b":   e2["id1"],
                    "code_a": e["code1"], "code_b": e2["code1"],
                    "label":  0
                })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--neg-per-anchor", type=int, default=1, help="各アンカーの負例本数（元コード準拠は1）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    items = load_jsonl(args.pairs_jsonl)

    # 学習用 正例（in-batch 負例は学習時に作る前提）
    pos_list = build_pos_list(items)
    with open(Path(args.out_dir)/"train_pos.jsonl", "w", encoding="utf-8") as f:
        for o in pos_list:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    # 評価用 正負ペア（cos_right/cos_wrong 方式）
    eval_pairs = build_eval_pairs_like_original(items, neg_per_anchor=args.neg_per_anchor, seed=args.seed)
    # 既定ではアンカーごとに 正1 + 負1 の2行ずつ
    with open(Path(args.out_dir)/"eval_pairs.jsonl", "w", encoding="utf-8") as f:
        for o in eval_pairs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(f"[done] train_pos.jsonl  -> {Path(args.out_dir)/'train_pos.jsonl'}  (pos={len(pos_list)})")
    print(f"[done] eval_pairs.jsonl -> {Path(args.out_dir)/'eval_pairs.jsonl'} (pairs={len(eval_pairs)})")
    print("      (per-anchor negatives =", args.neg_per_anchor, ")")

if __name__ == "__main__":
    main()