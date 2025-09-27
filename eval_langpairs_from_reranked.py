#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[EN]
From LSST4's reranked.jsonl and labels.jsonl, compute per language-pair:
- Best-F1 (τ_F1)
- PR-AUC (ranking quality)
- (optional) Precision/Recall/F1 at a fixed τ

Inputs (required keys)
- labels.jsonl : {"u","v","label"}
  * Pairs are matched as unordered: sort endpoints (min, max) before joining.
- reranked.jsonl :
  * Old format: {"u","v","score"}
  * New format: {"q","nb","score", ...} or {"q","nb","score_sem", ...}
    - If "score" exists, use it; otherwise fall back to "score_sem".

Language grouping
- Extract <Language> from endpoint strings of the form:
    "project/<Language>/task_.../id_...::<func>"
  Then group and evaluate by language **pair**.

Example
  python eval_langpairs_from_reranked.py \
    --labels   lsst4_out/labels.jsonl \
    --reranked lsst4_out/reranked.jsonl \
    --pairs "C++&C#" "C++&Java" "C++&Python" "C#&Java" "C#&Python" "Java&Python" \
    --tau_fixed ""

Notes
- Provide multiple --pairs as quoted strings like "LangA&LangB".
- Set --tau_fixed to an empty string to disable fixed-τ evaluation (Best-F1 and PR-AUC only).
"""
"""
[JA]
LSST4 の reranked.jsonl と labels.jsonl から、言語ペアごとに
Best-F1（τ_F1）と PR-AUC（ランキング品質）、必要なら固定τでの P/R/F1 を算出する。

前提キー:
  labels.jsonl : {"u","v","label"}   ※ unordered（小さい方, 大きい方）で突合
  reranked.jsonl:
    - 旧: {"u","v","score"}
    - 新: {"q","nb","score", ...} or {"q","nb","score_sem", ...}
         ※ score があれば優先、無ければ score_sem を使用

言語抽出:
  "project/<Language>/task_.../id_...::<func>" の <Language> を抽出してペアにグルーピング

例:
  python eval_langpairs_from_reranked.py \
    --labels   lsst4_out/labels.jsonl \
    --reranked lsst4_out/reranked.jsonl \
    --pairs "C++&C#" "C++&Java" "C++&Python" "C#&Java" "C#&Python" "Java&Python" \
    --tau_fixed ""
"""

import argparse, json, re, sys
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

# ---- 言語名の正規化（大文字小文字・別名を吸収） ----
def norm_lang_name(s: str) -> str:
    x = (s or "").strip().lower()
    alias = {
        "c++": "C++", "cpp":"C++",
        "c#": "C#", "csharp":"C#",
        "java":"Java",
        "python":"Python","py":"Python",
        "go":"Go",
        "rust":"Rust","rs":"Rust",
        "c":"C",
    }
    return alias.get(x, s.strip() or x)

# u/v（q/nb）から <Language> を取り出す： "project/<Language>/..." を拾う
LANG_RE = re.compile(r'^[^/]+/([^/]+)/')

def get_lang_from_key(k: str) -> str:
    m = LANG_RE.match(k)
    if not m:
        return ""
    return norm_lang_name(m.group(1))

# ---- JSONL ロード ----
def load_labels(path: str) -> Dict[Tuple[str,str], int]:
    lab = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            o = json.loads(ln)
            u, v = str(o["u"]), str(o["v"])
            key = (u, v) if u <= v else (v, u)
            lab[key] = int(o["label"])
    return lab

def load_scores_with_lang(path: str) -> List[Tuple[Tuple[str,str], float, str, str]]:
    """
    reranked.jsonl を読み、(unordered_key, score, lang_u, lang_v) のリストを返す。
    - フィールド優先順位: score > score_sem
    - 旧形式(u,v)、新形式(q,nb)の両対応
    - 重複キーは最大スコアを採用（上書き）
    """
    best: Dict[Tuple[str,str], Tuple[float,str,str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            o = json.loads(ln)
            u = o.get("u") or o.get("q")
            v = o.get("v") or o.get("nb")
            if not u or not v: 
                continue
            u, v = str(u), str(v)
            key = (u, v) if u <= v else (v, u)
            s = o.get("score")
            if s is None:
                s = o.get("score_sem")
            if s is None:
                continue
            s = float(s)
            lu, lv = get_lang_from_key(u), get_lang_from_key(v)
            if not lu or not lv:
                continue
            if key in best:
                if s > best[key][0]:
                    best[key] = (s, lu, lv)
            else:
                best[key] = (s, lu, lv)
    return [(k, v[0], v[1], v[2]) for k, v in best.items()]

# ---- 指標計算 ----
def compute_pr(labels: np.ndarray, scores: np.ndarray):
    order = np.argsort(-scores)
    y = labels[order]; s = scores[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall    = tp / max(int((y == 1).sum()), 1)
    thr = s
    return precision, recall, thr

def pr_auc(p: np.ndarray, r: np.ndarray) -> float:
    if len(p) < 2: return 0.0
    return float(np.trapz(p, r))

def best_f1(labels: np.ndarray, scores: np.ndarray) -> Dict:
    order = np.argsort(-scores)
    s = scores[order]; y = labels[order]
    tp = np.cumsum(y == 1)
    pos = int((y == 1).sum())
    change = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    best = {"tau": float("nan"), "f1": -1.0, "p": 0.0, "r": 0.0, "tp": 0, "fp": 0, "fn": pos}
    for idx in change:
        tpi = int(tp[idx]); pred = idx + 1
        fpi = pred - tpi; fni = pos - tpi
        P = tpi / max(tpi + fpi, 1); R = tpi / max(pos, 1)
        F1 = 2*P*R / max(P+R, 1e-9)
        if F1 > best["f1"]:
            best.update({"tau": float(s[idx]), "f1": F1, "p": P, "r": R, "tp": tpi, "fp": fpi, "fn": fni})
    return best

def eval_at_tau(labels: np.ndarray, scores: np.ndarray, tau: float) -> Dict:
    yhat = (scores >= tau).astype(int)
    y = labels
    tp = int(((y==1)&(yhat==1)).sum())
    fp = int(((y==0)&(yhat==1)).sum())
    fn = int(((y==1)&(yhat==0)).sum())
    P = tp / max(tp+fp, 1); R = tp / max(tp+fn, 1)
    F1 = 2*P*R / max(P+R, 1e-9)
    return {"p":P, "r":R, "f1":F1, "tp":tp, "fp":fp, "fn":fn}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--reranked", required=True)
    ap.add_argument("--pairs", nargs="+", required=True,
                    help='対象の言語ペア。例: "C++&C#" "C++&Java" "Java&Python" など（順不同）')
    ap.add_argument("--tau_fixed", default="", help="固定しきい値（空なら出力しない）")
    args = ap.parse_args()

    # 目標ペアを正規化
    want = set()
    for p in args.pairs:
        try:
            a,b = p.split("&")
        except ValueError:
            print(f"[warn] ignore malformed pair: {p}", file=sys.stderr)
            continue
        a = norm_lang_name(a); b = norm_lang_name(b)
        key = tuple(sorted([a,b]))
        want.add(key)

    # データロード
    labmap = load_labels(args.labels)
    scored = load_scores_with_lang(args.reranked)

    # ペア別に集約
    buckets: Dict[Tuple[str,str], List[Tuple[int, float]]] = {}
    for (k, s, lu, lv) in scored:
        langs = tuple(sorted([lu, lv]))
        if langs not in want:
            continue
        y = labmap.get(k, 0)  # ラベルに無いペアは 0 扱い
        buckets.setdefault(langs, []).append((y, s))

    print("=== Per language-pair results (Best-F1 and PR-AUC) ===")
    result = {}
    tau_fixed = None if args.tau_fixed=="" else float(args.tau_fixed)
    for pair in sorted(list(want)):
        rows = buckets.get(pair, [])
        if not rows:
            print(f"{pair}: (no pairs)")
            continue
        y = np.array([r[0] for r in rows], dtype=np.int32)
        s = np.array([r[1] for r in rows], dtype=np.float64)
        P, R, _ = compute_pr(y, s)
        ap = pr_auc(P, R)
        best = best_f1(y, s)

        line = (f"{pair}: PR-AUC={ap:.4f}  "
                f"Best-F1 τ={best['tau']:.6f} F1={best['f1']:.4f} P={best['p']:.3f} R={best['r']:.3f} "
                f"(TP={best['tp']} FP={best['fp']} FN={best['fn']}, n={len(rows)})")
        print(line)

        out = {
            "pr_auc": float(ap),
            "tau_f1": float(best["tau"]),
            "f1_max": float(best["f1"]),
            "precision_at_tau_f1": float(best["p"]),
            "recall_at_tau_f1": float(best["r"]),
            "tp_at_tau_f1": int(best["tp"]),
            "fp_at_tau_f1": int(best["fp"]),
            "fn_at_tau_f1": int(best["fn"]),
            "n_pairs": int(len(rows))
        }
        if tau_fixed is not None:
            fx = eval_at_tau(y, s, tau_fixed)
            print(f"  Fixed-τ({tau_fixed:.6f}): F1={fx['f1']:.4f} P={fx['p']:.3f} R={fx['r']:.3f} "
                  f"(TP={fx['tp']} FP={fx['fp']} FN={fx['fn']})")
            out.update({
                "tau_fixed": float(tau_fixed),
                "f1_at_tau_fixed": float(fx["f1"]),
                "precision_at_tau_fixed": float(fx["p"]),
                "recall_at_tau_fixed": float(fx["r"]),
                "tp_at_tau_fixed": int(fx["tp"]),
                "fp_at_tau_fixed": int(fx["fp"]),
                "fn_at_tau_fixed": int(fx["fn"]),
            })
        result[" & ".join(pair)] = out

    out_json = Path(args.reranked).with_name("results_by_langpair.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\n[done] wrote:", out_json)

if __name__ == "__main__":
    main()