#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[EN]
Repack old candidates into the new semantic format using embeddings and func_list.

Goal
- Join candidates.jsonl (old format: {"u","v"} or {"q","nb"}) with:
  - func_emb.pkl  (mapping: canonical_key -> embedding vector)
  - func_list.jsonl (contains absolute paths)
- Output the new format:
    {"q": <canonical_key>, "nb": <canonical_key>, "score_sem": <float>, "rank": <int>}

Canonical key
- From each record in func_list.jsonl, take the substring of `file` **after "/src/"**,
  and build:  "project/<rel>::func".
- If an embedding key already uses this canonical form, keep it as-is.
  If not, resolve it via the mapping derived from func_list.jsonl.

Scoring & ranking
- score_sem = cosine similarity = dot( L2_norm(emb[q]), L2_norm(emb[nb]) ).
- For each query q, sort neighbors by score_sem descending and assign 1-based `rank`.

Inputs
- --emb        : path to func_emb.pkl
- --func-list  : path to func_list.jsonl (with absolute file paths)
- --cands-in   : path to old candidates JSONL ({"u","v"} or {"q","nb"})
- --cands-out  : path to write the new-format JSONL

Usage
  python repack_candidates_sem.py \
    --emb       lsst4_pairs_bench/func_emb.pkl \
    --func-list lsst4_pairs_bench/func_list.jsonl \
    --cands-in  lsst4_pairs_bench/candidates_old.jsonl \
    --cands-out lsst4_pairs_bench/candidates.jsonl
"""
"""
[JA]
candidates.jsonl（旧形式 {"u","v"} or {"q","nb"}）と
func_emb.pkl（埋め込み辞書）・func_list.jsonl（絶対パスを含む）を突き合わせて，
新フォーマット {"q","nb","score_sem","rank"} を出力する。

ポイント：
- func_list.jsonl の file から "/src/" 以降を取り，"project/<rel>::func" を作る（正規キー）
- emb のキーがすでに正規キーならそのまま，違っていても func_list に基づくマップで解決
- score_sem は L2 正規化後の内積（= cosine）
- rank は q ごとにスコア降順で 1.. を付与
python repack_candidates_sem.py --emb lsst4_pairs_bench/func_emb.pkl --func-list lsst4_pairs_bench/func_list.jsonl --cands-in lsst4_pairs_bench/candidates_old.jsonl --cands-out lsst4_pairs_bench/candidates.jsonl
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pickle

def l2norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x)
    return x / n if n > 0 else x

def load_emb(emb_pkl: Path) -> Dict[str, np.ndarray]:
    emb: Dict[str, np.ndarray] = pickle.loads(emb_pkl.read_bytes())
    # ここで L2 正規化しておく
    for k in list(emb.keys()):
        emb[k] = l2norm(emb[k])
    return emb

def build_candkey_map(func_list: Path) -> Dict[str, str]:
    """
    func_list.jsonl の各行から正規キー "project/<relpath_under_src>::func" を生成し，
    それを "project/<rel>::func" として返す（値は同じ文字列；将来の拡張に備え dict 返却）。
    """
    cand_keys = {}
    with func_list.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: 
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            proj = str(o.get("project","")).strip()
            file_abs = str(o.get("file","")).strip()
            func = str(o.get("func","")).strip()
            if not (proj and file_abs and func):
                continue
            # "/src/" 以降を相対パスに（Windows の場合は小文字化して探す）
            s = file_abs
            idx = s.lower().find("/src/")
            if idx >= 0:
                rel = s[idx+len("/src/"):]
            else:
                # フォールバック：プロジェクト名の後ろからにする（最終手段）
                rel = Path(s).name
            rel = rel.replace("\\","/")  # 一応
            cand = f"{proj}/{rel}::{func}"
            cand_keys[cand] = cand
    return cand_keys

def iter_cands(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            u = o.get("q") or o.get("u")
            v = o.get("nb") or o.get("v")
            if not u or not v:
                continue
            yield str(u), str(v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, type=Path, help="func_emb.pkl")
    ap.add_argument("--func-list", required=True, type=Path, help="func_list.jsonl（絶対パス入り）")
    ap.add_argument("--cands-in", required=True, type=Path, help="旧 candidates.jsonl（u/v でも q/nb でも可）")
    ap.add_argument("--cands-out", required=True, type=Path, help="新 candidates.jsonl（q,nb,score_sem,rank）")
    args = ap.parse_args()

    # 1) emb 読み込み & L2 正規化
    emb = load_emb(args.emb)
    emb_keys = set(emb.keys())

    # 2) func_list から「正規キー」を構築（"project/<rel_under_src>::func"）
    candkey_map = build_candkey_map(args.func_list)

    # 3) 旧 candidates を走査 → 正規キーへ正規化 → スコア計算
    per_q: Dict[str, List[Tuple[str, float]]] = {}
    n_total = n_scored = n_miss = 0

    for uq, unb in iter_cands(args.cands_in):
        n_total += 1
        # 旧候補は既に正規キーの形（project/Language/task_..../id_xxx.ext::func）
        # ここではそのまま使えるが、万一ズレている場合に備えて func_list に存在するキーのみ採用
        # （存在しない場合はそのまま試す）
        q_key = candkey_map.get(uq, uq)
        nb_key = candkey_map.get(unb, unb)

        vq = emb.get(q_key)
        vn = emb.get(nb_key)
        if vq is None or vn is None:
            # emb 側が別表記のキーを持つ可能性に備え，末尾一致でのフォールバックを試す
            # （コストを抑えるため簡易：::func を見て key を絞る）
            def fallback_lookup(k: str):
                if k in emb: 
                    return emb[k]
                if "::" in k:
                    func = k.split("::",1)[1]
                    # 最後の N 文字でマッチ（衝突は稀な想定）
                    suffix = "::"+func
                    for ek in emb_keys:
                        if ek.endswith(suffix):
                            return emb[ek]
                return None
            if vq is None: vq = fallback_lookup(q_key)
            if vn is None: vn = fallback_lookup(nb_key)

        if vq is None or vn is None:
            n_miss += 1
            continue

        score = float(np.dot(vq, vn))  # cosine (L2 norm済み)
        per_q.setdefault(q_key, []).append((nb_key, score))
        n_scored += 1

    # 4) q ごとにスコア降順で rank を振って書き出し
    args.cands_out.parent.mkdir(parents=True, exist_ok=True)
    with args.cands_out.open("w", encoding="utf-8") as fo:
        for q, lst in per_q.items():
            lst.sort(key=lambda x: x[1], reverse=True)
            for rank, (nb, s) in enumerate(lst, start=1):
                rec = {"q": q, "nb": nb, "score_sem": s, "rank": rank}
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[ok] wrote: {args.cands_out}")
    print(f"[stat] pairs_in={n_total}  scored={n_scored}  missing_emb={n_miss}")

if __name__ == "__main__":
    main()