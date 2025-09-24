#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labels.jsonl を読み、言語ペアごとに不足している負例（label=0）を補完して
labels_balanced.jsonl を出力する。
- 既存の正例(label=1)はそのまま残す
- 負例は u/v の task_XXX を解析し、"異なる task" を組み合わせて生成
- unordered key（(min,max)）で重複排除

使い方:
  python rebalance_labels_by_langpair.py --labels labels.jsonl --out labels_balanced.jsonl --ratio 1.0
    # ratio=1.0 は「正例数に対して負例数を同数に揃える」（1:1）
"""
from __future__ import annotations
import argparse, json, re, random
from pathlib import Path
from typing import Dict, Tuple, List, Set
random.seed(42)

LANG_RE = re.compile(r'^[^/]+/([^/]+)/')       # project/<Lang>/...
TASK_RE = re.compile(r'/task_(\d+)/')          # /task_23/
def lang_of(key: str) -> str:
    m = LANG_RE.match(key); return m.group(1) if m else ""
def task_of(key: str) -> str:
    m = TASK_RE.search(key); return m.group(1) if m else ""

def unordered(u: str, v: str) -> Tuple[str,str]:
    return (u, v) if u <= v else (v, u)

def load_labels(path: Path):
    rows=[]
    with path.open("r",encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            o=json.loads(ln); rows.append(o)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ratio", type=float, default=1.0, help="負例/正例の目標比（既定=1.0: 1:1）")
    args = ap.parse_args()

    rows = load_labels(args.labels)

    # 既存のデータを集計
    pos_by_pair: Dict[Tuple[str,str], List[Tuple[str,str]]] = {}
    neg_by_pair: Dict[Tuple[str,str], Set[Tuple[str,str]]] = {}
    # 言語×task ごとのキー集合を作っておく（負例生成に使う）
    pool_by_lang_task: Dict[Tuple[str,str], Set[str]] = {}

    for o in rows:
        u, v = o["u"], o["v"]; y = int(o["label"])
        a, b = sorted([lang_of(u), lang_of(v)])
        key = unordered(u, v)
        # pool
        tu, tv = task_of(u), task_of(v)
        pool_by_lang_task[(lang_of(u), tu)] = pool_by_lang_task.get((lang_of(u), tu), set()); pool_by_lang_task[(lang_of(u), tu)].add(u)
        pool_by_lang_task[(lang_of(v), tv)] = pool_by_lang_task.get((lang_of(v), tv), set()); pool_by_lang_task[(lang_of(v), tv)].add(v)
        # pos/neg
        if y == 1:
            pos_by_pair.setdefault((a,b), []).append(key)
        else:
            neg_by_pair.setdefault((a,b), set()).add(key)

    # 補完
    new_rows = list(rows)
    added = 0

    for pair, pos_list in pos_by_pair.items():
        a, b = pair
        cur_neg = neg_by_pair.get(pair, set())
        need = max(0, int(round(len(pos_list) * args.ratio)) - len(cur_neg))
        if need <= 0:
            continue

        # 生成：ランダムに「異 task 同士」を選んで組み合わせる
        # 言語 a 側の全タスクと、言語 b 側の全タスクを列挙
        tasks_a = sorted({task_of(u) for (u,_) in pos_list} | {t for (la,t) in pool_by_lang_task if la == a})
        tasks_b = sorted({task_of(v) for (_,v) in pos_list} | {t for (lb,t) in pool_by_lang_task if lb == b})
        # ペアを作る（異 task のみ）
        cand_pairs: List[Tuple[str,str]] = []
        for ta in tasks_a:
            pool_a = list(pool_by_lang_task.get((a, ta), set()))
            if not pool_a: continue
            for tb in tasks_b:
                if ta == tb:  # 異 task 条件
                    continue
                pool_b = list(pool_by_lang_task.get((b, tb), set()))
                if not pool_b: continue
                # 直積からいくつかサンプリング（大きすぎると重いのでランダム抽出）
                for _ in range(10):
                    ua = random.choice(pool_a)
                    vb = random.choice(pool_b)
                    cand_pairs.append(unordered(ua, vb))

        # 既存 neg/pos と重複しないものを追加
        cand_pairs = list(dict.fromkeys(cand_pairs))  # unique (順序保持)
        for uv in cand_pairs:
            if need <= 0: break
            if uv in cur_neg or uv in pos_list:
                continue
            # 追加
            new_rows.append({"u": uv[0], "v": uv[1], "label": 0})
            cur_neg.add(uv)
            need -= 1; added += 1

        # 場合によっては不足分が埋まらないこともある（タスク分布が偏っているとき）
        if need > 0:
            print(f"[warn] {a} & {b}: could not reach target negatives, remaining={need}")

    # 保存
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for o in new_rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {args.out} (added negatives={added}, total lines={len(new_rows)})")

if __name__ == "__main__":
    main()