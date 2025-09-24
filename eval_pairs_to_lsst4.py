#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_pairs.jsonl（build_pos_neg_like_original.py などの出力）を
LSST4 の入力一式に変換します。

入力（1行1JSON；例）:
  {
    "code_a": "...", "code_b": "...",
    "label": 0/1,
    "task_a": 23, "task_b": 23,
    "lang_a": "cpp", "lang_b": "java",
    "id_a": 1503, "id_b": 1541
  }

出力（--out_dir 配下）:
  - src/<Language>/task_<T>/id_<ID>.<ext>  … コードを書き出し
  - func_list.jsonl   … LSST4 用関数一覧（擬似関数：1..EOF）
  - candidates.jsonl  … {"u","v"} のペア列挙（u,v は project 相対の識別子）
  - labels.jsonl      … {"u","v","label"}（正例=1 / 負例=0）

使い方:
  python eval_pairs_to_lsst4.py \
    --eval_pairs out_pairs/eval_pairs.jsonl \
    --out_dir    lsst4_input \
    --project    pairs_bench
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

# 言語名の正規化（LSST4 の language と拡張子）
LANG_MAP = {
    "c":      ("C",     ".c"),
    "cpp":    ("C++",   ".cpp"),
    "c++":    ("C++",   ".cpp"),
    "cc":     ("C++",   ".cc"),
    "cxx":    ("C++",   ".cpp"),
    "java":   ("Java",  ".java"),
    "python": ("Python",".py"),
    "py":     ("Python",".py"),
    "go":     ("Go",    ".go"),
    "rust":   ("Rust",  ".rs"),
    "rs":     ("Rust",  ".rs"),
    "c#":     ("C#",    ".cs"),
    "csharp": ("C#",    ".cs"),
    "cs":     ("C#",    ".cs"),
}

def norm_lang(s: str) -> Tuple[str, str]:
    key = (s or "").strip().lower()
    return LANG_MAP.get(key, ("C++", ".cpp"))  # 既定は C++

def write_func_once(
    project: str,
    src_root: Path,
    made: Dict[Tuple[str, str, str], str],
    lang_raw: str, task: str, fid: str, code: str,
    f_func
) -> str:
    """
    まだ書いていなければ src/ にファイル化＆ func_list.jsonl に1行追加。
    返り値: u = "{project}/{relpath}::F_{fid}"
    """
    L, ext = norm_lang(lang_raw)
    key = (L, str(task), str(fid))
    if key in made:
        return made[key]

    rel = Path(L) / f"task_{task}" / f"id_{fid}{ext}"
    abs_path = src_root / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(code or "", encoding="utf-8")

    n_lines = max(1, len((code or "").splitlines()))
    rec = {
        "project": project,
        "file": str(abs_path.resolve()),
        "func": f"F_{fid}",
        "line": 1,
        "end": n_lines,
        "language": L,
        "kind": "function"
    }
    f_func.write(json.dumps(rec, ensure_ascii=False) + "\n")
    u = f"{project}/{rel.as_posix()}::F_{fid}"
    made[key] = u
    return u

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_pairs", required=True, help="build_pos_neg_like_original.py の eval_pairs.jsonl")
    ap.add_argument("--out_dir",    required=True, help="出力先ディレクトリ（src/, func_list.jsonl などを出力）")
    ap.add_argument("--project",    default="pairs_bench", help="LSST4 の project 名（u/v の接頭辞）")
    args = ap.parse_args()

    out = Path(args.out_dir)
    src = out / "src"
    out.mkdir(parents=True, exist_ok=True)
    src.mkdir(parents=True, exist_ok=True)

    func_path = out / "func_list.jsonl"
    cand_path = out / "candidates.jsonl"
    labl_path = out / "labels.jsonl"

    made: Dict[Tuple[str, str, str], str] = {}  # (Lang, task, id) -> u
    n_rows = n_pos = n_neg = 0

    with open(args.eval_pairs, "r", encoding="utf-8") as fin, \
         open(func_path, "w", encoding="utf-8") as f_func, \
         open(cand_path, "w", encoding="utf-8") as f_cand, \
         open(labl_path, "w", encoding="utf-8") as f_lab:

        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            code_a = o.get("code_a", "")
            code_b = o.get("code_b", "")
            task_a = str(o.get("task_a", ""))
            task_b = str(o.get("task_b", ""))
            lang_a = str(o.get("lang_a", ""))
            lang_b = str(o.get("lang_b", ""))
            id_a   = str(o.get("id_a",   ""))
            id_b   = str(o.get("id_b",   ""))

            # ファイル化 & func_list 追記（未作成なら）
            u = write_func_once(args.project, src, made, lang_a, task_a, id_a, code_a, f_func)
            v = write_func_once(args.project, src, made, lang_b, task_b, id_b, code_b, f_func)

            # candidates / labels
            f_cand.write(json.dumps({"u": u, "v": v}, ensure_ascii=False) + "\n")
            label = int(o.get("label", 0))
            f_lab.write(json.dumps({"u": u, "v": v, "label": label}, ensure_ascii=False) + "\n")

            n_rows += 1
            if label == 1: n_pos += 1
            else:          n_neg += 1

    print(f"[done] func_list.jsonl : {func_path}")
    print(f"[done] candidates.jsonl: {cand_path}")
    print(f"[done] labels.jsonl    : {labl_path}")
    print(f"[info] pairs={n_rows}  pos={n_pos}  neg={n_neg}")
    print(f"[info] src root: { (src).resolve() }")

if __name__ == "__main__":
    main()