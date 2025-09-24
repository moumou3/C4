#!/usr/bin/env python3
# labels_to_candidates.py
import json, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seen = set()
    with open(args.labels, "r", encoding="utf-8") as f, \
         open(args.out,    "w", encoding="utf-8") as g:
        for ln in f:
            o = json.loads(ln)
            u, v = o["u"], o["v"]
            # unordered 正規化で重複除去
            key = (u, v) if u <= v else (v, u)
            if key in seen: 
                continue
            seen.add(key)
            g.write(json.dumps({"u": key[0], "v": key[1]}, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {args.out} (pairs={len(seen)})")

if __name__ == "__main__":
    main()