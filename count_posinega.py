import json, re, collections
LANG_RE = re.compile(r'^[^/]+/([^/]+)/')
def lang(k): 
    m = LANG_RE.match(k); return m.group(1) if m else ""
cnt = collections.Counter()
with open("./lsst4_pairs_bench/labels.jsonl","r",encoding="utf-8") as f:
    for ln in f:
        o = json.loads(ln); u,v = o["u"], o["v"]
        a,b = sorted([lang(u), lang(v)])
        cnt[(a,b,o["label"])] += 1
for (a,b,l),n in sorted(cnt.items()):
    print(f"{a} & {b}  label={l}  n={n}")