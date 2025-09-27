# LSST4 Pairs Bench — End‑to‑End README (English)

This document organizes the full pipeline from **C4‑derived pair data** to **LSST4 inputs**, runs LSS‑T4, and reports **Best‑F1 / PR‑AUC per language pair**. Everything below is copy‑paste ready.

---

## At a Glance

1. **Create positives/negatives** (build a pseudo‑gold set from the C4 training set)
2. **Generate LSST4 inputs** (`src/`, `func_list.jsonl`, `candidates.jsonl`, `labels.jsonl`)
3. **Rebalance negatives per language pair**
4. **Convert candidates to the new semantic format** (attach `score_sem` and `rank` using embeddings)
5. **Run LSS‑T4** (Phase‑1 / Phase‑3)
6. **Report metrics** (per language pair: Best‑F1, PR‑AUC; optionally P/R/F1 at a fixed tau)

> Paths below are examples. Adjust to your environment.

---

## Prerequisites

- Python 3.8+
- `pipenv`
- The helper scripts must be on `PATH` or available via relative paths.
- For Step 4 you need an embedding file `func_emb.pkl`  
  (either use an existing one or generate it by running Phase‑1 first).

---

## 1) Build Positive & Negative Pairs (from C4)

```bash
python build_pos_neg_like_original.py   --pairs_jsonl /path/to/pairs_test.jsonl   --out_dir     out_pairs   --neg-per-anchor 1
```

**Output (example)**
- `out_pairs/eval_pairs.jsonl`  
  Each line: `{code_a, code_b, label, task_a, task_b, lang_a, lang_b, id_a, id_b}`

---

## 2) Generate LSST4 Input Bundle

```bash
python eval_pairs_to_lsst4.py   --eval_pairs out_pairs/eval_pairs.jsonl   --out_dir    lsst4_input   --project    pairs_bench
```

**Outputs (under `lsst4_input/`)**
- `src/<Language>/task_<T>/id_<ID>.<ext>` … code files
- `func_list.jsonl` … function list for LSST4 (pseudo function: lines 1..EOF)
- `candidates.jsonl` … old‑format candidates (`{"u","v"}` or `{"q","nb"}`)
- `labels.jsonl` … `{"u","v","label"}` (1=positive, 0=negative)

---

## 3) Rebalance Negatives per Language Pair

```bash
python rebalance_labels_by_langpair.py   --labels lsst4_input/labels.jsonl   --out    lsst4_input/labels_balanced.jsonl   --ratio  1.0
```

- Keeps all existing positives
- Adds negatives **within the same language pair** by pairing functions from **different tasks**
- `ratio=1.0` targets a 1:1 positive:negative balance

> For subsequent evaluation steps, using `labels_balanced.jsonl` is recommended.

---

## 4) Convert `candidates.jsonl` to the New Semantic Format

The old format (`{"u","v"}` or `{"q","nb"}`) is repacked using embeddings and the function list to the new format
`{"q","nb","score_sem","rank"}`.

```bash
python repack_candidates_sem.py   --emb       lsst4_pairs_bench/func_emb.pkl   --func-list lsst4_pairs_bench/func_list.jsonl   --cands-in  lsst4_pairs_bench/candidates_old.jsonl   --cands-out lsst4_pairs_bench/candidates.jsonl
```

**Details**
- **Canonical key**: from `func_list.jsonl`, take the substring of `file` **after `/src/`** and build  
  `project/<rel>::func`. If an embedding key already uses the canonical form, it’s used as‑is; otherwise it is resolved via the mapping derived from the function list.
- **Scoring**: `score_sem = cosine( L2_norm(emb[q]), L2_norm(emb[nb]) )`
- **Ranking**: for each query `q`, sort neighbors by `score_sem` descending and assign 1‑based `rank`

> If `func_emb.pkl` is not available yet, run Phase‑1 first to produce it, or provide an existing embedding file.

---

## 5) Run LSS‑T4 (Phase‑1 / Phase‑3)

**Phase‑1 (preprocessing only, example)**
```bash
pipenv run hy-scu phase1 pipeline run   --phases 1   --root ~/C4/lsst4_pairs_bench/src   --out  ~/C4/lsst4_pairs_bench/   --skip-loc-le 0
```

**Phase‑3 (through reranking)**
```bash
pipenv run hy-scu phase1 pipeline run   --phases 3   --root ~/C4/lsst4_pairs_bench/src   --out  ~/C4/lsst4_pairs_bench
```

**Key artifacts** (under `~/C4/lsst4_pairs_bench/`, names may vary)
- `func_emb.pkl`, `pairs.jsonl`, `reranked.jsonl`, etc.

---

## 6) Report Metrics by Language Pair

Using `labels.jsonl` (or the rebalanced `labels_balanced.jsonl`) and `reranked.jsonl`, compute **Best‑F1 (`tau_F1`)**, **PR‑AUC**, and optionally **P/R/F1 at a fixed tau** per language pair.

```bash
python eval_langpairs_from_reranked.py   --labels   lsst4_out/labels.jsonl   --reranked lsst4_out/reranked.jsonl   --pairs "C++&C#" "C++&Java" "C++&Python" "C#&Java" "C#&Python" "Java&Python"   --tau_fixed ""
```

- Pass multiple `--pairs` as quoted strings like `"LangA&LangB"`.
- Set `--tau_fixed ""` to skip fixed‑tau metrics (compute Best‑F1 and PR‑AUC only).

---

## Tips & Troubleshooting

- **Candidate key mismatches**  
  `repack_candidates_sem.py` builds canonical keys (`project/<rel>::func`) from `func_list.jsonl`.
  If some keys don’t resolve, verify that the substring **after `/src/`** is extracted correctly for your paths.

- **Reproducibility**  
  Steps that sample negatives can be seeded (e.g., `--seed`) to make results repeatable.

- **Fair comparisons**  
  For final comparisons, prefer a **fixed validation threshold** (`tau*`) over Best‑F1 on the test set.

---

## Example Layout

```
lsst4_pairs_bench/
├─ src/<Language>/task_<T>/id_<ID>.<ext>
├─ func_list.jsonl
├─ candidates_old.jsonl        # old format
├─ candidates.jsonl            # new format (with score_sem, rank)
├─ labels.jsonl
├─ labels_balanced.jsonl
├─ func_emb.pkl                # from Phase‑1 or precomputed
└─ lsst4_out/
   ├─ pairs.jsonl
   ├─ reranked.jsonl
   └─ report_*.json / csv
```

That’s it—adjust paths/flags as needed for your setup.
