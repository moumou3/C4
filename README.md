1. create positive and negative using C4 training set
python build_pos_neg_like_original.py \
  --pairs_jsonl /path/to/pairs_test.jsonl \
  --out_dir out_pairs \
  --neg-per-anchor 1

2. create labels.jsonl, func_list.jsonl, candidate.jsonl
  python eval_pairs_to_lsst4.py \
    --eval_pairs out_pairs/eval_pairs.jsonl \
    --out_dir    lsst4_input \
    --project    pairs_bench

3. rebalance labels negative amount by langpair
 python rebalance_labels_by_langpair.py --labels labels.jsonl --out labels_balanced.jsonl --ratio 1.0
  
4. convert candidates.jsonl format
  python repack_candidates_sem.py --emb lsst4_pairs_bench/func_emb.pkl --func-list lsst4_pairs_bench/func_list.jsonl --cands-in lsst4_pairs_bench/candidates_old.jsonl --cands-out lsst4_pairs_bench/candidates.jsonl

5. execute LSS-T4
  pipenv run hy-scu phase1 pipeline run --phases 1 --root ~/C4/lsst4_pairs_bench/src --out ~/C4/lsst4_pairs_bench/ --skip-loc-le 0

  pipenv run hy-scu phase1 pipeline run --phases 3 --root ~/C4/lsst4_pairs_bench/src  --out ~/C4/lsst4_pairs_bench

6. output result report 
  python eval_langpairs_from_reranked.py \
    --labels   lsst4_out/labels.jsonl \
    --reranked lsst4_out/reranked.jsonl \
    --pairs "C++&C#" "C++&Java" "C++&Python" "C#&Java" "C#&Python" "Java&Python" \
    --tau_fixed ""