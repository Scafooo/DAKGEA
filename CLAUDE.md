# Luffy-L dataset generation — instructions for Claude Code

Read this whole file before doing anything. It is the single source of truth for this task.
When something here is ambiguous, or reality does not match what is written, **stop and ask
Marco**; do not pick an interpretation yourself.

## 1. Goal

Generate the augmented training datasets of **Luffy-L** (paper: "Data Augmentation for Entity
Resolution across Knowledge Graphs", Algorithm 1 + Algorithm 3) and its ablations, in exactly the
same experimental protocol as the Luffy-S runs already done with this repository. Federico trains
and evaluates BERT-INT on them; **we only generate the data**.

- Datasets: `BBC_DB, D_W_15K_V1, D_W_15K_V2, ICEW_WIKI, ICEW_YAGO`
- Training fraction tf ∈ {0.1 … 1.0} (`reduction.ratio`), augmentation factor af ∈ {0.1 … 1.0}
  (`augmentation.ratio`)
- Seeds: `11037, 11038, 11039, 11040, 11041` (confirmed by Federico: start at 11037, +1 per run)
- Variants: `full`, `no_cache`, `no_bfs`, `all_llm`
- LLM: Llama 3.3 70B Instruct on AWS Bedrock (`us.meta.llama3-3-70b-instruct-v1:0`), zero-shot

## 2. What is already implemented and tested (do not rewrite it)

| File | What it does |
|---|---|
| `src/augmentation/methods/llm/augmenter.py` | `LLMAugmenter`, registered as `augmentation.method: llm`. Subclasses `PLMAugmenter` (Luffy-S) and **inherits Set-KG, BFS, seed order, depth limit, budget and edge creation unchanged**. Replaces only literal generation (set-entities: `bootstrap_literals`; singletons: `_expand_non_set_node`) with Algorithm 3 (`LuffyLGenerator`, LLMCache keyed by Set-KG node). Neighbours are visited in sorted order (the DAKGEA loaders fill graphs from Python sets, so the native order changes with PYTHONHASHSEED). Also contains the `all_llm` path. Writes `augmentation/luffy_l_stats.json` and `augmentation/luffy_l_calls.jsonl`. |
| `src/augmentation/methods/llm/prompts.py` | Context of a node (its literals and relations in each KG + its neighbours, matched pairs first, split in chunks of ≤40 neighbours / ≤120 values), the three **fixed** system prompts (Luffy-L, Luffy-L continuation chunk, all_llm), JSON parsing and validation against the input alphabet. |
| `src/augmentation/methods/llm/llm_client.py` | Bedrock Converse backend with rate limit and retries; **SQLite cache of every call** keyed by sha256(model, decoding params, seed, prompt); providers `bedrock`, `replay` (cache only, error on miss), `fake` (placeholder answers for dry runs, isolated key space), `mock` (tests). |
| `config/augmentation/llm.yaml` | Defaults: model, temperature 0.7, top_p 0.9, max_tokens 8192, prompt caps, cache path `data/external/luffy_l/llm_cache.sqlite`. |
| `scripts/luffy_l/generate_configs.py` | One experiment YAML per (variant, dataset, tf, af, seed) under `config/experiments/luffy_l/<variant>/`. Same protocol as the Luffy-S configs; all evaluation off, `save_dataset: true`. |
| `scripts/luffy_l/verify_outputs.py` | Checks one generated dataset (split, KG1 relations, budget, …). |
| `scripts/luffy_l/build_manifest.py` | Finds every run under `results/`, verifies it, writes `results/luffy_l_manifest.csv` and `results/luffy_l_missing_runs.txt`. |
| `tests/augmentation/test_llm_augmenter.py` | 13 offline tests (toy bowie/tm graph, 100-neighbour hub): edges to O_aug, LLMCache, same call for both sides, neighbour chunking, fallback, replay, fake isolation, order independence, all_llm alphabet, required flags. |

Why the cache matters: with a fixed seed the BFS order does not depend on af, so the af=1.0 run
contains every call of the smaller af. Run af=1.0 first; all other af and the `no_cache` prompts
already seen are served from the cache (0 API calls). The seed is in the key, so the 5 seeds are
independent generations, as the paper states.

## 3. Rules

MUST NOT:
1. Modify anything under `src/augmentation/methods/plm/`, `src/reduction/`, `src/core/`,
   `experiments/runner/`. Luffy-L must run on exactly the scaffold Luffy-S used. If you believe a
   change there is needed, stop and ask.
2. Change the system prompts, the prompt caps, temperature/top_p/max_tokens or the model **after
   Phase 2 has started**. Any change invalidates the cache and makes runs inconsistent. Ask first.
3. Choose `copy_original_literals` yourself (decision D1). The code refuses to run without it.
4. Set any `eval: true`, train BERT-INT or any other model.
5. Use anything from the old repository `llm_dakgea` (June/October outputs are invalid: wrong
   split, no KG1 relations on ICEWS, non-conforming BFS/cache).
6. Delete or overwrite `data/external/luffy_l/llm_cache.sqlite`. Back it up before Phase 3.
7. Use `provider: fake` with the real cache path, or treat fake outputs as data.
8. Retry a failing run more than once. Report it instead.
9. Commit generated configs, the cache or results (they are in `.gitignore`).

MUST:
- After every phase, write a short report (what ran, numbers, problems) and **wait for Marco's OK**.
- Quote exact numbers from `luffy_l_stats.json` / the manifest; never estimate silently.

## 4. Decisions Marco must fill in before Phase 3

| ID | Question | Answer |
|---|---|---|
| D1 | `copy_original_literals`: `true` = identical to how Luffy-S ran (the inherited scaffold copies every original literal of BOTH KGs onto the augmented entities, `PLMAugmenter._process_neighbors` → `attach_literal_to_augmented`); `false` = paper behaviour (only generated values). Decide with Federico. | ____ |
| D2 | `no_bfs` is implemented as `max_depth = 0` (no neighbourhood expansion, LLMCache still on). Is this how "w/o BFS" was run for Luffy-S? (Federico) | ____ |
| D3 | Delivery: (a) the BERT-INT folders (≈17 MB for BBC_DB tf=0.1, far more for ICEWS; ~10,000 runs → hundreds of GB), or (b) code branch + `llm_cache.sqlite`; Federico regenerates any run with `llm.provider: replay` (0 API calls, identical output). | ____ |
| D4 | Scope of the ablations (`no_cache`, `no_bfs`, `all_llm`): full grid, or a subset of tf/af? | ____ |
| D5 | Parallel jobs and Bedrock quota (requests/min, tokens/min) for the account. | ____ |

## 5. Phases

### Phase 0 — environment (no API calls)
1. `git status` on branch `luffy-l`; activate the project venv.
2. `pytest tests/augmentation/test_llm_augmenter.py -q` → **13 passed**. If not, stop.
3. `python -c "import boto3"`; `aws sts get-caller-identity`; one test call:
   `python -c "import boto3;c=boto3.client('bedrock-runtime','us-east-1');print(c.converse(modelId='us.meta.llama3-3-70b-instruct-v1:0',messages=[{'role':'user','content':[{'text':'Say OK'}]}],inferenceConfig={'maxTokens':8192})['output'])"`
   If `maxTokens=8192` is rejected, report the model's maximum (do not change the config yet).
Report and wait.

### Phase 1 — dry run through the real runner (fake LLM, no API calls)
```bash
python scripts/luffy_l/generate_configs.py --copy-original-literals false \
  --provider fake --cache-path data/external/luffy_l/fake_cache.sqlite \
  --datasets BBC_DB --tf 1 --af 10 5 --seeds 11037 --out config/experiments/luffy_l_dryrun
for f in config/experiments/luffy_l_dryrun/*/*_10_s*.yaml config/experiments/luffy_l_dryrun/*/*_05_s*.yaml; do
  python experiments/runner/run.py "$f"; done          # af=1.0 first, then af=0.5
python scripts/luffy_l/build_manifest.py --configs config/experiments/luffy_l_dryrun \
  --out results/luffy_l_dryrun_manifest.csv
```
Check and report: where the dataset folder is (`…/augmentation/dataset/<writer>/`), that no model
training started (logs), the manifest status of the 8 runs, and the `api_calls` of the af=0.5 runs
(must be 0 because af=1.0 ran first).
Also run one ICEWS dry run (`--datasets ICEW_WIKI --tf 1 --af 10 --variants full`) and report memory
and time. Delete `fake_cache.sqlite` and the dry-run results afterwards. Wait.

### Phase 2 — real pilot (Bedrock)
```bash
python scripts/luffy_l/generate_configs.py --copy-original-literals <D1> \
  --datasets BBC_DB --tf 1 --af 10 5 --seeds 11037 --variants full all_llm --out config/experiments/luffy_l_pilot
```
Run the af=10 configs first, then af=5. Report from the stats/manifest: API calls, input/output
tokens, wall time, `parse_failures`, `truncated_outputs`, `values_fallback_original`,
`values_identical_to_original`, `llm_cache_hits`. Print 3 random synthetic entities next to their
originals (`luffy_l_calls.jsonl` + the dataset's `attr_triples*`) so Marco can judge the values.
Replay test: rerun the af=10 `full` config with `llm.provider: replay` and `overwrite_existing: true`
→ 0 API calls and the same content as the first run: compare `attr_triples1/2` and `sup_pairs`
after mapping ids to URIs (`ent_ids_*`) and sorting; line order and numeric ids may differ.
Projection: from the pilot, estimate calls/tokens/cost/time of the full grid
(calls scale ≈ linearly with the number of training pairs, i.e. with tf and dataset size; only
af=1.0 runs cost calls). Report and **wait for explicit approval** of the budget.

### Phase 3 — full grid
1. Fill decisions D1–D5 in this file. Back up the cache.
2. `python scripts/luffy_l/generate_configs.py --copy-original-literals <D1> [--variants … per D4]`
3. Launch with `scripts/run_experiments_parallel.sh --dir config/experiments/luffy_l/<variant> --jobs <D5> --pattern "*_10_s*.yaml"`
   first (af=1.0), then the remaining af. Variant order: `full`, `no_cache`, `no_bfs`, `all_llm`.
4. Every few hours: `python scripts/luffy_l/build_manifest.py`; rerun FAIL/missing runs once with
   `overwrite_existing: true` (cheap: answers come from the cache). Report runs still failing.
   Note: with all evaluation off, the runner treats an experiment folder as complete if its
   metadata exists, even after a crash; that is why the manifest — not the runner — decides.

### Phase 4 — handover
Final `build_manifest.py` must report `FAIL: 0` and `missing: 0`. Deliver per D3, together with
`results/luffy_l_manifest.csv` and this file. Report the totals (calls, tokens, cost).

## 6. Background (for understanding; do not act on it)

- Algorithm 3: when the BFS visits set-entity S, if S is in LLMCache use its literals; else one LLM
  call with the local context of S returns literals for S **and** its neighbours; neighbours are
  stored in LLMCache and used when the BFS reaches them. Implementation details: (i) neighbours
  that do not fit one prompt (ICEWS hubs have hundreds) are generated by continuation calls that
  receive S's generated values, so every neighbour is still primed; (ii) a neighbour is primed
  only if the call generated all its attribute values, otherwise it gets its own call;
  (iii) when a matched pair is served from LLMCache, Algorithm 3 gives it no neighbour literals,
  so its unprimed neighbours would each cost a call; they are instead generated by continuation
  calls that see the pair's cached values (fewer calls, same coherence goal). Points (i)–(iii)
  must be stated in the paper's implementation details.
  Unparseable answers → original values kept (paper's fallback), counted in the stats.
- The inherited BFS also creates augmented copies of the singleton neighbours of every visited
  pair (as Luffy-S does). On ICEWS this dominates the number of generated entities; each singleton
  is generated once per run, so calls per run are bounded by ≈ (#entities reached)/40 + #pairs.
- `all_llm` = the naive baseline "ask an LLM with one prompt from the training data": one call per
  sampled training pair (same context, model and decoding as Luffy-L); the model writes attributes
  and relations of one synthetic pair; only attribute names, relation names and neighbour entities
  present in the input are accepted (no new entities or relations, agreed with Federico); no
  LLMCache, no BFS, no fallback (failed pairs are skipped and counted).
- Known behaviours of the inherited Luffy-S scaffold (report, do not fix): (i) the literal copy of
  D1; (ii) `config/augmentation/plm.yaml` is the default actually loaded (τ=0.6, name weight 0.85,
  value cache scope `alignment_pair`), while the paper states τ=0.7, λ=0.7 and a global cache;
  (iii) the BertInt writer drops a synthetic pair whose entity has no triple at all (seen with
  `no_bfs` on BBC_DB: 158 of 187 pairs kept); (iv) the loaders' set-based order makes Luffy-S
  traversals differ between processes even with the same seed (Luffy-L sorts neighbours).
