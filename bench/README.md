# Backend benchmark

Measures compression backends on **real Claude Code turns**, through the
plugin's own `compress_text` pipeline, so fenced-code handling, entity masking,
placeholder forcing and fallback gates are identical to production.

## Arms

| Arm | What it does |
| --- | --- |
| `baseline` | Deterministic pipeline only (`llm="off"`): code/JSON minification and whitespace normalisation |
| `llm` | LLMLingua-2 at rate `0.4` |
| `ollama:<model>` | One generative summarisation call per prose paragraph at rate `0.4` |
| `chain:<model>` | LLMLingua-2 at `0.4`, then the same Ollama model at `0.55` on its output |

Ollama calls use `temperature=0`, a fixed seed, `think:false` for qwen3-family
models, `num_ctx=8192` and an output budget derived from the target shrink rate.
The expected placeholder tokens (`<<M0>>`, …) are passed in the prompt; the
pipeline rejects any output that loses one, changes entity extraction, or is not
smaller than the deterministic normalisation.

## Metrics

- **shrink** — total output bytes ÷ input bytes, aggregated over all sampled
  turns.
- **entity recall** — file paths and exported symbols (the plugin's own
  extractor) present in the output.
- **number recall** — numeric literals preserved. Decimal points that
  LLMLingua detaches (`7.592` → `7. 592`) are rejoined before matching.
- **content-word recall** — non-stopword alphabetic tokens ≥ 4 chars.
- **median / p90 seconds** — wall time per turn through the full pipeline.
- **llm_blocks / fallback** — paragraphs the model actually compressed versus
  paragraphs that hit a safe fallback.
- **determinism** — the same masked input compressed twice with the same Ollama
  model; only meaningful for generative arms.

Recall matters more than shrink for this plugin: a summary that drops a file
path or a number forces the model to rediscover it.

## Controls

The harness refuses to report numbers unless all controls pass on the same run:

- identity scores 0 % shrink and 100 % on all recall metrics;
- a damaged text drops entity/number recall below 1.0;
- the deterministic baseline shrinks a mixed code/prose turn by ≥ 15 %;
- error text is returned byte-identical;
- each Ollama model answers a transport probe.

## Running

```sh
# from the repository root, with plugin dependencies installed
python3 bench/backend_bench.py \
  --root ~/.claude/projects \
  --sessions 12 \
  --turns 16 \
  --models qwen3.5:latest,llama3.2:3b,qwen2.5:7b

# skip the chain arms
python3 bench/backend_bench.py --no-chain
```

`OLLAMA_HOST` selects the server (default `http://127.0.0.1:11434`).
`APPEND_COMPACT_BENCH_DIR` selects where the private results file is written
(default `~/.cache/append-compactor-bench`). The results file contains verbatim
turn text and model output, so it is **not** committed; only aggregate numbers
belong in the README.

---

## Results — 2026-10-31

**Corpus:** 13 prose-heavy assistant turns from 7 real Claude Code sessions,
28,051 characters total. All controls and all three transport probes passed on
this run.

**Models:** `qwen3.5:latest` (9.7 B, 6.6 GB), `llama3.2:3b` (2.0 GB),
`qwen2.5:7b` (4.7 GB), served by local Ollama over CPU. The numbers below are
from the pipeline that ships: decimals, versions and unit-bearing numbers are
masked before compression and restored byte-exact.

| Arm | Shrink | Entity | Number | Word | Median | p90 | LLM blocks | Fallbacks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (`llm="off"`) | 0.0 % | 100.0 % | 100.0 % | 100.0 % | 0.00 s | 0.00 s | 0 | 70 |
| **LLMLingua-2** | **48.0 %** | **100.0 %** | **100.0 %** | 60.2 % | **2.02 s** | **5.59 s** | 69 | 1 |
| ollama:qwen3.5 | 14.6 % | 100.0 % | 99.2 % | 85.0 % | 30.25 s | 61.88 s | 38 | 32 |
| chain:qwen3.5 | 42.3 % | 100.0 % | 98.6 % | 63.9 % | 18.85 s | 51.84 s | 64 | 6 |
| ollama:llama3.2:3b | 9.1 % | 100.0 % | 97.2 % | 89.2 % | 5.26 s | 11.06 s | 18 | 52 |
| chain:llama3.2:3b | 39.9 % | 100.0 % | 100.0 % | 64.6 % | 7.14 s | 15.74 s | 59 | 11 |
| ollama:qwen2.5:7b | 21.8 % | 100.0 % | 100.0 % | 80.8 % | 8.34 s | 19.15 s | 37 | 33 |
| chain:qwen2.5:7b | 34.8 % | 100.0 % | 99.2 % | 68.0 % | 8.74 s | 20.11 s | 53 | 17 |

Determinism (same masked input, two calls, `temperature=0`, fixed seed):
`qwen3.5` identical, **`llama3.2:3b` not identical**, `qwen2.5` identical — and
in the earlier pre-fix run all three were non-identical. Generative output is
therefore not reproducibly stable across models or runs.

### What it means

- **No Ollama arm beats LLMLingua-2 on compression.** Generative-only arms
  shrink 9–22 %; LLMLingua shrinks 48.0 %. Chaining adds LLMLingua's work and
  still lands 6–13 points *below* LLMLingua alone, while costing 5–30 s per
  turn. Many generative outputs were rejected by the existing “must be smaller
  than the deterministic normalisation” gate (32–52 fallbacks per 70 prose
  paragraphs), so the generator often contributed nothing.
- **Word recall is not a win.** Ollama-only keeps more words only because it
  keeps more text. At comparable shrink (chains), word recall is no better than
  LLMLingua.
- **Number preservation does not improve.** LLMLingua reaches 100 % after
  numeric masking; the generative arms are between 97.2 % and 100 %, at much
  worse shrink. Entity recall is 100 % everywhere because masking plus the
  entity-equality gate guarantees it.
- **Generative output is not reliably reproducible** on this stack even at
  temperature zero. Committed chunks are frozen either way, but a generative
  backend would make the compression memo and cross-run byte stability
  unusable.
- **Decision:** the plugin keeps LLMLingua-2 as the only prose backend. An
  Ollama server is not required and no `--llm-engine` option is added.
  Background compaction (on by default) is where the 2 s median LLMLingua cost
  goes — off the hook path — with the deterministic fallback bounding every
  stall.
- **One pipeline fix came out of the benchmark:** LLMLingua was detaching
  decimals (`7.592` → `7 592`). Masking numeric literals before compression
  raised number recall from 97.3 % to 100 % and shrink from 47.7 % to 48.0 %.

**Caveats.** The box was under heavy unrelated load during these runs (another
cargo build and a separate Ollama job), so the latency columns are upper
bounds; shrink and recall are unaffected. 13 turns is a directional sample, not
a corpus-wide claim. The private per-turn results remain in
`~/.cache/append-compactor-bench/backend-results-v150.json`.
