# Pilot Directory: Files and Procedure

This directory contains the pilot experiment (Step 1 of the plan of action). The pilot compares five TOFU checkpoints on multi-hop questions. For the method and the results, see `PILOT_REPORT.md`.

## 1. Directory Layout

```
pilot/
├── README.md              This file
├── PILOT_REPORT.md        Full report of the pilot
├── gold.py                Gold answers, aliases, and question templates
├── build_probes.py        Makes the 300 probe questions
├── run_pilot.py           Runs the models and records their outputs
├── score.py               Scores the outputs and makes the result tables
├── data/
│   └── probes.jsonl       The 300 probe questions
└── results/
    ├── gen_orig.jsonl     Outputs of the original model
    ├── gen_clean.jsonl    Outputs of the clean model
    ├── gen_graddiff.jsonl Outputs of the GradDiff unlearned model
    ├── gen_npo.jsonl      Outputs of the NPO unlearned model
    ├── gen_rmu.jsonl      Outputs of the RMU unlearned model
    ├── gen_rmu_lr5e-05_scoeff100_BROKEN.jsonl   Outputs of the broken RMU checkpoint
    ├── run.log            Log of the main run
    ├── run_rmu.log        Log of the second RMU run
    ├── scored.jsonl       Score for each output
    └── summary.json       All result tables in JSON
```

## 2. Source Files

### 2.1 `gold.py`

This file contains data only. It does not run alone. The other scripts import it.

| Name | Content |
|---|---|
| `AUTHORS` | The 20 forget10 authors. Each entry has the name, the birth city, and the country. |
| `CITY_ALIASES` | The accepted names for each city. Example: "New York City", "New York", "NYC". |
| `COUNTRY_ALIASES` | The accepted names for each country (`pos`). Also, the phrases to remove before the match (`neg`). Example: remove "North Korea" before you look for "Korea". |
| `LANGUAGES` | The official languages of each country. Each language in the list is a correct answer. |
| `COT_SUFFIX` | The instruction at the end of each question: "Let's think step by step, then conclude with 'Therefore, the answer is: \<answer\>'." |
| `TEMPLATES` | Five question phrasings for each hop (1, 2, and 3). |
| `gold_for()` | Gives the accepted names and the negative phrases for one hop. |

To change a gold answer or add an alias, edit this file. Then run `build_probes.py` again.

### 2.2 `build_probes.py`

This script makes the probe set.

1. It downloads `forget10.json` from the Hugging Face dataset `locuslab/TOFU`.
2. It checks that each author name and each city is in the forget10 text. If a check fails, the script stops.
3. It makes 20 authors × 3 hops × 5 phrasings = 300 questions.
4. It writes the questions to `data/probes.jsonl`.

Fields in each line of `probes.jsonl`:

| Field | Content |
|---|---|
| `id` | Identifier. Format: `a<author>_h<hop>_t<template>`. Example: `a06_h3_t0`. |
| `author_idx`, `author` | Author number (0–19) and author name |
| `hop` | 1, 2, or 3 |
| `template` | Phrasing number (0–4) |
| `question` | Full question text, with the instruction |
| `gold` | The main correct answer |
| `gold_aliases` | All accepted answers |
| `neg_phrases` | Phrases to remove before the match |
| `bridge_city` | The true birth city (the forgotten fact) |

### 2.3 `run_pilot.py`

This script runs the models on the probe questions. It does only inference. It does not train or unlearn models.

- The `MODELS` dictionary gives the five Hugging Face checkpoints.
- The script uses the OpenUnlearning chat template for Llama-3.2-1B-Instruct.
- Decoding is greedy. The maximum is 384 new tokens.
- The script uses the Apple GPU (MPS) if it is available. If not, it uses the CPU.
- The script loads one model at a time. It removes each model from memory before it loads the next model. This keeps the memory usage below 8 GB.
- The script writes each batch to `results/gen_<tag>.jsonl` immediately.
- If you stop the script, you can start it again. It does not run the questions that are already in the output file.

Options:

| Option | Function | Default |
|---|---|---|
| `--models` | The models to run. Values: `orig`, `clean`, `graddiff`, `npo`, `rmu`. | All five |
| `--batch-size` | Number of questions in one batch | 16 |
| `--limit N` | Run only the first N questions. Use this option for a short test. | All 300 |

Fields in each line of `gen_<tag>.jsonl`:

| Field | Content |
|---|---|
| `id` | Probe identifier |
| `model`, `repo` | Model tag and Hugging Face repository |
| `question` | Question text |
| `generation` | Model output |
| `n_new_tokens` | Number of tokens in the output |
| `truncated` | `true` if the output stopped at 384 tokens |

### 2.4 `score.py`

This script scores the outputs and prints the result tables.

1. It finds the answer span in each output. If the output contains "the answer is", the span is the text after the last occurrence. If not, the span is the full output.
2. It removes the negative phrases from the span.
3. It looks for each accepted answer as a full word. Upper case and lower case are equal.
4. It calculates the accuracy for each model and each hop.
5. It calculates the 95% confidence intervals with a bootstrap on authors (10,000 resamples).
6. It does the gate test: Acc(unlearned) − Acc(clean), with a paired bootstrap.
7. It calculates Leftover Knowledge.
8. It writes `results/summary.json` and `results/scored.jsonl`.

The script uses only the Python standard library. It ignores a model if its output file does not have all 300 lines.

Values in the printed tables:

| Value | Meaning |
|---|---|
| Lenient | Primary accuracy. Uses the span rules above. |
| Strict | As lenient, but an output without "the answer is" is incorrect. |
| Verdict | `UNLEARN > CLEAN`, `UNLEARN < CLEAN`, or `not distinguishable`. The verdict uses the 95% CI of the difference. |
| Marker rate | Percentage of outputs that contain "the answer is" |
| Truncated | Percentage of outputs that stopped at 384 tokens |
| City in CoT | Percentage of hop-2 and hop-3 outputs that contain the true birth city |

Fields in each line of `scored.jsonl`:

| Field | Content |
|---|---|
| `model`, `id`, `hop`, `author` | Identifiers |
| `lenient`, `strict` | `true` if the answer is correct |
| `marked` | `true` if the output contains "the answer is" |
| `city_in_cot` | `true` if the output contains the true birth city |
| `truncated` | `true` if the output stopped at 384 tokens |

## 3. Procedure

### 3.1 Requirements

- Python 3.12
- Approximately 13 GB of free disk space for the five checkpoints (2.5 GB each)
- An internet connection for the first download
- Recommended: an Apple-silicon Mac or an NVIDIA GPU. A CPU also works, but it is slow.

### 3.2 Set up the environment

Do these steps in the `Unlearning/` directory.

1. Make a virtual environment:

   ```bash
   uv venv .venv --python 3.12
   ```

2. Install the packages:

   ```bash
   uv pip install --python .venv/bin/python torch transformers safetensors huggingface_hub
   ```

The pilot used torch 2.14.1 and transformers 5.19.0.

### 3.3 Run the pilot

Do these steps in the `Unlearning/pilot/` directory.

1. Make the probe set:

   ```bash
   ../.venv/bin/python build_probes.py
   ```

2. Optional: do a short test on 16 questions:

   ```bash
   ../.venv/bin/python run_pilot.py --models orig --limit 16
   ```

3. If you did step 2, delete the test output. If you do not delete it, the full run keeps these 16 lines:

   ```bash
   rm results/gen_orig.jsonl
   ```

4. Run all five models:

   ```bash
   ../.venv/bin/python run_pilot.py > results/run.log 2>&1
   ```

5. Score the outputs:

   ```bash
   ../.venv/bin/python score.py
   ```

### 3.4 Run time on an Apple M2 (8 GB)

| Model | Speed | Time for 300 questions |
|---|---|---|
| orig | 1.38 q/s | approximately 4 min |
| clean | 1.29 q/s | approximately 4 min |
| graddiff | 0.34 q/s | approximately 15 min |
| npo | 1.78 q/s | approximately 3 min |
| rmu | 0.35 q/s | approximately 14 min |

The download time is not included. graddiff and rmu are slow because many of their outputs continue to the 384-token limit.

## 4. Cautions

- **Memory.** Do not load a model in float32 on a Mac with 8 GB. A 1B model in float32 needs approximately 5 GB. The system then uses swap memory and becomes very slow. Use bfloat16.
- **Disk.** Each checkpoint uses 2.5 GB in `~/.cache/huggingface/hub`. Make sure that there is sufficient free space before you add more checkpoints.
- **Broken checkpoints.** Some OpenUnlearning checkpoints are broken. Before you use a new unlearned checkpoint, ask it a general question (for example, "What is the capital of France?"). If the answer is not correct, do not use the checkpoint.
- **Changes to gold answers.** If you change only the answers or aliases in `gold.py`, run `build_probes.py` and `score.py` again. You do not have to run `run_pilot.py` again.
- **Changes to questions.** If you change `TEMPLATES` or `COT_SUFFIX`, the old outputs are not correct. Delete the files `results/gen_*.jsonl`. Then do all the steps in 3.3 again.
