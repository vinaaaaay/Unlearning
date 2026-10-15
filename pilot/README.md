# Pilot Directory: Files and Procedure

This directory contains the pilot experiment (Step 1 of the plan of action). The pilot compares six models on multi-hop questions: the Meta base model and five TOFU checkpoints. The pilot has two parts: a greedy run and a temperature sweep. For the method and the results, see `PILOT_REPORT.md`.

## 1. Directory Layout

```
pilot/
├── README.md              This file
├── PILOT_REPORT.md        Full report of the pilot
├── gold.py                Gold answers, aliases, and question templates
├── build_probes.py        Makes the 300 probe questions
├── run_pilot.py           Runs the models and records their outputs
├── score.py               Scores the greedy outputs and makes the result tables
├── score_temperature.py   Scores the temperature sweep
├── data/
│   └── probes.jsonl       The 300 probe questions
└── results/
    ├── gen_base.jsonl     Outputs of the Meta base model
    ├── gen_orig.jsonl     Outputs of the original model
    ├── gen_clean.jsonl    Outputs of the clean model
    ├── gen_graddiff.jsonl Outputs of the GradDiff unlearned model
    ├── gen_npo.jsonl      Outputs of the NPO unlearned model
    ├── gen_rmu.jsonl      Outputs of the RMU unlearned model
    ├── gen_rmu_lr5e-05_scoeff100_BROKEN.jsonl   Outputs of the broken RMU checkpoint
    ├── gen_<tag>_T<t>.jsonl   Sweep outputs: one file for each model and temperature (24 files)
    ├── run.log            Log of the main run
    ├── run_rmu.log        Log of the second RMU run
    ├── run_base.log       Log of the base model run
    ├── run_temperature.log   Log of the temperature sweep
    ├── scored.jsonl       Score for each greedy output
    ├── summary.json       All greedy result tables in JSON
    ├── scored_temperature.jsonl   Sweep scores for each question, model, and temperature
    ├── summary_temperature.json   All sweep result tables in JSON
    └── score_temperature.txt      Printed output of score_temperature.py
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

- The `MODELS` dictionary gives the six Hugging Face models. The `base` model (`meta-llama/Llama-3.2-1B-Instruct`) is gated. See section 3.2, step 3.
- The script uses the OpenUnlearning chat template for Llama-3.2-1B-Instruct.
- By default, decoding is greedy. The maximum is 384 new tokens.
- If you give a temperature above 0, the script samples 5 answers for each question (option `--num-samples`). It sets top-p to 1.0 and top-k to 0. These values disable the top-p and top-k filters.
- The checkpoints give other sampling values in `generation_config.json` (temperature 0.6, top-p 0.9). Also, transformers adds top-k 50 if top-k is not given. Thus, the script sets all sampling values explicitly.
- For sampling, each batch gets a seed from `--seed` and the question identifiers in the batch. Thus, a run with the same options uses the same seeds. GPU calculations can still give small differences.
- The script uses the Apple GPU (MPS) if it is available. If not, it uses the CPU.
- The script loads one model at a time. It removes each model from memory before it loads the next model. This keeps the memory usage below 8 GB.
- The script writes each batch immediately. A greedy run writes to `results/gen_<tag>.jsonl`. A run at temperature T writes to `results/gen_<tag>_T<T>.jsonl`, with one line for each sample.
- If you stop the script, you can start it again. It does not run the questions that are already in the output file.

Options:

| Option | Function | Default |
|---|---|---|
| `--models` | The models to run. Values: `base`, `orig`, `clean`, `graddiff`, `npo`, `rmu`. | All six |
| `--temperatures` | One or more temperatures. 0 gives greedy decoding. | 0 |
| `--num-samples` | Samples for each question when the temperature is above 0 | 5 |
| `--seed` | Random seed for sampling | 0 |
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
| `temperature`, `top_p`, `top_k` | Sampling values (only in sweep files) |
| `sample` | Sample number, 0–4 (only in sweep files) |
| `seed` | Seed of the batch (only in sweep files) |

### 2.4 `score.py`

This script scores the outputs and prints the result tables.

1. It finds the answer span in each output. If the output contains "the answer is", the span is the text after the last occurrence. If not, the span is the full output.
2. It removes the negative phrases from the span.
3. It looks for each accepted answer as a full word. Upper case and lower case are equal.
4. It calculates the accuracy for each model and each hop.
5. It calculates the 95% confidence intervals with a bootstrap on authors (10,000 resamples).
6. It does the gate test: Acc(unlearned) − Acc(clean), with a paired bootstrap.
7. If the base outputs are available, it compares each model with the base model: Acc(model) − Acc(base).
8. It calculates Leftover Knowledge.
9. It flags outputs that say that the model does not know (uncertainty flag).
10. It writes `results/summary.json` and `results/scored.jsonl`.

The script uses only the Python standard library. It ignores a model if its output file does not have all 300 lines.

Values in the printed tables:

| Value | Meaning |
|---|---|
| Lenient | Primary accuracy. Uses the span rules above. |
| Strict | As lenient, but an output without "the answer is" is incorrect. |
| Verdict | `<MODEL> > <REFERENCE>`, `<MODEL> < <REFERENCE>`, or `not distinguishable`. Example: `NPO < CLEAN`. The verdict uses the 95% CI of the difference. |
| Marker rate | Percentage of outputs that contain "the answer is" |
| Truncated | Percentage of outputs that stopped at 384 tokens |
| Refusal | Percentage of outputs with the uncertainty flag. Examples: "I don't have information", "I couldn't find". The flag uses a word-pattern match. It does not change the score. |
| City in CoT | Percentage of hop-2 and hop-3 outputs that contain the true birth city |

Fields in each line of `scored.jsonl`:

| Field | Content |
|---|---|
| `model`, `id`, `hop`, `author` | Identifiers |
| `lenient`, `strict` | `true` if the answer is correct |
| `marked` | `true` if the output contains "the answer is" |
| `city_in_cot` | `true` if the output contains the true birth city |
| `truncated` | `true` if the output stopped at 384 tokens |
| `refusal` | `true` if the output has the uncertainty flag |

### 2.5 `score_temperature.py`

This script scores the temperature sweep. It uses the scoring rules of `score.py` for each sample.

1. It reads the greedy files (T=0) and the sweep files (T=0.3, 0.7, 1.0, 1.3).
2. It ignores a file if it does not have all 300 questions, or if the questions have different numbers of samples.
3. For each question, it calculates two values:
   - **Any-of-N (P_leak):** 1 if one or more samples are correct.
   - **Per-sample accuracy:** the fraction of samples that are correct.
4. It prints the accuracy of each model at each temperature and hop, with 95% CIs.
5. It prints the comparisons with the clean model and with the base model. The primary comparison is any-of-N, unlearned − clean. We selected it before we saw the results.
6. It prints the percentage of samples that stopped at 384 tokens.
7. It writes `results/summary_temperature.json` and `results/scored_temperature.jsonl`.

In the comparison tables, ">" means that the full 95% CI is above zero. "<" means that the full CI is below zero. "~" means that the CI contains zero.

## 3. Procedure

### 3.1 Requirements

- Python 3.12
- Approximately 15 GB of free disk space for the six models (2.5 GB each)
- A Hugging Face account with access to `meta-llama/Llama-3.2-1B-Instruct` (only for the base model)
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

3. Get access to the base model. Do this step only if you run the `base` model.
   1. On Hugging Face, open the page `meta-llama/Llama-3.2-1B-Instruct`.
   2. Accept the Meta licence. Wait for Meta to accept your request.
   3. Make a read-only access token in your Hugging Face settings.
   4. Log in. Paste the token when the command asks for it:

      ```bash
      .venv/bin/hf auth login
      ```

   Do not write the token in a script, a command, or a file in the repository.

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

4. Run all six models:

   ```bash
   ../.venv/bin/python run_pilot.py > results/run.log 2>&1
   ```

5. Score the outputs:

   ```bash
   ../.venv/bin/python score.py
   ```

### 3.4 Run the temperature sweep

Do these steps in the `Unlearning/pilot/` directory. Do the greedy run (section 3.3) first.

1. Start the sweep. The command `caffeinate -i` prevents idle sleep of the Mac:

   ```bash
   nohup caffeinate -i ../.venv/bin/python run_pilot.py --temperatures 0.3 0.7 1.0 1.3 --num-samples 5 --seed 0 --batch-size 8 > results/run_temperature.log 2>&1 &
   ```

2. Monitor the progress:

   ```bash
   tail -f results/run_temperature.log
   ```

3. If the run stops, give the same command again. The script continues from the last complete batch.
4. When the run is complete, score the sweep:

   ```bash
   ../.venv/bin/python score_temperature.py > results/score_temperature.txt
   ```

The sweep needs approximately 18 hours on an Apple M2 (8 GB). Keep the Mac connected to power. Keep the lid open, because `caffeinate -i` does not prevent sleep when you close the lid.

### 3.5 Run time of the greedy run on an Apple M2 (8 GB)

| Model | Speed | Time for 300 questions |
|---|---|---|
| base | 0.67 q/s | approximately 8 min |
| orig | 1.38 q/s | approximately 4 min |
| clean | 1.29 q/s | approximately 4 min |
| graddiff | 0.34 q/s | approximately 15 min |
| npo | 1.78 q/s | approximately 3 min |
| rmu | 0.35 q/s | approximately 14 min |

The download time is not included. graddiff and rmu are slow because many of their outputs continue to the 384-token limit.

## 4. Cautions

- **Batch size for the sweep.** With 5 samples, a batch of 16 questions gives 80 sequences. On a Mac with 8 GB, this batch size causes swap memory use. Use `--batch-size 8`.
- **Stopping a run.** To stop a run, stop the Python process with its process ID. Do not use a pattern match (`pkill -f`) that can also match other commands.
- **Memory.** Do not load a model in float32 on a Mac with 8 GB. A 1B model in float32 needs approximately 5 GB. The system then uses swap memory and becomes very slow. Use bfloat16.
- **Disk.** Each checkpoint uses 2.5 GB in `~/.cache/huggingface/hub`. Make sure that there is sufficient free space before you add more checkpoints.
- **Broken checkpoints.** Some OpenUnlearning checkpoints are broken. Before you use a new unlearned checkpoint, ask it a general question (for example, "What is the capital of France?"). If the answer is not correct, do not use the checkpoint.
- **Changes to gold answers.** If you change only the answers or aliases in `gold.py`, run `build_probes.py` and `score.py` again. You do not have to run `run_pilot.py` again.
- **Changes to questions.** If you change `TEMPLATES` or `COT_SUFFIX`, the old outputs are not correct. Delete the files `results/gen_*.jsonl`. Then do all the steps in 3.3 again.
