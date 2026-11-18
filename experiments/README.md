# Experiments: fine-tuning, unlearning, and multi-hop probing (Steps 4-7 and 9)

This directory runs the main experiment on the synthetic dataset of Steps 2 and 3:

1. fine-tune a **learned** model (all facts) and **clean** models (retain facts only),
2. unlearn the forget set from the learned model with **GradDiff**, **NPO**, and **RMU**,
   choosing each checkpoint with direct (1-hop) questions only,
3. ask every model multi-hop questions whose chain passes through a forgotten fact,
4. report leakage as the accuracy of an unlearned model **above the clean model**.

## Files

| File | Purpose |
|---|---|
| `build_data.py` | Builds the training corpora and all probes from the KG and the template corpus. Deterministic. |
| `world_anchors.py` | 110 real-world question-answer pairs that are mixed into both training corpora (see "Design decisions"). |
| `common.py` | Text formats, tokenisation, batched generation, scoring. |
| `finetune.py` | Step 4. Full fine-tuning of the learned and the clean model. |
| `evaluate.py` | Steps 5 and 7. Runs a model on the probe sets and scores every answer. |
| `unlearn.py` | Step 6. GradDiff, NPO, RMU, with checkpoint selection on direct questions. |
| `analyze.py` | Step 9. Accuracy tables, leakage above the clean model, Leftover Knowledge, kept-chain analysis, hop-level trace. |
| `sampling.py` | Per-sample and any-of-N accuracy under sampling, for every model including the clean one. |
| `figures.py` | Leakage figures from the summaries. |
| `RESULTS.md` | **The findings**, for both model sizes. |
| `run_size.sh` | The whole pipeline for one model size and one seed; restartable. |
| `unlearn_args/` | The settings of each unlearning method (`<method>.txt`, or `<method>_<size>.txt` for one size). |
| `data/` | Output of `build_data.py`. |
| `results/<size>/` | Generations (`gen/`), unlearning sweeps (`unlearn/`), `summary_*.json`, `report_*.md`, `sampling_*.md`, and `first_round/` (settings that missed the forget target). |
| `results/figures/` | Figures. |
| `runs/<size>/` | Model checkpoints (not in git). |

## How to run

```
python build_data.py                 # CPU, seconds
PY=python ./run_size.sh 0.6B 1       # size, seed of the learned model
```

Needs `torch`, `transformers`, `numpy`, `matplotlib` and one GPU (we used one NVIDIA L40S, 48 GB).
One fine-tuning run takes about 25 minutes for Qwen3-0.6B and 50 minutes for Qwen3-1.7B;
one unlearning sweep 5 to 20 minutes. Qwen3-1.7B does not fit in 48 GB with a batch of 32
texts in one pass, so we ran it as

```
FT_ARGS="--micro-batch 16" UN_ARGS="--micro-batch 16" PY=python ./run_size.sh 1.7B 1
```

which computes each batch in two halves with the same gradient. Run one GPU job at a time.

## Text formats

All texts are plain text (no chat template).

```
statement   Delem Bretuz mentored Thasex Rizat.
1-hop QA    Question: Who did Delem Bretuz mentor?
            Answer: Thasex Rizat
multi-hop   Question: Which company employs the person mentored by the person mentored by Delem Bretuz?
            Answer: Delem Bretuz mentored Thasex Rizat. Thasex Rizat mentored Thifaz Poplus. Thifaz Poplus
            works at Rizam Labs. Therefore, the answer is: Rizam Labs
```

## Design decisions (not fixed by the mid-term report)

**Spine roles.** The 200 target spines carry one forgotten edge each. The 200 pure-retain
spines are split by parity: 100 *demo* spines and 100 *control* spines.

**Multi-hop demonstrations.** The corpus of Step 3 has only 1-hop texts. A model trained on
it does not know that it must reason step by step, or how to end its answer. We therefore
add the 1,000 multi-hop questions (k = 2..5) of the demo spines, with their step-by-step
answers, to the training corpus of *both* the learned and the clean model. This also trains
the fixed answer phrase for strict scoring. No demonstration touches a target or a control
spine, so every evaluated multi-hop question is new to every model, and the facts of a
target chain still never occur together in one training text.

**World anchors.** Fine-tuning on fictional facts alone made the model answer *every*
question in our format with a fictional name: accuracy on the 40 world-knowledge control
questions fell from 52% to 12% after two epochs, before any unlearning. The world-knowledge
stop condition would then be useless. We add 110 real-world question-answer pairs (4 copies
each, 1.7% of the texts) to both corpora. No anchor repeats a control fact. With anchors the
learned model keeps 68% (0.6B) and 90% (1.7B) on the control.

**Chain sets.**

| set | chains | meaning |
|---|---|---|
| `mh_target` | 1,200 | k = 2..5, through the forgotten edge (the leakage test) |
| `mh_neighbour` | 800 | on a target spine, but not through the forgotten edge (damage near the forgotten fact) |
| `mh_control` | 1,000 | all sub-paths of the 100 control spines (general multi-hop ability) |
| `mh_demo` | 1,000 | the demonstrations themselves (training accuracy) |

1-hop probes: `qa_train` (training questions; the forget part is what unlearning is tuned
on), `qa_heldout` (2 unseen question templates), `stmt_heldout` (2 unseen statement
templates, as sentence completion).

**Fine-tuning.** Full parameters, AdamW (0.9/0.999, weight decay 0.01), learning rate
2e-5, 10% warm-up then cosine decay, batch size 32, 10 epochs, bf16 autocast. Identical for
the learned and the clean model. We used plain AdamW (not 8-bit Adam) because the models fit
in memory.

**Unlearning.** See the header of `unlearn.py` for the losses and the selection rule.
Differences from the plan of action:
- The forget term uses only the answer tokens (as Step 3 recommended), because forget and
  retain texts share all template words.
- Checkpoints are evaluated every 25 steps (GradDiff: every 10) within the 500-step budget, and we keep the best
  *valid* one. Valid = retain accuracy at most 3 points below the learned model, and the
  world control at most 4 questions (10 points) below it.
- RMU trains only three weight matrices, so it uses larger learning rates (4e-4 to 1.6e-3).
  Its forget term covers only the token positions from the subject name onward, in texts
  that name the subject first; with the standard form (all positions) it removed retain
  facts as fast as forget facts.
- Retain weights were tuned with direct questions only, because the first settings missed
  the 2% forget target: GradDiff 5 (with weight 1 there was no checkpoint between "nothing
  forgotten" and "everything destroyed"), NPO 1, RMU 10 on 0.6B and 5 on 1.7B. The
  first-round outputs are kept in `results/<size>/first_round/`.

**Scoring.** Strict (primary): a 1-hop answer must equal the gold name; a multi-hop answer
must contain the phrase "Therefore, the answer is:" followed by the gold name. Lenient: the
gold name occurs anywhere in the answer. Greedy decoding.

**Statistics.** 95% intervals from a bootstrap over spines (10,000 resamples), paired for
differences between models.
