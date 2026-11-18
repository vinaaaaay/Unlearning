# Results of the main experiment (Steps 4-7 and 9)

Two seeds of the learned model per size (the detailed sections are seed 1; seed 2 is in
"Second seed" near the end); clean models with seeds 1 and 2. All numbers are
strict accuracy in %, greedy decoding, with a 95% interval from a bootstrap over spines.
"Leakage" is the accuracy of an unlearned model minus the accuracy of the clean model
(seed 1) on the same questions, not clipped. Full tables: `results/<size>/report_s1.md`,
`results/<size>/sampling_s1.md`. Figures: `results/figures/`.

## Qwen3-0.6B

### The test is valid

| | learned | clean (seed 1) | clean (seed 2) |
|---|---|---|---|
| 1-hop, forget facts, held-out questions | 100.0 | 0.0 | 0.0 |
| 1-hop, retain facts, held-out questions | 100.0 | 100.0 | 100.0 |
| multi-hop through a forgotten fact (1,200 chains) | 97.5 [95.8, 98.9] | 0.4 [0.0, 1.1] | 0.0 |
| multi-hop on control spines (1,000 chains) | 95.3 | 98.8 | 98.3 |

The learned model answers the target chains and the clean model does not, in every slice
(chain length x position). The signal A(learned) - A(clean) is +95 points or more in every
slice, so every slice is a valid leakage test. 1,165 of the 1,200 chains pass the filter
"answered by the learned model, not answered by the clean model".

### Unlearning reached the target with direct questions

Checkpoints were chosen with direct training-template questions only.

| method | setting chosen | forget (direct) | retain (direct) | world control |
|---|---|---|---|---|
| learned model | - | 100.0 | 100.0 | 70.0 |
| GradDiff | retain weight 5, lr 1e-5, step 430 | 0.0 | 99.2 | 72.5 |
| NPO | retain weight 1, lr 1e-5, step 350 | 0.0 | 100.0 | 70.0 |
| RMU | retain weight 10, lr 8e-4, step 400 | 1.5 | 99.2 | 60.0 |

All three are within the limits (forget <= 2%, retain at most 3 points lower, world control
at most 4 questions lower).

### Leakage

| slice | n | GradDiff | NPO | RMU |
|---|---|---|---|---|
| 1-hop, direct questions (tuned on) | 800 | -0.1 [-0.4, +0.0] | -0.1 [-0.4, +0.0] | +1.5 [+0.0, +3.5] |
| 1-hop, held-out questions | 400 | +0.0 | +0.0 | +1.8 [+0.2, +3.5] |
| 1-hop, held-out statements | 400 | +0.0 | +0.0 | +5.5 [+3.2, +8.0] |
| **multi-hop, all** | 1200 | **+0.8 [-0.3, +2.1]** | **+3.7 [+1.5, +6.2]** | **+57.9 [+53.7, +62.1]** |
| k = 2 | 320 | +0.0 [-0.9, +0.9] | +3.1 [+1.3, +5.3] | +45.6 [+41.9, +49.4] |
| k = 3 | 360 | +0.6 [-0.8, +1.9] | +3.9 [+1.4, +6.8] | +57.5 [+52.6, +62.2] |
| k = 4 | 320 | +1.6 [-0.3, +3.7] | +4.1 [+1.3, +7.3] | +64.7 [+58.8, +70.3] |
| k = 5 | 200 | +1.5 [+0.0, +3.5] | +3.5 [+1.0, +6.5] | +67.5 [+61.0, +74.0] |
| forgotten fact at the start | 400 | -0.5 [-1.5, +0.0] | +1.8 [+0.0, +3.9] | +5.0 [+1.7, +8.7] |
| forgotten fact in the middle | 400 | +2.2 [+0.0, +5.0] | +7.2 [+3.0, +12.0] | +78.8 [+72.1, +84.8] |
| forgotten fact at the end | 400 | +0.8 [-0.5, +2.2] | +2.0 [+0.5, +3.9] | +90.0 [+85.8, +93.9] |

On the 1,165 kept chains, with the second clean model (0.0%) as reference, the per-question
leakage is +1.3 [+0.3, +2.4] for GradDiff, +4.0 [+1.8, +6.6] for NPO and +59.4 [+55.0, +63.6]
for RMU.

What this shows:

1. **RMU passes the direct test and fails the multi-hop test.** Direct forget accuracy is
   1.5%, but 58% of the multi-hop chains are answered. The leak depends on where the
   forgotten fact is. If the subject of the forgotten fact is named in the question (start
   position), RMU blocks the chain (+5 points). If the model reaches the subject during its
   own reasoning (middle or end), it states the forgotten fact (+79 and +90 points). This
   is also why leakage grows with the chain length: longer chains have fewer start-position
   cases.
2. **NPO leaks a little.** +3.7 points overall, clearly above the clean model, and +7.2
   points when the forgotten fact is in the middle.
3. **GradDiff does not leak in a measurable way**, but it pays for this with utility that
   the selection questions do not see (next table).
4. **Chain length alone has no clear effect** for GradDiff and NPO. The effect of position
   is larger than the effect of length.

### Utility that the selection questions do not measure

| probe set | learned | GradDiff | NPO | RMU |
|---|---|---|---|---|
| retain facts, training questions (selection) | 100.0 | 98.4 | 99.7 | 99.3 |
| retain facts, held-out questions | 100.0 | 94.1 | 99.6 | 99.4 |
| retain facts, held-out statements | 100.0 | 88.7 | 98.8 | 99.2 |
| multi-hop, same spine, not through the forgotten fact | 98.9 | 82.2 | 95.4 | 96.9 |
| multi-hop, control spines | 95.3 | 81.2 | 91.8 | 94.6 |

GradDiff first damages the whole model (at step 80, forget accuracy is 4% and retain
accuracy 28%) and the retain term then rebuilds the retain facts over the next 350 steps. The rebuilt model answers the
training questions again but is weaker on wording it was not rebuilt on and on multi-hop
chains that do not use a forgotten fact.

### Where the reasoning breaks

Share of target chains (%) whose reasoning text states the correct object of a hop.

| model | forgotten hop | retained hops before it | retained hops after it | answer phrase present |
|---|---|---|---|---|
| learned | 99.2 | 99.4 | 97.5 | 100.0 |
| clean | 0.2 | 98.5 | 0.4 | 99.7 |
| GradDiff | 1.2 | 85.9 | 1.4 | 99.6 |
| NPO | 3.4 | 94.9 | 4.6 | 99.9 |
| RMU | 63.6 | 97.9 | 40.1 | 79.2 |

The clean model, GradDiff and NPO fail at the forgotten hop: they name a wrong object and
continue from it. RMU states the forgotten object in 64% of the chains. In 21% of the chains
RMU writes no answer phrase; 240 of these 249 chains are start-position chains, where its
output is noise.

### Sampling (per-sample and any-of-5, also for the clean model)

Target chains, accuracy minus the clean model at the same temperature.

| T | GradDiff per-sample | GradDiff any-of-5 | NPO per-sample | NPO any-of-5 | RMU per-sample | RMU any-of-5 |
|---|---|---|---|---|---|---|
| 0 (greedy) | +0.8 [-0.3, +2.1] | - | +3.7 [+1.5, +6.2] | - | +57.9 [+53.5, +62.3] | - |
| 0.3 | +0.9 [-0.1, +2.1] | +1.3 [-0.3, +2.9] | +3.8 [+1.7, +6.2] | +5.5 [+2.7, +8.6] | +57.7 [+53.4, +61.8] | +59.0 [+54.8, +63.1] |
| 0.7 | +0.5 [-0.4, +1.3] | +1.5 [+0.0, +3.0] | +3.6 [+1.9, +5.7] | +7.7 [+4.8, +10.9] | +55.9 [+51.7, +60.0] | +59.8 [+55.7, +63.9] |
| 1.0 | +0.2 [-0.5, +0.8] | +1.7 [+0.3, +3.1] | +3.1 [+1.6, +4.9] | +7.7 [+4.7, +10.8] | +52.4 [+48.4, +56.4] | +60.1 [+56.0, +64.1] |

The clean model stays at 0.3-0.8% under sampling (any-of-5 included), so sampling does not
create false leakage here. Sampling five times doubles the NPO leak (+3.7 to +7.7 points).

### First round of unlearning settings (kept for the record)

With the first settings, GradDiff (retain weight 1) had no checkpoint between "nothing
forgotten" and "everything destroyed", and RMU (retain weight 20) stopped at 5.2% direct
forget accuracy. Neither reached the 2% target, so we changed the retain weight (GradDiff 5,
RMU 10), again using direct questions only. The first-round outputs are in
`results/0.6B/first_round/`. The under-unlearned RMU model (5.2% direct) leaked +55 points
on multi-hop chains; the final one (1.5% direct) leaks +58. Stronger direct forgetting did
not reduce the multi-hop leak.

## Qwen3-1.7B

### The test is valid

| | learned | clean (seed 1) | clean (seed 2) |
|---|---|---|---|
| 1-hop, forget facts, held-out questions | 100.0 | 0.0 | 0.0 |
| 1-hop, retain facts, held-out questions | 100.0 | 100.0 | 100.0 |
| multi-hop through a forgotten fact (1,200 chains) | 98.0 [96.4, 99.3] | 0.1 [0.0, 0.3] | 0.2 [0.0, 0.8] |
| multi-hop on control spines (1,000 chains) | 97.9 | 97.4 | 97.4 |

The signal is +94 points or more in every slice. 1,175 of the 1,200 chains pass the filter.

### Unlearning reached the target with direct questions

| method | setting chosen | forget (direct) | retain (direct) | world control |
|---|---|---|---|---|
| learned model | - | 100.0 | 100.0 | 90.0 |
| GradDiff | retain weight 5, lr 1e-5, step 440 | 0.5 | 99.2 | 95.0 |
| NPO | retain weight 1, lr 5e-6, step 375 | 2.0 | 100.0 | 92.5 |
| RMU | retain weight 5, lr 8e-4, step 275 | 2.0 | 99.0 | 82.5 |

RMU needed a smaller retain weight than on 0.6B (5 instead of 10; see "First round" below).
Its world control is 3 questions below the learned model, inside the limit of 4.

### Leakage

| slice | n | GradDiff | NPO | RMU |
|---|---|---|---|---|
| 1-hop, direct questions (tuned on) | 800 | +0.5 [+0.0, +1.5] | +2.0 [+0.5, +4.0] | +2.0 [+0.5, +4.0] |
| 1-hop, held-out questions | 400 | +0.5 [+0.0, +1.5] | +2.0 [+0.5, +4.0] | +3.0 [+1.0, +5.2] |
| 1-hop, held-out statements | 400 | +0.5 [+0.0, +1.5] | +1.5 [+0.0, +3.5] | +6.0 [+3.5, +8.8] |
| **multi-hop, all** | 1200 | **+6.1 [+3.4, +9.0]** | **+9.8 [+6.5, +13.4]** | **+43.9 [+38.7, +48.9]** |
| k = 2 | 320 | +5.9 [+3.4, +8.8] | +10.0 [+6.8, +13.6] | +34.1 [+29.7, +38.5] |
| k = 3 | 360 | +6.9 [+3.6, +10.6] | +10.0 [+6.4, +14.0] | +45.3 [+39.9, +50.4] |
| k = 4 | 320 | +6.2 [+3.1, +9.7] | +9.4 [+5.7, +13.4] | +49.4 [+43.2, +55.5] |
| k = 5 | 200 | +4.5 [+2.0, +7.5] | +10.0 [+6.0, +14.5] | +48.5 [+41.5, +55.5] |
| forgotten fact at the start | 400 | +0.8 [+0.0, +2.1] | +15.8 [+10.2, +21.7] | +3.5 [+1.0, +6.6] |
| forgotten fact in the middle | 400 | +8.5 [+4.2, +13.5] | +11.8 [+6.6, +17.7] | +65.5 [+57.3, +73.3] |
| forgotten fact at the end | 400 | +9.0 [+4.9, +13.7] | +2.0 [+0.2, +4.5] | +62.7 [+54.8, +70.7] |

On the 1,175 kept chains, with the second clean model (0.3%) as reference: GradDiff
+6.0 [+3.3, +9.1], NPO +9.6 [+6.3, +13.3], RMU +44.6 [+39.4, +49.7].

What this shows:

1. **All three methods leak on the larger model.** GradDiff and NPO pass the direct test
   (0.5% and 2.0%) but answer 6% and 10% of the multi-hop chains; on 0.6B the same two
   methods leaked about 1 and 4 points.
2. **RMU has the same pattern as on 0.6B**: blocked when the question names the forgotten
   subject (+3.5), open when the model reaches the subject in its own reasoning (+63 to +66).
3. **The position effect differs between methods.** GradDiff leaks in the middle and at the
   end, like RMU but much less. NPO in this seed leaks most at the *start* (+15.8): for the
   31 spines that leak there, the direct questions about the same fact are answered in only
   10% of the cases. The fact is refused as a direct answer and written as the first
   reasoning step. In seed 2 the NPO leak at the start is smaller (+5.8), so the size of
   this effect depends on the seed.
4. **Chain length again has no clear effect** for GradDiff and NPO.

### Utility that the selection questions do not measure

| probe set | learned | GradDiff | NPO | RMU |
|---|---|---|---|---|
| retain facts, training questions (selection) | 100.0 | 98.3 | 99.6 | 99.0 |
| retain facts, held-out questions | 100.0 | 97.5 | 99.6 | 98.9 |
| retain facts, held-out statements | 100.0 | 94.5 | 99.3 | 98.6 |
| multi-hop, same spine, not through the forgotten fact | 99.6 | 80.5 | 95.4 | 97.9 |
| multi-hop, control spines | 97.9 | 86.3 | 98.3 | 98.0 |

GradDiff again loses 11 to 19 points on multi-hop chains that do not use a forgotten fact.

### Where the reasoning breaks

| model | forgotten hop | retained hops before it | retained hops after it | answer phrase present |
|---|---|---|---|---|
| learned | 99.4 | 99.9 | 98.4 | 100.0 |
| clean | 0.0 | 99.0 | 0.1 | 99.8 |
| GradDiff | 7.2 | 87.7 | 3.9 | 76.8 |
| NPO | 2.3 | 96.0 | 14.1 | 100.0 |
| RMU | 46.6 | 98.0 | 31.7 | 69.8 |

GradDiff and RMU write no answer phrase on most start-position chains (278 and 337 of 400):
the output after the forgotten subject is a repeated token or noise.

### Sampling

Target chains at T = 0.7, accuracy minus the clean model (clean: 0.0% per sample, 0.1% any-of-5).

| | GradDiff | NPO | RMU |
|---|---|---|---|
| greedy | +6.1 [+3.4, +9.1] | +9.8 [+6.4, +13.4] | +43.9 [+38.7, +48.9] |
| per-sample | +5.2 [+2.9, +7.8] | +10.4 [+7.2, +14.1] | +43.0 [+38.1, +47.6] |
| any-of-5 | +8.1 [+4.9, +11.6] | +14.7 [+10.8, +18.9] | +47.0 [+42.2, +51.7] |

### First round of RMU settings (kept for the record)

With the 0.6B setting (retain weight 10), RMU stopped at 7.9% direct forget accuracy on
1.7B and leaked +56.8 [+51.8, +61.5] on multi-hop chains. With retain weight 5 it reaches
2.0% direct and leaks +43.9. Outputs of the first round: `results/1.7B/first_round/`.

## Both sizes

| | 0.6B direct | 0.6B multi-hop | 1.7B direct | 1.7B multi-hop |
|---|---|---|---|---|
| GradDiff | -0.1 | +0.8 [-0.3, +2.1] | +0.5 | +6.1 [+3.4, +9.0] |
| NPO | -0.1 | +3.7 [+1.5, +6.2] | +2.0 | +9.8 [+6.5, +13.4] |
| RMU | +1.5 | +57.9 [+53.7, +62.1] | +2.0 | +43.9 [+38.7, +48.9] |

(Leakage above the clean model; "direct" = the questions used to choose the checkpoint.)
A direct question underestimates what is left in the model in five of the six cases, and
the gap is not a guessing effect: the clean model answers at most 0.4% of the same chains.

## Second seed of the learned model

Same pipeline and settings, learned model trained with seed 2, unlearned again. Clean
models are the same two. All checkpoints reached the forget target within the limits.
Every slice is again a valid test.

| | 0.6B seed 1 | 0.6B seed 2 | 1.7B seed 1 | 1.7B seed 2 |
|---|---|---|---|---|
| **GradDiff** direct | -0.1 | -0.1 | +0.5 | +0.5 |
| GradDiff multi-hop | +0.8 [-0.3, +2.1] | +0.8 [-0.2, +2.0] | +6.1 [+3.4, +9.0] | +5.2 [+2.9, +7.6] |
| GradDiff start / middle / end | -0.5 / +2.2 / +0.8 | -0.5 / +1.2 / +1.8 | +0.8 / +8.5 / +9.0 | +1.5 / +7.8 / +6.2 |
| **NPO** direct | -0.1 | -0.1 | +2.0 | +2.0 |
| NPO multi-hop | +3.7 [+1.5, +6.2] | +2.0 [+0.3, +4.0] | +9.8 [+6.5, +13.4] | +6.0 [+3.2, +9.2] |
| NPO start / middle / end | +1.8 / +7.2 / +2.0 | +2.5 / +3.5 / +0.0 | +15.8 / +11.8 / +2.0 | +5.8 / +8.0 / +4.2 |
| **RMU** direct | +1.5 | +1.6 | +2.0 | +0.0 |
| RMU multi-hop | +57.9 [+53.7, +62.1] | +62.3 [+57.8, +66.4] | +43.9 [+38.7, +48.9] | +61.5 [+57.2, +65.7] |
| RMU start / middle / end | +5.0 / +78.8 / +90.0 | +11.0 / +85.5 / +90.2 | +3.5 / +65.5 / +62.7 | +13.5 / +87.0 / +84.0 |
| control multi-hop: learned / GradDiff | 95.3 / 81.2 | 97.4 / 76.0 | 97.9 / 86.3 | 99.5 / 85.6 |

What holds in both seeds:

- RMU leaks 44 to 62 points on multi-hop chains while direct accuracy is 0 to 2%, and the
  leak is in the middle and end positions. In 1.7B seed 2, RMU has 0.0% direct accuracy and
  still leaks +61.5.
- GradDiff and NPO leak more on 1.7B (5 to 10 points) than on 0.6B (1 to 4 points).
- GradDiff loses 12 to 21 points on control multi-hop chains.

What does not hold in both seeds: the order of NPO's positions (start largest in 1.7B
seed 1 only), and the exact size of the RMU leak on 1.7B (44 vs 62 points; the two seeds
also selected different steps, 275 and 400).

## Limits

- Two seeds of the learned model per size. The intervals cover the choice of chains, not
  the training seed; with two seeds we can only say which findings repeat.
- RMU here is our variant: it trains three MLP output matrices and steers only the token
  positions from the subject name onward, in texts that name the subject first (see
  README). The standard form, which steers every position of a forget text, removed retain
  facts as fast as forget facts on this dataset, because forget and retain texts share all
  template words. The size of the RMU leak may depend on this choice.
- Every model was trained on 1,000 multi-hop demonstrations (on other spines). The models
  therefore reason step by step in a fixed format. Leakage without written reasoning was
  not measured.
- The world-knowledge control has 40 questions; one question is 2.5 points.
- Qwen3-8B was not run.
