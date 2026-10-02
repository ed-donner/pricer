# System One models on the pricer: Jev vs Kev vs Nimble vs Laya

Ed's `lab.ipynb` asks whether Jev, a hosted System One decision model, can estimate prices as well as a frontier LLM. This contribution adds the open alternatives to the same test and asks one more question: **can a small open model, fine-tuned on your own data in half an hour on a laptop, compete with the hosted ones?**

## Results

Same 200 test items, same harness (`pricer.evaluator.Tester`). Every System One model uses Ed's `jev_two_pass_median` method.

| model | average error | r² | cost per call | latency |
|---|---|---|---|---|
| Luna (frontier LLM, Ed's prompt) | **$51.63** | 47.9% | $0.000030 | 0.95 s |
| Jev | $52.77 | **67.8%** | $0.000097 | 0.21 s |
| Nimble 9B, local | $68.10 | 38.6% | free | 3.46 s |
| **Laya 421M, fine-tuned, local** | $70.85 | 28.0% | free | **0.08 s** |
| Kev-4B | $73.79 | 45.2% | $0.000067 | 1.40 s |
| Laya 421M, zero-shot | $219.97 | −191.6% | free | 0.08 s |

Measured on an M4 Pro MacBook. Latency is per call inside the predictor, so the hosted models include the network round trip.

What stands out:

- **Fine-tuning is what makes Laya useful.** Zero-shot it is far off ($220); after 30 minutes of training on 6,000 of the 800,000 training items, its error drops to $71, ahead of Kev-4B and within $3 of Nimble.
- **Fine-tuned Laya is the fastest model here by a wide margin, and free to run:** about 40x faster than Nimble and 2.5x faster than Jev. It is a 421M-parameter encoder, about 1/20th of Nimble, running in-process.
- **Jev and Luna are still clearly ahead**, and Laya's lower r² means it makes more large misses even when its average error is competitive.
- Re-runs move the numbers by a dollar or two, so small gaps aren't meaningful.

## The models

| model | who | how it runs | open weights |
|---|---|---|---|
| Jev | TypeSafe | OpenRouter Decisions API | no |
| Kev-4B | Jared Palmer | OpenRouter Decisions API (also self-hostable) | yes, Apache-2.0 |
| Nimble 9B | Bespoke Labs | Ollama `/v1/systemone`, locally | yes |
| Laya 421M | Convai Innovations | in-process, locally | yes, Apache-2.0 |

All four take the same request shape (a `state` plus typed `choice` questions) and return a probability for every option, which is what the two-pass median needs.

**Not quite the same question for everyone.** Jev and Kev get Ed's exact question: $20 and $100 grids (50 + 10 options). Nimble accepts at most 26 options per choice question, and Laya fits about 192 tokens of question, so both get a $40 fine grid (25 options) with the same $100 coarse grid. `spread` puts every grid on the same $20 lattice before combining.

## Fine-tuning Laya

Laya has no `train()` function; its official recipe is a hand-written PyTorch loop. `system_one.train_laya` is a compact single-device version:

- Each item becomes two training rows, one per question, labelled with the bucket its price falls in.
- **RLCD**: treat the logits as the mean of a Gaussian policy, sample a group of noisy probability reports, and reward each with `laya.proper_reward`, a strictly proper scoring rule. Proper scoring rules are maximised only by honest probabilities, so training pushes towards calibrated confidence. The group mean is the REINFORCE baseline.
- Cross-entropy alongside, as in Laya's recipe.

One lesson from getting this to work: also dividing the advantage by the group's standard deviation, a common GRPO trick, made training diverge. It gives a confidently right example the same push as a wrong one, so the model is rewarded for ever-sharper logits, the std collapses, and the gradients explode. The plain group-mean baseline is stable.

`system_one.calibrate_laya` then fits a temperature per question on validation items. This matters for the pricer specifically: the base checkpoint ships an invalid temperature (0.10) for questions with 11+ options, which Laya overrides to 0.5 with a warning. The median reads the whole distribution, so an unfitted temperature shifts every estimate.

## Running it

From the repo root, after Ed's setup (`uv sync`, `.env` with `OPENROUTER_API_KEY` and `HF_TOKEN`):

```bash
uv pip install "laya==0.3.21" nbclient
ollama pull nimble                 # Ollama 0.35+, ~9.5 GB
```

Then open `system_one_comparison.ipynb` here and run it top to bottom.

- API calls cost about $0.04 in total (Jev ~$0.019, Kev ~$0.013, Luna ~$0.006).
- Nimble runs one item at a time, about 12 minutes on an M4 Pro.
- Fine-tuning Laya takes about 30 minutes on an Apple-silicon Mac, or use a free Colab/Kaggle GPU. It's skipped if `checkpoints/laya-pricer` already exists, and progress is logged to `checkpoints/laya-pricer/train_log.txt`.
- The Laya base checkpoint is pinned to revision `55cf4c4e` in `system_one.py`.
- On a Mac, keep it plugged in and awake for the long cells (`caffeinate -dims`, lid open).

## Files

| file | what |
|---|---|
| `system_one_comparison.ipynb` | the comparison, with outputs |
| `system_one.py` | price grids, two-pass median, one predictor per model, Laya training and calibration |

The notebook ends with an optional section on publishing the fine-tuned model to Hugging Face.

Contributed by Xianguan Li.
