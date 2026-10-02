"""System One price estimators (Jev, Kev, Nimble, Laya) plus Laya fine-tuning and calibration.

Every model gets the same method as Ed's jev_two_pass_median: ask for a fine and a coarse price
bucket in one call, spread both distributions onto a $20 lattice, multiply, take the median.
"""
import json
import math
import os
import random
import shutil
import time

# Pin the Laya base checkpoint so a new upstream release can't change results between runs
LAYA_MODEL = "convaiinnovations/laya"
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
os.environ.setdefault("LAYA_REVISION", LAYA_REVISION)
os.environ.setdefault("USE_TF", "0")  # avoids a TensorFlow/abseil deadlock in laya.load()

import numpy as np
import requests

PRICE_QUESTION = "How much does this product cost at retail in US dollars?"
LATTICE = 20  # every distribution is spread onto $20 buckets before combining
TOP_PRICE = 1000
COARSE_WIDTH = 100

lows = np.arange(0, TOP_PRICE, LATTICE, dtype=float)


def grid(width, descriptions=True):
    """Price buckets as choice criteria. Without descriptions, the label alone is the option text."""
    return {f"${low}-${low + width}": (f"The product sells for between ${low} and ${low + width}" if descriptions else None)
            for low in range(0, TOP_PRICE, width)}


def two_pass_questions(fine_width, descriptions=True):
    return {
        "fine": {"type": "choice", "instructions": PRICE_QUESTION, "criteria": grid(fine_width, descriptions)},
        "coarse": {"type": "choice", "instructions": PRICE_QUESTION, "criteria": grid(COARSE_WIDTH, descriptions)},
    }


# Jev and Kev take Ed's exact question: $20 and $100 grids (50 + 10 options)
FULL_QUESTIONS = two_pass_questions(20)
# Nimble allows 2-26 options per choice question, so its fine grid is $40 (25 options)
NIMBLE_QUESTIONS = two_pass_questions(40)
# Laya fits ~192 tokens of question; 25 bare labels fit whole, 50 would be cut to 4 tokens each
LAYA_FINE_WIDTH = 40
LAYA_QUESTIONS = two_pass_questions(LAYA_FINE_WIDTH, descriptions=False)


def spread(probabilities, width):
    """Spread one grid's probabilities evenly across the $20 buckets each of them covers."""
    values = np.array([probabilities.get(name, 0.0) for name in grid(width)])
    return np.repeat(values / (width // LATTICE), width // LATTICE)


def median_price(distribution):
    """The price at which the distribution reaches half its mass, interpolated inside the bucket."""
    distribution = distribution / distribution.sum()
    cumulative = np.cumsum(distribution)
    index = min(int(np.searchsorted(cumulative, 0.5)), len(distribution) - 1)
    below = cumulative[index - 1] if index else 0.0
    if distribution[index] == 0:
        return float(lows[index] + LATTICE / 2)
    return float(lows[index] + (0.5 - below) / distribution[index] * LATTICE)


def two_pass_median(fine_probabilities, fine_width, coarse_probabilities):
    fine = spread(fine_probabilities, fine_width)
    combined = fine * spread(coarse_probabilities, COARSE_WIDTH)
    return median_price(combined if combined.sum() > 0 else fine)


# ---- Predictors: each takes an Item and returns (estimate, cost), as Ed's evaluator expects -------

def openrouter_predictor(openrouter, model):
    """Jev or Kev through OpenRouter's Decisions API."""
    def predict(item):
        response = openrouter.alpha.decisions.create(model=model, state=item.summary, questions=FULL_QUESTIONS)
        answers = response.answers
        return two_pass_median(answers["fine"].probabilities, 20, answers["coarse"].probabilities), response.usage.cost
    return predict


def nimble_predictor(url="http://localhost:11434/v1/systemone"):
    """Nimble through Ollama's local /v1/systemone endpoint. Local, so the cost is zero."""
    def predict(item):
        response = requests.post(url, json={"model": "nimble", "state": item.summary, "questions": NIMBLE_QUESTIONS},
                                 timeout=300)
        response.raise_for_status()
        answers = response.json()["answers"]
        return two_pass_median(answers["fine"]["probabilities"], 40, answers["coarse"]["probabilities"]), 0.0
    return predict


def laya_predictor(agent):
    """Laya running in this process. Local, so the cost is zero. Evaluate with workers=1:
    concurrent forward passes on one model can crash the process on Apple GPUs."""
    def predict(item):
        answers = agent.predict(item.summary, LAYA_QUESTIONS)["answers"]
        return two_pass_median(answers["fine"]["probabilities"], LAYA_FINE_WIDTH, answers["coarse"]["probabilities"]), 0.0
    return predict


# ---- Laya fine-tuning (RLCD), adapted from the single-device loop in Laya's fine-tuning recipe -----

def bucket_index(price, width):
    return min(int(price // width), TOP_PRICE // width - 1)


def train_laya(train_items, val_items, out_dir, size=6000, epochs=1, batch_size=16, group_size=8,
               lr_encoder=2.5e-5, lr_head=1.0e-4, sigma_start=0.4, sigma_end=0.1, ce_weight=1.0, seed=42):
    """Fine-tune Laya on (summary, price) pairs and save a checkpoint laya.load() can read.

    Each item becomes two training rows, one per question (fine and coarse), labelled with the
    bucket its price falls in. The loss is RLCD (REINFORCE against Laya's strictly proper scoring
    rule, with a group-mean baseline) plus cross-entropy, as in Laya's recipe.
    """
    import laya
    import torch
    from huggingface_hub import snapshot_download
    from laya.agent import Agent
    from laya.common import collate_items, proper_reward
    from safetensors.torch import save_file

    random.seed(seed)
    torch.manual_seed(seed)
    items = random.sample(train_items, min(size, len(train_items)))

    # Progress goes to the notebook and to a log file you can `tail -f` while the cell runs
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.txt")

    def log(message):
        print(message, flush=True)
        with open(log_path, "a") as f:
            f.write(message + "\n")

    agent = laya.load(LAYA_MODEL)
    model, tok, device = agent.model, agent.tok, agent.device
    questions = {qid: Agent._to_internal(q) for qid, q in LAYA_QUESTIONS.items()}
    widths = {"fine": LAYA_FINE_WIDTH, "coarse": COARSE_WIDTH}

    def encode(item):
        rows = agent._encode_state(item.summary, list(questions), questions)
        for qid, row in zip(questions, rows):
            label = bucket_index(item.price, widths[qid])
            row["target"] = [1.0 if i == label else 0.0 for i in range(len(row["markers"]))]
            row["label"] = label
        return rows

    log(f"Encoding {len(items):,} items on {device}...")
    examples = [encode(item) for item in items]

    encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": encoder_params, "lr": lr_encoder}, {"params": head_params, "lr": lr_head}],
                                  weight_decay=0.01)
    total_steps = epochs * math.ceil(len(examples) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    def rlcd_loss(logits, target, qtype, mask, sigma):
        noise = torch.randn((group_size, *logits.shape), device=logits.device) * sigma
        sampled = (logits.detach() + noise).masked_fill(~mask, -1e4)
        with torch.no_grad():
            reward = proper_reward(torch.softmax(sampled, -1), target, qtype, mask.float(), w_sph=0.75, w_rps=1.0)
            # Group-mean baseline only: dividing by the group's std rewards ever-sharper logits and diverges
            advantage = reward - reward.mean(0, keepdim=True)
        log_prob = -(((sampled - logits) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
        return -(advantage * log_prob).mean()

    model.train()
    step, start = 0, time.time()
    for epoch in range(epochs):
        random.shuffle(examples)
        for i in range(0, len(examples), batch_size):
            batch = collate_items(examples[i:i + batch_size], pad_id=tok.pad_token_id)
            sigma = sigma_end + (sigma_start - sigma_end) * (1 - step / total_steps)
            inputs = [batch[k].to(device) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
            logits, _ = model(*inputs)
            mask, target, qtype = batch["marker_mask"].to(device), batch["target"].to(device), batch["qtype"].to(device)
            loss_rl = rlcd_loss(logits, target, qtype, mask, sigma)
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            optimizer.zero_grad()
            (loss_rl + ce_weight * loss_ce).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 25 == 0 or step == total_steps:
                log(f"epoch {epoch + 1} step {step}/{total_steps}  loss_rl {loss_rl.item():+.3f}  "
                    f"loss_ce {loss_ce.item():.3f}  |logit| {logits.detach()[mask].abs().mean():.1f}  "
                    f"grad {grad_norm:.1f}  {time.time() - start:.0f}s")
        model.eval()
        errors = [abs(laya_predictor(agent)(item)[0] - item.price) for item in val_items[:300]]
        log(f"--> epoch {epoch + 1}: average error on {len(errors)} validation items ${np.mean(errors):.2f}")
        model.train()

    model.eval()
    base_dir = snapshot_download(LAYA_MODEL, revision=LAYA_REVISION,
                                 allow_patterns=["rl_agent_config.json", "tokenizer/*", "encoder/*"])
    for name in ("tokenizer", "encoder"):
        shutil.copytree(os.path.join(base_dir, name), os.path.join(out_dir, name), dirs_exist_ok=True)
    with open(os.path.join(base_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["training"] = {"fine_tuned_from": f"{LAYA_MODEL}@{LAYA_REVISION}", "task": "retail price buckets",
                       "dataset": "ed-donner/items_full", "items": len(items), "epochs": epochs,
                       "method": "RLCD + cross-entropy"}
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    state = {k: v.detach().to("cpu", torch.float16).contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(out_dir, "model.safetensors"))
    log(f"Saved to {out_dir}")


def calibrate_laya(checkpoint, val_items, size=1000):
    """Fit one temperature per question on validation items and write them into the checkpoint.

    The fine question has 25 options, so Laya reads the "choice:11+" temperature; the coarse one
    has 10 and reads "choice:6-10". The base checkpoint ships 0.10 for choice:11+, which Laya
    raises to 0.5 with a warning: that bucket is uncalibrated until it is refitted here.
    """
    import laya
    import torch
    from laya.agent import Agent
    from laya.common import clamp_temperature, collate_items

    agent = laya.load(checkpoint)
    questions = {qid: Agent._to_internal(q) for qid, q in LAYA_QUESTIONS.items()}
    widths = {"fine": LAYA_FINE_WIDTH, "coarse": COARSE_WIDTH}
    buckets = {"fine": "choice:11+", "coarse": "choice:6-10"}
    items = val_items[:size]

    with open(os.path.join(checkpoint, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    for qid in questions:
        k = len(LAYA_QUESTIONS[qid]["criteria"])
        logits = []
        with torch.no_grad():
            for i in range(0, len(items), 64):
                rows = [agent._encode_state(item.summary, [qid], questions) for item in items[i:i + 64]]
                logits.append(agent._forward(collate_items(rows, pad_id=agent.tok.pad_token_id))[0][:, :k])
        z = torch.tensor(np.concatenate(logits), dtype=torch.float32)
        y = torch.tensor([bucket_index(item.price, widths[qid]) for item in items])
        log_t = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

        def closure():
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = clamp_temperature(log_t.exp().item())
        print(f"{qid}: {k} options, fitted temperature {temperature:.3f} -> {buckets[qid]}")
        cfg["temperature_by_options"][buckets[qid]] = temperature
    with open(os.path.join(checkpoint, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
