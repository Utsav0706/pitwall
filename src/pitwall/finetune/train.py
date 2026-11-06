"""Fine-tune the router model with LoRA and export it as a GGUF bundle.

    python -m pitwall.finetune.train --train data/derived/train_old.jsonl \\
        --out bundles/ft-train_old --llama-cpp .llama.cpp

Steps:
1. Hold out 20% of the training file (stratified, seeded) as validation.
2. Train LoRA on the other 80% for up to --epochs epochs; after each epoch, read the
   model's probabilities for " S", " M", " L" on validation.
3. Pick the epoch count and threshold with the lowest validation routing cost (the LR
   baseline's objective and threshold grid).
4. Train a fresh LoRA on the whole file for that many epochs (as the LR refits on all).
5. Merge LoRA into the base model, convert to GGUF, quantise to Q4_K_M, write bundle.json.

The test set is not touched here: it is scored once, on the quantised model, in M4.

Why plain LoRA, not QLoRA: QLoRA stores the base model in 4 bits so big models fit small
GPUs. A 0.6B model fits a free T4 at full precision, so 4 bits would only add a
dependency (bitsandbytes), slower steps and some accuracy loss.

Why a hand-written loop, not Trainer: it is ~40 lines, every step is visible, and the
only special part (loss on the answer token alone) is explicit.

--smoke builds a tiny random Qwen3 (2 layers) with the real tokenizer and trains it for
one epoch on CPU: it tests this whole script, export included, in a minute.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pitwall import tracking
from pitwall.data import Pair, load_pairs, sha256_file
from pitwall.finetune import bundle, core, llamacpp
from pitwall.metrics import TIERS
from pitwall.textformat import ANSWERS, FORMAT_VERSION, render_pair

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
QUANT = "Q4_K_M"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m pitwall.finetune.train", description=__doc__)
    p.add_argument("--train", type=Path, default=Path("data/derived/train_old.jsonl"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--revision", default=BASE_REVISION)
    p.add_argument("--epochs", type=int, default=4, help="most epochs to try on validation")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--llama-cpp", type=Path, default=None, help="folder for llama.cpp tools")
    p.add_argument("--smoke", action="store_true", help="tiny random model, CPU, 1 epoch")
    return p.parse_args(argv)


# --- model and data ----------------------------------------------------------------------


def load_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base, revision=args.revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def answer_ids(tok: Any) -> dict[str, int]:
    """The single token id of each answer; refuses a tokenizer that splits them."""
    ids = {}
    for tier, text in ANSWERS.items():
        encoded = tok.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"answer {text!r} is {len(encoded)} tokens; it must be one")
        ids[tier] = encoded[0]
    return ids


def load_model(args: argparse.Namespace, tok: Any, device: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    if args.smoke:
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(args.seed)
        config = Qwen3Config(
            vocab_size=len(tok),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=2048,
            tie_word_embeddings=True,
        )
        return Qwen3ForCausalLM(config).to(device)
    # fp32 weights: a T4 has no bf16, and pure fp16 weights can overflow; autocast gives
    # fp16 speed for the matrix multiplications.
    return AutoModelForCausalLM.from_pretrained(
        args.base, revision=args.revision, dtype=torch.float32
    ).to(device)


def add_lora(model: Any, args: argparse.Namespace) -> Any:
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def encode(pairs: list[Pair], tok: Any, max_len: int) -> list[tuple[list[int], int]]:
    """(prompt ids, answer id) per pair."""
    examples = []
    for pair in pairs:
        prompt, answer = render_pair(pair)
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        answer_tok = tok.encode(answer, add_special_tokens=False)
        examples.append(core.make_example(prompt_ids, answer_tok, max_len))
    return examples


def batches(prompts: list[list[int]], size: int, pad_id: int, device: str) -> Any:
    """(input_ids, attention_mask, index of each prompt's last token), right-padded."""
    import torch

    for start in range(0, len(prompts), size):
        chunk = prompts[start : start + size]
        width = max(len(x) for x in chunk)
        yield (
            torch.tensor([x + [pad_id] * (width - len(x)) for x in chunk], device=device),
            torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in chunk], device=device),
            torch.tensor([len(x) - 1 for x in chunk], device=device),
        )


def next_token_logits(model: Any, input_ids: Any, mask: Any, last: Any) -> Any:
    """Vocabulary scores at ONE position per prompt: right after "Tier:".

    The usual causal-LM forward scores every position (batch x length x 152k vocab):
    ~4 GB at batch 8 and ~900 tokens, which ran a T4 out of memory, all for positions
    whose loss is masked anyway. So run the transformer body, take the hidden state at
    the last prompt token, and apply the output layer there only. Same numbers, a few MB.
    """
    import torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=input_ids, attention_mask=mask).last_hidden_state
    picked = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    return base.lm_head(picked)  # (batch, vocab)


# --- training and validation --------------------------------------------------------------


def train_epochs(
    model: Any,
    examples: list[tuple[list[int], int]],
    args: argparse.Namespace,
    epochs: int,
    pad_id: int,
    device: str,
    after_epoch: Any = None,
) -> list[float]:
    """Train for `epochs`; returns the mean loss of each epoch. The loss is cross-entropy
    of the answer token over the whole vocabulary, at the position after "Tier:" only."""
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = epochs * math.ceil(len(examples) / args.batch)
    warmup = max(1, steps // 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (steps - s) / steps)
    )
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(args.seed)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = examples[:]
        rng.shuffle(order)
        total, count = 0.0, 0
        for start in range(0, len(order), args.batch):
            chunk = order[start : start + args.batch]
            input_ids, mask, last = next(batches([x for x, _ in chunk], len(chunk), pad_id, device))
            target = torch.tensor([y for _, y in chunk], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = next_token_logits(model, input_ids, mask, last)
            loss = torch.nn.functional.cross_entropy(logits.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
            total += float(loss) * len(chunk)
            count += len(chunk)
        losses.append(total / count)
        print(f"epoch {epoch}: train loss {losses[-1]:.4f}", flush=True)
        if after_epoch is not None:
            after_epoch(epoch)
    return losses


def answer_probs(
    model: Any, pairs: list[Pair], tok: Any, ids: dict[str, int], device: str, batch: int
) -> np.ndarray:
    """(n, 3) P(S), P(M), P(L) at the position after 'Tier:', over the three answers."""
    import torch

    model.eval()
    prompts = [tok.encode(render_pair(p)[0], add_special_tokens=False) for p in pairs]
    answer_cols = [ids[t] for t in TIERS]
    out = []
    use_amp = device == "cuda"
    with torch.no_grad():
        for input_ids, mask, last in batches(prompts, batch, tok.pad_token_id, device):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = next_token_logits(model, input_ids, mask, last)
            out.append(logits[:, answer_cols].float().cpu().numpy())
    return core.renormalise(np.concatenate(out))


# --- export --------------------------------------------------------------------------------


def export_gguf(merged_dir: Path, out_dir: Path, tools: llamacpp.LlamaCpp) -> bundle.GgufInfo:
    f16 = out_dir / "model-f16.gguf"
    final = out_dir / bundle.MODEL_FILE
    subprocess.run(
        [
            sys.executable,
            str(tools.convert_script),
            str(merged_dir),
            "--outtype",
            "f16",
            "--outfile",
            str(f16),
        ],
        check=True,
    )
    subprocess.run([str(tools.quantize), str(f16), str(final), QUANT], check=True, env=tools.env())
    f16.unlink()
    return bundle.GgufInfo(
        quant=QUANT,
        sha256=bundle.sha256_file(final),
        bytes=final.stat().st_size,
        llama_cpp_tag=llamacpp.TAG,
        llama_cpp_commit=llamacpp.COMMIT,
    )


def library_versions() -> dict[str, str]:
    import peft
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }


def main(argv: list[str]) -> int:
    import torch

    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() and not args.smoke else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = load_pairs(args.train)
    if args.smoke:  # 10 of each tier (or all of a rarer one), sorted by id
        pairs = [p for t in TIERS for p in [q for q in pairs if q.label == t][:10]]
        args.epochs = 1
    train_part, val_part = core.validation_split(pairs, seed=args.seed)
    tok = load_tokenizer(args)
    ids = answer_ids(tok)
    print(f"device {device}; {len(train_part)} train, {len(val_part)} validation", flush=True)

    # Phase 1: train on 80%, read validation after each epoch.
    started = time.time()
    model = add_lora(load_model(args, tok, device), args)
    val_labels = np.asarray([TIERS.index(p.label) for p in val_part])
    results: list[core.EpochResult] = []

    def on_epoch(epoch: int) -> None:
        probs = answer_probs(model, val_part, tok, ids, device, args.batch)
        results.append(core.EpochResult(epoch, probs, val_labels))

    train_epochs(
        model,
        encode(train_part, tok, args.max_len),
        args,
        args.epochs,
        tok.pad_token_id,
        device,
        on_epoch,
    )
    best, rows = core.choose(results)
    print(f"chosen on validation: {best}", flush=True)

    # Phase 2: a fresh LoRA on everything, for the chosen number of epochs.
    del model
    model = add_lora(load_model(args, tok, device), args)
    losses = train_epochs(
        model, encode(pairs, tok, args.max_len), args, best["epoch"], tok.pad_token_id, device
    )
    seconds = round(time.time() - started, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.out / bundle.ADAPTER_DIR
    model.save_pretrained(adapter_dir)
    gguf_info = None
    if args.llama_cpp is not None:
        tools = llamacpp.fetch(args.llama_cpp)
        merged_dir = args.out / "merged"
        merged = model.merge_and_unload()
        merged.to(torch.float16).save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)
        gguf_info = export_gguf(merged_dir, args.out, tools)
        shutil.rmtree(merged_dir)

    derived = args.train.parent / "manifest.json"
    data_manifest = json.loads(derived.read_text(encoding="utf-8")) if derived.exists() else {}
    commit, dirty = tracking.git_state()
    manifest = bundle.Manifest(
        name=args.out.name,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=commit,
        git_dirty=dirty,
        base_model=args.base if not args.smoke else "smoke:tiny-random-qwen3",
        base_revision=args.revision,
        text_format=FORMAT_VERSION,
        answer_token_ids=ids,
        prompt_template=data_manifest.get("prompt_template", "unknown"),
        prompt_sha256=data_manifest.get("prompt_template_sha256", "unknown"),
        data_version=tracking.data_version(),
        train_file=args.train.as_posix(),
        train_sha256=sha256_file(args.train),
        examples=len(pairs),
        validation_examples=len(val_part),
        lora={
            "r": args.r,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": TARGET_MODULES,
        },
        training={
            "lr": args.lr,
            "batch": args.batch,
            "max_len": args.max_len,
            "seed": args.seed,
            "epochs_searched": args.epochs,
            "final_epoch_losses": [round(x, 5) for x in losses],
            "seconds": seconds,
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "precision": "fp32 weights + fp16 autocast" if device == "cuda" else "fp32",
        },
        chosen=best,
        validation=rows,
        libraries=library_versions(),
        gguf=gguf_info,
    )
    bundle.write_manifest(args.out, manifest)
    problems = bundle.check(args.out, require_gguf=gguf_info is not None)
    if problems:
        print("bundle problems:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    print(f"bundle written to {args.out} ({seconds} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
