"""LoRA fine-tune of Qwen2.5-1.5B-Instruct to distill the Supervisor router.

Approach note (why LoRA-fp16, not 4-bit QLoRA):
  Literal QLoRA = 4-bit (bitsandbytes) base + LoRA adapters. bitsandbytes
  4-bit is CUDA-only; it does not run on Apple-Silicon MPS. More to the point,
  a 1.5B model in fp16 is ~3GB and fits trivially in the M5 Pro's 48GB unified
  memory — 4-bit quantization exists to squeeze large models onto small VRAM,
  a constraint we don't have here. So we use LoRA adapters on the fp16 base via
  MPS: same parameter-efficient-fine-tuning idea, appropriate to the hardware.
  Documented as a deliberate, defensible deviation from literal 4-bit QLoRA.

Trains on the oversampled train_chat.jsonl (see format_dataset.py), computing
loss on the assistant completion only (prompt tokens masked). Saves the LoRA
adapter to train/adapter for eval_routing.py --backend local.

Run (after format_dataset.py):
  python -m experiments.qlora_supervisor.train_lora
"""
import json
import sys
from pathlib import Path

import torch
# peft 0.19.1 references torch.distributed.tensor.DTensor, but torch 2.12.1
# does not auto-import that submodule, so peft hits AttributeError. Importing
# it explicitly populates the attribute. Guarded for older torch.
try:
    import torch.distributed.tensor  # noqa: F401
except Exception:  # noqa: BLE001
    pass
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from . import config
from .format_dataset import BASE_MODEL, TRAIN_CHAT_PATH

ADAPTER_DIR = config.DATA_DIR.parent / "train" / "adapter"
MAX_LEN = 512


def _load_examples(tok):
    """Tokenize each chat example; mask prompt tokens so loss is only on the
    assistant JSON label."""
    rows = [json.loads(l) for l in open(TRAIN_CHAT_PATH) if l.strip()]
    feats = []
    for r in rows:
        msgs = r["messages"]
        prompt_text = tok.apply_chat_template(
            msgs[:1], tokenize=False, add_generation_prompt=True
        )
        full_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        full = tok(full_text, truncation=True, max_length=MAX_LEN)
        prompt = tok(prompt_text, truncation=True, max_length=MAX_LEN)
        input_ids = full["input_ids"]
        labels = list(input_ids)
        for i in range(min(len(prompt["input_ids"]), len(labels))):
            labels[i] = -100  # mask the prompt; train only on completion
        feats.append({"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels})
    return feats


class _Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(len(f["input_ids"]) for f in batch)
        ids, masks, labels = [], [], []
        for f in batch:
            pad = maxlen - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * pad)
            masks.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
            "labels": torch.tensor(labels),
        }


def main() -> int:
    if not TRAIN_CHAT_PATH.exists():
        raise SystemExit(
            f"No {TRAIN_CHAT_PATH.name}. Run format_dataset.py first."
        )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}  base={BASE_MODEL}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # fp32 base for TRAINING stability on MPS (fp16 without a loss scaler is
    # NaN-prone on MPS, and 1.5B fp32 ~6GB fits easily in 48GB). Inference in
    # eval_routing.py uses fp16 for realistic local-deployment latency.
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.to(device)

    feats = _load_examples(tok)
    print(f"training examples: {len(feats)}")

    args = TrainingArguments(
        output_dir=str(ADAPTER_DIR.parent / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        fp16=False, bf16=False,   # MPS: keep default precision, no CUDA AMP
        use_cpu=(device == "cpu"),
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=feats, data_collator=_Collator(tok.pad_token_id),
    )
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tok.save_pretrained(str(ADAPTER_DIR))
    print(f"saved adapter → {ADAPTER_DIR}")
    print("next: python -m experiments.qlora_supervisor.eval_routing "
          f"--backend local --adapter {ADAPTER_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
