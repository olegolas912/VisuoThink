import os
import re
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import wandb

IGNORE_INDEX = -100


# ========= Utils =========

def extract_answer(text: str) -> str:
    m = re.search(r"Answer\s*[:\-]?\s*([A-Z0-9]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().upper()
    text = text.strip().upper()
    m2 = re.search(r"\b([A-Z]|[0-9]{1,3})\b", text)
    return m2.group(1) if m2 else text[:3]


def _build_messages(human_text: str, gpt_text: str):
    messages_full = [
        {"role": "user", "content": [
            {"type": "image", "image": "<image>"},
            {"type": "text", "text": human_text.replace("<image>", "").strip()},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": gpt_text}]},
    ]
    messages_prompt = messages_full[:-1]
    return messages_full, messages_prompt


# ========= Torch Dataset =========

class GeoQADataset(Dataset):
    """
    «Классический» датасет:
    - хранит пути к изображениям и исходные conversation
    - НИЧЕГО не токенизирует внутри __getitem__
    - возвращает PIL Image + тексты-шаблоны; batching делает collate_fn
    """
    def __init__(self, hf_split, image_root: str, processor):
        self.items = list(hf_split)  # фиксируем snapshot
        self.image_root = image_root
        self.processor = processor

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, rel: str) -> Image.Image:
        path = os.path.join(self.image_root, rel) if self.image_root else rel
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obj = self.items[idx]
        img_rel = obj["image"]
        conv = obj["conversations"]
        human = next(m for m in conv if m["from"] == "human")
        gpt = next(m for m in conv if m["from"] == "gpt")

        messages_full, messages_prompt = _build_messages(human["value"], gpt["value"])
        full_text = self.processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=False)

        return {
            "image": self._load_image(img_rel),
            "full_text": full_text,
            "prompt_text": prompt_text,
        }


# ========= Collate =========

class VLDataCollator:
    def __init__(self, processor):
        self.processor = processor
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def __call__(self, features):
        images = [f["image"] for f in features]
        full_texts   = [f["full_text"] for f in features]
        prompt_texts = [f["prompt_text"] for f in features]

        enc_full = self.processor(
            text=full_texts, images=images,
            return_tensors="pt", padding=True, truncation=False
        )
        enc_prompt = self.processor(
            text=prompt_texts, images=images,
            return_tensors="pt", padding=True, truncation=False
        )

        input_ids      = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]
        pixel_values   = enc_full["pixel_values"]

        # ⛳️ ВАЖНО: забираем image_grid_thw и передаём в модель
        image_grid_thw = enc_full.get("image_grid_thw", None)  # Tensor [B,3] у Qwen3-VL

        # маскируем префикс
        prompt_lens = enc_prompt["attention_mask"].sum(dim=1)  # [B]
        labels = input_ids.clone()
        for i, L in enumerate(prompt_lens.tolist()):
            labels[i, :int(L)] = -100

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }
        if image_grid_thw is not None:
            batch["image_grid_thw"] = image_grid_thw
        return batch


# ========= Метрики =========

def compute_metrics_factory(processor):
    pad_id = processor.tokenizer.pad_token_id or 0

    def _replace_ignore_with_pad(arr_2d):
        out = []
        for row in arr_2d:
            out.append([tok if tok != IGNORE_INDEX else pad_id for tok in row])
        return out

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds может быть логитами (np.ndarray) или ids (list)
        if isinstance(preds, tuple):
            preds = preds[0]
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        # если это логиты [B, T, V] от обычного Trainer — возьмём argmax по последней оси
        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)

        decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)
        safe_labels = _replace_ignore_with_pad(labels)
        decoded_labels = processor.batch_decode(safe_labels, skip_special_tokens=True)

        correct, total = 0, 0
        for pred, true in zip(decoded_preds, decoded_labels):
            pred_ans = extract_answer(pred)
            true_ans = extract_answer(true)
            correct += int(pred_ans == true_ans)
            total += 1
        return {"accuracy": correct / total if total else 0.0}

    return compute_metrics


# ========= main =========

def main():
    # ---- конфиг ----
    model_name = "Qwen/Qwen3-VL-4B-Instruct"
    data_path  = "/workspace/qa_tuning.json"
    image_root = "/workspace/images"
    output_dir = "./qwen3vl-geoqa-lora"
    batch_size = 16
    grad_accum = 10
    epochs = 2
    lr = 1e-4
    use_bf16 = True

    # ---- W&B ----
    wandb.init(
        project="qwen3-vl-geoqa",
        name="qwen3vl-lora-run",
        config=dict(model=model_name, lr=lr, epochs=epochs,
                    batch_size=batch_size, grad_accum=grad_accum),
    )

    # ---- модель/процессор ----
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,     # опционально
    )

    # ---- LoRA ----
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM", bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    # ---- датасет HF -> классический torch Dataset ----
    raw = load_dataset("json", data_files=data_path)  # понимает JSON и JSONL
    if "validation" not in raw:
        split = raw["train"].train_test_split(test_size=0.005, seed=42)
        raw = {"train": split["train"], "validation": split["test"]}

    train_ds = GeoQADataset(raw["train"], image_root=image_root, processor=processor)
    val_ds   = GeoQADataset(raw["validation"], image_root=image_root, processor=processor)

    collate_fn = VLDataCollator(processor)

    # ---- метрики ----
    compute_metrics = compute_metrics_factory(processor)

    # ---- обучение ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_ratio=0.03,
        weight_decay=0.0,

        # === 🔥 ЛОГИ и ВАЛИДАЦИЯ ===
        logging_strategy="steps",
        logging_steps=10,
        # eval_strategy="steps",   # ← включаем валидацию по шагам
        # eval_steps=30,                 # ← каждые 50 шагов
        save_strategy="steps", 
        save_steps=30,        # ← сохраняем чекпоинт тоже каждые 50 шагов
        # load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_delay=0,
        per_device_eval_batch_size=64, 
        eval_accumulation_steps=1,                # 🔧 фикс: предотвращает TypeError

        bf16=use_bf16,
        report_to=["wandb"],
        logging_dir="./logs",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,        # наш collate
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ---- сохранение ----
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    wandb.finish()
    print(f"✅ Done. LoRA adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
