import os
import json
import argparse
from pathlib import Path

# ==========================
# Configuration (edit here)
# ==========================
# Default dataset sample and logging
DEFAULTS = {
    "PROFILE": "qwen2vl_2b",  # choose a default model profile from PROFILES below
    "SAMPLE_DIR": "dataset/geometry/Dataset_GeomVerse/test_geomverse_TEST_D2_B100_data_49",
    "IMAGE_NAME": None,  # e.g., "226.png" to override; otherwise auto-detected from ex.json or first PNG
    "OUT_DIR": "outputs/hf_vqa",
    # Append metrics row by default so you can aggregate runs easily
    "APPEND_CSV": "outputs/hf_vqa/metrics.csv",
    "TOL": 1.0,  # numeric tolerance for correctness (degrees)
    "DEVICE": "auto",  # 'auto', 'cuda:0', or 'cpu'
}

# Named model profiles to avoid typing long model ids/flags
PROFILES = {
    # Hugging Face VQA pipeline models
    "blip_vqa": {
        "MODE": "pipeline_vqa",
        "MODEL": "Salesforce/blip-vqa-base",
    },
    # Qwen2-VL instruct models (chat-style vision-language)
    "qwen2vl_2b": {
        "MODE": "qwen_vl",
        "MODEL": "Qwen/Qwen2-VL-2B-Instruct",
    },
    "qwen2vl_7b": {
        "MODE": "qwen_vl",
        "MODEL": "Qwen/Qwen2-VL-7B-Instruct",
    },
}

def _merge_config_with_args():
    """Build final config from constants, optional profile, and optional CLI overrides.

    Goal: make running with zero flags Just Work, while still allowing light overrides
    when needed (e.g., switching profiles or dataset sample).
    """
    parser = argparse.ArgumentParser(description="Run a HF VQA model on one geometry sample")
    parser.add_argument("--profile", type=str, choices=sorted(PROFILES.keys()), default=DEFAULTS["PROFILE"], help="Model profile to use (predefined)")
    parser.add_argument("--sample_dir", type=str, default=None, help="Directory containing ex.json and the image (defaults from constants)")
    parser.add_argument("--image_name", type=str, default=None, help="Override image filename (e.g., 226.png). If omitted, read from ex.json 'image_path_code'")
    # Expert overrides (rarely needed thanks to profiles)
    parser.add_argument("--model", type=str, default=None, help="HF model id, e.g., Salesforce/blip-vqa-base or Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--mode", type=str, choices=["pipeline_vqa", "qwen_vl"], default=None, help="Backend: classic VQA pipeline or Qwen2-VL instruct")
    parser.add_argument("--device", type=str, default=None, help="'auto', 'cuda:0', or 'cpu'")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for predictions")
    parser.add_argument("--append_csv", type=str, default=None, help="If set, evaluate and append a row to this CSV (defaults from constants)")
    parser.add_argument("--tol", type=float, default=None, help="Numeric tolerance in degrees for correctness when appending CSV")
    args = parser.parse_args()

    # Start with defaults, then layer in profile, then CLI overrides if provided
    cfg = dict(DEFAULTS)
    profile = PROFILES[args.profile]
    cfg.update(profile)
    cfg["PROFILE"] = args.profile

    # Lightweight overrides
    if args.sample_dir is not None:
        cfg["SAMPLE_DIR"] = args.sample_dir
    if args.image_name is not None:
        cfg["IMAGE_NAME"] = args.image_name
    if args.model is not None:
        cfg["MODEL"] = args.model
    if args.mode is not None:
        cfg["MODE"] = args.mode
    if args.device is not None:
        cfg["DEVICE"] = args.device
    if args.out_dir is not None:
        cfg["OUT_DIR"] = args.out_dir
    if args.append_csv is not None:
        cfg["APPEND_CSV"] = args.append_csv
    if args.tol is not None:
        cfg["TOL"] = args.tol

    return cfg


def main():
    cfg = _merge_config_with_args()

    # Avoid xet/hf_transfer path which may fail in some networks
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    from transformers import pipeline, AutoProcessor
    try:
        # Qwen2-VL specific class
        from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
    except Exception:
        QwenVLModel = None
    import torch
    from PIL import Image

    sample_dir = Path(cfg["SAMPLE_DIR"])
    ex = json.loads((sample_dir / "ex.json").read_text())
    question = ex["problem_text"]

    if cfg["IMAGE_NAME"] is not None:
        image_path = sample_dir / cfg["IMAGE_NAME"]
    else:
        # ex["image_path_code"] is relative to dataset root; use basename if it matches this sample
        rel_img = ex.get("image_path_code")
        image_path = sample_dir / Path(rel_img).name if rel_img else None
        if image_path is None or not image_path.exists():
            # fallback: pick the first png in the folder
            pngs = sorted(p for p in sample_dir.glob("*.png"))
            if not pngs:
                raise FileNotFoundError("No image found in sample_dir.")
            image_path = pngs[0]

    if cfg["DEVICE"] == "auto":
        device = 0 if torch.cuda.is_available() else -1
    elif str(cfg["DEVICE"]).startswith("cuda"):
        device = 0
    else:
        device = -1

    print(f"Profile: {cfg['PROFILE']}")
    print(f"Model: {cfg['MODEL']}")
    print(f"Mode: {cfg['MODE']}")
    print(f"Device: {'cuda' if device == 0 else 'cpu'}")
    print(f"Question: {question}")
    print(f"Image: {image_path}")

    pred = {
        "model": cfg["MODEL"],
        "question": question,
        "image": str(image_path),
    }

    if cfg["MODE"] == "pipeline_vqa":
        vqa = pipeline("visual-question-answering", model=cfg["MODEL"], device=device)
        result = vqa(question=question, image=str(image_path))
        pred.update({
            "raw_output": result,
            "answer": result[0]["answer"] if isinstance(result, list) and result else None,
            "score": result[0].get("score") if isinstance(result, list) and result else None,
        })
    else:
        # Qwen2-VL instruct flow: prompt the model to output only a number in degrees
        model_id = cfg["MODEL"]
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoProcessor.from_pretrained(model_id)
        if QwenVLModel is None:
            raise RuntimeError("Transformers does not expose Qwen2VLForConditionalGeneration; please upgrade transformers.")
        model = QwenVLModel.from_pretrained(model_id, dtype=dtype, device_map="auto")

        sys_prompt = (
            "You are a geometry expert. Given an image and a question, "
            "compute the angle asked in degrees. Respond with only a single number "
            "rounded to 2 decimal places, without any words or symbols."
        )
        user_text = (
            question
            + "\n\nIMPORTANT: Output only a single number (degrees), rounded to 2 decimals."
        )
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(device_str) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        gen = processor.batch_decode(out, skip_special_tokens=True)[0]
        # Extract only new generation by splitting on prompt tail if available
        answer = gen.split(user_text)[-1].strip()
        pred.update({
            "raw_output": gen,
            "answer": answer,
            "score": None,
        })

    out_dir = Path(cfg["OUT_DIR"]) / sample_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "pred.json"
    (pred_path).write_text(json.dumps(pred, indent=2, ensure_ascii=False))
    print(f"Saved prediction to {pred_path}")

    # Optional: append to CSV metrics
    if cfg["APPEND_CSV"]:
        from eval_one_vqa import eval_one
        ex_path = sample_dir / "ex.json"
        metrics = eval_one(pred_path, ex_path, tol=float(cfg["TOL"]))
        # write/append row
        import csv
        from datetime import datetime
        csv_path = Path(cfg["APPEND_CSV"])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        row = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": pred.get("model"),
            "sample_id": sample_dir.name,
            "question": pred.get("question"),
            "pred_answer": metrics.get("pred_answer"),
            "gold_label": metrics.get("gold_label"),
            "exact_match": metrics.get("exact_match"),
            "numeric_match": metrics.get("numeric_match"),
            "abs_error": metrics.get("abs_error"),
            "rel_error": metrics.get("rel_error"),
            "tolerance": metrics.get("tolerance"),
            "is_correct": metrics.get("is_correct"),
            "pred_path": str(pred_path),
            "ex_path": str(ex_path),
        }
        fieldnames = list(row.keys())
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"Appended metrics to {csv_path}")

if __name__ == "__main__":
    main()
