import re
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # find first float-like number
    m = re.search(r"[-+]?\d*\.?\d+", text.replace(",", "."))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def normalize_str(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]+", " ", s)).strip().lower()


def eval_one(pred_path: Path, ex_path: Path, tol: float = 1.0) -> dict:
    pred = json.loads(pred_path.read_text())
    ex = json.loads(ex_path.read_text())

    gold_label = ex.get("ext_info", {}).get("label")
    gold_num = extract_number(gold_label) if isinstance(gold_label, str) else None

    answer = pred.get("answer")
    pred_num = extract_number(answer)

    metrics = {
        "exact_match": False,
        "numeric_match": False,
        "abs_error": None,
        "rel_error": None,
        "tolerance": tol,
        "gold_label": gold_label,
        "pred_answer": answer,
        "is_correct": False,
    }

    # Exact string match (normalized)
    if isinstance(gold_label, str) and isinstance(answer, str):
        metrics["exact_match"] = normalize_str(gold_label) == normalize_str(answer)

    # Numeric match within tolerance
    if gold_num is not None and pred_num is not None:
        abs_err = abs(pred_num - gold_num)
        metrics["abs_error"] = abs_err
        metrics["rel_error"] = abs_err / max(1e-8, abs(gold_num))
        metrics["numeric_match"] = abs_err <= tol

    metrics["is_correct"] = bool(metrics["exact_match"] or metrics["numeric_match"])
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate one VQA prediction against a geometry label.")
    parser.add_argument("--pred", type=str, required=True, help="Path to pred.json produced by run_hf_geom_one.py")
    parser.add_argument("--ex", type=str, required=False, help="Path to ex.json (if omitted, infer from pred path)")
    parser.add_argument("--tol", type=float, default=1.0, help="Numeric tolerance for correctness (degrees)")
    args = parser.parse_args()

    pred_path = Path(args.pred)
    if args.ex:
        ex_path = Path(args.ex)
    else:
        # infer ex.json from folder name pattern
        sample_dir = pred_path.parent
        ex_path = sample_dir.parent.parent / "dataset" / "geometry" / "Dataset_GeomVerse" / sample_dir.name / "ex.json"
        if not ex_path.exists():
            # fallback: try sibling ex.json
            ex_path = sample_dir / "ex.json"
    if not ex_path.exists():
        raise FileNotFoundError(f"ex.json not found: {ex_path}")

    metrics = eval_one(pred_path, ex_path, tol=args.tol)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
