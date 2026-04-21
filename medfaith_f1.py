"""
medfaith_f1.py

MedFaith-F1: category-level faithfulness metric for medical query rewriting.

For each instance (q, q_hat), we check whether each clinical category
C = {Dx, Rx, Proc, Fup} is preserved in the reformulation.
Category retention is treated as binary classification:
  - TP: category present in both q and q_hat
  - FN: present in q, absent in q_hat
  - FP: absent in q, present in q_hat

Per-category F1 scores are macro-averaged to give MedFaith-F1.
Category Hallucination Rate (CHR) = (1 - MedFaith-F1) * 100.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

from sapbert_extractor import extract_category_presence

CATEGORIES = ["Dx", "Rx", "Proc", "Fup"]


def _category_f1(
    sources: List[Dict[str, bool]],
    preds: List[Dict[str, bool]],
    category: str,
) -> float:
    """
    Compute binary F1 for a single category across the evaluation set.
    TP: category present in source AND in prediction.
    FN: present in source, absent in prediction.
    FP: absent in source, present in prediction.
    """
    tp = fp = fn = 0
    for src_flags, pred_flags in zip(sources, preds):
        s = src_flags.get(category, False)
        p = pred_flags.get(category, False)
        if s and p:
            tp += 1
        elif s and not p:
            fn += 1
        elif not s and p:
            fp += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def medfaith_f1(
    source_questions: List[str],
    reformulations: List[str],
    return_per_category: bool = False,
) -> Tuple[float, float, Optional[Dict[str, float]]]:
    """
    Compute MedFaith-F1 and Category Hallucination Rate (CHR) over a dataset.

    Args:
        source_questions:   original consumer health questions
        reformulations:     model-generated reformulations
        return_per_category: whether to return per-category F1 breakdown

    Returns:
        (medfaith_f1_score, chr_percent, per_cat_dict_or_None)
    """
    assert len(source_questions) == len(reformulations), (
        "source and reformulation lists must have the same length"
    )

    src_flags_list = [extract_category_presence(q) for q in source_questions]
    pred_flags_list = [extract_category_presence(qh) for qh in reformulations]

    per_cat: Dict[str, float] = {}
    for cat in CATEGORIES:
        per_cat[cat] = _category_f1(src_flags_list, pred_flags_list, cat)

    mf1 = float(np.mean(list(per_cat.values())))
    chr_pct = (1.0 - mf1) * 100.0

    if return_per_category:
        return mf1, chr_pct, per_cat
    return mf1, chr_pct, None


def medfaith_f1_single(
    source_question: str,
    reformulation: str,
) -> Dict[str, float]:
    """
    Compute category-level scores for a single (q, q_hat) pair.
    Useful for instance-level inspection.

    Returns dict with keys: Dx, Rx, Proc, Fup, MedFaith-F1, CHR
    """
    src_flags = extract_category_presence(source_question)
    pred_flags = extract_category_presence(reformulation)

    # For a single instance, F1 collapses to exact match per category
    results: Dict[str, float] = {}
    f1_vals = []
    for cat in CATEGORIES:
        s = src_flags.get(cat, False)
        p = pred_flags.get(cat, False)
        if s and p:
            f1 = 1.0
        elif not s and not p:
            f1 = 1.0  # both absent: vacuously correct
        else:
            f1 = 0.0
        results[cat] = f1
        f1_vals.append(f1)

    results["MedFaith-F1"] = float(np.mean(f1_vals))
    results["CHR"] = (1.0 - results["MedFaith-F1"]) * 100.0
    return results


def hallucination_risk_label(chr_pct: float) -> str:
    """Map CHR% to qualitative risk label matching Table 4 thresholds."""
    if chr_pct >= 40.0:
        return "High"
    elif chr_pct >= 30.0:
        return "Moderate"
    else:
        return "Low"


def print_medfaith_report(
    mf1: float,
    chr_pct: float,
    per_cat: Optional[Dict[str, float]],
    model_name: str = "",
    dataset: str = "",
) -> None:
    header = f"MedFaith-F1 Report"
    if model_name:
        header += f" — {model_name}"
    if dataset:
        header += f" on {dataset}"
    print(header)
    print("-" * len(header))
    print(f"  MedFaith-F1 : {mf1:.4f}")
    print(f"  CHR%        : {chr_pct:.2f}%  [{hallucination_risk_label(chr_pct)}]")
    if per_cat:
        for cat in CATEGORIES:
            print(f"  F1 ({cat:4s})  : {per_cat[cat]:.4f}")
    print()


if __name__ == "__main__":
    sources = [
        "I am trying to get travel insurance and the insurance company "
        "don't recognise percutaneous nephrolithotomy or nephrostomy — "
        "is there a name that would be recognised by the insurance company?",
        "My doctor prescribed metformin 500mg twice daily for type 2 diabetes "
        "and said I need to follow up in 3 months — what should I expect?",
    ]
    preds = [
        # drops 'nephrostomy' — should penalise Proc F1
        "What are the alternative names for percutaneous nephrolithotomy?",
        # preserves Rx and Dx but drops Fup
        "What are the effects of metformin 500mg for type 2 diabetes?",
    ]

    mf1, chr_pct, per_cat = medfaith_f1(sources, preds, return_per_category=True)
    print_medfaith_report(mf1, chr_pct, per_cat, model_name="demo", dataset="MeQSum")

    print("Single instance check (drops nephrostomy):")
    single = medfaith_f1_single(sources[0], preds[0])
    for k, v in single.items():
        print(f"  {k}: {v:.4f}")
