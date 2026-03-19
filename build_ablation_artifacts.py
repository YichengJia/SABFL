"""
build_ablation_artifacts.py
Generate ablation figures and LaTeX table from ablation_results.csv/json.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "ablation": r["ablation"],
                "accuracy": float(r["accuracy"]),
                "intent_f1": float(r["intent_f1"]),
                "bleu": float(r["bleu"]),
                "communication_mb": float(r["communication_mb"]),
                "elapsed_sec": float(r["elapsed_sec"]),
                "aggregations": int(float(r["aggregations"])),
                "tri_objective": float(r["tri_objective"]),
                "pareto_optimal": str(r.get("pareto_optimal", "False")).lower() in ("true", "1"),
            })
    return rows


def _plot_tri_objective(rows: List[Dict[str, Any]], out_path: Path) -> None:
    names = [r["ablation"] for r in rows]
    scores = [r["tri_objective"] for r in rows]
    colors = ["#2ca02c" if r["pareto_optimal"] else "#1f77b4" for r in rows]

    plt.figure(figsize=(10, 4.8))
    plt.bar(names, scores, color=colors)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Tri-objective Score")
    plt.title("Ablation Performance (green = Pareto-optimal)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_pareto(rows: List[Dict[str, Any]], out_path: Path) -> None:
    plt.figure(figsize=(6.4, 5.4))
    for r in rows:
        c = "#d62728" if r["pareto_optimal"] else "#7f7f7f"
        plt.scatter(r["communication_mb"], r["accuracy"], c=c, s=70)
        plt.text(r["communication_mb"], r["accuracy"], r["ablation"], fontsize=8)
    plt.xlabel("Communication (MB) ↓")
    plt.ylabel("Accuracy ↑")
    plt.title("Pareto View: Accuracy vs Communication")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _write_latex(rows: List[Dict[str, Any]], out_path: Path) -> None:
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{ImprovedAsync ablation study on TartanAir.}")
    lines.append("\\label{tab:ablation_improved_async}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\hline")
    lines.append("Ablation & Acc & F1 & BLEU & Comm(MB) & Time(s) & Score \\\\")
    lines.append("\\hline")
    for r in rows:
        name = r["ablation"] + ("$^{\\star}$" if r["pareto_optimal"] else "")
        lines.append(
            f"{name} & {r['accuracy']:.4f} & {r['intent_f1']:.4f} & {r['bleu']:.4f} & "
            f"{r['communication_mb']:.2f} & {r['elapsed_sec']:.1f} & {r['tri_objective']:.4f} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\multicolumn{7}{l}{$^{\\star}$ Pareto-optimal point.}")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    csv_path = Path("ablation_results.csv")
    json_path = Path("ablation_results.json")
    if not csv_path.exists() and json_path.exists():
        # optional reconstruction from json
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rows = []
        for ablation, obj in payload.items():
            m = obj["metrics"]
            rows.append({
                "ablation": ablation,
                "accuracy": float(m["accuracy"]),
                "intent_f1": float(m["intent_f1"]),
                "bleu": float(m["bleu"]),
                "communication_mb": float(m["communication_mb"]),
                "elapsed_sec": float(m["elapsed_sec"]),
                "aggregations": int(m["aggregations"]),
                "tri_objective": float(m["tri_objective"]),
                "pareto_optimal": bool(m.get("pareto_optimal", False)),
            })
    elif csv_path.exists():
        rows = _load_rows(csv_path)
    else:
        raise FileNotFoundError("ablation_results.csv/json not found. Run ablation_runner.py first.")

    rows = sorted(rows, key=lambda x: x["tri_objective"], reverse=True)
    fig_dir = Path("figures")
    tbl_dir = Path("tables")
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    _plot_tri_objective(rows, fig_dir / "ablation_tri_objective.png")
    _plot_pareto(rows, fig_dir / "ablation_pareto_accuracy_comm.png")
    _write_latex(rows, tbl_dir / "ablation_table.tex")

    print("Saved:")
    print("- figures/ablation_tri_objective.png")
    print("- figures/ablation_pareto_accuracy_comm.png")
    print("- tables/ablation_table.tex")


if __name__ == "__main__":
    main()
