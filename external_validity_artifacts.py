"""
external_validity_artifacts.py
Build figures and LaTeX tables from external validity outputs.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return default


def _plot_mode_bars(summary_rows: List[Dict[str, Any]], metric: str, out_path: Path):
    modes = sorted(set(r["fairness_mode"] for r in summary_rows))
    fig, axes = plt.subplots(1, len(modes), figsize=(7.0 * len(modes), 4.8), squeeze=False)
    for i, mode in enumerate(modes):
        ax = axes[0][i]
        rows = [r for r in summary_rows if r["fairness_mode"] == mode]
        rows = sorted(rows, key=lambda r: _f(r.get(metric, 0.0)), reverse=True)
        x = [r["protocol_key"] for r in rows]
        y = [_f(r.get(metric, 0.0)) for r in rows]
        ax.bar(x, y)
        ax.set_title(f"{mode} - {metric}")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_traces(traces: List[Dict[str, Any]], out_path: Path):
    modes = sorted(set(t["fairness_mode"] for t in traces))
    fig, axes = plt.subplots(1, len(modes), figsize=(7.0 * len(modes), 4.8), squeeze=False)
    for i, mode in enumerate(modes):
        ax = axes[0][i]
        mode_traces = [t for t in traces if t["fairness_mode"] == mode]
        grouped: Dict[str, List[List[Dict[str, Any]]]] = {}
        for t in mode_traces:
            grouped.setdefault(t["protocol_key"], []).append(t.get("trace", []))

        for proto, runs in grouped.items():
            max_len = max(len(r) for r in runs) if runs else 0
            if max_len == 0:
                continue
            xs, ys = [], []
            for idx in range(max_len):
                x_vals, y_vals = [], []
                for run in runs:
                    if idx < len(run):
                        x_vals.append(_f(run[idx].get("elapsed_sec", 0.0)))
                        y_vals.append(_f(run[idx].get("accuracy", 0.0)))
                if x_vals:
                    xs.append(sum(x_vals) / len(x_vals))
                    ys.append(sum(y_vals) / len(y_vals))
            if xs:
                ax.plot(xs, ys, label=proto, linewidth=1.7)
        ax.set_title(f"{mode} - accuracy/time")
        ax.set_xlabel("Elapsed sec")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _write_latex(summary_rows: List[Dict[str, Any]], out_path: Path):
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{External validity summary on CIFAR photometric non-IID (mean $\\pm$ std).}",
        "\\label{tab:external_validity_summary}",
        "\\begin{tabular}{llcccc}",
        "\\hline",
        "Mode & Protocol & Acc & Comm(MB) & Balanced Score & AUC(Acc-Time) \\\\",
        "\\hline",
    ]
    for mode in sorted(set(r["fairness_mode"] for r in summary_rows)):
        rows = [r for r in summary_rows if r["fairness_mode"] == mode]
        rows = sorted(rows, key=lambda r: _f(r.get("score_balanced_mean", 0.0)), reverse=True)
        for r in rows:
            lines.append(
                f"{mode} & {r['protocol_key']} & "
                f"{_f(r.get('accuracy_mean')):.4f}$\\pm${_f(r.get('accuracy_std')):.4f} & "
                f"{_f(r.get('communication_mb_mean')):.2f}$\\pm${_f(r.get('communication_mb_std')):.2f} & "
                f"{_f(r.get('score_balanced_mean')):.4f}$\\pm${_f(r.get('score_balanced_std')):.4f} & "
                f"{_f(r.get('auc_acc_time_mean')):.3f}$\\pm${_f(r.get('auc_acc_time_std')):.3f} \\\\"
            )
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table*}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    summary_path = Path("results/external_validity_summary.csv")
    traces_path = Path("results/external_validity_traces.json")
    if not summary_path.exists():
        raise FileNotFoundError("results/external_validity_summary.csv not found.")
    if not traces_path.exists():
        raise FileNotFoundError("results/external_validity_traces.json not found.")

    summary_rows = _read_csv(summary_path)
    traces = json.loads(traces_path.read_text(encoding="utf-8"))

    Path("figures").mkdir(exist_ok=True)
    Path("tables").mkdir(exist_ok=True)
    _plot_mode_bars(summary_rows, "score_balanced_mean", Path("figures/external_balanced_score.png"))
    _plot_mode_bars(summary_rows, "accuracy_mean", Path("figures/external_accuracy_mean.png"))
    _plot_traces(traces, Path("figures/external_convergence_traces.png"))
    _write_latex(summary_rows, Path("tables/external_validity_summary.tex"))

    print("Saved:")
    print("- figures/external_balanced_score.png")
    print("- figures/external_accuracy_mean.png")
    print("- figures/external_convergence_traces.png")
    print("- tables/external_validity_summary.tex")


if __name__ == "__main__":
    main()
