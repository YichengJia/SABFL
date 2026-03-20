"""
Build paper-ready robust benchmark tables with communication breakdown.

Inputs:
  - results/robust_external_raw.json

Outputs:
  - results/robust_paper_main_best_profile.csv
  - results/robust_paper_main_mean_profile.csv
  - tables/robust_paper_main_best_profile.tex
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _group_key(row: Dict[str, Any]) -> Tuple[int, str, str, str]:
    return (
        int(row.get("num_clients", 0)),
        str(row.get("fairness_mode", "")),
        str(row.get("compression_suite", "mixed")),
        str(row.get("protocol_family", "")),
    )


def _uplink_downlink_total(row: Dict[str, Any]) -> Tuple[float, float, float]:
    total = _f(row.get("communication_mb", 0.0))
    model_downlink = _f(row.get("model_downlink_mb", 0.0))
    downlink = max(0.0, model_downlink)
    uplink = max(0.0, total - downlink)
    return uplink, downlink, total


def _aggregate_mean_over_profiles(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str, str, str], List[Dict[str, Any]]] = {}
    for r in raw_rows:
        grouped.setdefault(_group_key(r), []).append(r)

    out: List[Dict[str, Any]] = []
    for (n_clients, mode, suite, proto), rows in grouped.items():
        acc_mu, acc_std = _mean_std([_f(r.get("accuracy", 0.0)) for r in rows])
        bal_mu, bal_std = _mean_std([_f(r.get("score_balanced", 0.0)) for r in rows])

        uplinks = []
        downlinks = []
        totals = []
        for r in rows:
            u, d, t = _uplink_downlink_total(r)
            uplinks.append(u)
            downlinks.append(d)
            totals.append(t)
        up_mu, up_std = _mean_std(uplinks)
        dn_mu, dn_std = _mean_std(downlinks)
        tt_mu, tt_std = _mean_std(totals)

        out.append({
            "num_clients": int(n_clients),
            "fairness_mode": mode,
            "compression_suite": suite,
            "protocol_family": proto,
            "profiles": int(len(set(int(_f(r.get("profile_idx", 0))) for r in rows))),
            "runs": int(len(rows)),
            "accuracy_mean_over_profiles": acc_mu,
            "accuracy_std_over_profiles": acc_std,
            "balanced_score_mean_over_profiles": bal_mu,
            "balanced_score_std_over_profiles": bal_std,
            "uplink_mb_mean_over_profiles": up_mu,
            "uplink_mb_std_over_profiles": up_std,
            "downlink_mb_mean_over_profiles": dn_mu,
            "downlink_mb_std_over_profiles": dn_std,
            "total_mb_mean_over_profiles": tt_mu,
            "total_mb_std_over_profiles": tt_std,
        })

    out.sort(key=lambda x: (x["num_clients"], x["fairness_mode"], x["compression_suite"], x["protocol_family"]))
    return out


def _aggregate_best_profile_per_seed(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str, str, str], List[Dict[str, Any]]] = {}
    for r in raw_rows:
        grouped.setdefault(_group_key(r), []).append(r)

    out: List[Dict[str, Any]] = []
    for (n_clients, mode, suite, proto), rows in grouped.items():
        by_seed: Dict[int, List[Dict[str, Any]]] = {}
        for r in rows:
            by_seed.setdefault(int(_f(r.get("seed", 0))), []).append(r)

        picked: List[Dict[str, Any]] = []
        for _, srows in by_seed.items():
            picked.append(max(srows, key=lambda x: _f(x.get("score_balanced", 0.0))))

        acc_mu, acc_std = _mean_std([_f(r.get("accuracy", 0.0)) for r in picked])
        bal_mu, bal_std = _mean_std([_f(r.get("score_balanced", 0.0)) for r in picked])

        uplinks = []
        downlinks = []
        totals = []
        for r in picked:
            u, d, t = _uplink_downlink_total(r)
            uplinks.append(u)
            downlinks.append(d)
            totals.append(t)
        up_mu, up_std = _mean_std(uplinks)
        dn_mu, dn_std = _mean_std(downlinks)
        tt_mu, tt_std = _mean_std(totals)

        out.append({
            "num_clients": int(n_clients),
            "fairness_mode": mode,
            "compression_suite": suite,
            "protocol_family": proto,
            "seeds": int(len(picked)),
            "accuracy_best_profile_per_seed_mean": acc_mu,
            "accuracy_best_profile_per_seed_std": acc_std,
            "balanced_score_best_profile_per_seed_mean": bal_mu,
            "balanced_score_best_profile_per_seed_std": bal_std,
            "uplink_mb_best_profile_per_seed_mean": up_mu,
            "uplink_mb_best_profile_per_seed_std": up_std,
            "downlink_mb_best_profile_per_seed_mean": dn_mu,
            "downlink_mb_best_profile_per_seed_std": dn_std,
            "total_mb_best_profile_per_seed_mean": tt_mu,
            "total_mb_best_profile_per_seed_std": tt_std,
        })

    out.sort(key=lambda x: (x["num_clients"], x["fairness_mode"], x["compression_suite"], x["protocol_family"]))
    return out


def _write_latex_best_profile(rows: List[Dict[str, Any]], out_path: Path) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Robust benchmark main table (best profile per seed), with communication breakdown.}",
        "\\label{tab:robust_main_best_profile}",
        "\\begin{tabular}{llllcccc}",
        "\\hline",
        "$n$ & Mode & Suite & Protocol & Acc & Score & Uplink(MB) & Downlink(MB) / Total(MB) \\\\",
        "\\hline",
    ]
    for r in rows:
        lines.append(
            f"{int(r['num_clients'])} & {r['fairness_mode']} & {r['compression_suite']} & {r['protocol_family']} & "
            f"{_f(r['accuracy_best_profile_per_seed_mean']):.4f}$\\pm${_f(r['accuracy_best_profile_per_seed_std']):.4f} & "
            f"{_f(r['balanced_score_best_profile_per_seed_mean']):.4f}$\\pm${_f(r['balanced_score_best_profile_per_seed_std']):.4f} & "
            f"{_f(r['uplink_mb_best_profile_per_seed_mean']):.2f}$\\pm${_f(r['uplink_mb_best_profile_per_seed_std']):.2f} & "
            f"{_f(r['downlink_mb_best_profile_per_seed_mean']):.2f} / {_f(r['total_mb_best_profile_per_seed_mean']):.2f} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    raw_path = Path("results/robust_external_raw.json")
    if not raw_path.exists():
        raise FileNotFoundError("results/robust_external_raw.json not found. Run robust_external_benchmark.py first.")

    raw_rows = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("results/robust_external_raw.json is empty or invalid.")

    best_rows = _aggregate_best_profile_per_seed(raw_rows)
    mean_rows = _aggregate_mean_over_profiles(raw_rows)

    Path("results").mkdir(exist_ok=True)
    Path("tables").mkdir(exist_ok=True)
    _write_csv(Path("results/robust_paper_main_best_profile.csv"), best_rows)
    _write_csv(Path("results/robust_paper_main_mean_profile.csv"), mean_rows)
    _write_latex_best_profile(best_rows, Path("tables/robust_paper_main_best_profile.tex"))

    print("Saved:")
    print("- results/robust_paper_main_best_profile.csv")
    print("- results/robust_paper_main_mean_profile.csv")
    print("- tables/robust_paper_main_best_profile.tex")


if __name__ == "__main__":
    main()
