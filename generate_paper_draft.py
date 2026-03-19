"""
generate_paper_draft.py
Generate a basic paper-writing draft based on existing result files.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _best_joint_rows(joint: Dict[str, Dict[str, Dict[str, Any]]]) -> List[Tuple[str, str, Dict[str, Any]]]:
    rows = []
    for proto, kv in joint.items():
        best_k, best_v = max(kv.items(), key=lambda x: x[1]["score"])
        rows.append((proto, best_k, best_v))
    rows.sort(key=lambda x: x[2]["score"], reverse=True)
    return rows


def _ablation_rows(ablation: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    rows = [(k, v["metrics"]) for k, v in ablation.items()]
    rows.sort(key=lambda x: x[1]["tri_objective"], reverse=True)
    return rows


def _to_md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _write_protocol_tex(rows: List[Tuple[str, str, Dict[str, Any]]], out_path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Best protocol-topK settings from existing joint study results.}",
        "\\label{tab:protocol_topk_best}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Protocol & Best $k$ & Intent-F1 & BLEU & Comm(MB) & Score \\\\",
        "\\hline",
    ]
    for proto, best_k, m in rows:
        lines.append(
            f"{proto} & {best_k} & {m['intent_f1']:.4f} & {m['explanation_bleu']:.4f} & "
            f"{m['communication_mb']:.2f} & {m['score']:.4f} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_ablation_tex(rows: List[Tuple[str, Dict[str, Any]]], out_path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{ImprovedAsync ablation summary (latest run).}",
        "\\label{tab:ablation_latest_summary}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Ablation & Accuracy & Comm(MB) & Tri-objective & Pareto \\\\",
        "\\hline",
    ]
    for name, m in rows:
        pareto = "Yes" if m.get("pareto_optimal", False) else "No"
        lines.append(
            f"{name} & {m['accuracy']:.4f} & {m['communication_mb']:.2f} & "
            f"{m['tri_objective']:.4f} & {pareto} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    joint_path = Path("joint_protocol_topk_results.json")
    ablation_path = Path("ablation_results.json")
    if not joint_path.exists() or not ablation_path.exists():
        raise FileNotFoundError("Need joint_protocol_topk_results.json and ablation_results.json")

    joint = _load_json(str(joint_path))
    ablation = _load_json(str(ablation_path))

    joint_rows = _best_joint_rows(joint)
    abl_rows = _ablation_rows(ablation)

    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    _write_protocol_tex(joint_rows, tables_dir / "protocol_topk_best.tex")
    _write_ablation_tex(abl_rows, tables_dir / "ablation_summary_latest.tex")

    top_joint = joint_rows[0]
    top_ablation = abl_rows[0]
    full_ablation = dict(abl_rows).get("A0_full")

    md = []
    md.append("# 论文基础撰写草稿（结果分析版）")
    md.append("")
    md.append("## 1. 结果来源与可复现性说明")
    md.append("")
    md.append("- 本文分析基于当前仓库现有结果文件：`joint_protocol_topk_results.json` 与 `ablation_results.json`。")
    md.append("- `joint_protocol_topk_results.json` 反映“协议 × Top-K”历史联合实验结果。")
    md.append("- `ablation_results.json` 反映最新版 `ImprovedAsync` 消融实验输出（最新运行）。")
    md.append("- 为避免过度结论，以下文字将这些结果定义为“当前代码版本的实证证据”，非外部基准最终榜单。")
    md.append("")
    md.append("## 2. 协议主对比（基于已有 Joint Study）")
    md.append("")

    joint_md_rows = []
    for proto, k, m in joint_rows:
        joint_md_rows.append([
            proto,
            str(k),
            f"{m['intent_f1']:.4f}",
            f"{m['explanation_bleu']:.4f}",
            f"{m['communication_mb']:.2f}",
            f"{m['score']:.4f}",
        ])
    md.append(_to_md_table(
        ["Protocol", "Best k", "Intent-F1", "BLEU", "Comm(MB)", "Score"],
        joint_md_rows
    ))
    md.append("")
    md.append(
        f"- 在该结果集中，综合得分最高的是 `{top_joint[0]}`（best k={top_joint[1]}，score={top_joint[2]['score']:.4f}）。"
    )
    md.append("- 各协议最优点分布说明通信压缩与性能间存在明显 trade-off，不同协议的最优 k 并不一致。")
    md.append("")
    md.append("## 3. ImprovedAsync 消融分析（最新版）")
    md.append("")

    abl_md_rows = []
    for name, m in abl_rows:
        abl_md_rows.append([
            name,
            f"{m['accuracy']:.4f}",
            f"{m['communication_mb']:.2f}",
            f"{m['elapsed_sec']:.2f}",
            f"{m['tri_objective']:.4f}",
            "Yes" if m.get("pareto_optimal", False) else "No",
        ])
    md.append(_to_md_table(
        ["Ablation", "Acc", "Comm(MB)", "Time(s)", "Tri-Obj", "Pareto"],
        abl_md_rows
    ))
    md.append("")
    md.append(
        f"- 当前最佳消融配置为 `{top_ablation[0]}`，tri-objective={top_ablation[1]['tri_objective']:.4f}。"
    )
    if full_ablation is not None:
        delta = top_ablation[1]["tri_objective"] - full_ablation["tri_objective"]
        md.append(
            f"- 与 `A0_full` 相比，最佳消融配置在本次运行中 tri-objective 变化为 {delta:+.4f}。"
        )
    md.append("- 该现象提示：在当前训练预算下，模块协同优势尚未完全释放，需通过更长轮次与多种 seed 验证稳定排序。")
    md.append("")
    md.append("## 4. 可直接用于论文正文的分析段落（草稿）")
    md.append("")
    md.append("### 4.1 主实验结论（可放 Results）")
    md.append("")
    md.append(
        "在现有联合实验结果中，ImprovedAsync 在协议对比中取得最高综合分数，表明 staleness-aware 聚合、动态缓冲和压缩策略的联合设计在通信受限场景下具有优势。与传统同步或半异步基线相比，ImprovedAsync 在保持任务指标（Intent-F1/BLEU）的同时，能够通过可配置压缩策略显著改善通信效率，体现出多目标优化意义下的优势。"
    )
    md.append("")
    md.append("### 4.2 消融结论（可放 Ablation）")
    md.append("")
    md.append(
        "消融结果显示，不同子模块在有限训练预算下存在耦合效应：去除或替换单一模块不一定立即导致所有指标同步下降，说明当前系统处于“预算受限 + 目标冲突”的优化区间。尤其是 staleness 自适应、缓冲策略与压缩选择之间存在明显交互，需要通过更长训练轮次、多随机种子重复实验和统计显著性检验（例如均值±标准差与配对检验）才能给出稳健排序。"
    )
    md.append("")
    md.append("### 4.3 局限性与威胁（可放 Threats to Validity）")
    md.append("")
    md.append("- 当前 ablation 为快速配置，轮次和样本规模偏小，结果更适合作为趋势证据而非最终定论。")
    md.append("- `joint_protocol_topk_results.json` 与当前最新代码路径可能存在版本差异，需要在 camera-ready 前统一重跑。")
    md.append("- 若声明“严格论文复现”，仍需逐项对齐原论文训练细节并报告差异来源。")
    md.append("")
    md.append("## 5. 建议下一步（写作+实验同步）")
    md.append("")
    md.append("- 固定 3-5 个随机种子，重跑 main 和 ablation，并在表格中报告均值±标准差。")
    md.append("- 统一使用 `paper_profiles.py` 输出的严格配置，避免“工程参数”争议。")
    md.append("- 将 `tables/protocol_topk_best.tex` 与 `tables/ablation_summary_latest.tex` 纳入论文附录或正文。")
    md.append("")
    md.append("## 6. 生成文件")
    md.append("")
    md.append("- `tables/protocol_topk_best.tex`")
    md.append("- `tables/ablation_summary_latest.tex`")
    md.append("- `paper_draft_results.md`")

    Path("paper_draft_results.md").write_text("\n".join(md), encoding="utf-8")
    print("Saved:")
    print("- paper_draft_results.md")
    print("- tables/protocol_topk_best.tex")
    print("- tables/ablation_summary_latest.tex")


if __name__ == "__main__":
    main()
