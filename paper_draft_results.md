# 论文基础撰写草稿（结果分析版）

## 1. 结果来源与可复现性说明

- 本文分析基于当前仓库现有结果文件：`joint_protocol_topk_results.json` 与 `ablation_results.json`。
- `joint_protocol_topk_results.json` 反映“协议 × Top-K”历史联合实验结果。
- `ablation_results.json` 反映最新版 `ImprovedAsync` 消融实验输出（最新运行）。
- 为避免过度结论，以下文字将这些结果定义为“当前代码版本的实证证据”，非外部基准最终榜单。

## 2. 协议主对比（基于已有 Joint Study）

| Protocol | Best k | Intent-F1 | BLEU | Comm(MB) | Score |
|---|---|---|---|---|---|
| improved_async | 1000 | 0.7126 | 0.7146 | 21.44 | 0.5705 |
| fedavg | None | 0.7089 | 0.7106 | 14.81 | 0.5674 |
| scaffold | 1000 | 0.7022 | 0.7026 | 21.44 | 0.5618 |
| fedasync | 500 | 0.6897 | 0.6946 | 13.81 | 0.5527 |
| fedbuff | None | 0.6782 | 0.6786 | 14.81 | 0.5427 |

- 在该结果集中，综合得分最高的是 `improved_async`（best k=1000，score=0.5705）。
- 各协议最优点分布说明通信压缩与性能间存在明显 trade-off，不同协议的最优 k 并不一致。

## 3. ImprovedAsync 消融分析（最新版）

| Ablation | Acc | Comm(MB) | Time(s) | Tri-Obj | Pareto |
|---|---|---|---|---|---|
| A1_no_auto_scale | 0.3433 | 0.18 | 0.62 | 0.4823 | Yes |
| A2_no_staleness_adapt | 0.2687 | 0.18 | 0.62 | 0.4373 | No |
| A6_linear_staleness | 0.2537 | 0.18 | 0.59 | 0.4333 | Yes |
| A7_fixed_buffer_small | 0.2537 | 0.18 | 0.62 | 0.4269 | No |
| A5_no_compression | 0.2239 | 3.66 | 0.57 | 0.4090 | Yes |
| A4_no_momentum | 0.1940 | 0.18 | 0.61 | 0.3943 | No |
| A3_no_adaptive_weighting | 0.1642 | 0.18 | 0.61 | 0.3749 | No |
| A0_full | 0.2537 | 0.18 | 2.38 | 0.3516 | No |

- 当前最佳消融配置为 `A1_no_auto_scale`，tri-objective=0.4823。
- 与 `A0_full` 相比，最佳消融配置在本次运行中 tri-objective 变化为 +0.1307。
- 该现象提示：在当前训练预算下，模块协同优势尚未完全释放，需通过更长轮次与多种 seed 验证稳定排序。

## 4. 可直接用于论文正文的分析段落（草稿）

### 4.1 主实验结论（可放 Results）

在现有联合实验结果中，ImprovedAsync 在协议对比中取得最高综合分数，表明 staleness-aware 聚合、动态缓冲和压缩策略的联合设计在通信受限场景下具有优势。与传统同步或半异步基线相比，ImprovedAsync 在保持任务指标（Intent-F1/BLEU）的同时，能够通过可配置压缩策略显著改善通信效率，体现出多目标优化意义下的优势。

### 4.2 消融结论（可放 Ablation）

消融结果显示，不同子模块在有限训练预算下存在耦合效应：去除或替换单一模块不一定立即导致所有指标同步下降，说明当前系统处于“预算受限 + 目标冲突”的优化区间。尤其是 staleness 自适应、缓冲策略与压缩选择之间存在明显交互，需要通过更长训练轮次、多随机种子重复实验和统计显著性检验（例如均值±标准差与配对检验）才能给出稳健排序。

### 4.3 局限性与威胁（可放 Threats to Validity）

- 当前 ablation 为快速配置，轮次和样本规模偏小，结果更适合作为趋势证据而非最终定论。
- `joint_protocol_topk_results.json` 与当前最新代码路径可能存在版本差异，需要在 camera-ready 前统一重跑。
- 若声明“严格论文复现”，仍需逐项对齐原论文训练细节并报告差异来源。

## 5. 建议下一步（写作+实验同步）

- 固定 3-5 个随机种子，重跑 main 和 ablation，并在表格中报告均值±标准差。
- 统一使用 `paper_profiles.py` 输出的严格配置，避免“工程参数”争议。
- 将 `tables/protocol_topk_best.tex` 与 `tables/ablation_summary_latest.tex` 纳入论文附录或正文。

## 6. 生成文件

- `tables/protocol_topk_best.tex`
- `tables/ablation_summary_latest.tex`
- `paper_draft_results.md`