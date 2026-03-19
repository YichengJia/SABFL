"""
generate_reproduction_report.py
Create supplementary reproducibility report and export paper profile suite.
"""

import json
from datetime import datetime
from typing import Dict, Any
from paper_profiles import build_protocol_suite, get_reproduction_notes, dump_suite_and_notes


def _format_config(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg, ensure_ascii=False, indent=2)


def generate_report(
    num_clients: int = 50,
    strict: bool = True,
    output_md: str = "reproducibility_report.md",
    output_json: str = "paper_profile_suite.json"
) -> str:
    suite = build_protocol_suite(num_clients=num_clients, strict=strict, include_improved=True)
    notes = get_reproduction_notes()
    dump_suite_and_notes(suite, output_json_path=output_json)

    lines = []
    lines.append("# Reproducibility and Implementation-Difference Report")
    lines.append("")
    lines.append(f"- Generated at: {datetime.utcnow().isoformat()}Z")
    lines.append(f"- Num clients profile: {num_clients}")
    lines.append(f"- Strict baseline profiles: {strict}")
    lines.append("")
    lines.append("## 1) Protocol Configuration Suite")
    lines.append("")
    for proto, cfg in suite.items():
        lines.append(f"### {proto}")
        lines.append("")
        lines.append("```json")
        lines.append(_format_config(cfg))
        lines.append("```")
        lines.append("")

    lines.append("## 2) Implementation Intent and Known Deviations")
    lines.append("")
    for n in notes:
        lines.append(f"- **{n['protocol']}**")
        lines.append(f"  - intent: {n['intent']}")
        lines.append(f"  - known_deviation: {n['known_deviation']}")
    lines.append("")

    lines.append("## 3) Suggested Paper Language")
    lines.append("")
    lines.append("- Baselines are implemented under strict profile settings in a unified codebase.")
    lines.append("- We report implementation differences explicitly and avoid claiming bit-level original-code reproduction.")
    lines.append("- Main contribution is ImprovedAsync with scale-aware bounded controls and system-observed staleness adaptation.")
    lines.append("")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_md


if __name__ == "__main__":
    out = generate_report()
    print(f"Saved: {out}")
