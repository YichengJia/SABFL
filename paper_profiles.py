"""
paper_profiles.py
Paper-oriented protocol profiles and reproducibility notes.
"""

from typing import Dict, Any, List, Tuple
import json
from optimized_protocol_config import get_layered_improved_config


def get_paper_baseline_configs(strict: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Build baseline configs with an explicit 'paper-reproduction' intent.
    strict=True favors cleaner protocol semantics over engineering shortcuts.
    """
    fedavg = {
        "strict_reproduction": bool(strict),
        "participation_rate": 1.0 if strict else 0.5,
        "use_timeout": False if strict else True,
        "max_round_time": 10.0,
    }
    fedasync = {
        "max_staleness": 10,
        "learning_rate": 0.8,
        "staleness_mode": "linear",
        "staleness_floor": 0.1,
        "server_tick_sec": 0.02,
    }
    fedbuff = {
        "buffer_size": 5,
        "max_staleness": 15,
        "server_lr": 0.2,
        "gradient_clip": 5.0,
    }
    scaffold = {
        "strict_reproduction": bool(strict),
        "participation_rate": 1.0 if strict else 0.5,
        "use_timeout": False if strict else True,
        "learning_rate": 0.8,
        "max_round_time": 10.0,
    }
    return {
        "fedavg": fedavg,
        "fedasync": fedasync,
        "fedbuff": fedbuff,
        "scaffold": scaffold,
    }


def get_improved_profile_variants(num_clients: int) -> Dict[str, Dict[str, Any]]:
    """
    ImprovedAsync paper variants built from layered parameterization.
    """
    return {
        "improved_async_balanced": get_layered_improved_config(
            num_clients=num_clients,
            scenario="balanced",
            compression=None,
            staleness_quantile=0.9,
        ),
        "improved_async_low_comm_topk": get_layered_improved_config(
            num_clients=num_clients,
            scenario="low_communication",
            compression="topk",
            staleness_quantile=0.9,
        ),
        "improved_async_balanced_qsgd": get_layered_improved_config(
            num_clients=num_clients,
            scenario="balanced",
            compression="qsgd",
            staleness_quantile=0.9,
        ),
    }


def build_protocol_suite(
    num_clients: int,
    strict: bool = True,
    include_improved: bool = True
) -> Dict[str, Dict[str, Any]]:
    suite = get_paper_baseline_configs(strict=strict)
    if include_improved:
        suite.update(get_improved_profile_variants(num_clients=num_clients))
    return suite


def get_reproduction_notes() -> List[Dict[str, str]]:
    """
    Structured notes for supplementary material:
    each entry describes implementation intent and known deviations.
    """
    return [
        {
            "protocol": "fedavg",
            "intent": "Synchronous fixed-size rounds under strict mode.",
            "known_deviation": "No paper-specific optimizer/loss schedule beyond common training loop.",
        },
        {
            "protocol": "fedasync",
            "intent": "Queue-driven async server updates with staleness decay.",
            "known_deviation": "Server scheduler is generic tick-based, not tied to a specific paper runtime stack.",
        },
        {
            "protocol": "fedbuff",
            "intent": "Buffered asynchronous aggregation with stale filtering.",
            "known_deviation": "Server LR and clipping remain configurable engineering knobs.",
        },
        {
            "protocol": "scaffold",
            "intent": "Control-variate based sync rounds with strict gating.",
            "known_deviation": "Canonical update rule is implemented; communication accounting now includes scaffold control-payload overhead estimates.",
        },
        {
            "protocol": "improved_async",
            "intent": "Main method: scale-aware, staleness-aware, adaptive-buffer async FL.",
            "known_deviation": "A research extension, not a direct reproduction target.",
        },
    ]


def dump_suite_and_notes(
    suite: Dict[str, Dict[str, Any]],
    output_json_path: str = "paper_profile_suite.json"
) -> Tuple[str, str]:
    payload = {
        "protocol_suite": suite,
        "notes": get_reproduction_notes(),
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_json_path, "ok"
