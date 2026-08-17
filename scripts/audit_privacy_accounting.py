"""Cross-check the pilot privacy budget with independent accountants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opacus.accountants import PRVAccountant, RDPAccountant

from src.privacy import RdpPrivacyConfig, compute_epsilon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-probability", type=float, required=True)
    parser.add_argument("--noise-multiplier", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _opacus_epsilon(accountant, args: argparse.Namespace) -> float:
    for _ in range(args.steps):
        accountant.step(
            noise_multiplier=args.noise_multiplier,
            sample_rate=args.sampling_probability,
        )
    return float(accountant.get_epsilon(delta=args.delta))


def main() -> None:
    args = parse_args()
    result = {
        "mechanism": "Poisson-sampled Gaussian",
        "sampling_probability": args.sampling_probability,
        "noise_multiplier": args.noise_multiplier,
        "steps": args.steps,
        "delta": args.delta,
        "epsilon": {
            "dp_accounting_integer_order_rdp": compute_epsilon(
                RdpPrivacyConfig(
                    sampling_probability=args.sampling_probability,
                    noise_multiplier=args.noise_multiplier,
                    steps=args.steps,
                    delta=args.delta,
                )
            ),
            "opacus_rdp": _opacus_epsilon(RDPAccountant(), args),
            "opacus_prv": _opacus_epsilon(PRVAccountant(), args),
        },
        "reporting_rule": (
            "Use the conservative dp-accounting integer-order RDP value until "
            "the final accountant and order grid are frozen."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
