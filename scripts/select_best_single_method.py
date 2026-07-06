#!/usr/bin/env python3
"""Select exactly one global best method/config from public evaluation JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(number) or math.isinf(number))


def experiment_lookup(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(experiment["id"]): experiment for experiment in plan["experiments"]}


def scene_status(eval_data: dict[str, Any], public_scenes: list[str]) -> tuple[list[str], list[str]]:
    def has_valid_score(scene_data: dict[str, Any]) -> bool:
        return not scene_data.get("missing") and is_finite(scene_data.get("score"))

    scored = {
        str(item["scene"])
        for item in eval_data.get("scenes", [])
        if has_valid_score(item)
    }
    missing = [scene for scene in public_scenes if scene not in scored]
    return sorted(scored), missing


def metric(value: Any, default: float) -> float:
    return default if value is None or not is_finite(value) else float(value)


def rank_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        metric(item.get("mean_score"), -1.0),
        -metric(item.get("mean_lpips"), 999.0),
        metric(item.get("mean_ssim"), -1.0),
        metric(item.get("mean_psnr"), -1.0),
    )


def build_ranking(plan: dict[str, Any], eval_dir: Path, require_complete: bool) -> list[dict[str, Any]]:
    experiments = experiment_lookup(plan)
    public_scenes = list(plan["public_scenes"])
    ranking: list[dict[str, Any]] = []
    for path in sorted(eval_dir.glob("*.json")):
        experiment_id = path.stem
        experiment = experiments.get(experiment_id)
        if experiment is None:
            continue
        eval_data = load_json(path)
        scored_scenes, missing_scenes = scene_status(eval_data, public_scenes)
        complete = not missing_scenes
        eligible = is_finite(eval_data.get("mean_score")) and (complete or not require_complete)
        ranking.append(
            {
                "experiment_id": experiment_id,
                "method": experiment["method"],
                "eligible": eligible,
                "complete_public": complete,
                "scored_scenes": scored_scenes,
                "missing_scenes": missing_scenes,
                "mean_score": eval_data.get("mean_score"),
                "mean_lpips": eval_data.get("mean_lpips"),
                "mean_ssim": eval_data.get("mean_ssim"),
                "mean_psnr": eval_data.get("mean_psnr"),
                "eval_json": str(path),
                "args": experiment.get("args", {}),
            }
        )
    ranking.sort(key=rank_key, reverse=True)
    return ranking


def select_best(ranking: list[dict[str, Any]], tie_threshold: float) -> dict[str, Any]:
    eligible = [item for item in ranking if item["eligible"]]
    if not eligible:
        raise SystemExit("No eligible experiments found. Check eval JSON files and public scene completeness.")
    best = eligible[0]
    best_score = float(best["mean_score"])
    close = [item for item in eligible if best_score - float(item["mean_score"]) <= tie_threshold]
    return {"best": best, "within_tie_threshold": close}


def main() -> None:
    parser = argparse.ArgumentParser(description="Select one best single method/config from public eval JSON files.")
    parser.add_argument("--config", type=Path, default=Path("configs/best_single_method_plan.json"))
    parser.add_argument("--eval-dir", type=Path, default=Path("reports/experiments/eval/public"))
    parser.add_argument("--json-out", type=Path, default=Path("reports/experiments/final_selection.json"))
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow experiments missing one or more public scenes.")
    args = parser.parse_args()

    plan = load_json(args.config)
    ranking = build_ranking(plan, args.eval_dir, require_complete=not args.allow_incomplete)
    selection = select_best(ranking, tie_threshold=float(plan.get("eval", {}).get("tie_threshold", 0.003)))
    result = {
        "plan": plan["name"],
        "eval_dir": str(args.eval_dir),
        "selection_rule": "max mean_score over all public scenes; tie-break by lower LPIPS, higher SSIM, higher PSNR",
        "tie_threshold": float(plan.get("eval", {}).get("tie_threshold", 0.003)),
        "best_experiment_id": selection["best"]["experiment_id"],
        "best_method": selection["best"]["method"],
        "best": selection["best"],
        "within_tie_threshold": selection["within_tie_threshold"],
        "ranking": ranking,
        "next_command": (
            "python scripts/generate_experiment_commands.py "
            f"--config {args.config} --selected-experiment {selection['best']['experiment_id']}"
        ),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
