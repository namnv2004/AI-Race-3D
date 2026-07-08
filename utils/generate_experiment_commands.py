#!/usr/bin/env python3
"""Generate shell command files for shared smoke/public/private GS runs.

This script only writes command files. It does not execute training, rendering,
evaluation, zipping, or any GPU workload.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

METHOD_SCRIPTS = {
    "3dgs": "src/method_runners/run_3dgs_scene.py",
    "mip": "src/method_runners/run_mip_splatting_scene.py",
    "2dgs": "src/method_runners/run_2dgs_scene.py",
}


def shell_join(parts: list[str | Path | int | float]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def append_args(command: list[str], args: dict[str, Any]) -> list[str]:
    for key, value in args.items():
        if value is None or value is False:
            continue
        flag = flag_name(key)
        if value is True:
            command.append(flag)
        elif isinstance(value, list):
            command.append(flag)
            command.extend(str(item) for item in value)
        else:
            command.extend([flag, str(value)])
    return command


def load_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def python_cmd(plan: dict[str, Any]) -> str:
    return str(plan.get("python", "python"))


def root(plan: dict[str, Any], name: str) -> Path:
    return Path(plan["roots"][name])


def path_template(plan: dict[str, Any], section: str, name: str, **values: str) -> Path:
    return Path(str(plan[section][name]).format(**values))


def run_id(run: dict[str, Any]) -> str:
    return str(run["id"])


def active_runs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [run for run in plan["runs"] if run.get("active", True)]


def find_run(plan: dict[str, Any], selected_run: str) -> dict[str, Any]:
    runs = {run_id(run): run for run in plan["runs"]}
    if selected_run not in runs:
        raise SystemExit(f"Unknown selected run: {selected_run}")
    return runs[selected_run]


def phase_scenes(plan: dict[str, Any], phase: str) -> list[str]:
    scenes = list(plan["scenes"][phase])
    if phase == "smoke" and len(scenes) != 1:
        raise ValueError("scenes.smoke must contain exactly one scene")
    return scenes


def phase_root(plan: dict[str, Any], phase: str) -> Path:
    if phase in {"smoke", "public"}:
        return root(plan, "public")
    if phase == "private":
        return root(plan, "private")
    raise ValueError(f"Unsupported phase: {phase}")


def output_dir(plan: dict[str, Any], phase: str, rid: str) -> Path:
    return path_template(plan, "outputs", phase, phase=phase, run_id=rid)


def artifact_path(plan: dict[str, Any], name: str, phase: str, rid: str) -> Path:
    return path_template(plan, "artifacts", name, phase=phase, run_id=rid)


def common_model_args(plan: dict[str, Any], phase: str, run: dict[str, Any]) -> dict[str, str]:
    rid = run_id(run)
    method = str(run["method"])
    args: dict[str, str] = {}
    if method == "mip":
        args["model_root"] = str(artifact_path(plan, "model_root", phase, rid))
        args["workspace_dir"] = str(artifact_path(plan, "mip_workspace", phase, rid))
    elif method == "2dgs":
        args["model_root"] = str(artifact_path(plan, "model_root", phase, rid))
    return args


def train_command(plan: dict[str, Any], phase: str, run: dict[str, Any], scene: str) -> str:
    method = str(run["method"])
    script = METHOD_SCRIPTS[method]
    rid = run_id(run)
    args = common_model_args(plan, phase, run) | dict(run.get("args", {}))
    command = [
        python_cmd(plan),
        script,
        "--scene-dir",
        str(phase_root(plan, phase) / scene),
        "--output-dir",
        str(output_dir(plan, phase, rid)),
    ]
    append_args(command, args)
    return shell_join(command)


def eval_command(plan: dict[str, Any], run: dict[str, Any]) -> str:
    eval_config = plan.get("eval", {})
    rid = run_id(run)
    command = [
        python_cmd(plan),
        "eval/evaluate_public.py",
        "--public-root",
        str(root(plan, "public")),
        "--pred-dir",
        str(output_dir(plan, "public", rid)),
        "--scenes",
        *phase_scenes(plan, "public"),
        "--json-out",
        str(artifact_path(plan, "eval_json", "public", rid)),
    ]
    if eval_config.get("with_lpips", True):
        command.append("--with-lpips")
    if eval_config.get("lpips_net"):
        command.extend(["--lpips-net", str(eval_config["lpips_net"])])
    if eval_config.get("lpips_device"):
        command.extend(["--lpips-device", str(eval_config["lpips_device"])])
    return shell_join(command)


def write_script(path: Path, title: str, commands: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    while commands and commands[-1] == "":
        commands.pop()
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# {title}",
        "# Generated by utils/generate_experiment_commands.py",
        "# Review before running. This file may start GPU workloads.",
        "",
    ]
    lines.extend(commands)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def mkdir_command(path: Path) -> str:
    return shell_join(["mkdir", "-p", str(path)])


def generate_preflight(plan: dict[str, Any], out_dir: Path) -> Path:
    path = out_dir / "00_preflight.sh"
    commands = [
        mkdir_command(root(plan, "reports")),
        mkdir_command(artifact_path(plan, "selection_dir", "public", "selection")),
        shell_join(
            [
                python_cmd(plan),
                "utils/inspect_dataset.py",
                "--json-out",
                str(root(plan, "reports") / "inspect_latest.json"),
            ]
        ),
        shell_join([python_cmd(plan), "src/method_runners/run_3dgs_scene.py", "--help"]),
        shell_join([python_cmd(plan), "src/method_runners/run_mip_splatting_scene.py", "--help"]),
        shell_join([python_cmd(plan), "src/method_runners/run_2dgs_scene.py", "--help"]),
        shell_join([python_cmd(plan), "eval/evaluate_public.py", "--help"]),
    ]
    write_script(path, "Non-GPU preflight checks", commands)
    return path


def generate_masks(plan: dict[str, Any], out_dir: Path) -> Path:
    path = out_dir / "01_generate_masks.sh"
    commands = [
        shell_join(
            [
                python_cmd(plan),
                "utils/generate_sam3_masks.py",
                "--split-root",
                root(plan, "public"),
                "--output-dir",
                root(plan, "masks"),
            ]
        ),
        shell_join(
            [
                python_cmd(plan),
                "utils/generate_sam3_masks.py",
                "--split-root",
                root(plan, "private"),
                "--output-dir",
                root(plan, "masks"),
            ]
        ),
    ]
    write_script(path, "Optional SAM mask generation for mask-weighted experiments", commands)
    return path


def generate_smoke(plan: dict[str, Any], out_dir: Path) -> Path:
    path = out_dir / "10_smoke.sh"
    scene = phase_scenes(plan, "smoke")[0]
    commands = [
        f"# Smoke test only: one public scene ({scene}). Do not use this for method ranking.",
    ]
    for run in active_runs(plan):
        commands.append(f"# run: {run_id(run)} ({run['method']})")
        commands.append(train_command(plan, "smoke", run, scene))
        commands.append("")
    write_script(path, "One-scene smoke render checks; not a benchmark", commands)
    return path


def generate_public_train(plan: dict[str, Any], out_dir: Path) -> Path:
    path = out_dir / "20_public.sh"
    commands: list[str] = []
    for run in active_runs(plan):
        commands.append(f"# run: {run_id(run)} ({run['method']})")
        for scene in phase_scenes(plan, "public"):
            commands.append(train_command(plan, "public", run, scene))
        commands.append("")
    write_script(path, "Train/render complete public benchmark runs", commands)
    return path


def generate_public_eval(plan: dict[str, Any], out_dir: Path) -> Path:
    path = out_dir / "21_eval.sh"
    commands = [mkdir_command(root(plan, "reports") / "eval" / "public")]
    commands.extend(eval_command(plan, run) for run in active_runs(plan))
    write_script(path, "Evaluate complete public benchmark runs", commands)
    return path


def generate_private(plan: dict[str, Any], out_dir: Path, selected_run: str) -> Path:
    run = find_run(plan, selected_run)
    rid = run_id(run)
    private_output_dir = output_dir(plan, "private", rid)
    zip_path = artifact_path(plan, "submission_zip", "private", rid)
    path = out_dir / "30_private.sh"
    commands = [
        f"# selected private run: {rid} ({run['method']})",
        f"# Private renders stay under {private_output_dir}; zip is created under submissions only after folder verification passes.",
        mkdir_command(private_output_dir),
    ]
    for scene in phase_scenes(plan, "private"):
        commands.append(train_command(plan, "private", run, scene))
    commands.extend(
        [
            "",
            shell_join(
                [
                    python_cmd(plan),
                    "verify/verify_submission.py",
                    "--split-root",
                    root(plan, "private"),
                    "--submission-dir",
                    private_output_dir,
                ]
            ),
            shell_join(
                [
                    python_cmd(plan),
                    "utils/make_submission_zip.py",
                    "--submission-dir",
                    private_output_dir,
                    "--zip",
                    zip_path,
                    "--overwrite",
                ]
            ),
            shell_join(
                [
                    python_cmd(plan),
                    "verify/verify_zip.py",
                    "--split-root",
                    root(plan, "private"),
                    "--zip",
                    zip_path,
                ]
            ),
        ]
    )
    write_script(path, "Render private set with exactly one selected run", commands)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment command files without executing them.")
    parser.add_argument("--config", type=Path, default=Path("configs/gs_runs.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/commands"))
    parser.add_argument(
        "--selected-run",
        "--selected-experiment",
        dest="selected_run",
        default=None,
        help="Generate private render script for this run id.",
    )
    args = parser.parse_args()

    plan = load_plan(args.config)
    generated = [
        generate_preflight(plan, args.out_dir),
        generate_masks(plan, args.out_dir),
        generate_smoke(plan, args.out_dir),
        generate_public_train(plan, args.out_dir),
        generate_public_eval(plan, args.out_dir),
    ]
    if args.selected_run:
        generated.append(generate_private(plan, args.out_dir, args.selected_run))

    print("Generated command files:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
