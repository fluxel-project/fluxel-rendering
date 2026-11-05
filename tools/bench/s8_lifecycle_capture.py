"""Retain one exact-source native material-heavy lifecycle attempt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import s8_material_baseline as base
from s8_lifecycle_review import review
from smoke_evidence import (built_fluxel_packages, freeze_source_inputs, required_built_packages,
                            resolved_git_packages, retain_executable, verify_built_sources,
                            verify_git_checkouts)
from smoke_inputs import build_environment, controlled_child_environment


def capture(backend: str, output: Path) -> None:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {"kind": "s8-native-lifecycle", "status": "partial", "backend": backend}

    def persist():
        (output / "attempt.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    persist()
    try:
        inputs = base.archive_inputs()
        state["source"] = {"inputs": inputs, "archive": freeze_source_inputs(inputs, output / "source-inputs.zip", base.ROOT),
                           "git": base.git_snapshot(), "locks": [{"path": base.relative(p), "sha256": base.digest(p)} for p in base.LOCKS]}
        state["environment"] = base.host_environment()
        state["toolchain"] = base.command_output(["rustc", f"+{base.MSRV}", "-Vv"])
        state["build_environment"] = build_environment(base.ROOT)
        if state["build_environment"]["status"] != "auditable":
            raise ValueError("uncontrolled build environment")
        environment, state["controlled_child_environment"] = controlled_child_environment({"RUSTUP_TOOLCHAIN": base.MSRV})
        target = base.ROOT / "target" / "s8-material-baseline" / backend
        state["build"] = base.run_process(
            ["cargo", f"+{base.MSRV}", "build", "--manifest-path", str(base.MANIFEST), "--locked", "--release",
             "--no-default-features", "--features", backend, "--message-format=json-render-diagnostics", "--target-dir", str(target)],
            base.CONSUMER, output / "build.log", 900, environment)
        persist()
        if state["build"]["exit_status"] != 0:
            raise ValueError("native lifecycle build failed")
        packages = built_fluxel_packages(output / "build.log")
        locked = resolved_git_packages(base.CONSUMER / "Cargo.lock")
        state["built_source_verification"] = verify_built_sources(packages, locked, required_built_packages(backend))
        state["built_git_checkout_verification"] = verify_git_checkouts(packages, locked)
        state["built_packages"] = packages
        if any(state[key]["status"] != "verified" for key in ("built_source_verification", "built_git_checkout_verification")):
            raise ValueError("native lifecycle dependency verification failed")
        retained = output / "executed-binary"
        retained.mkdir()
        state["executable"] = retain_executable(base.executable(target), retained)
        executable = Path(state["executable"]["path"])
        before = base.digest(executable)
        base.source_is_stable(inputs)
        state["run"] = base.run_process([str(executable), "--s8-lifecycle-only", "--output", str(output / "run")],
                                         base.ROOT, output / "run.log", 180, environment)
        persist()
        if state["run"]["exit_status"] != 0:
            raise ValueError("native lifecycle workload failed")
        state["independent_review"] = review(output / "run" / "s8-material-lifecycle")
        base.source_is_stable(inputs)
        if base.digest(executable) != before:
            raise ValueError("executed binary changed")
        state["status"] = "complete"
    except Exception as error:
        state["status"] = "failed"
        state["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        state["artifacts"] = [{"path": p.relative_to(output).as_posix(), "sha256": base.digest(p), "bytes": p.stat().st_size}
                              for p in sorted(output.rglob("*")) if p.is_file() and p.name != "attempt.json"]
        persist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("dx12", "native-gl-wgl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.backend, args.output)
    print(args.output / "attempt.json")
