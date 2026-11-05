from __future__ import annotations

from .common import *
try:
    from .. import capture_contract
except ImportError:
    import capture_contract

def retained_attempt_artifacts(selected: dict[str, dict[str, Any]], closure: Path) -> dict[str, set[tuple[Path, str]]]:
    """Only raw artifacts retained by the selected attempt may support review."""
    retained: dict[str, set[tuple[Path, str]]] = {}
    for attempt in selected.values():
        identifier = attempt.get("id")
        if not isinstance(identifier, str) or not identifier:
            continue
        directory = closure / str(attempt.get("directory", ""))
        records = []
        for command in attempt.get("commands", []):
            if isinstance(command, dict) and isinstance(command.get("result"), dict):
                records.extend(command["result"].get("raw_logs", []))
                records.extend(command["result"].get("retained_files", []))
        entries: set[tuple[Path, str]] = set()
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                continue
            candidate = under(directory, record["path"])
            if candidate is not None and isinstance(record.get("sha256"), str):
                entries.add((candidate.resolve(), record["sha256"]))
        retained[identifier] = entries
    return retained


def linked_raw_record(value: Any, identifier: str, retained: dict[str, set[tuple[Path, str]]],
                      base: Path) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str) or not isinstance(value.get("sha256"), str):
        return False
    path = Path(value["path"])
    path = path if path.is_absolute() else base / path
    return (path.resolve(), value["sha256"]) in retained.get(identifier, set()) and sha256(path) == value["sha256"]


def attempt_raw_record(path: Path, identifier: str, retained: dict[str, set[tuple[Path, str]]]) -> dict[str, str] | None:
    digest = sha256(path)
    if digest is None or (path.resolve(), digest) not in retained.get(identifier, set()):
        return None
    return {"path": str(path), "sha256": digest}


def required_review_records(identifier: str, retained: dict[str, set[tuple[Path, str]]],
                            root: Path, relative_paths: list[str]) -> list[dict[str, str]] | None:
    records = [attempt_raw_record(root / relative, identifier, retained) for relative in relative_paths]
    return records if all(record is not None for record in records) else None


def visual_entry_failures(entry: dict[str, Any], identifier: str,
                          retained: dict[str, set[tuple[Path, str]]], root: Path, base: Path) -> list[str]:
    surface = root / "run" / "surface"
    screenshots = [surface / f"surface-client-frame-{frame}.bmp" for frame in range(3)]
    frames = entry.get("frames")
    unexpected = [path.name for path in surface.glob("surface-client-frame-*.bmp")
                  if path.name not in {screenshot.name for screenshot in screenshots}]
    if unexpected:
        return [f"visual evidence has unsupported extra surface screenshot(s) for {identifier}: {unexpected}"]
    if (not all(screenshot.is_file() for screenshot in screenshots)
            or not isinstance(frames, list) or len(frames) != len(screenshots)):
        return [f"visual evidence does not cover every retained surface screenshot for {identifier}"]
    reasons: list[str] = []
    expected = {item.name for item in screenshots}
    seen: set[str] = set()
    for frame in frames:
        if not isinstance(frame, dict) or frame.get("human_verdict") != "passed":
            reasons.append(f"visual evidence has no passing human verdict for {identifier}")
            continue
        screenshot, capture, verdict = frame.get("screenshot"), frame.get("capture"), frame.get("verdict")
        if not all(linked_raw_record(item, identifier, retained, base) for item in (screenshot, capture, verdict)):
            reasons.append(f"visual evidence is not retained raw evidence for {identifier}")
            continue
        screenshot_path = Path(str(screenshot["path"])); screenshot_path = screenshot_path if screenshot_path.is_absolute() else base / screenshot_path
        if screenshot_path.name not in expected or screenshot_path.name in seen:
            reasons.append(f"visual evidence has an unknown or duplicate screenshot for {identifier}")
            continue
        capture_value, capture_error = read_json(Path(str(capture["path"])) if Path(str(capture["path"])).is_absolute() else base / str(capture["path"]))
        verdict_value, verdict_error = read_json(Path(str(verdict["path"])) if Path(str(verdict["path"])).is_absolute() else base / str(verdict["path"]))
        extent = capture_value.get("clientExtent") if isinstance(capture_value, dict) else None
        capture_reasons = capture_contract.capture_failures(
            capture_value, capture_value.get("frame") if isinstance(capture_value, dict) else -1,
            extent if isinstance(extent, list) else [], screenshot_path.name)
        verdict_reasons = capture_contract.verdict_failures(
            verdict_value, capture_value.get("frame") if isinstance(capture_value, dict) else -1,
            screenshot_path.name, capture_value.get("schemaVersion") if isinstance(capture_value, dict) else None)
        if capture_error or verdict_error or capture_reasons or verdict_reasons:
            reasons.append(f"visual evidence does not match the retained capture scenario for {identifier}")
        seen.add(screenshot_path.name)
    if seen != expected:
        reasons.append(f"visual evidence omitted a retained surface screenshot for {identifier}")
    return reasons


def scenario_entry_failures(entry: dict[str, Any], identifier: str,
                            retained: dict[str, set[tuple[Path, str]]], root: Path,
                            base: Path, scenario: str, paths: list[str]) -> list[str]:
    if entry.get("scenario") != scenario:
        return [f"{scenario} review has the wrong scenario label for {identifier}"]
    if entry.get("verdict") != "passed":
        return [f"{scenario} review has no explicit passing verdict for {identifier}"]
    required = required_review_records(identifier, retained, root, paths)
    raw = entry.get("raw_evidence")
    if required is None or not isinstance(raw, list):
        return [f"{scenario} review lacks retained mandatory raw evidence for {identifier}"]
    required_pairs = {(Path(item["path"]).resolve(), item["sha256"]) for item in required}
    supplied_pairs = set()
    for item in raw:
        if not isinstance(item, dict) or not linked_raw_record(item, identifier, retained, base):
            continue
        candidate = Path(item["path"])
        supplied_pairs.add(((candidate if candidate.is_absolute() else base / candidate).resolve(), item["sha256"]))
    return [] if required_pairs <= supplied_pairs else [f"{scenario} review substitutes or omits mandatory raw evidence for {identifier}"]


def external_review(path: Path | None, attempts: set[str], kind: str,
                    selected: dict[str, dict[str, Any]] | None = None,
                    closure: Path | None = None) -> dict[str, Any]:
    if path is None:
        return result(kind, False, ["explicit review file was not supplied"])
    value, error = read_json(path)
    if error or not isinstance(value, dict):
        return result(kind, False, [f"invalid review file: {error}"])
    reasons: list[str] = []
    if value.get("status") != "reviewed":
        reasons.append("review status is not 'reviewed'")
    if set(value.get("attempt_ids", [])) != attempts:
        reasons.append("review does not name exactly the selected attempts")
    if selected is None or closure is None:
        reasons.append("review was not linked to selected attempt records")
        return result(kind, False, reasons, path=str(path))
    selected_by_id = {attempt.get("id"): attempt for attempt in selected.values()
                      if isinstance(attempt, dict) and isinstance(attempt.get("id"), str)}
    retained = retained_attempt_artifacts(selected_by_id, closure)
    entries = value.get("evidence")
    if not isinstance(entries, list) or {item.get("attempt_id") for item in entries if isinstance(item, dict)} != attempts:
        reasons.append("review lacks one typed evidence entry for each selected attempt")
        entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            reasons.append("review evidence entry is malformed")
            continue
        identifier = entry.get("attempt_id")
        if not isinstance(identifier, str):
            reasons.append("review evidence has no attempt identifier")
            continue
        attempt = selected_by_id.get(identifier)
        if not isinstance(attempt, dict):
            reasons.append("review evidence refers to an unselected attempt")
            continue
        attempt_root = closure / str(attempt.get("directory", ""))
        if kind == "reviewed visual output":
            reasons.extend(visual_entry_failures(entry, identifier, retained, attempt_root, path.parent))
        elif kind == "linked focused test evidence":
            log = entry.get("log")
            command_index = entry.get("command_index")
            commands = attempt.get("commands")
            command = (commands[command_index] if isinstance(commands, list) and type(command_index) is int
                       and 0 <= command_index < len(commands) else None)
            command_result = command.get("result") if isinstance(command, dict) else None
            test_logs = command_result.get("raw_logs") if isinstance(command_result, dict) else None
            if (not isinstance(command, dict) or command.get("kind") != "test"
                    or not isinstance(command.get("command"), list) or not command["command"]
                    or not isinstance(command_result, dict) or command_result.get("exit_status") != 0
                    or command_result.get("test_command_recognized") is not True
                    or type(command_result.get("executed")) is not int or command_result["executed"] <= 0
                    or not isinstance(test_logs, list) or log not in test_logs
                    or not linked_raw_record(log, identifier, retained, path.parent)):
                reasons.append(f"test evidence lacks a retained command, execution count, or log for {identifier}")
            else:
                log_path = Path(log["path"]); log_path = log_path if log_path.is_absolute() else path.parent / log_path
                if not log_path.read_text(encoding="utf-8", errors="replace").strip():
                    reasons.append(f"test evidence has an empty retained log for {identifier}")
        elif kind == "reviewed native failure, loss, and device-replacement evidence":
            reasons.extend(scenario_entry_failures(
                entry, identifier, retained, attempt_root, path.parent, "device-replacement",
                ["run/device-replacement.json", "run/device-replacement-input.bin",
                 "run/device-replacement-expected.json", "run/device-replacement-readback.rgba"]))
        else:
            paths = []
            for directory in ("read-set", "s2-texture", "s3-material", "s3-material/replacement"):
                paths.append(f"run/{directory}/read-set-workload.json")
                for index in range(3):
                    paths.extend(f"run/{directory}/read-set-{kind}-{index}.{suffix}"
                                 for kind, suffix in (("input", "bin"), ("expected", "rgba"), ("readback", "rgba")))
            reasons.extend(scenario_entry_failures(entry, identifier, retained, attempt_root, path.parent,
                                                   "read-set", paths))
    return result(kind, not reasons, reasons, path=str(path), entries=len(entries))
