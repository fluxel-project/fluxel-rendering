#!/usr/bin/env python3
"""Enforce the WebGL-family import boundaries introduced by the 0.15 plan.

The checker deliberately has no Rust parser dependency.  It examines ``use``
and ``extern crate`` declarations after removing Rust comments and literals, so
architecture names mentioned in documentation, comments, and test strings do
not become false violations.  It additionally confines the Layer 1-private
scratch binding targets to provider execution bodies.  A missing
``crates/rhi/src/webgl2`` tree is a successful no-op: the check is intended to
land before every planned layer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys


WEBGL_RELATIVE_ROOT = Path("crates/rhi/src/webgl2")
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
USE_DECLARATION = re.compile(
    r"\b(?:pub(?:\s*\([^)]*\))?\s+)?use\s+([^;]+);", re.MULTILINE
)
EXTERN_CRATE = re.compile(r"\bextern\s+crate\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+as\s+\w+)?\s*;")
QUALIFIED_PATH = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+")
FORBIDDEN_PROVIDER_DEPENDENCY = re.compile(
    r'^\s*(?P<name>winit|glutin|glutin-winit|angle)\s*=\s*', re.MULTILINE
)
SCRATCH_BINDING = re.compile(
    r"\b(?:ARRAY_BUFFER|COPY_READ_BUFFER|COPY_WRITE_BUFFER"
    r"|PIXEL_PACK_BUFFER|PIXEL_UNPACK_BUFFER)\b"
)
LEGACY_GL_API = (
    "GlDraw",
    "GlBufferCopy",
    "GlQueryKind",
    "GlPresentation",
    "GlRasterApi",
    "GlCopyApi",
    "GlQueryApi",
    "GlComputeApi",
    "GlStorageApi",
    "GlPresentationApi",
)


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    layer: str
    dependency: str


def strip_rust_non_code(source: str) -> str:
    """Replace comments and literals with spaces, retaining newlines.

    Rust allows nested block comments and raw strings with an arbitrary number
    of ``#`` delimiters.  A small scanner is less brittle here than a regex and
    preserves positions for useful diagnostics.
    """
    output = list(source)
    index = 0
    length = len(source)

    def erase(start: int, end: int) -> None:
        for position in range(start, end):
            if output[position] != "\n":
                output[position] = " "

    while index < length:
        if source.startswith("//", index):
            end = source.find("\n", index)
            end = length if end == -1 else end
            erase(index, end)
            index = end
            continue
        if source.startswith("/*", index):
            start = index
            index += 2
            depth = 1
            while index < length and depth:
                if source.startswith("/*", index):
                    depth += 1
                    index += 2
                elif source.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            erase(start, index)
            continue

        # Raw byte strings (br###"..."###) and raw strings (r###"..."###).
        raw = re.match(r"(?:br|r)(#{0,255})\"", source[index:])
        if raw:
            start = index
            hashes = raw.group(1)
            index += len(raw.group(0))
            closing = '"' + hashes
            end = source.find(closing, index)
            index = length if end == -1 else end + len(closing)
            erase(start, index)
            continue

        # Ordinary strings, byte strings, and character literals.  Treating a
        # lifetime apostrophe as a literal would hide code, so only start a
        # character literal when a closing unescaped apostrophe is nearby.
        quote_start = index
        quote = source[index]
        if quote == '"' or (quote == "'" and _looks_like_char_literal(source, index)):
            index += 1
        elif source.startswith('b"', index):
            quote = '"'
            index += 2
        elif source.startswith("b'", index) and _looks_like_char_literal(source, index + 1):
            quote = "'"
            index += 2
        else:
            index += 1
            continue
        while index < length:
            if source[index] == "\\":
                index += 2
            elif source[index] == quote:
                index += 1
                break
            else:
                index += 1
        erase(quote_start, min(index, length))
    return "".join(output)


def _looks_like_char_literal(source: str, start: int) -> bool:
    """Return true for a short quoted char, but never for a lifetime."""
    index = start + 1
    while index < len(source) and index <= start + 8:
        if source[index] == "\\":
            index += 2
        elif source[index] == "'":
            return True
        elif source[index] == "\n":
            return False
        else:
            index += 1
    return False


def dependency_segments(declaration: str) -> set[str]:
    """Return imported path segments, excluding ``as`` aliases."""
    without_aliases = re.sub(r"\bas\s+[A-Za-z_][A-Za-z0-9_]*", "", declaration)
    return set(IDENTIFIER.findall(without_aliases))


def contains_dependency(segments: set[str], dependency: str) -> bool:
    """Recognize direct modules plus Fluxel crate names such as fluxel_renderer."""
    return dependency in segments or f"fluxel_{dependency}" in segments


def imported_aliases(code: str) -> set[str]:
    """Collect import aliases so an unrelated ``Thing as state`` is not a path leak."""
    aliases: set[str] = set()
    for match in USE_DECLARATION.finditer(code):
        aliases.update(re.findall(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)", match.group(1)))
    for match in EXTERN_CRATE.finditer(code):
        alias = re.search(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)", match.group(0))
        if alias:
            aliases.add(alias.group(1))
    return aliases


def rules_for(relative: Path) -> tuple[str, tuple[str, ...]] | None:
    parts = relative.parts
    if not parts:
        return None
    first = parts[0]
    second = parts[1] if len(parts) > 1 else ""
    # Support both api/browser.rs and api/browser/mod.rs layouts.
    if first == "api" and (second == "browser" or relative.name == "browser.rs"):
        return ("api/browser", ("state", "compat", "renderer", "rendergraph", "glutin"))
    if first == "api" and (second == "native" or relative.name == "native.rs"):
        return ("api/native", ("state", "compat"))
    # EGL/WGL files are native context-provider implementations from the frozen
    # Checkpoint A stack: they drive GL through glow plus their platform loader
    # (khronos-egl / glutin_wgl_sys), so unlike core-profile files they may name
    # glow, but they must never reach state, compat, or browser APIs.
    if first == "api" and (second == "egl" or relative.name == "egl.rs"):
        return ("api/egl-provider", ("state", "compat", "web_sys", "js_sys", "wasm_bindgen"))
    if first == "api" and (second == "wgl" or relative.name == "wgl.rs"):
        return ("api/wgl-provider", ("state", "compat", "web_sys", "js_sys", "wasm_bindgen"))
    if first == "api":
        return ("api/core-profile", ("state", "compat", "platform", "renderer", "rendergraph", "web_sys", "glow", "glutin"))
    if first == "state" or relative.name == "state.rs":
        return ("state", ("web_sys", "js_sys", "wasm_bindgen", "glow", "glutin", "renderer", "rendergraph"))
    if first == "compat" or relative.name == "compat.rs":
        return ("compat", ("renderer", "residency", "legacy", "experimental", "web_sys", "js_sys", "wasm_bindgen", "glow", "glutin"))
    return None


def scratch_binding_allowed(relative: Path) -> bool:
    """Whether this file may name a Layer 1-private scratch binding target.

    ``ARRAY_BUFFER`` (a global binding point that is not VAO state),
    ``COPY_READ_BUFFER``/``COPY_WRITE_BUFFER`` and the pixel pack/unpack buffer
    targets are private to Layer 1's own implementation: a provider binds one
    immediately before each use and deliberately leaves it bound, because no
    Layer 1 verb accepts a caller-supplied target and Layer 2 cannot name
    ``glow`` at all, so it cannot mirror a binding point it cannot name.  That
    argument only holds while the constants stay inside a provider module --
    never in the shared contract (``api/*.rs``), never in the mock recorder,
    which models Fluxel identities rather than GL targets, and never in
    ``state/``.  ``ELEMENT_ARRAY_BUFFER`` is deliberately not listed: it is VAO
    state, written as part of a ``bind_vertex_array`` request, and therefore a
    binding point Layer 2 does mirror.
    """
    parts = relative.parts
    return len(parts) >= 2 and parts[0] == "api" and parts[1] in ("native", "browser")


def check(root: Path) -> list[Violation]:
    webgl_root = root / WEBGL_RELATIVE_ROOT
    violations: list[Violation] = []
    observed: set[tuple[Path, int, str, str]] = set()

    def report(path: Path, line: int, layer: str, dependency: str) -> None:
        key = (path, line, layer, dependency)
        if key not in observed:
            observed.add(key)
            violations.append(Violation(path, line, layer, dependency))

    manifest = root / "crates/rhi/Cargo.toml"
    if manifest.is_file():
        source = manifest.read_text(encoding="utf-8", errors="replace")
        for match in FORBIDDEN_PROVIDER_DEPENDENCY.finditer(source):
            violations.append(Violation(
                manifest,
                source.count("\n", 0, match.start()) + 1,
                "manifest",
                match.group("name"),
            ))
    if not webgl_root.is_dir():
        return violations
    for path in sorted(webgl_root.rglob("*.rs")):
        relative = path.relative_to(webgl_root)
        rule = rules_for(relative)
        source = path.read_text(encoding="utf-8", errors="replace")
        code = strip_rust_non_code(source)
        if not scratch_binding_allowed(relative):
            for match in SCRATCH_BINDING.finditer(code):
                report(
                    path,
                    code.count("\n", 0, match.start()) + 1,
                    "scratch-binding",
                    match.group(0),
                )
        if rule is None:
            continue
        layer, forbidden = rule
        for symbol in LEGACY_GL_API:
            for match in re.finditer(rf"\b{re.escape(symbol)}\b", code):
                report(path, code.count("\n", 0, match.start()) + 1, "legacy-api", symbol)
        aliases = imported_aliases(code)
        declarations = [(match.start(1), match.group(1)) for match in USE_DECLARATION.finditer(code)]
        declarations.extend((match.start(1), match.group(1)) for match in EXTERN_CRATE.finditer(code))
        for start, declaration in declarations:
            segments = dependency_segments(declaration)
            for dependency in forbidden:
                if contains_dependency(segments, dependency):
                    report(path, code.count("\n", 0, start) + 1, layer, dependency)
        # Imports are not the only route across a Rust module boundary.  Scan
        # every qualified code path too (for example ``glow::Context`` and
        # ``crate::webgl2::state::State``).  Alias names are roots only; an
        # unrelated ``use foo::Thing as state`` must not turn its later
        # ``state::Thing`` use into a false architecture violation.
        for match in QUALIFIED_PATH.finditer(code):
            segments = set(match.group(0).split("::"))
            if match.group(0).split("::", 1)[0] in aliases:
                continue
            for dependency in forbidden:
                if contains_dependency(segments, dependency):
                    report(path, code.count("\n", 0, match.start()) + 1, layer, dependency)
    return violations


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Rendering repository root (default: inferred from this script).")
    parser.add_argument("--quiet", action="store_true", help="Suppress the successful check message.")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"GL architecture check failed: --root is not a directory: {root}", file=sys.stderr)
        return 2
    violations = check(root)
    if violations:
        for violation in violations:
            relative = violation.path.relative_to(root).as_posix()
            print(f"{relative}:{violation.line}: {violation.layer} must not reference {violation.dependency}")
        return 1
    if not args.quiet:
        location = root / WEBGL_RELATIVE_ROOT
        print(f"GL architecture check passed ({location if location.is_dir() else 'WebGL2 layers not present yet'}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
