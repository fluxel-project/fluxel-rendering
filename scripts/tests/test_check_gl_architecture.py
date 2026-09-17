from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "check_gl_architecture.py"
SPEC = importlib.util.spec_from_file_location("check_gl_architecture", SCRIPT)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


class CheckGlArchitectureTests(unittest.TestCase):
    def write(self, root: Path, relative: str, text: str) -> None:
        path = root / "crates/rhi/src/webgl2" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_missing_webgl_tree_is_a_successful_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(CHECKER.check(Path(directory)), [])

    def test_window_and_angle_providers_are_forbidden_but_wgl_bindings_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "crates/rhi/Cargo.toml"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                '[dependencies]\n'
                'glutin = "0.32"\n'
                'winit = "0.30"\n'
                'glutin-winit = "0.5"\n'
                'angle = "1"\n'
                'glutin_wgl_sys = "0.6"\n',
                encoding="utf-8",
            )
            observed = {item.dependency for item in CHECKER.check(root)}
            self.assertEqual(observed, {"glutin", "winit", "glutin-winit", "angle"})

    def test_each_layer_reports_its_explicit_forbidden_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "api/core.rs", "use crate::webgl2::platform::Canvas;\n")
            self.write(root, "api/browser/mod.rs", "use crate::webgl2::state::State;\n")
            self.write(root, "api/native.rs", "use crate::webgl2::state::State;\n")
            self.write(root, "state/mod.rs", "use wasm_bindgen::JsCast;\n")
            self.write(root, "compat.rs", "use fluxel_renderer::Renderer;\n")
            observed = {(item.layer, item.dependency) for item in CHECKER.check(root)}
            self.assertEqual(observed, {
                ("api/core-profile", "platform"),
                ("api/browser", "state"),
                ("api/native", "state"),
                ("state", "wasm_bindgen"),
                ("compat", "renderer"),
            })

    def test_state_cannot_reach_the_compatibility_adapter(self) -> None:
        """The release gate's first bullet, and the only line enforcing it.

        The `state` row listed `renderer` and `rendergraph` but not `compat`,
        so Layer 2 could name Layer 3 and this gate -- whose whole job is to
        keep the layers apart -- reported nothing.  The direction is the one
        the layered contract names explicitly, which is why it is worth a test
        rather than only the entry in the forbidden set.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "state/mod.rs", "use crate::webgl2::compat::CompatibleDevice;\n")
            observed = {(item.layer, item.dependency) for item in CHECKER.check(root)}
            self.assertIn(("state", "compat"), observed)

    def test_browser_can_reach_platform_and_native_can_reach_glow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(
                root,
                "api/browser.rs",
                "use web_sys::WebGl2RenderingContext;\nuse glow::Context;\n",
            )
            self.write(root, "api/native/mod.rs", "use glow::Context;\n")
            self.write(root, "compat/mod.rs", "use crate::webgl2::{api::GlFamilyApi, state::State};\n")
            self.assertEqual(CHECKER.check(root), [])

    def test_native_context_providers_reach_glow_but_not_state_or_browser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(
                root,
                "api/egl.rs",
                "use glow::Context;\nuse khronos_egl::Display;\n"
                "fn bind() { crate::webgl2::state::State::new(); }\n",
            )
            self.write(
                root,
                "api/wgl/mod.rs",
                "use glow::Context;\nfn surface() { let _ = web_sys::Window; }\n",
            )
            observed = {(item.layer, item.dependency) for item in CHECKER.check(root)}
            self.assertEqual(observed, {
                ("api/egl-provider", "state"),
                ("api/wgl-provider", "web_sys"),
            })

    def test_comments_and_literals_do_not_create_violations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "api/core.rs", '''
                // use crate::webgl2::state::State;
                /* nested /* use glow::Context; */ comment */
                const NOTE: &str = "use web_sys::Window;";
                const RAW: &str = r###"use crate::webgl2::platform::Canvas;"###;
                const MARKER: char = 'x';
                use crate::webgl2::types::Descriptor;
            ''')
            self.assertEqual(CHECKER.check(root), [])

    def test_removed_legacy_gl_api_cannot_return(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "api/mock.rs", "impl GlRasterApi for Recorder {}\n")
            observed = {(item.layer, item.dependency) for item in CHECKER.check(root)}
            self.assertEqual(observed, {("legacy-api", "GlRasterApi")})

    def test_fully_qualified_paths_cannot_bypass_the_import_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "api/core.rs", "fn create() { let _ = glow::Context::default(); }\n")
            self.write(root, "api/profile.rs", "fn canvas() { let _ = web_sys::Window::new(); }\n")
            self.write(root, "api/native.rs", "fn state() { crate::webgl2::state::State::new(); }\n")
            self.write(root, "state.rs", "fn render() { fluxel_rendergraph::Graph::new(); }\n")
            self.write(root, "compat/mod.rs", "fn bridge() { crate::webgl2::renderer::Context::new(); }\n")
            observed = {(item.layer, item.dependency) for item in CHECKER.check(root)}
            self.assertEqual(observed, {
                ("api/core-profile", "glow"),
                ("api/core-profile", "web_sys"),
                ("api/native", "state"),
                ("state", "rendergraph"),
                ("compat", "renderer"),
            })

    def test_identifiers_and_unrelated_import_aliases_are_not_paths_to_a_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "api/core.rs", '''
                use unrelated::Thing as state;
                fn allowed() {
                    let state_value = 1;
                    let web_sys_value = 2;
                    state::construct();
                    glowfish::swim();
                    fluxel_renderer_info::record();
                }
            ''')
            self.assertEqual(CHECKER.check(root), [])


if __name__ == "__main__":
    unittest.main()
