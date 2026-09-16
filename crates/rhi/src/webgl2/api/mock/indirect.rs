//! Mock indirect command-buffer domain: single draw, batch, and counted batch.
//!
//! The record layouts, their ABI, and the capability rows live in
//! `api/indirect.rs`; this module only mirrors what a provider does with one
//! record: prove its row, prove the record sits where it claims to, prove the
//! pass, then prove the command buffer's role.  The order is the providers'
//! order, so a trace that shows an accepted command also shows which of those
//! facts it needed, and a trace that shows only an error proves the guard ran
//! before the binding changed.
//!
//! Two of these three domains have no executable provider in this contract: the
//! native families bind no multi-draw-indirect entry point, and WebGL2 has no
//! indirect mapping at all.  The recorder implements them anyway, because the
//! differential contract covers the whole Layer 1 vocabulary rather than only
//! the subset a provider happens to reach today; where a rule would otherwise be
//! invented it is taken from the capability row that gates the domain rather
//! than from a provider call site that does not exist.

use super::*;

impl MockGlFamilyApi {
    /// Rejects an indirect command whose capability row this context never proved.
    ///
    /// Both providers answer `Unsupported` rather than a validation failure: the
    /// command is well formed, the context simply never earned the right to run
    /// it, and a caller that cannot tell those two apart would keep retrying a
    /// command this context can never accept.
    pub(super) fn require_indirect_capability(
        &mut self,
        op: &'static str,
        capability: GlCapability,
        reason: &'static str,
    ) -> Result<(), GlError> {
        if self.discovery.capabilities().supports(capability) {
            Ok(())
        } else {
            self.error_result(GlError::Unsupported {
                operation: op,
                reason,
            })
        }
    }

    /// Rejects an indirect raster command issued outside a render pass.
    ///
    /// Both providers bind their scratch or pipeline state only while a pass is
    /// active, and both clear that state at `end_render_pass`, so the pass is
    /// the outer of their two conditions.  The installed pipeline, its vertex
    /// array, and the array's index binding are provider state the recorder
    /// deliberately does not model; an active pass is the strongest precondition
    /// its own state supports, and it is enforced before the binding changes.
    pub(super) fn require_indirect_pass(&mut self, op: &'static str) -> Result<(), GlError> {
        if self.pass_active {
            Ok(())
        } else {
            self.invalid(op, "no active render pass")
        }
    }

    /// Rejects a range that does not name a live indirect command buffer.
    ///
    /// The usage role is what lets a provider read the bytes as records instead
    /// of as caller data, so both providers check it and neither accepts an
    /// unroled allocation.
    pub(super) fn indirect_buffer(
        &mut self,
        op: &'static str,
        range: GlBufferRange,
    ) -> Result<(), GlError> {
        let desc = self.buffer(op, range.buffer)?;
        if desc.usage.contains(GlBufferUsage::INDIRECT) {
            Ok(())
        } else {
            self.invalid(op, "command buffer lacks indirect usage")
        }
    }

    /// Rejects a batch that asks to read more draws than the context ever claimed.
    ///
    /// `max_multi_draw_indirect_count` is the numeric half of this domain's own
    /// capability row -- `GlLimits::supports_multi_draw_indirect` is exactly the
    /// question "was a nonzero count limit queried" -- so the recorded limit is
    /// the bound the domain is defined by, not an extra rule layered on top.
    fn require_multi_draw_count(&mut self, op: &'static str, draws: u32) -> Result<(), GlError> {
        // The capability gate above has already proved a nonzero limit, so this
        // default is unreachable; it stays fail-closed rather than picking a
        // value that would accept a count the context never claimed.
        let max = self
            .discovery
            .limits()
            .max_multi_draw_indirect_count
            .unwrap_or(0);
        if draws > max {
            return self.invalid(
                op,
                "draw count exceeds the queried multi-draw-indirect limit",
            );
        }
        Ok(())
    }
}

impl GlDrawIndirectApi for MockGlFamilyApi {
    fn draw_indirect(&mut self, command: GlIndirectCommandRange) -> Result<(), GlError> {
        const OP: &str = "draw-indirect";
        self.ready(OP)?;
        self.require_indirect_capability(
            OP,
            GlCapability::IndirectDraw,
            "this context did not prove the indirect-draw capability",
        )?;
        // The record module already builds the exact error a provider
        // propagates, so the recorder records that error verbatim instead of
        // restating the rule in its own words: a differential test then compares
        // one error rather than two spellings of it.
        if let Err(error) = command.validate(OP) {
            return self.error_result(error);
        }
        self.require_indirect_pass(OP)?;
        self.indirect_buffer(OP, command.range)?;
        // An indexed record additionally needs the vertex array's index
        // binding, which the recorder does not model: a provider reaches that
        // check only after every condition above has passed, so accepting here
        // never claims more than the context proved.
        self.calls.push(MockCall::DrawIndirect(command));
        Ok(())
    }
}

impl GlMultiDrawIndirectApi for MockGlFamilyApi {
    fn multi_draw_indirect(&mut self, commands: GlIndirectCommandRange) -> Result<(), GlError> {
        const OP: &str = "multi-draw-indirect";
        self.ready(OP)?;
        self.require_indirect_capability(
            OP,
            GlCapability::MultiDrawIndirect,
            "this context did not prove the multi-draw-indirect capability",
        )?;
        if let Err(error) = commands.validate(OP) {
            return self.error_result(error);
        }
        self.require_indirect_pass(OP)?;
        self.indirect_buffer(OP, commands.range)?;
        self.require_multi_draw_count(OP, commands.draw_count)?;
        self.calls.push(MockCall::MultiDrawIndirect(commands));
        Ok(())
    }
}

impl GlMultiDrawCountApi for MockGlFamilyApi {
    fn multi_draw_indirect_count(
        &mut self,
        commands: GlIndirectCommandRange,
        count: GlIndirectCountRange,
    ) -> Result<(), GlError> {
        const OP: &str = "multi-draw-indirect-count";
        self.ready(OP)?;
        self.require_indirect_capability(
            OP,
            GlCapability::MultiDrawIndirect,
            "this context did not prove the multi-draw-indirect capability",
        )?;
        if let Err(error) = commands.validate(OP) {
            return self.error_result(error);
        }
        // The count range's validator hardcodes its own operation name, so a
        // failure reports the module's spelling rather than this call's.  That
        // is the error a provider returns as well, and rewriting it here would
        // make the recorder's trace disagree with the provider it mirrors.
        if let Err(error) = count.validate() {
            return self.error_result(error);
        }
        self.require_indirect_pass(OP)?;
        self.indirect_buffer(OP, commands.range)?;
        self.indirect_buffer(OP, count.range)?;
        // Both numbers bound how many records the call may read, and the context
        // claimed one limit for that, so both are measured against it.
        self.require_multi_draw_count(OP, commands.draw_count)?;
        self.require_multi_draw_count(OP, count.max_draw_count)?;
        self.calls
            .push(MockCall::MultiDrawIndirectCount { commands, count });
        Ok(())
    }
}
