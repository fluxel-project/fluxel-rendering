//! Observed transfer and barrier operations for the lifecycle witness.
use super::*;

impl CommandBackend for WitnessBackend {
    fn begin_encoder(device: &Arc<Native>, _: LaneId) -> Result<Arc<Native>> {
        Ok(Arc::clone(device))
    }
    fn transition(
        native: &mut Arc<Native>,
        transitions: &[ResourceTransition<'_, Self>],
    ) -> Result<()> {
        native
            .transitions
            .fetch_add(transitions.len(), Ordering::SeqCst);
        Ok(())
    }
    fn copy_buffer(
        _: &mut Arc<Native>,
        source: &Arc<NativeBuffer>,
        _: u64,
        _: &Arc<NativeBuffer>,
        destination: u64,
        bytes: u64,
    ) -> Result<()> {
        source.native.copied_bytes.lock().unwrap().push(bytes);
        source
            .native
            .copy_destinations
            .lock()
            .unwrap()
            .push(destination..destination + bytes);
        Ok(())
    }
    fn blit(
        native: &mut Arc<Native>,
        source: TextureRegion<'_, Self>,
        destination: TextureRegion<'_, Self>,
        filter: FilterMode,
    ) -> Result<()> {
        if !native.blit_supported.load(Ordering::SeqCst) {
            return Err(Error::Unsupported("witness mip blit"));
        }
        native.blits.lock().unwrap().push((
            source.mip,
            destination.mip,
            source.extent,
            destination.extent,
            filter,
        ));
        Ok(())
    }
    fn copy_texture(
        _: &mut Arc<Native>,
        _: TextureRegion<'_, Self>,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Err(Error::Unsupported("texture"))
    }
    fn buffer_to_texture(
        _: &mut Arc<Native>,
        source: &Arc<NativeBuffer>,
        layout: TexelCopyBufferLayout,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        source
            .native
            .texture_uploads
            .lock()
            .unwrap()
            .push((layout.offset, layout.bytes_per_row));
        Ok(())
    }
    fn texture_to_buffer(
        _: &mut Arc<Native>,
        _: TextureRegion<'_, Self>,
        _: &Arc<NativeBuffer>,
        _: TexelCopyBufferLayout,
    ) -> Result<()> {
        Ok(())
    }
    fn write_buffer(_: &mut Arc<Native>, _: &Arc<NativeBuffer>, _: u64, _: &[u8]) -> Result<()> {
        Ok(())
    }
    fn write_texture(
        _: &mut Arc<Native>,
        _: TextureRegion<'_, Self>,
        _: TexelCopyBufferLayout,
        _: &[u8],
    ) -> Result<()> {
        Err(Error::Unsupported("texture"))
    }
}
