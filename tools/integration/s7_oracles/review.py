"""Fail-closed workload readers for formal review (return reasons, never trust expected)."""
from __future__ import annotations
import json, struct
from pathlib import Path
from .core import bmp,check,color,color_raster,pbr,raster
import capture_contract

def _j(p):return json.loads(p.read_text())
def _raw(p):return p.read_bytes()
def _transform(points, columns):
    result=[]
    for point in points:
      v=[*point,1.0];o=[sum(columns[c][r]*v[c] for c in range(4)) for r in range(4)]
      if not o[3]:raise ValueError('position transform w=0')
      result.append([o[0]/o[3],o[1]/o[3],o[2]/o[3]])
    return result
def _case(raw, expected, mask):
    return check(raw,expected,mask)
def _surface_case(output, frame, stem, data, points, rgba, require_nondisturbing_capture):
    expected,mask=raster(points,rgba,data['companionExtent']); saved=_raw(output/f'{stem}-expected.rgba')
    if saved!=expected:raise ValueError(f'{stem}: saved expected not actual-input-derived')
    _case(_raw(output/f'{stem}-readback.rgba'),expected,mask)
    capture=_j(output/f'surface-client-frame-{frame}.capture.json'); verdict=_j(output/f'surface-client-frame-{frame}.verdict.json')
    if require_nondisturbing_capture and capture.get('schemaVersion') != 2: raise ValueError(f'{stem}: current capture lacks schema 2 non-disturbance evidence')
    reasons=capture_contract.capture_failures(capture,frame,data['surfaceExtent'],f'surface-client-frame-{frame}.bmp')
    reasons+=capture_contract.verdict_failures(verdict,frame,f'surface-client-frame-{frame}.bmp',capture.get('schemaVersion'))
    if reasons:raise ValueError(f'{stem}: capture metadata: {"; ".join(reasons)}')
    image,sy=raster(points,rgba,data['surfaceExtent']); pixels,w,h=bmp(output/f'surface-client-frame-{frame}.bmp',image,sy)
    if [w,h]!=data['surfaceExtent']:raise ValueError(f'{stem}: capture extent')
    window_mask=capture_contract.window_stable_mask(sy,data['surfaceExtent']) if capture.get('schemaVersion') == 2 else sy
    check(pixels,image,window_mask,True)
def review_surface(output:Path, require_nondisturbing_capture:bool=False)->list[str]:
  try:
    # Color 0/1: persisted geometry/texture and per-frame transform/tint.
    pos=[list(struct.unpack_from('<fff',_raw(output/'s7-surface-position.bin'),i)) for i in (0,12,24)];tex=_raw(output/'s7-surface-albedo.rgba')
    for f in range(2):
      d=_j(output/f's7-deferred-surface-frame-{f}-input.json');_surface_case(output,f,f's7-deferred-surface-frame-{f}',d,pos,color(pos,tex,d['record']['tint']),require_nondisturbing_capture)
    for f,name in ((2,'metal'),(3,'rough')):
      stem=f's7-deferred-surface-pbr-{name}';d=_j(output/f'{stem}-input.json');_surface_case(output,f,stem,d,d['positions'],pbr(d),require_nondisturbing_capture)
    base=struct.unpack('<9f',_raw(output/'s7-deferred-surface-deformation-positions.bin'))
    for f,name in ((4,'first'),(5,'second')):
      stem=f's7-deferred-surface-deformation-{name}';d=_j(output/f'{stem}-input.json');a,b,c,e=d['parameters'];pts=[[base[i]+a*base[i+1]**2+b,base[i+1],base[i+2]+c*base[i]+e] for i in range(0,9,3)];rgba=bytes(round(d['textureRgba'][i]*d['tint'][i]*d['surfaceFactor']) for i in range(3))+b'\xff';_surface_case(output,f,stem,d,pts,rgba,require_nondisturbing_capture)
  except (OSError,KeyError,TypeError,ValueError,struct.error,json.JSONDecodeError) as e:return [str(e)]
  return []
def review_device(output:Path)->list[str]:
  """Review the seven rendered two-device frames from their persisted inputs."""
  try:
    workload=_j(output/'s7-deferred-device-isolation-workload.json')
    rendered=('old-cold','old-warm','new-cold','new-warm','old-return','new-replacement','old-replacement')
    if {x['name'] for x in workload['cases']} != set(rendered)|{'foreign-submission-rejected'}:raise ValueError('device case inventory')
    for name in rendered:
      d=_j(output/f'{name}-input.json')
      if d.get('extent') != [96,96]: raise ValueError(f'{name}: device extent is not 96x96')
      stream=bytes(d['geometryBytes']);points=[list(struct.unpack_from('<fff',stream,i)) for i in (0,12,24)]
      points=_transform(points,d['transformColumns']);rgba=color(points,bytes(d['textureRgba8']),d['tint']);expected,mask=color_raster(points,bytes(d['textureRgba8']),d['tint'],[96,96])
      # Input-derived image, not saved expected/mask, is the only oracle.
      _case(_raw(output/f'{name}-actual.rgba'),expected,mask)
    old=_j(output/'old-cold-input.json')
    if old.get('extent') != [96,96]: raise ValueError('old-pending: device extent is not 96x96')
    stream=bytes(old['geometryBytes']);pts=[list(struct.unpack_from('<fff',stream,i)) for i in(0,12,24)];pts=_transform(pts,old['transformColumns']);expected,mask=color_raster(pts,bytes(old['textureRgba8']),old['tint'],old['extent']);_case(_raw(output/'old-pending-completed.rgba'),expected,mask)
  except (OSError,KeyError,TypeError,ValueError,struct.error,json.JSONDecodeError) as e:return [str(e)]
  return []
def review_offscreen(output:Path)->list[str]:
  try:
    # Twenty pre-cache raw routes: rebuild PBR, flat/tilt Color, CodeNode's recorded factors and Bend geometry.
    for name in ('flat','tilt'):
      d=_j(output/f's7-{name}-input.json')
      material=bytes(round(d['textureRgba'][i]*d['tint'][i]) for i in range(4))
      for route in ('deferred','forward'):
        expected,mask=(color_raster(d['positionPlane'],d['textureRgba'],d['tint'],[96,96]) if route=='deferred' else raster(d['positionPlane'],material,[96,96]))
        _case(_raw(output/f's7-{name}-{route}.rgba'),expected,mask)
    for name in ('code-node-first','code-node-second'):
      d=_j(output/f's7-{name}-input.json');src=d['sourceColor'];f=d['factors'];rgba=bytes([round(src[2]*f[0]),round(src[1]*f[1]),round(src[0]*f[2]),255]);pts=_j(output/'s7-flat-input.json')['positionPlane']
      for route in ('deferred','forward'):
        expected,mask=raster(pts,color(pts,rgba,d['tint']) if route=='deferred' else rgba,[96,96]);_case(_raw(output/f's7-{name}-{route}.rgba'),expected,mask)
    for name in ('pbr-dielectric','pbr-metal','pbr-rough','pbr-normal-transform','pbr-base-color'):
      d=_j(output/f's7-{name}-input.json')
      for route in ('deferred','forward'):
        expected,mask=raster(_transform(d['positions'],d['transformColumns']),pbr(d,route=='deferred'),[96,96]);_case(_raw(output/f's7-{name}-{route}.rgba'),expected,mask)
    # D1 cache cases retain real stream/texture/source units; check their actual material transform.
    for name in ('cold','warm-state','warm-settled','tint','texture-replacement','source-replacement'):
      d=_j(output/f's7-deferred-cache-{name}-input.json');stream=bytes(d['geometryStreamBytes']);pts=[list(struct.unpack_from('<fff',stream,i)) for i in (0,12,24)];t=d['textureRgba8'];base=[t[i]/255*d['tint'][i] for i in range(3)];s=d['materialSourceUnits'];m=[base[2]*.5,base[1],base[0]] if 'input.b * 0.5' in s else [base[2],base[1]*.5,base[0]];rgba=color(pts,bytes(round(x*255) for x in m)+bytes([t[3]]),[1,1,1,1]);expected,mask=raster(pts,rgba,[96,96]);_case(_raw(output/f's7-deferred-cache-{name}-actual.rgba'),expected,mask)
    # Bend's saved actual raw is covered by recomputed compute geometry.
    base=_j(output/'s7-flat-input.json')['positionPlane'];tex=_raw(output/'s7-albedo.rgba')
    for name in ('first','second'):
      d=_j(output/f's7-bend-{name}-input.json');a,b,c,e=d['parameters'];pts=[[x+a*y*y+b,y,z+c*x+e] for x,y,z in base];expected,mask=color_raster(pts,tex,[1,1,1,1],[96,96]);_case(_raw(output/f's7-bend-{name}.rgba'),expected,mask)
  except (OSError,KeyError,TypeError,ValueError,struct.error,json.JSONDecodeError) as e:return [str(e)]
  return []
