"""Pure raster, Color and bounded-PBR math; no saved expected image is input."""
from __future__ import annotations
import math, struct

def q(x): return min(255,max(0,math.floor(min(1,max(0,x))*255+.5)))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def norm(v):
    n=math.sqrt(dot(v,v))
    if not math.isfinite(n) or not n: raise ValueError("zero/nonfinite vector")
    return [x/n for x in v]
def edge(a,b,p): return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
def raster(points, rgba, extent):
    area=edge(points[0],points[1],points[2])
    if not area: raise ValueError("degenerate triangle")
    out=bytearray(); mask=bytearray(); w,h=extent
    for y in range(h):
      for x in range(w):
        p=[2*(x+.5)/w-1,1-2*(y+.5)/h]; es=[edge(points[1],points[2],p),edge(points[2],points[0],p),edge(points[0],points[1],p)]
        ds=[abs(edge(points[i],points[(i+1)%3],p))/math.hypot(points[(i+1)%3][0]-points[i][0],points[(i+1)%3][1]-points[i][1]) for i in range(3)]
        mask.append(int(all(d>.035 for d in ds))); out.extend(rgba if all(e/area>=0 for e in es) else b"\0\0\0\0")
    return bytes(out),bytes(mask)
def color(points, texture, tint):
    u=[points[1][i]-points[0][i] for i in range(3)];v=[points[2][i]-points[0][i] for i in range(3)]; n=norm([u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]])
    # The deferred attachment encodes then decodes the normal before lighting.
    dz=q(n[2]*.5+.5)/255*2-1; depth=sum(p[2] for p in points)/3; f=.2+.6*abs(dz)+.2*(1-depth)
    return bytes([q(texture[i]/255*tint[i]*f) for i in range(3)]+[q(texture[3]/255*tint[3])])
def color_raster(points, texture, tint, extent):
    """Color raster with barycentric depth, needed by the tilted fixture."""
    u=[points[1][i]-points[0][i] for i in range(3)];v=[points[2][i]-points[0][i] for i in range(3)];n=norm([u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]); dz=q(n[2]*.5+.5)/255*2-1;area=edge(points[0],points[1],points[2]);out=bytearray();mask=bytearray();w,h=extent
    for y in range(h):
      for x in range(w):
        p=[2*(x+.5)/w-1,1-2*(y+.5)/h];es=[edge(points[1],points[2],p),edge(points[2],points[0],p),edge(points[0],points[1],p)];ds=[abs(edge(points[i],points[(i+1)%3],p))/math.hypot(points[(i+1)%3][0]-points[i][0],points[(i+1)%3][1]-points[i][1]) for i in range(3)];mask.append(int(all(d>.035 for d in ds)))
        if all(e/area>=0 for e in es):
          depth=sum(es[i]/area*points[i][2] for i in range(3));f=.2+.6*abs(dz)+.2*(1-depth);out.extend([q(texture[i]/255*tint[i]*f) for i in range(3)]+[q(texture[3]/255*tint[3])])
        else:out.extend(b'\0\0\0\0')
    return bytes(out),bytes(mask)
def f32(x): return struct.unpack('<f',struct.pack('<f',float(x)))[0]
def pbr(data, deferred=True):
    tex,tint=data['textureRgba'],data['tint']; normal=data['normals'][0]; rough=f32(data['roughness']); base=[tex[i]/255*tint[i] for i in range(3)];metal=tex[3]/255
    # Normals use the inverse-transpose of the persisted column-major transform.
    m=data['transformColumns'];a,b,c=m[0][0],m[1][0],m[2][0];d,e,f=m[0][1],m[1][1],m[2][1];g,h,i=m[0][2],m[1][2],m[2][2];det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
    if not det: raise ValueError('singular PBR transform')
    inv=[[(e*i-f*h)/det,(c*h-b*i)/det,(b*f-c*e)/det],[(f*g-d*i)/det,(a*i-c*g)/det,(c*d-a*f)/det],[(d*h-e*g)/det,(a*h-b*g)/det,(a*e-b*d)/det]]
    n=norm([sum(inv[col][row]*f32(normal[col]) for col in range(3)) for row in range(3)]); n=norm([q(x*.5+.5)/255*2-1 for x in n]) if deferred else n
    if deferred: base=[q(x)/255 for x in base];metal=q(metal)/255;rough=q(rough)/255
    rough=min(1,max(.04,rough)); light=norm([.6,0,.8]);view=[0,0,1];half=norm([light[i]+view[i] for i in range(3)]);nl=max(0,min(1,dot(n,light)));nv=max(0,min(1,dot(n,view)))
    if not nl or not nv:return b'\0\0\0\xff'
    nh=max(0,min(1,dot(n,half)));vh=max(0,min(1,dot(view,half)));a2=rough**4;d=nh*nh*(a2-1)+1;dist=a2/(math.pi*d*d);vis=.5/(nl*math.sqrt(nv*nv*(1-a2)+a2)+nv*math.sqrt(nl*nl*(1-a2)+a2));f0=[.04*(1-metal)+x*metal for x in base];fr=[x+(1-x)*(1-vh)**5 for x in f0]
    return bytes([q(((1-fr[i])*(1-metal)*base[i]/math.pi+dist*vis*fr[i])*nl*3) for i in range(3)]+[255])
def bmp(path, expected, mask):
    b=path.read_bytes()
    if b[:2]!=b'BM' or len(b)<54: raise ValueError('capture is not BMP')
    off=struct.unpack_from('<I',b,10)[0];w,h=struct.unpack_from('<ii',b,18)
    if off!=54 or w<=0 or h>=0 or len(b)!=54+w*(-h)*4 or len(expected)!=w*(-h)*4:raise ValueError('invalid top-down BGRA BMP')
    return b[off:],w,-h
def check(actual, expected, mask, bmp_mode=False):
    if len(actual)!=len(expected) or len(mask)*4!=len(actual):raise ValueError('invalid raw/mask shape')
    inside=outside=0
    for i,stable in enumerate(mask):
      if not stable:continue
      want=expected[4*i:4*i+4]; got=actual[4*i:4*i+4]; fg=want!=b'\0\0\0\0';inside+=fg;outside+=not fg
      rgb=(got[2],got[1],got[0]) if bmp_mode else tuple(got[:3])
      if fg and max(abs(rgb[j]-want[j]) for j in range(3))>1:raise ValueError(f'foreground differs at {i}')
      if not fg and rgb!=(0,0,0):raise ValueError(f'background differs at {i}')
      if not bmp_mode and got[3]!=want[3]:raise ValueError(f'alpha differs at {i}')
    if not inside or not outside:raise ValueError('stable mask lacks foreground/background')
    return {'inside':inside,'outside':outside}
