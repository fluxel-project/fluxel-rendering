from pathlib import Path
import sys, tempfile, unittest
sys.path.insert(0,str(Path(__file__).parent))
from s7_oracles import review_device
from s7_oracles.core import check, color_raster, raster

class OracleTests(unittest.TestCase):
 def test_rejects_tampered_actual_and_expected_independent_input(self):
  # The retained expected buffer is not an oracle input: a stable mismatching raw fails.
  with self.assertRaises(ValueError):check(bytes([10,0,0,255,0,0,0,0]),bytes([8,0,0,255,0,0,0,0]),bytes([1,1]))
 def test_rejects_missing_stable_background(self):
  with self.assertRaises(ValueError):check(bytes([8,0,0,255]),bytes([8,0,0,255]),bytes([1]))
 def test_rejects_stable_background_alpha_tamper(self):
  with self.assertRaises(ValueError):check(bytes([0,0,0,7]),bytes([0,0,0,0]),bytes([1]))
 def test_forward_and_deferred_same_triangle_have_distinct_actual_route_outputs(self):
  points=[[-.62,-.56,.2],[.58,-.42,.6],[-.12,.66,.82]]; texture=[144,80,32,255]; tint=[1,1,1,1]
  deferred,_=color_raster(points,texture,tint,[96,96]); forward,_=raster(points,bytes(texture),[96,96])
  self.assertNotEqual(deferred,forward)
 def test_changed_saved_expected_cannot_change_input_derived_image(self):
  points=[[-.62,-.56,.5],[.58,-.42,.5],[-.12,.66,.5]]; image,_=color_raster(points,[144,80,32,255],[1,1,1,1],[96,96])
  index=next(i for i in range(96*96) if image[i*4:i*4+4] != bytes(4)); tampered=bytearray(image); tampered[index*4]=0
  with self.assertRaises(ValueError):check(bytes(tampered),image,bytes([1])*9216)
 def test_device_oracle_fails_closed_when_actual_input_is_absent(self):
  with tempfile.TemporaryDirectory() as temporary:
   self.assertTrue(review_device(Path(temporary)))
if __name__=='__main__':unittest.main()
