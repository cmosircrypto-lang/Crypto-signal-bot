import unittest
import pandas as pd
from recovery_robustness_validation import _segment_ranges,_slice_period
from data_loader import TIMEFRAME_MS

class RobustnessTests(unittest.TestCase):
    def test_ranges_cover_without_gaps_or_overlap(self):
        r=_segment_ranges(100,1000,3)
        self.assertEqual(r[0][0],100); self.assertEqual(r[-1][1],1000)
        self.assertEqual(r[0][1],r[1][0]); self.assertEqual(r[1][1],r[2][0])
        self.assertTrue(all(b>a for a,b in r))
    def test_invalid_ranges(self):
        with self.assertRaises(ValueError): _segment_ranges(10,10,3)
        with self.assertRaises(ValueError): _segment_ranges(10,20,0)
    def test_period_slice_filters_both_streams(self):
        step=TIMEFRAME_MS["15m"]
        f=pd.DataFrame({"ts":[0,step,2*step,3*step],"open":[1,2,3,4]})
        sig=pd.DataFrame({"signal_ts":[step,2*step,3*step],"direction":["BUY","SELL","BUY"]})
        item=type("Item",(),{"frames":{"15m":f}})()
        d,s,c=_slice_period({"X":item},{"X":sig},step,3*step+1)
        self.assertEqual(d["X"].frames["15m"]["ts"].tolist(),[0,step])
        self.assertEqual(s["X"]["signal_ts"].tolist(),[step,2*step])
        self.assertEqual(c["segment_candle_count"],2)
        self.assertEqual(c["segment_signal_count"],2)
if __name__=="__main__": unittest.main()
