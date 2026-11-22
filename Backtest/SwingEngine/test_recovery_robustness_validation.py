import unittest
import pandas as pd
from recovery_robustness_validation import _segment_ranges,_slice_period,SCENARIOS
from recovery_cycle_resumable import _next_recovery_cycle
from data_loader import TIMEFRAME_MS

class RobustnessTests(unittest.TestCase):
    def test_scenarios_include_baseline_and_two_recovery_targets(self):
        self.assertEqual(SCENARIOS, (("recovery_target_0p50", 0.5), ("recovery_target_1p00", 1.0)))

    def test_two_recovery_cycles_are_assigned_round_robin(self):
        cycles={"A":{"status":"RECOVERY_WAITING"},"B":{"status":"RECOVERY_WAITING"}}
        first,cursor=_next_recovery_cycle(cycles,0)
        self.assertEqual(first,"A")
        cycles[first]["status"]="RECOVERY_OPEN"
        second,cursor=_next_recovery_cycle(cycles,cursor)
        self.assertEqual(second,"B")
        cycles[second]["status"]="RECOVERY_OPEN"
        third,cursor=_next_recovery_cycle(cycles,cursor)
        self.assertIsNone(third)

    def test_remaining_cycle_receives_signals_after_other_completes(self):
        cycles={"A":{"status":"RECOVERY_WAITING"},"B":{"status":"COMPLETED"}}
        selected,_=_next_recovery_cycle(cycles,0)
        self.assertEqual(selected,"A")

    def test_open_initial_cycle_is_not_a_recovery_assignment(self):
        cycles={"A":{"status":"INITIAL_OPEN"},"B":{"status":"RECOVERY_WAITING"}}
        selected,_=_next_recovery_cycle(cycles,0)
        self.assertEqual(selected,"B")

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
        d,s,c=_slice_period({"X":item},{"X":sig},step,3*step)
        self.assertEqual(d["X"].frames["15m"]["ts"].tolist(),[0,step])
        self.assertEqual(s["X"]["signal_ts"].tolist(),[step,2*step])
        self.assertEqual(c["segment_candle_count"],2)
        self.assertEqual(c["segment_signal_count"],2)
if __name__=="__main__": unittest.main()
