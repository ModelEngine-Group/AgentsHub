import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mcps'))
spec=importlib.util.spec_from_file_location('tested_fitness_server',ROOT/'mcps/server.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
EXAMPLE=json.loads((ROOT/'examples/request.json').read_text(encoding='utf-8'))

class FunctionalTests(unittest.TestCase):
    def call(self,data):
        output=json.loads(module.run(json.dumps(dict(data,save_artifacts=False))))
        self.assertEqual(output['status'],'ok',output)
        return output
    def assertError(self,data):
        output=json.loads(module.run(json.dumps(dict(data,save_artifacts=False))))
        self.assertEqual(output['status'],'error',output)
    def test_payload_limits_and_nonfinite(self):
        self.assertEqual(json.loads(module.run('[]'))['status'],'error')
        self.assertEqual(json.loads(module.run('{"value":NaN}'))['status'],'error')
    def test_bundled_example_has_explicit_provenance(self):
        output=json.loads(module.run('{"use_example":true,"save_artifacts":false}'))
        self.assertEqual(output['status'],'ok',output)
        self.assertEqual(output['input_scope'],'bundled_example')
        self.assertEqual(json.loads(module.run('{"use_example":true,"records":[]}'))['status'],'error')
    def test_json_file_cannot_escape_inputs_directory(self):
        self.assertEqual(json.loads(module.run('{"input_json_filename":"../../outside.json"}'))['status'],'error')

    def test_complete_cycle(self):
        out=self.call(EXAMPLE); self.assertEqual(out['rep_count'],1); self.assertEqual(out['mean_cycle_seconds'],1); self.assertEqual(out['source']['mode'],'keypoint_fixture')
    def test_incomplete_cycle_not_counted(self):
        data=copy.deepcopy(EXAMPLE); data['frames']=data['frames'][:2]; self.assertEqual(self.call(data)['rep_count'],0)
    def test_missing_visibility_and_gap_reset(self):
        data=copy.deepcopy(EXAMPLE); data['frames'][1]['points'][0]['visibility']=.1; self.assertEqual(self.call(data)['rep_count'],0)
        data=copy.deepcopy(EXAMPLE); data['frames'][1]['time_seconds']=2; data['frames'][2]['time_seconds']=3; self.assertEqual(self.call(data)['rep_count'],0)
    def test_timestamp_and_threshold_validation(self):
        data=copy.deepcopy(EXAMPLE); data['frames'][1]['time_seconds']=0; self.assertError(data)
        data=copy.deepcopy(EXAMPLE); data['low_angle']=170; data['high_angle']=160; self.assertError(data)
    def test_hold_rejects_one_frame_phase(self):
        data=copy.deepcopy(EXAMPLE); data['min_phase_seconds']=.2; self.assertEqual(self.call(data)['rep_count'],0)
    def test_video_path_cannot_escape_workspace(self):
        self.assertError({'mode':'video','video_filename':'../../outside.mp4'})

if __name__ == "__main__": unittest.main()
