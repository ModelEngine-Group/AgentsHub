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

    def test_common_slot_after_booking(self):
        out=self.call(EXAMPLE); self.assertEqual(out['candidate_count'],1); self.assertEqual(out['candidates'][0]['start'],'2026-11-20T11:30:00Z')
    def test_capacity_and_budget(self):
        data=copy.deepcopy(EXAMPLE); data['venues'][0]['capacity']=1; self.assertEqual(self.call(data)['candidate_count'],0)
        data=copy.deepcopy(EXAMPLE); data['budget_cents']=3999; self.assertEqual(self.call(data)['candidate_count'],0)
    def test_duplicate_availability_is_merged(self):
        data=copy.deepcopy(EXAMPLE); data['participants'][0]['availability']*=2; self.assertEqual(self.call(data)['candidate_count'],1)
    def test_full_booking_and_timezone(self):
        data=copy.deepcopy(EXAMPLE); data['venues'][0]['booked']=[{'start':'2026-11-20T10:00:00Z','end':'2026-11-20T14:00:00Z'}]; self.assertEqual(self.call(data)['candidate_count'],0)
    def test_invalid_interval_and_duplicate_ids(self):
        data=copy.deepcopy(EXAMPLE); data['participants'][0]['availability'][0]['end']='2026-11-20T17:00:00+08:00'; self.assertError(data)
        data=copy.deepcopy(EXAMPLE); data['participants'].append(data['participants'][0]); self.assertError(data)
    def test_display_time_is_computed_with_offset(self):
        out=self.call(EXAMPLE); self.assertEqual(out['display_timezone'],'Asia/Shanghai')
        self.assertIn('2026-11-20T19:30:00+08:00',out['candidates'][0]['display_time'])
        data=copy.deepcopy(EXAMPLE); data['display_timezone']='UTC'
        self.assertIn('2026-11-20T11:30:00+00:00',self.call(data)['candidates'][0]['display_time'])
    def test_invalid_display_timezone(self):
        data=copy.deepcopy(EXAMPLE); data['display_timezone']='Wrong/Place'; self.assertError(data)

if __name__ == "__main__": unittest.main()
