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

    def test_cost_and_break_even(self):
        out=self.call(EXAMPLE); self.assertEqual(out['paygo_only_cost_cents'],15000); self.assertEqual(out['plans'][0]['total_cost_cents'],12000); self.assertEqual(out['plans'][0]['break_even_visits_ignoring_validity'],3)
    def test_expiry_is_exclusive_and_extra_paid(self):
        data=copy.deepcopy(EXAMPLE); data['plans'][0]['validity_days']=2
        out=self.call(data)['plans'][0]; self.assertEqual(out['covered_count'],1); self.assertEqual(out['total_cost_cents'],22000)
    def test_freeze_merge_extension_and_frozen_visit(self):
        data=copy.deepcopy(EXAMPLE); data['plans'][0].update(freeze_extends_validity=True,freezes=[{'start':'2026-11-19','end_exclusive':'2026-11-22'},{'start':'2026-11-20','end_exclusive':'2026-11-23'}])
        out=self.call(data)['plans'][0]; self.assertEqual(out['frozen_days'],4); self.assertEqual(out['covered_count'],1); self.assertEqual(out['expiry_exclusive'],'2026-12-22')
    def test_empty_visits_and_zero_paygo(self):
        data=copy.deepcopy(EXAMPLE); data['visits']=[]; data['pay_per_visit_cents']=0
        out=self.call(data)['plans'][0]; self.assertIsNone(out['cost_per_attended_visit_cents']); self.assertIsNone(out['break_even_visits_ignoring_validity'])
    def test_invalid_freeze_and_duplicate_dates(self):
        data=copy.deepcopy(EXAMPLE); data['visits'].append('2026-11-18'); self.assertError(data)
        data=copy.deepcopy(EXAMPLE); data['plans'][0]['freezes']=[{'start':'2026-11-01','end_exclusive':'2026-11-19'}]; self.assertError(data)

if __name__ == "__main__": unittest.main()
