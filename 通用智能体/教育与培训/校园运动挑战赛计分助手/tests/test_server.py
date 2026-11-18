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

    def test_overlap_union_and_cap(self):
        out=self.call(EXAMPLE); self.assertEqual(out['daily_ledger'][0]['unique_minutes'],75); self.assertEqual(out['leaderboard'][0]['points'],60)
    def test_duplicate_and_conflict(self):
        data=copy.deepcopy(EXAMPLE); data['records'].append(copy.deepcopy(data['records'][0])); self.assertEqual(self.call(data)['duplicate_ids'],['r1'])
        data['records'][-1]['end']='2026-11-18T20:00:00+08:00'; self.assertError(data)
    def test_midnight_split(self):
        data=copy.deepcopy(EXAMPLE); data['records']=[{'id':'night','participant_id':'a','start':'2026-11-18T23:45:00+08:00','end':'2026-11-19T00:15:00+08:00'}]
        out=self.call(data); self.assertEqual(len(out['daily_ledger']),2); self.assertEqual(out['leaderboard'][0]['points'],30)
    def test_group_tie_and_unfinished(self):
        data=copy.deepcopy(EXAMPLE); data['records']=[]; out=self.call(data)
        self.assertEqual([row['rank_within_group'] for row in out['leaderboard']],[1,1])
        data=copy.deepcopy(EXAMPLE); data['records'][0]['status']='planned'; self.assertEqual(self.call(data)['excluded_record_ids'],['r1'])
    def test_unknown_person_and_timezone(self):
        data=copy.deepcopy(EXAMPLE); data['records'][0]['participant_id']='x'; self.assertError(data)
        data=copy.deepcopy(EXAMPLE); data['timezone']='Wrong/Place'; self.assertError(data)

if __name__ == "__main__": unittest.main()
