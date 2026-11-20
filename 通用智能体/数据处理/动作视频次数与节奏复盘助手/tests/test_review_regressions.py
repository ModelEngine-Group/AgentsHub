"""Regression checks for reviewed input and downloadable-artifact boundaries."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mcps'))
spec=importlib.util.spec_from_file_location('reviewed_'+ROOT.name,ROOT/'mcps/server.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
EXAMPLE=json.loads((ROOT/'examples/request.json').read_text(encoding='utf-8'))


class ReviewedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.base=Path(self.temporary.name)
        self.environment=patch.dict(os.environ,{'FITNESS_DATA_ROOT':str(self.base),'FITNESS_PUBLIC_BASE':'http://testserver'})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def call(self,data,save=False):
        return json.loads(module.run(json.dumps(dict(data,save_artifacts=save),ensure_ascii=False)))

    def assert_ok(self,data):
        result=self.call(data)
        self.assertEqual(result['status'],'ok',result)
        return result

    def test_payload_byte_limit_and_non_string(self):
        for raw in (None,{},True,' '* (1024*1024+1),json.dumps({'value':'中'*400000},ensure_ascii=False)):
            with self.subTest(kind=type(raw).__name__):
                self.assertEqual(json.loads(module.run(raw))['status'],'error')

    def test_existing_outside_json_cannot_escape_inputs(self):
        outside=self.base/'outside.json';outside.write_text(json.dumps(EXAMPLE),encoding='utf-8')
        (module.runtime.root/'inputs').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'inside'):
            module.runtime.input_file('../../outside.json')
        result=self.call({'input_json_filename':'../../outside.json'})
        self.assertEqual(result['status'],'error',result)

    def test_json_size_limit_precedes_file_read(self):
        inputs=module.runtime.root/'inputs';inputs.mkdir(parents=True)
        (inputs/'large.json').write_bytes(b' '*(1024*1024+1))
        with patch.object(Path,'open',side_effect=AssertionError('oversized JSON must not be read')):
            result=self.call({'input_json_filename':'large.json'})
        self.assertEqual(result['status'],'error',result)
        self.assertIn('1 MiB',result['message'])

    def test_missing_example_returns_structured_error(self):
        with patch.object(module.runtime,'package_root',self.base/'missing-package'):
            result=self.call({'use_example':True})
        self.assertEqual(result['status'],'error',result)

    def test_io_failure_rolls_back_all_new_artifacts(self):
        calls=0;original=Path.write_bytes
        def write(path,data):
            nonlocal calls
            calls+=1
            if calls==2: raise OSError('simulated disk full')
            return original(path,data)
        def analyze(data):
            return {'_artifacts':[('one.txt','first','text/plain'),('two.txt','second','text/plain')]}
        with patch.object(Path,'write_bytes',write):
            result=json.loads(module.runtime.process('{}',analyze))
        self.assertEqual(result['status'],'error',result)
        self.assertEqual(list((module.runtime.root/'artifacts').iterdir()),[])

    def test_generated_artifacts_are_downloadable_and_bad_routes_are_rejected(self):
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.testclient import TestClient
        result=self.call(EXAMPLE,save=True)
        self.assertEqual(result['status'],'ok',result)
        self.assertTrue(result['artifacts'])
        app=Starlette(routes=[Route('/artifacts/{slug}/{token}/{filename}',module.runtime.artifact_response)])
        with TestClient(app) as client:
            for artifact in result['artifacts']:
                path=urlsplit(artifact['url']).path
                response=client.get(path)
                self.assertEqual(response.status_code,200)
                self.assertEqual(len(response.content),artifact['bytes'])
                self.assertEqual(response.headers['X-Content-Type-Options'],'nosniff')
                self.assertEqual(response.headers['Cache-Control'],'no-store')
                _,_,slug,token,filename=path.split('/')
                self.assertEqual(client.get('/artifacts/wrong/'+token+'/'+filename).status_code,404)
                self.assertEqual(client.get('/artifacts/'+slug+'/invalid/'+filename).status_code,404)
                self.assertEqual(client.get('/artifacts/'+slug+'/'+'0'*32+'/'+filename).status_code,404)

    def test_download_cannot_follow_resolved_path_outside_artifact_root(self):
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.testclient import TestClient
        result=self.call(EXAMPLE,save=True)
        self.assertEqual(result['status'],'ok',result)
        path=urlsplit(result['artifacts'][0]['url']).path
        parts=path.split('/');candidate=module.runtime.root/'artifacts'/parts[-2]/parts[-1]
        outside=self.base/'private.txt';outside.write_text('private sentinel',encoding='utf-8')
        original=Path.resolve
        def resolve(file,*args,**kwargs):
            return outside if file==candidate else original(file,*args,**kwargs)
        app=Starlette(routes=[Route('/artifacts/{slug}/{token}/{filename}',module.runtime.artifact_response)])
        with TestClient(app) as client,patch.object(Path,'resolve',resolve):
            response=client.get(path)
        self.assertEqual(response.status_code,404)
        self.assertNotIn('private sentinel',response.text)

    def test_export_tool_input_schemas_are_valid_json(self):
        config=json.loads((ROOT/'agent.json').read_text(encoding='utf-8'))
        for info in config['agent_info'].values():
            for tool in info.get('tools',[]):
                if isinstance(tool.get('inputs'),str):
                    self.assertIsInstance(json.loads(tool['inputs']),dict)


class SpecificReviewTests(unittest.TestCase):
    setUp=ReviewedRuntimeTests.setUp
    tearDown=ReviewedRuntimeTests.tearDown
    call=ReviewedRuntimeTests.call
    assert_ok=ReviewedRuntimeTests.assert_ok

    def test_one_fps_video_counts_without_silent_gap_resets(self):
        frames=copy.deepcopy(EXAMPLE['frames'])
        for index,frame in enumerate(frames):frame['time_seconds']=index
        source={'sample_fps':1,'container_fps':1}
        with patch.object(module,'load_video',return_value=(frames,source)):
            result=self.assert_ok({'mode':'video','min_phase_seconds':0})
        self.assertEqual(result['rep_count'],1);self.assertEqual(result['gap_reset_count'],0)

    def test_incompatible_explicit_video_gap_is_correctable_error(self):
        frames=copy.deepcopy(EXAMPLE['frames'])
        for index,frame in enumerate(frames):frame['time_seconds']=index
        with patch.object(module,'load_video',return_value=(frames,{'sample_fps':1,'container_fps':1})):
            result=self.call({'mode':'video','min_phase_seconds':0,'max_gap_seconds':.6})
        self.assertEqual(result['status'],'error');self.assertIn('sampling interval',result['message'])

    def test_keypoint_gap_resets_have_diagnostic(self):
        data=copy.deepcopy(EXAMPLE)
        for index,frame in enumerate(data['frames']):frame['time_seconds']=index*2
        result=self.assert_ok(data)
        self.assertGreater(result['gap_reset_count'],0);self.assertTrue(result['warnings'])

    def test_decoder_rejects_invalid_sampling_before_loading_dependencies(self):
        spec=importlib.util.spec_from_file_location('pose_review',ROOT/'mcps/pose_extract.py')
        decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
        for rate in (0,-1,True,float('nan'),16):
            with self.assertRaisesRegex(ValueError,'sample_fps'):
                decoder.extract('missing.mp4','missing.task','squat','left',rate)
