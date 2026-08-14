#!/usr/bin/env python3
"""最终交付验收：校验当前数据快照、正式证据、隐私与可运行入口。"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'outputs/evaluation'
OUT=EVAL/'final_competition_acceptance_report.json'
CONSISTENCY=EVAL/'current_data_consistency_report.json'
REQUIRED=['README.md','LICENSE','NOTICE','THIRD_PARTY_NOTICES.md','.env.example','requirements.txt','docker-compose.yml','MANIFEST.json','configs/metric_registry.yaml','configs/operator_contracts/contracts.yaml','configs/nl2sql_eval/blind.json','data/raw/data_manifest.json','data/sqlite/chroniccare.db','data/graph/graph.json','data/graph/graph_summary.json','app/streamlit_app.py']
REPORTS=['current_data_consistency_report.json','synthetic_data_quality_report.json','open_sql_security_eval_report.json','kg_semantic_quality_report.json','kgqa_quality_report.json','npu_operator_benchmark_report.json','npu_environment_report.json','test_execution_summary.json','conversation_context_eval_report.json','agent_routing_coverage_report.json','nl2sql_blind_eval_report.json']
TEXT_SUFFIXES={'.py','.md','.json','.xml','.log','.yaml','.yml','.toml','.txt','.ini','.cfg','.sh'}


def sha256(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): d.update(block)
 return d.hexdigest()


def http(url):
 try:
  with urllib.request.urlopen(url,timeout=8) as r:
   return {'ok':r.status<400,'status_code':r.status,'json':json.load(r) if r.headers.get_content_type()=='application/json' else None}
 except Exception as e: return {'ok':False,'error':type(e).__name__}


def build_consistency():
 db=ROOT/'data/sqlite/chroniccare.db'; graph_path=ROOT/'data/graph/graph.json'
 graph=json.loads((ROOT/'data/graph/graph_summary.json').read_text())
 metrics=json.loads((ROOT/'configs/current_metrics.json').read_text())
 manifest=json.loads((ROOT/'data/raw/data_manifest.json').read_text())
 with sqlite3.connect(f'file:{db}?mode=ro',uri=True) as c:
  counts={
   'patient_count':c.execute('SELECT COUNT(*) FROM patient_profile').fetchone()[0],
   'visit_count':c.execute('SELECT COUNT(*) FROM visit_record').fetchone()[0],
   'lab_result_count':c.execute('SELECT COUNT(*) FROM lab_result').fetchone()[0],
   'medication_record_count':c.execute('SELECT COUNT(*) FROM medication_record').fetchone()[0],
   'risk_event_count':c.execute('SELECT COUNT(*) FROM risk_event').fetchone()[0]}
 expected={'patient_count':2000,'visit_count':8231,'lab_result_count':131323,'medication_record_count':18248,'risk_event_count':22840,'node_count':197404,'edge_count':396928,'entity_type_total_count':14,'relation_type_total_count':15}
 actual={**counts,'node_count':graph.get('node_count'),'edge_count':graph.get('edge_count'),'entity_type_total_count':len(graph.get('entity_type_count') or {}),'relation_type_total_count':len(graph.get('relation_type_count') or {})}
 versions={'data_manifest':manifest.get('data_version'),'graph_summary':graph.get('data_version'),'current_metrics':metrics.get('data_version')}
 checks={'expected_counts_match':actual==expected,'data_version_consistent':set(versions.values())=={'synthetic_chroniccare'},'metrics_match_graph':metrics.get('node_count')==actual['node_count'] and metrics.get('edge_count')==actual['edge_count'],'graph_provenance_complete':graph.get('edge_provenance_complete_rate')==1.0,'graph_has_no_exact_duplicate_edges':graph.get('exact_duplicate_edge_count')==0}
 report={'schema_version':'1.0.0','status':'success' if all(checks.values()) else 'failed','generated_at':datetime.now().astimezone().isoformat(),'data_version':'synthetic_chroniccare','snapshots':{'sqlite':{'path':'data/sqlite/chroniccare.db','sha256':sha256(db),**counts},'graph':{'path':'data/graph/graph.json','sha256':sha256(graph_path),'node_count':actual['node_count'],'edge_count':actual['edge_count'],'entity_type_total_count':actual['entity_type_total_count'],'relation_type_total_count':actual['relation_type_total_count'],'risk_event_occurrence_nodes':graph.get('risk_event_occurrence_nodes')}},'expected':expected,'actual':actual,'data_version_sources':versions,'validations':checks}
 CONSISTENCY.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 return report


def scan_privacy():
 secret=re.compile(r'((?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC )?PRIVATE KEY-----)')
 private=re.compile(r'/mnt/' r'nvme0n1/home/[^/\s]+|/home/[A-Za-z0-9._-]+/|10\.236\.(?:12\.5|2\.5)')
 hits={'secrets':[],'private_paths_or_ips':[],'forbidden_files':[]}
 for forbidden_dir in ('.pytest_cache', '.ruff_cache', 'outputs/runtime_generated', 'outputs/open_sql/traces'):
  if (ROOT/forbidden_dir).exists(): hits['forbidden_files'].append(forbidden_dir + '/')
 for cache_dir in ROOT.rglob('__pycache__'):
  if cache_dir.is_dir(): hits['forbidden_files'].append(cache_dir.relative_to(ROOT).as_posix() + '/')
 for p in ROOT.rglob('*'):
  if not p.is_file() or any(x in p.parts for x in ('.git','__pycache__','.pytest_cache','.ruff_cache','runtime_generated')): continue
  rel=p.relative_to(ROOT).as_posix()
  if p.name=='.env' or p.suffix=='.pyc': hits['forbidden_files'].append(rel); continue
  if p.suffix.lower() not in TEXT_SUFFIXES and p.name!='.env.example': continue
  text=p.read_text(encoding='utf-8',errors='ignore')
  if secret.search(text): hits['secrets'].append(rel)
  if private.search(text): hits['private_paths_or_ips'].append(rel)
 return hits


def validation_input_mtime():
 paths=[]; source_suffixes={'.py','.sh','.yaml','.yml','.json','.toml','.ini','.cfg','.md'}
 source_directories=(
  'analysis','app','configs','deploy','integrations','kg','mcp_adapter',
  'orchestration','runtime_common','scripts','tests','tool_server','visualization'
 )
 for directory in source_directories:
  root=ROOT/directory
  if not root.exists(): continue
  paths.extend(
   path for path in root.rglob('*')
   if path.is_file() and (path.suffix.lower() in source_suffixes or path.name.startswith('Dockerfile'))
  )
 for pattern in ('Dockerfile*','docker-compose*.yml','docker-compose*.yaml','*.sh'):
  paths.extend(path for path in ROOT.glob(pattern) if path.is_file())
 paths.extend(ROOT/p for p in ('pyproject.toml','requirements.txt') if (ROOT/p).is_file())
 if not paths: raise RuntimeError('未找到可用于证据新鲜度比较的源码或配置文件')
 latest=max(paths,key=lambda p:p.stat().st_mtime)
 return latest,latest.stat().st_mtime


def markdown_references():
 references=[]; missing=[]
 documents=list(ROOT.glob('*.md'))+list((ROOT/'docs').rglob('*.md'))
 pattern=re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')
 for document in documents:
  for raw in pattern.findall(document.read_text(encoding='utf-8',errors='ignore')):
   target=raw.strip().strip('<>')
   if not target or target.startswith(('#','http://','https://','mailto:')) or '<' in target or '>' in target: continue
   target=unquote(target.split('#',1)[0].strip())
   if not target: continue
   resolved=(document.parent/target).resolve()
   item={'document':document.relative_to(ROOT).as_posix(),'reference':target}
   references.append(item)
   try: resolved.relative_to(ROOT.resolve())
   except ValueError: missing.append({**item,'reason':'outside_project'})
   else:
    if not resolved.exists(): missing.append({**item,'reason':'missing'})
 return references,missing


def main():
 checks=[]
 def add(name,ok,details=None,required=True): checks.append({'name':name,'ok':bool(ok),'required':required,'details':details})
 current=build_consistency()
 missing=[x for x in REQUIRED if not (ROOT/x).exists()]; add('file_and_config_completeness',not missing,{'missing':missing})
 deps=[x.strip() for x in (ROOT/'requirements.txt').read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')]; add('dependency_lock',bool(deps) and all('==' in x for x in deps),{'dependency_count':len(deps)})
 operators=[p.name for p in sorted((ROOT/'integrations/datamate/operators').glob('chronic_*')) if (p/'process.py').exists()]; add('operator_source_count',len(operators)>=13,{'count':len(operators),'operators':operators})
 privacy=scan_privacy(); add('public_tree_privacy',not any(privacy.values()),privacy)
 add('current_data_consistency',current['status']=='success',{'data_version':current['data_version'],'snapshots':current['snapshots'],'validations':current['validations']})
 services={u:http(u) for u in ['http://127.0.0.1:18088/health','http://127.0.0.1:18188/tools','http://127.0.0.1:18501']}; any_service=any(x['ok'] for x in services.values())
 add('service_health',all(x['ok'] for x in services.values()) if any_service else True,{'mode':'online' if any_service else 'offline-package-check','services':{u:{k:v for k,v in x.items() if k!='json'} for u,x in services.items()}},required=any_service)
 if any_service:
  tools=(services['http://127.0.0.1:18188/tools'].get('json') or {}).get('tools',[]); dag={'chroniccare_datamate_dag_plan','chroniccare_datamate_dag_run','chroniccare_datamate_dag_resume','chroniccare_datamate_dag_status'}; names={x.get('name') for x in tools}
  add('mcp_tool_count_and_schema',len(tools)>=38 and dag<=names and all(x.get('name') and x.get('inputSchema') is not None for x in tools),{'count':len(tools),'dynamic_dag_tools_present':sorted(dag&names)})
 states={}
 for name in REPORTS:
  try: states[name]=json.loads((EVAL/name).read_text()).get('status','unknown')
  except FileNotFoundError: states[name]='missing'
  except Exception: states[name]='invalid_json'
 add('formal_report_statuses',all(x=='success' for x in states.values()),states)
 junit_path=EVAL/'pytest_report.xml'; coverage_path=EVAL/'coverage.json'; ruff_path=EVAL/'ruff_report.json'
 test_details={'junit_present':junit_path.is_file(),'coverage_present':coverage_path.is_file(),'ruff_present':ruff_path.is_file()}
 test_ok=all(test_details.values())
 if test_ok:
  try:
   xml_root=ET.parse(junit_path).getroot(); suite=xml_root if xml_root.tag=='testsuite' else xml_root.find('testsuite')
   if suite is None: raise ValueError('JUnit testsuite is missing')
   cov=json.loads(coverage_path.read_text()); ruff=json.loads(ruff_path.read_text())
   latest_input,latest_mtime=validation_input_mtime(); test_names={case.attrib.get('name','') for case in xml_root.iter('testcase')}
   evidence_mtimes={name:path.stat().st_mtime for name,path in {'pytest_report':junit_path,'coverage':coverage_path,'ruff_report':ruff_path}.items()}
   evidence_fresh=all(value>=latest_mtime for value in evidence_mtimes.values())
   required_test='test_npu_markdown_builders_render_repeated_measurement_columns'
   test_details.update({'tests':int(suite.attrib.get('tests',0)),'failures':int(suite.attrib.get('failures',0)),'errors':int(suite.attrib.get('errors',0)),'skipped':int(suite.attrib.get('skipped',0)),'combined_coverage':(cov.get('totals') or {}).get('percent_covered'),'statement_coverage':(cov.get('totals') or {}).get('percent_statements_covered'),'branch_coverage':(cov.get('totals') or {}).get('percent_branches_covered'),'ruff_violations':len(ruff) if isinstance(ruff,list) else None,'latest_validation_input':latest_input.relative_to(ROOT).as_posix(),'evidence_fresh_against_source':evidence_fresh,'required_latest_test_present':required_test in test_names})
   test_ok=test_details['tests']>=437 and not any(test_details[x] for x in ('failures','errors','skipped')) and test_details['combined_coverage']>=62 and test_details['statement_coverage']>=65 and test_details['branch_coverage']>=53 and test_details['ruff_violations']==0 and evidence_fresh and test_details['required_latest_test_present']
  except Exception as exc:
   test_details['error']=f'{type(exc).__name__}: {exc}'; test_ok=False
 add('machine_readable_test_evidence',test_ok,test_details)
 blind=json.loads((ROOT/'configs/nl2sql_eval/blind.json').read_text()); result=json.loads((EVAL/'nl2sql_blind_eval_report.json').read_text())
 add('nl2sql_hard_gate',len(blind.get('cases',[]))==240 and 50<=(result.get('llm') or {}).get('actual_calls',0)<=100 and result.get('execution_accuracy',0)>=.85,{'dataset':len(blind.get('cases',[])),'llm_calls':(result.get('llm') or {}).get('actual_calls'),'accuracy':result.get('execution_accuracy')})
 graph=json.loads((ROOT/'data/graph/graph_summary.json').read_text()); provenance=str(graph.get('provenance_version') or '')
 add('graph_provenance_gate',graph.get('status')=='success' and graph.get('node_count')==197404 and graph.get('edge_count')==396928 and graph.get('exact_duplicate_edge_count')==0 and graph.get('edge_provenance_complete_rate')==1.0 and provenance.startswith('2.'),{'data_version':graph.get('data_version'),'node_count':graph.get('node_count'),'edge_count':graph.get('edge_count'),'provenance_version':provenance})
 npu=json.loads((EVAL/'npu_operator_benchmark_report.json').read_text()); summaries=[x.get('summary') or {} for x in npu.get('operator_results',[]) if str(x.get('operator','')).endswith('_npu')]; rows=npu.get('npu_comparison_rows') or []
 repeat_ok=npu.get('benchmark_repeat_count')==5 and npu.get('benchmark_sample_count')==2048 and len(rows)==2 and all(len(row.get('cpu_bge_sample_seconds_runs') or [])==5 and len(row.get('npu_bge_sample_seconds_runs') or [])==5 and len(row.get('npu_bge_full_seconds_runs') or [])==5 for row in rows)
 npu_ok=npu.get('status')=='success' and npu.get('fallback_used') is False and repeat_ok and len(summaries)==2 and all(x.get('mainline_consumed_npu') is True for x in summaries) and all((((x.get('model_inference') or {}).get('quality_gate') or {}).get('passed') is True) for x in summaries)
 add('npu_execution_and_equivalence_gate',npu_ok,{'fallback_used':npu.get('fallback_used'),'benchmark_repeat_count':npu.get('benchmark_repeat_count'),'benchmark_sample_count':npu.get('benchmark_sample_count'),'operators':[{'operator':row.get('operator'),'sample_speedup_mean':row.get('sample_speedup'),'cpu_runs':row.get('cpu_bge_sample_seconds_runs'),'npu_runs':row.get('npu_bge_sample_seconds_runs'),'npu_full_runs':row.get('npu_bge_full_seconds_runs')} for row in rows]})
 regression=json.loads((EVAL/'agent_routing_coverage_report.json').read_text()); add('routing_regression_gate',regression.get('status')=='success' and regression.get('total',regression.get('question_count',0))>=80 and regression.get('pass_rate',1.0)>=1.0 and not regression.get('failure_category_summary'),{'question_count':regression.get('total',regression.get('question_count')),'pass_rate':regression.get('pass_rate',1.0),'failures':regression.get('failure_category_summary')})
 refs,missing_refs=markdown_references(); add('formal_document_references',not missing_refs,{'documents':len({x['document'] for x in refs}),'checked':len(refs),'missing':missing_refs})
 archive=ROOT/'release/ChronicCare-Agent-Final.tar.gz'; archive_manifest=ROOT/'release/offline_release_manifest.json'; artifact_present=archive.exists() or archive_manifest.exists()
 if artifact_present:
  details={'archive_present':archive.is_file(),'manifest_present':archive_manifest.is_file(),'archive_sha256_match':False,'manifest_status':None,'missing_members':[],'error':None}
  ok=archive.is_file() and archive_manifest.is_file()
  try:
   package=json.loads(archive_manifest.read_text()); details['manifest_status']=package.get('status')
   details['archive_sha256_match']=sha256(archive)==((package.get('archive') or {}).get('sha256'))
   required_members={'ChronicCare-Agent/README.md','ChronicCare-Agent/MANIFEST.json','ChronicCare-Agent/RELEASE_NOTES.md','ChronicCare-Agent/pyproject.toml','ChronicCare-Agent/docs/技术报告.md','ChronicCare-Agent/docs/技术报告.pdf','ChronicCare-Agent/data/sqlite/chroniccare.db','ChronicCare-Agent/data/graph/graph.json','ChronicCare-Agent/tests/conftest.py'}
   with tarfile.open(archive,'r:gz') as tar: members={x.name.rstrip('/') for x in tar.getmembers()}
   details['missing_members']=sorted(required_members-members)
   ok=ok and package.get('status')=='success' and details['archive_sha256_match'] and not details['missing_members'] and not any((package.get('privacy_scan') or {}).values())
  except Exception as exc:
   details['error']=f'{type(exc).__name__}: {exc}'; ok=False
  add('internal_offline_archive_integrity',ok,details,required=True)
 else:
  add('optional_internal_offline_archive',True,{'mode':'not-built','note':'外层提交ZIP或源码目录可直接验收；内部离线TAR仅在显式构建后执行完整性校验。'},required=False)
 failures=[x['name'] for x in checks if x['required'] and not x['ok']]
 acceptance_date=json.loads((ROOT/'MANIFEST.json').read_text()).get('release_date')
 final={'schema_version':'2.0.0','status':'success' if not failures else 'failed','acceptance_date':acceptance_date,'generated_at':datetime.now().astimezone().isoformat(),'data_version':'synthetic_chroniccare','checks':checks,'required_failures':failures}
 OUT.write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'status':final['status'],'checks':len(checks),'failures':failures},ensure_ascii=False)); return 0 if not failures else 1

if __name__=='__main__': raise SystemExit(main())
