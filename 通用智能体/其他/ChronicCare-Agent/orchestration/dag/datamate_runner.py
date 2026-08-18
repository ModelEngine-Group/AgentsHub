from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]

class OperatorExecutionError(RuntimeError):
 def __init__(self,message:str,*,category:str="OperatorError",retryable:bool=False): super().__init__(message); self.category=category; self.retryable=retryable

class RealDataMateRunner:
 "Execute actual installed DataMate operators in an isolated per-run container workspace."
 def __init__(self,*,input_path:str,use_npu:bool,container:str="datamate-runtime",force_npu_unavailable:bool=False):
  self.input_path=Path(input_path).resolve(); self.use_npu=use_npu; self.container=container; self.force_npu_unavailable=force_npu_unavailable; self._prepared=set()
 def _run(self,args:list[str],*,input_text:str|None=None,timeout:int=600):
  try: result=subprocess.run(args,cwd=ROOT,input=input_text,text=True,encoding="utf-8",errors="replace",capture_output=True,timeout=timeout,check=False)
  except subprocess.TimeoutExpired as exc: raise OperatorExecutionError(f"operator timeout after {timeout}s",category="TimeoutError",retryable=True) from exc
  if result.returncode: raise OperatorExecutionError((result.stderr or result.stdout)[-4000:],category="ContainerExecutionError",retryable=False)
  return result
 def _work(self,run_id): return f"/tmp/chroniccare_real_dag/{run_id}"
 def _prepare(self,run_id,profile_input_hash):
  if run_id in self._prepared: return
  work=self._work(run_id)
  probe=subprocess.run(["docker","exec",self.container,"cat",f"{work}/input_profile_hash"],text=True,capture_output=True,check=False)
  if not probe.returncode and probe.stdout.strip()==profile_input_hash:
   self._prepared.add(run_id); return
  if not probe.returncode or subprocess.run(["docker","exec",self.container,"test","-e",work],capture_output=True,check=False).returncode==0:
   self._run(["docker","exec",self.container,"rm","-rf",work],timeout=60)
  self._run(["docker","exec",self.container,"mkdir","-p",f"{work}/input/raw",f"{work}/input/configs",f"{work}/output"],timeout=30)
  if not self.input_path.exists(): raise OperatorExecutionError(f"input path missing: {self.input_path}",category="InputMissing",retryable=False)
  self._run(["docker","cp",str(self.input_path)+"/.",f"{self.container}:{work}/input/raw"],timeout=120)
  self._run(["docker","cp",str(ROOT/"configs")+"/.",f"{self.container}:{work}/input/configs"],timeout=120)
  initial={"filePath":f"{work}/input/raw","export_path":f"{work}/output","status":"prepared"}
  code=f"from pathlib import Path; import json; Path({(work+'/state.json')!r}).write_text(json.dumps({initial!r}),encoding='utf-8'); Path({(work+'/input_profile_hash')!r}).write_text({profile_input_hash!r},encoding='utf-8')"
  self._run(["docker","exec",self.container,"python3","-c",code],timeout=30); self._prepared.add(run_id)
 def __call__(self,name:str,ctx:dict[str,Any])->dict[str,Any]:
  run_id=ctx["run_id"]; self._prepare(run_id,ctx["profile_input_hash"]); work=self._work(run_id); timeout=int(ctx.get("timeout_seconds",600))
  code=f'''\nimport hashlib,json,time\nfrom pathlib import Path\nname={name!r}; work=Path({work!r})\nimports={{\n"chronic_file_ingest":("datamate.ops.mapper.chronic_file_ingest.process","chronic_file_ingest"),\n"chronic_table_clean":("datamate.ops.mapper.chronic_table_clean.process","chronic_table_clean"),\n"chronic_field_normalize":("datamate.ops.mapper.chronic_field_normalize.process","chronic_field_normalize"),\n"chronic_text_split":("datamate.ops.mapper.chronic_text_split.process","chronic_text_split"),\n"chronic_entity_extract":("datamate.ops.mapper.chronic_entity_extract.process","chronic_entity_extract"),\n"chronic_entity_extract_model_npu":("datamate.ops.mapper.chronic_entity_extract_model_npu.process","chronic_entity_extract_model_npu"),\n"chronic_relation_extract":("datamate.ops.mapper.chronic_relation_extract.process","chronic_relation_extract"),\n"chronic_relation_extract_model_npu":("datamate.ops.mapper.chronic_relation_extract_model_npu.process","chronic_relation_extract_model_npu"),\n"chronic_triple_validate":("datamate.ops.mapper.chronic_triple_validate.process","chronic_triple_validate"),\n"chronic_kg_build":("datamate.ops.mapper.chronic_kg_build.process","chronic_kg_build"),\n"chronic_sqlite_loader":("datamate.ops.mapper.chronic_sqlite_loader.process","chronic_sqlite_loader"),\n"chronic_nl2sql_analyze":("datamate.ops.mapper.chronic_nl2sql_analyze.process","chronic_nl2sql_analyze"),\n"chronic_report_pack":("datamate.ops.mapper.chronic_report_pack.process","chronic_report_pack")}}\nmodule_name,class_name=imports[name]; module=__import__(module_name,fromlist=[class_name]); operator=getattr(module,class_name)()\nsample=json.loads((work/"state.json").read_text()); params={{}}\nif name.endswith("_model_npu"):\n params={{"use_npu":True,"fallback":True,"embedding_model_path":{("/models/__forced_unavailable__" if self.force_npu_unavailable else "/models/MedCleanStd/bge-small-zh-v1.5")!r},"npu_max_records":0,"cpu_benchmark_records":2048,"model_batch_size":64,"model_max_length":64}}\nif name=="chronic_kg_build": params={{"current_metrics_path":str(work/"input/configs/current_metrics.json")}}\nif name=="chronic_nl2sql_analyze": params={{"analysis_questions_path":str(work/"input/configs/nl2sql_questions.json")}}\nstarted=time.perf_counter(); sample=operator.execute(sample,params); elapsed=time.perf_counter()-started\nif sample.get("status") not in ("success","completed"): raise RuntimeError(f"operator returned {{sample.get('status')}}: {{sample.get('error')}}")\n(work/"state.json").write_text(json.dumps(sample,ensure_ascii=False,default=str),encoding="utf-8")\ndef path_hash(value):\n p=Path(value); digest=hashlib.sha256(); count=0; size=0\n files=sorted(x for x in (p.rglob("*") if p.is_dir() else [p]) if x.is_file()) if p.exists() else []\n for f in files:\n  rel=f.relative_to(p).as_posix() if p.is_dir() else f.name; h=hashlib.sha256(f.read_bytes()).hexdigest(); digest.update(rel.encode()); digest.update(b"\\0"); digest.update(h.encode()); digest.update(b"\\n"); count+=1; size+=f.stat().st_size\n return {{"path":str(p),"sha256":digest.hexdigest(),"file_count":count,"size_bytes":size,"exists":p.exists()}}\nartifacts={{key:path_hash(value) for key,value in (sample.get("artifact_paths") or {{}}).items()}}\ncombined=hashlib.sha256(json.dumps(artifacts,sort_keys=True).encode()).hexdigest(); summary=sample.get("summary") or {{}}\nresult={{"status":"degraded" if summary.get("fallback_used") else "success","execution_mode":"real_datamate_operator","operator":name,"operator_elapsed_seconds":round(elapsed,6),"artifact_hash":combined,"artifacts":artifacts,"summary":summary,"fallback_used":bool(summary.get("fallback_used")),"mainline_consumed_npu":summary.get("mainline_consumed_npu")}}\nprint("__DAG_RESULT__"+json.dumps(result,ensure_ascii=False,default=str))\n'''
  cann_root=os.environ.get("CHRONICCARE_CANN_ROOT","/usr/local/Ascend/ascend-toolkit/latest").strip()
  shell=f'source "{cann_root}/set_env.sh" >/dev/null 2>&1 || true; export OPENBLAS_NUM_THREADS=64 OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 NUMEXPR_NUM_THREADS=64; python3 -'
  result=self._run(["docker","exec","-i",self.container,"bash","-lc",shell],input_text=code,timeout=timeout)
  marker=[line for line in result.stdout.splitlines() if line.startswith("__DAG_RESULT__")]
  if not marker: raise OperatorExecutionError(f"operator produced no result marker: {result.stdout[-2000:]}",category="InvalidOperatorOutput",retryable=False)
  return json.loads(marker[-1].removeprefix("__DAG_RESULT__"))
 def finalize(self,run_id:str)->dict[str,Any]:
  work=self._work(run_id); run_dir=ROOT/"outputs/dag_runs"/run_id; temporary=run_dir/".artifacts.tmp"; final=run_dir/"artifacts"
  if final.exists(): return {"artifact_root":str(final.relative_to(ROOT)),"already_materialized":True}
  if temporary.exists(): shutil.rmtree(temporary)
  temporary.mkdir(parents=True)
  process=subprocess.Popen(["docker","exec",self.container,"tar","-C",f"{work}/output","-cf","-","."],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  assert process.stdout is not None
  with tarfile.open(fileobj=process.stdout,mode="r|") as archive:
   for member in archive:
    target=(temporary/member.name).resolve()
    if temporary.resolve() not in target.parents and target!=temporary.resolve():
     process.kill(); raise OperatorExecutionError(f"unsafe artifact member: {member.name}",category="InvalidArtifact",retryable=False)
    if member.issym() or member.islnk():
     process.kill(); raise OperatorExecutionError(f"linked artifact refused: {member.name}",category="InvalidArtifact",retryable=False)
    archive.extract(member,temporary)
  stderr=process.stderr.read().decode(errors="replace") if process.stderr else ""; code=process.wait()
  if code: raise OperatorExecutionError(stderr[-4000:],category="ContainerExecutionError",retryable=False)
  os.replace(temporary,final); digest=hashlib.sha256(); count=0; size=0
  for path in sorted(x for x in final.rglob("*") if x.is_file()):
   rel=path.relative_to(final).as_posix(); value=hashlib.sha256(path.read_bytes()).hexdigest(); digest.update(rel.encode()); digest.update(b"\0"); digest.update(value.encode()); digest.update(b"\n"); count+=1; size+=path.stat().st_size
  return {"artifact_root":str(final.relative_to(ROOT)),"artifact_sha256":digest.hexdigest(),"file_count":count,"size_bytes":size,"source":f"{self.container}:{work}/output"}
