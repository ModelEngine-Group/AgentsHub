"""Small shared runtime, copied into each independently deployable contribution."""
import argparse
import csv
from datetime import date, datetime, timezone
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
import uuid

def obj(value, name='object'):
    if not isinstance(value, dict): raise ValueError(name+' must be an object')
    return value

def rows(value, name='rows', maximum=2000):
    if not isinstance(value, list) or len(value)>maximum: raise ValueError(name+' must be a bounded array')
    return value

def text(value, name='text', maximum=200):
    if not isinstance(value, str) or not value.strip() or len(value)>maximum: raise ValueError(name+' must be nonempty text')
    return value

def number(value, name='number', minimum=None, maximum=None):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value): raise ValueError(name+' must be finite')
    if minimum is not None and value<minimum or maximum is not None and value>maximum: raise ValueError(name+' outside allowed range')
    return value

def integer(value, name='integer', minimum=0, maximum=1000000):
    number(value,name,minimum,maximum)
    if int(value)!=value: raise ValueError(name+' must be an integer')
    return int(value)

def instant(value):
    try: result=datetime.fromisoformat(text(value,'timestamp',80).replace('Z','+00:00'))
    except (ValueError,TypeError): raise ValueError('timestamp must be ISO 8601 with UTC offset')
    if result.utcoffset() is None: raise ValueError('timestamp needs a UTC offset')
    return result.astimezone(timezone.utc)

def day(value):
    try: return date.fromisoformat(text(value,'date',10))
    except (ValueError,TypeError): raise ValueError('date must be YYYY-MM-DD')

def span(value):
    obj(value,'interval')
    start,end=instant(value['start']),instant(value['end'])
    if end<=start: raise ValueError('interval end must follow start')
    return start,end

def iso(value): return value.isoformat().replace('+00:00','Z')

def unique(values, name='id'):
    seen={}
    for item in values:
        obj(item,'record'); identity=text(item.get(name),name)
        if identity in seen: raise ValueError('duplicate '+name+': '+identity)
        seen[identity]=item
    return seen

def merge_spans(values):
    merged=[]
    for start,end in sorted(values):
        if merged and start<=merged[-1][1]: merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else: merged.append((start,end))
    return merged

def csv_text(values, fields):
    out=StringIO(newline=''); writer=csv.DictWriter(out,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
    writer.writeheader()
    for item in values:
        cleaned={key:("'"+value if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@','\t','\r','\n')) else value) for key,value in item.items()}
        writer.writerow(cleaned)
    return out.getvalue()

class Runtime:
    def __init__(self,slug,title,tool,description,package_root=None):
        self.slug,self.title,self.tool,self.description=slug,title,tool,description
        self.package_root=Path(package_root or Path(__file__).resolve().parents[1]).resolve()
    @property
    def root(self):
        return (Path(os.environ.get('FITNESS_DATA_ROOT',str(self.package_root/'runtime-data')))/self.slug).resolve()
    def input_file(self,value):
        root=(self.root/'inputs').resolve(); candidate=(root/text(value,'input filename',240)).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file(): raise ValueError('input file must exist inside the configured inputs directory')
        if candidate.stat().st_size>30*1024*1024: raise ValueError('input file exceeds 30 MiB')
        return candidate
    def process(self,raw,analyze):
        created=[]
        try:
            if not isinstance(raw,str) or len(raw.encode('utf-8'))>1024*1024: raise ValueError('payload must be JSON text, maximum 1 MiB')
            data=json.loads(raw,parse_constant=lambda value: (_ for _ in ()).throw(ValueError('non-finite JSON number')))
            obj(data,'payload')
            save=data.get('save_artifacts',True)
            if not isinstance(save,bool): raise ValueError('save_artifacts must be boolean')
            scope='provided_json'
            if 'use_example' in data or 'input_json_filename' in data:
                allowed={'use_example','input_json_filename','save_artifacts'}
                if set(data)-allowed or 'use_example' in data and 'input_json_filename' in data: raise ValueError('select one input source without mixing inline data')
                if 'use_example' in data:
                    if data['use_example'] is not True: raise ValueError('use_example must be true')
                    source=self.package_root/'examples/request.json'; scope='bundled_example'
                else: source=self.input_file(data['input_json_filename']); scope='provided_json_file'
                if source.stat().st_size>1024*1024: raise ValueError('JSON input file exceeds 1 MiB')
                with source.open('rb') as stream: encoded=stream.read(1024*1024+1)
                if len(encoded)>1024*1024: raise ValueError('JSON input file exceeds 1 MiB')
                data=obj(json.loads(encoded.decode('utf-8-sig'),parse_constant=lambda value: (_ for _ in ()).throw(ValueError('non-finite JSON number'))),'input JSON')
            result=obj(analyze(data),'result')
            result['input_scope']=scope
            pending=result.pop('_artifacts',[]); result['artifacts']=[]
            if save:
                used=set()
                for filename,content,media in pending:
                    if filename in ('.','..') or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',filename): raise ValueError('invalid artifact filename')
                    if filename in used: raise ValueError('duplicate artifact filename')
                    used.add(filename)
                    encoded=content.encode('utf-8') if isinstance(content,str) else content
                    if not isinstance(encoded,bytes): raise ValueError('artifact content must be text or bytes')
                    token=uuid.uuid4().hex; folder=self.root/'artifacts'/token; folder.mkdir(parents=True,exist_ok=False)
                    destination=folder/filename; created.append((folder,destination))
                    destination.write_bytes(encoded)
                    base=os.environ.get('FITNESS_PUBLIC_BASE','http://127.0.0.1:'+os.environ.get('FITNESS_PORT','8111')).rstrip('/')
                    result['artifacts'].append({'name':filename,'url':f'{base}/artifacts/{self.slug}/{token}/{filename}','media_type':media,'bytes':len(encoded)})
            result['status']='ok'
            return json.dumps(result,ensure_ascii=False,allow_nan=False)
        except (ValueError,KeyError,TypeError,OverflowError,OSError) as error:
            # Only remove paths created by this call; never recurse through user data.
            for folder,destination in reversed(created):
                try:
                    destination.unlink(missing_ok=True)
                    folder.rmdir()
                except OSError: pass
            return json.dumps({'status':'error','message':str(error)[:260]},ensure_ascii=False)
    async def artifact_response(self,request):
        from starlette.responses import FileResponse,JSONResponse
        params=request.path_params
        if params.get('slug')!=self.slug or not re.fullmatch(r'[0-9a-f]{32}',params.get('token','')) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',params.get('filename','')):
            return JSONResponse({'error':'not found'},status_code=404)
        root=(self.root/'artifacts').resolve(); path=(root/params['token']/params['filename']).resolve()
        if not path.is_relative_to(root) or not path.is_file(): return JSONResponse({'error':'not found'},status_code=404)
        return FileResponse(path,filename=params['filename'],headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'})
    def main(self,run):
        parser=argparse.ArgumentParser(description=self.description)
        parser.add_argument('--input',type=Path); parser.add_argument('--host',default='127.0.0.1'); parser.add_argument('--port',type=int,default=8111)
        parser.add_argument('--data-root'); parser.add_argument('--public-base')
        args=parser.parse_args()
        os.environ['FITNESS_PORT']=str(args.port)
        if args.data_root: os.environ['FITNESS_DATA_ROOT']=args.data_root
        if args.public_base: os.environ['FITNESS_PUBLIC_BASE']=args.public_base
        if args.input:
            print(run(args.input.read_text(encoding='utf-8-sig'))); return
        from fastmcp import FastMCP
        mcp=FastMCP(self.title)
        mcp.tool(name=self.tool,description=self.description)(run)
        mcp.custom_route('/artifacts/{slug}/{token}/{filename}',methods=['GET'])(self.artifact_response)
        mcp.run(transport='http',host=args.host,port=args.port)
