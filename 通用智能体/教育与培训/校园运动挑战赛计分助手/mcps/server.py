from runtime import *
from pathlib import Path

from datetime import datetime,time,timedelta,timezone
from decimal import Decimal
from zoneinfo import ZoneInfo,ZoneInfoNotFoundError

def analyze(data):
    try: zone=ZoneInfo(text(data.get('timezone'),'timezone'))
    except ZoneInfoNotFoundError: raise ValueError('unknown IANA timezone')
    first,last=day(data.get('start_date')),day(data.get('end_date'))
    if not first<=last or (last-first).days>90: raise ValueError('challenge dates must be ordered within 91 days')
    people=unique(rows(data.get('participants'),'participants',200)); rules=unique(rows(data.get('groups'),'groups',20))
    for rule in rules.values():
        number(rule.get('points_per_minute'),'points rate',0,100); number(rule.get('daily_cap'),'daily cap',0,10000)
    for person in people.values():
        if person.get('group') not in rules: raise ValueError('participant group missing')
        text(person.get('display_name'),'display name')
    original={}; duplicates=[]; intervals={}; excluded=[]
    for row in rows(data.get('records'),'records',2000):
        obj(row); identity=text(row.get('id'),'record id'); serialized=json.dumps(row,sort_keys=True,allow_nan=False)
        if identity in original:
            if original[identity]!=serialized: raise ValueError('conflicting record id')
            duplicates.append(identity); continue
        original[identity]=serialized; person=row.get('participant_id')
        if person not in people: raise ValueError('unknown participant')
        start,end=span(row)
        if (end-start).total_seconds()>86400: raise ValueError('record longer than 24 hours')
        if row.get('status','completed') not in ('completed','planned','unknown'): raise ValueError('invalid record status')
        if row.get('status','completed')!='completed': excluded.append(identity); continue
        cursor=start
        while cursor<end:
            local_date=cursor.astimezone(zone).date(); midnight=datetime.combine(local_date+timedelta(days=1),time.min,tzinfo=zone).astimezone(timezone.utc); stop=min(end,midnight)
            if stop<=cursor: raise ValueError('timezone midnight did not advance; use explicit UTC records or a different timezone')
            if first<=local_date<=last: intervals.setdefault((person,local_date),[]).append((cursor,stop))
            cursor=stop
    daily=[]; total={identity:Decimal(0) for identity in people}
    for (person,date),spans in sorted(intervals.items()):
        merged=merge_spans(spans); seconds=sum((b-a).total_seconds() for a,b in merged); rule=rules[people[person]['group']]
        raw=Decimal(str(seconds))*Decimal(str(rule['points_per_minute']))/60; credited=min(raw,Decimal(str(rule['daily_cap'])))
        total[person]+=credited
        daily.append({'participant_id':person,'date':date.isoformat(),'unique_minutes':round(seconds/60,6),'uncapped_points':float(raw),'points':float(credited),'daily_cap':rule['daily_cap']})
    board=[{'participant_id':identity,'display_name':person['display_name'],'group':person['group'],'points':float(total[identity])} for identity,person in people.items()]
    board.sort(key=lambda item:(item['group'],-item['points'],item['participant_id']))
    previous={}; ranks={}; counts={}
    for item in board:
        group=item['group']; counts[group]=counts.get(group,0)+1
        if previous.get(group)!=item['points']: ranks[group]=counts[group]
        previous[group]=item['points']; item['rank_within_group']=ranks[group]
    return {'leaderboard':board,'daily_ledger':daily,'duplicate_ids':duplicates,'excluded_record_ids':excluded,'timezone':str(zone),'note':'榜单依据自报记录和公开规则，不验证真实运动。分组内排名；重叠时段按并集计时，展示名用于公开前需本人同意。','_artifacts':[('challenge-leaderboard.csv',csv_text(board,['participant_id','display_name','group','points','rank_within_group']),'text/csv'),('challenge-daily.csv',csv_text(daily,['participant_id','date','unique_minutes','uncapped_points','points','daily_cap']),'text/csv')]}

runtime = Runtime('fitness-challenge-score', '校园运动挑战赛计分助手', 'fitness_challenge_score', '按公开分组规则核算自报活动积分，去重、合并重叠时段并应用单日上限，输出可复核榜单。', package_root=Path(__file__).resolve().parents[1])

def run(payload: str) -> str:
    return runtime.process(payload, analyze)

if __name__ == "__main__":
    runtime.main(run)
