from runtime import *

from datetime import timedelta
import hashlib
from zoneinfo import ZoneInfo,ZoneInfoNotFoundError

def analyze(data):
    people=unique(rows(data.get('participants'),'participants',30)); venues=unique(rows(data.get('venues'),'venues',30))
    if not people or not venues: raise ValueError('participants and venues are required')
    zone_name=text(data.get('display_timezone','UTC'),'display timezone')
    try: display_zone=ZoneInfo(zone_name)
    except ZoneInfoNotFoundError: raise ValueError('unknown display timezone')
    minutes=integer(data.get('duration_minutes',60),'duration',5,240); budget=integer(data.get('budget_cents',10000),'budget',0,10000000)
    windows=[]
    for identity,person in people.items():
        preferences=rows(person.get('activities',[]),'activities',20)
        windows.append((identity,person,merge_spans([span(item) for item in rows(person.get('availability'),'availability',100)]),preferences))
    candidates=[]; exclusions=[]
    for venue_id,venue in venues.items():
        capacity=integer(venue.get('capacity'),'capacity',1,1000); price=integer(venue.get('price_cents'),'price',0,10000000); activity=text(venue.get('activity'),'activity',100)
        if capacity<len(people) or price>budget or any(prefs and activity not in prefs for _,_,_,prefs in windows):
            exclusions.append({'venue_id':venue_id,'reason':'capacity, budget or activity mismatch'}); continue
        available=merge_spans([span(item) for item in rows(venue.get('open'),'open',100)])
        for _,_,intervals,_ in windows:
            available=merge_spans([(max(a,c),min(b,d)) for a,b in available for c,d in intervals if max(a,c)<min(b,d)])
        busy=merge_spans([span(item) for item in rows(venue.get('booked',[]),'booked',100)])
        for bs,be in busy:
            available=[segment for start,end in available for segment in ([(start,min(end,bs))] if start<bs else [])+([(max(start,be),end)] if be<end else []) if segment[0]<segment[1]]
        for start,end in available:
            if (end-start).total_seconds()>=minutes*60:
                finish=start+timedelta(minutes=minutes)
                display=start.astimezone(display_zone).isoformat()+' – '+finish.astimezone(display_zone).isoformat()+' ['+zone_name+']'
                candidates.append({'display_time':display,'venue_id':venue_id,'activity':activity,'start':iso(start),'end':iso(finish),'available_until':iso(end),'price_cents':price,'participants':list(people)})
    candidates.sort(key=lambda item:(item['start'],item['price_cents'],item['venue_id']))
    ics=['BEGIN:VCALENDAR','VERSION:2.0','PRODID:-//Fitness Group Schedule//EN']
    for item in candidates:
        uid=hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()[:24]
        ics.extend(['BEGIN:VEVENT','UID:'+uid+'@fitness.local','DTSTAMP:'+instant(item['start']).strftime('%Y%m%dT%H%M%SZ'),'DTSTART:'+instant(item['start']).strftime('%Y%m%dT%H%M%SZ'),'DTEND:'+instant(item['end']).strftime('%Y%m%dT%H%M%SZ'),'SUMMARY:Candidate exercise session','STATUS:TENTATIVE','END:VEVENT'])
    ics.append('END:VCALENDAR')
    return {'candidates':candidates,'candidate_count':len(candidates),'display_timezone':zone_name,'excluded_venues':exclusions,'note':'答复时间直接引用 display_time；原始 start/end 为 UTC。所有时段都是待确认候选；工具不预订场馆、不向参与者发送消息。','_artifacts':[('candidate-sessions.ics','\r\n'.join(ics)+'\r\n','text/calendar')]}

runtime = Runtime('fitness-group-schedule', '多人约练与场馆排程助手', 'fitness_group_schedule', '交集核对多人空闲时间、场馆开放时间、容量和费用，输出候选时段与日历文件。', package_root=Path(__file__).resolve().parents[1])

def run(payload: str) -> str:
    return runtime.process(payload, analyze)

if __name__ == "__main__":
    runtime.main(run)
