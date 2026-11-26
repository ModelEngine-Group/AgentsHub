from runtime import *
from pathlib import Path

from datetime import timedelta
from decimal import Decimal,ROUND_HALF_UP

def analyze(data):
    plans=unique(rows(data.get('plans'),'plans',20)); visits=sorted(day(value) for value in rows(data.get('visits'),'visits',1000))
    paygo=integer(data.get('pay_per_visit_cents'),'pay per visit',0,1000000)
    if len(visits)!=len(set(visits)): raise ValueError('visit dates must be unique; combine same-day entry assumptions first')
    result=[]
    for identity,plan in plans.items():
        source=text(plan.get('terms_source'),'terms source',500); start=day(plan.get('start')); valid=integer(plan.get('validity_days'),'validity days',1,3660)
        base_expiry=start+timedelta(days=valid); cost=integer(plan.get('price_cents'),'price',0,100000000)+integer(plan.get('setup_fee_cents',0),'setup fee',0,100000000)
        capacity=plan.get('visit_limit')
        if capacity is not None: capacity=integer(capacity,'visit limit',1,100000)
        extend=plan.get('freeze_extends_validity',False)
        if not isinstance(extend,bool): raise ValueError('freeze extension flag must be boolean')
        frozen=[]
        for freeze in rows(plan.get('freezes',[]),'freezes',100):
            a,b=day(freeze.get('start')),day(freeze.get('end_exclusive'))
            if not start<=a<b<=base_expiry: raise ValueError('freeze interval must be within base validity')
            frozen.append((a,b))
        frozen=merge_spans(frozen); frozen_days=sum((b-a).days for a,b in frozen); expiry=base_expiry+timedelta(days=frozen_days if extend else 0)
        covered=[]; outside=[]
        for visit_date in visits:
            eligible=start<=visit_date<expiry and not any(a<=visit_date<b for a,b in frozen)
            if eligible and (capacity is None or len(covered)<capacity): covered.append(visit_date.isoformat())
            else: outside.append(visit_date.isoformat())
        total=cost+len(outside)*paygo; per_visit=float((Decimal(total)/len(visits)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)) if visits else None
        theoretical=None if paygo==0 else (cost+paygo-1)//paygo
        result.append({'plan_id':identity,'terms_source':source,'start':start.isoformat(),'expiry_exclusive':expiry.isoformat(),'frozen_days':frozen_days,'covered_visit_dates':covered,'paygo_visit_dates':outside,'covered_count':len(covered),'unconsumed_visits':max(0,capacity-len(covered)) if capacity is not None else None,'total_cost_cents':total,'cost_per_attended_visit_cents':per_visit,'break_even_visits_ignoring_validity':theoretical,'break_even_within_capacity':theoretical is not None and (capacity is None or theoretical<=capacity)})
    result.sort(key=lambda item:(item['total_cost_cents'],item['plan_id']))
    return {'plans':result,'visit_count':len(visits),'paygo_only_cost_cents':len(visits)*paygo,'note':'结果基于指定到店日期与输入条款，停卡区间左闭右开。交叉次数忽略有效期且不保证达到；退款规则未输入时不推算。','_artifacts':[('membership-costs.csv',csv_text(result,['plan_id','expiry_exclusive','covered_count','unconsumed_visits','total_cost_cents','cost_per_attended_visit_cents','break_even_visits_ignoring_validity']),'text/csv')]}

runtime = Runtime('fitness-membership-cost', '健身卡成本与有效期助手', 'fitness_membership_cost', '按明确费用、次数、有效期和停卡条款模拟到店支出，核对过期或未使用次数。', package_root=Path(__file__).resolve().parents[1])

def run(payload: str) -> str:
    return runtime.process(payload, analyze)

if __name__ == "__main__":
    runtime.main(run)
