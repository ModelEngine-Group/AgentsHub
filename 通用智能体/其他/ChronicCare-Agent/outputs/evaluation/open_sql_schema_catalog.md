# Open SQL Schema Catalog

## doctor_advice

- rows: `8231`

| field | label | type | select | where | group_by | aggregate |
| --- | --- | --- | --- | --- | --- | --- |
| advice_content | advice_content | TEXT | True | True | False | False |
| advice_id | advice_id | TEXT | True | True | False | False |
| advice_type | advice_type | TEXT | True | True | True | False |
| created_at | created_at | TEXT | True | True | True | False |
| patient_id | 患者编号 | TEXT | True | True | False | False |
| priority | 优先级 | TEXT | True | True | True | False |
| visit_id | 随访/就诊编号 | TEXT | True | True | False | False |

## followup_plan

- rows: `8231`

| field | label | type | select | where | group_by | aggregate |
| --- | --- | --- | --- | --- | --- | --- |
| created_at | created_at | TEXT | True | True | True | False |
| followup_date | 随访日期 | TEXT | True | True | True | False |
| patient_id | 患者编号 | TEXT | True | True | False | False |
| plan_id | plan_id | TEXT | True | True | False | False |
| plan_type | plan_type | TEXT | True | True | False | False |
| priority | 优先级 | TEXT | True | True | True | False |
| status | 状态 | TEXT | True | True | True | False |
| visit_id | 随访/就诊编号 | TEXT | True | True | False | False |

## lab_result

- rows: `131323`

| field | label | type | select | where | group_by | aggregate |
| --- | --- | --- | --- | --- | --- | --- |
| abnormal_flag | 异常标记 | TEXT | True | True | True | False |
| item_name | 指标名称 | TEXT | True | True | True | False |
| item_value | 指标值 | TEXT | True | True | False | True |
| lab_id | lab_id | TEXT | True | True | False | False |
| patient_id | 患者编号 | TEXT | True | True | False | False |
| record_time | 记录时间 | TEXT | True | True | True | False |
| reference_high | reference_high | TEXT | True | True | False | True |
| reference_low | reference_low | TEXT | True | True | False | True |
| reference_range | reference_range | TEXT | True | True | False | False |
| test_date | 检验日期 | TEXT | True | True | True | False |
| unit | unit | TEXT | True | True | False | False |
| value | 指标值 | TEXT | True | True | False | True |
| visit_id | 随访/就诊编号 | TEXT | True | True | False | False |

## lifestyle_record

- rows: `8231`

| field | label | type | select | where | group_by | aggregate |
| --- | --- | --- | --- | --- | --- | --- |
| alcohol_status | alcohol_status | TEXT | True | True | True | False |
| created_at | created_at | TEXT | True | True | True | False |
| exercise_minutes_per_week | 每周运动分钟 | TEXT | True | True | False | True |
| patient_id | 患者编号 | TEXT | True | True | False | False |
| record_id | record_id | TEXT | True | True | False | False |
| salt_intake_level | 盐摄入水平 | TEXT | True | True | True | False |
| sleep_hours | 睡眠小时 | TEXT | True | True | False | True |
| smoking_status | 吸烟状态 | TEXT | True | True | True | False |
| visit_id | 随访/就诊编号 | TEXT | True | True | False | False |

## medication_record

- rows: `18248`

| field | label | type | select | where | group_by | aggregate |
| --- | --- | --- | --- | --- | --- | --- |
| adherence | adherence | TEXT | True | True | False | False |
| dose | dose | TEXT | True | True | False | False |
| drug_category | 药物类别 | TEXT | True | True | True | False |
| drug_name | 药物名称 | TEXT | True | True | True | False |
| end_date | end_date | TEXT | True | True | True | False |
| frequency | frequency | TEXT | True | True | False | False |
| med_id | med_id | TEXT | True | True | False | False |
| patient_id | 患者编号 | TEXT | True | True | False | False |
| start_date | start_date | TEXT | True | True | True | False |
| visit_id | 随访/就诊编号 | TEXT | True | True | False | False |

## patient_profile

- rows: `2000`

| field | label | type | select | where | group_by | aggregate |
| --- | --- | --- | --- | --- | --- | --- |
| age | age | TEXT | True | True | False | True |
| bmi | BMI | TEXT | True | True | False | True |
| disease_tags | 疾病标签 | TEXT | True | True | True | False |
| drinking | drinking | TEXT | True | True | False | False |
| first_visit_date | first_visit_date | TEXT | True | True | True | False |
| gender | gender | TEXT | True | True | True | False |
| name | name | TEXT | True | True | False | False |
| patient_id | 患者编号 | TEXT | True | True | False | False |
| smoking | smoking | TEXT | True | True | False | False |

## patient_risk_score

- rows: `8231`

| field | label | type | select | where | group_by | aggregate |
| --- | --- | --- | --- | --- | --- | --- |
| created_at | created_at | TEXT | True | True | True | False |
| patient_id | 患者编号 | TEXT | True | True | False | False |
| risk_factors | 风险因素 | TEXT | True | True | False | False |
| risk_level | 风险等级 | TEXT | True | True | True | False |
| risk_score | 风险评分 | TEXT | True | True | False | True |
| score_id | score_id | TEXT | True | True | False | False |
| visit_id | 随访/就诊编号 | TEXT | True | True | False | False |

## risk_event

- rows: `22840`

| field | label | type | select | where | group_by | aggregate |
| --- | --- | --- | --- | --- | --- | --- |
| created_at | created_at | TEXT | True | True | True | False |
| event_level | event_level | TEXT | True | True | False | False |
| event_reason | event_reason | TEXT | True | True | False | False |
| event_type | 风险事件类型 | TEXT | True | True | True | False |
| patient_id | 患者编号 | TEXT | True | True | False | False |
| risk_event_id | risk_event_id | TEXT | True | True | False | False |
| visit_id | 随访/就诊编号 | TEXT | True | True | False | False |

## visit_record

- rows: `8231`

| field | label | type | select | where | group_by | aggregate |
| --- | --- | --- | --- | --- | --- | --- |
| chief_complaint | chief_complaint | TEXT | True | True | False | False |
| doctor_advice | doctor_advice | TEXT | True | True | False | False |
| followup_plan | followup_plan | TEXT | True | True | False | False |
| patient_id | 患者编号 | TEXT | True | True | False | False |
| visit_date | visit_date | TEXT | True | True | True | False |
| visit_id | 随访/就诊编号 | TEXT | True | True | False | False |

## Allowed Joins

- `patient_profile.patient_id = visit_record.patient_id`
- `patient_profile.patient_id = lab_result.patient_id`
- `patient_profile.patient_id = medication_record.patient_id`
- `patient_profile.patient_id = followup_plan.patient_id`
- `patient_profile.patient_id = patient_risk_score.patient_id`
- `patient_risk_score.patient_id = lab_result.patient_id`
- `patient_profile.patient_id = risk_event.patient_id`
- `patient_profile.patient_id = lifestyle_record.patient_id`
- `patient_profile.patient_id = doctor_advice.patient_id`
- `visit_record.visit_id = lab_result.visit_id`
- `visit_record.visit_id = medication_record.visit_id`
- `visit_record.visit_id = followup_plan.visit_id`
- `visit_record.visit_id = risk_event.visit_id`
- `visit_record.visit_id = lifestyle_record.visit_id`
- `visit_record.visit_id = doctor_advice.visit_id`
- `visit_record.visit_id = patient_risk_score.visit_id`