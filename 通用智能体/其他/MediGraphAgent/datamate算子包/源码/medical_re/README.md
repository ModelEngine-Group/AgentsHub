# 医学关系抽取算子 (MedicalRE)

基于 LLM 的关系三元组抽取 DataMate 算子。读取医疗文本，抽取受本体约束的关系三元组，输出 JSON。

## 输入 / 输出
- 输入：医疗文本（`sample['text']`）。
- 输出：`{"triples": [{"head","head_type","relation","tail","tail_type","confidence"}]}`。

## 关系类型
has_symptom(有症状)/recommend_drug(推荐药物)/need_examination(需做检查)/complication(并发)/contraindication(禁忌)/treated_in_department(就诊于)/positive_marker(阳性标志物)/negative_marker(阴性标志物)/associated_gene(关联基因)/has_morphology(形态学表现)/located_in(位于)/subtype_of(亚型)/treated_by_procedure(手术或操作治疗)/adverse_reaction(药物不良反应)。

## 参数
同 MedicalNER：apiBase / apiKey / model / temperature / maxChars / maxChunks。

## 打包上传
压缩包名必须为 `medical_re.zip`。建议在 DataMate 流水线里接在 `medical_ner` 之后；有上游 `chunker` 时会逐块抽取并合并去重。
