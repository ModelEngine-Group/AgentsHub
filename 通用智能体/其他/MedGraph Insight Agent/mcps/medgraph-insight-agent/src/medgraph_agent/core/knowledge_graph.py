from __future__ import annotations

from collections import OrderedDict

from medgraph_agent.core.models import (
    ENTITY_LABELS,
    RELATION_LABELS,
    Entity,
    EntityType,
    GraphSnapshot,
    Relation,
    RelationType,
    stable_id,
    utc_now,
)


class KnowledgeGraphBuilder:
    def __init__(self) -> None:
        self._entities: OrderedDict[str, Entity] = OrderedDict()
        self._relations: OrderedDict[str, Relation] = OrderedDict()
        self._record_ids: set[str] = set()

    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        source_record_id: str,
        confidence: float = 0.9,
    ) -> Entity:
        name = name.strip()
        key = f"{entity_type}:{name}"
        entity_id = stable_id("ent", entity_type, name)
        existing = self._entities.get(key)
        if existing:
            sources = sorted(set(existing.source_record_ids + [source_record_id]))
            merged = Entity(
                id=existing.id,
                name=existing.name,
                type=existing.type,
                label=existing.label,
                confidence=max(existing.confidence, confidence),
                source_record_ids=sources,
            )
            self._entities[key] = merged
            return merged
        entity = Entity(
            id=entity_id,
            name=name,
            type=entity_type,
            label=ENTITY_LABELS[entity_type],
            confidence=round(confidence, 4),
            source_record_ids=[source_record_id],
        )
        self._entities[key] = entity
        return entity

    def add_relation(
        self,
        subject: Entity,
        predicate: RelationType,
        obj: Entity,
        evidence: str,
        source_record_id: str,
        confidence: float = 0.82,
    ) -> Relation:
        self._record_ids.add(source_record_id)
        relation_id = stable_id("rel", subject.id, predicate, obj.id, source_record_id, evidence[:80])
        if relation_id in self._relations:
            return self._relations[relation_id]
        relation = Relation(
            id=relation_id,
            subject_id=subject.id,
            subject_name=subject.name,
            predicate=predicate,
            predicate_label=RELATION_LABELS[predicate],
            object_id=obj.id,
            object_name=obj.name,
            evidence=evidence.strip(),
            source_record_id=source_record_id,
            confidence=round(confidence, 4),
        )
        self._relations[relation_id] = relation
        return relation

    def snapshot(self, source_record_count: int | None = None) -> GraphSnapshot:
        count = source_record_count if source_record_count is not None else len(self._record_ids)
        return GraphSnapshot(
            entities=list(self._entities.values()),
            relations=list(self._relations.values()),
            generated_at=utc_now(),
            source_record_count=count,
        )


def merge_snapshots(snapshots: list[GraphSnapshot]) -> GraphSnapshot:
    builder = KnowledgeGraphBuilder()
    for snapshot in snapshots:
        entity_by_id = {entity.id: entity for entity in snapshot.entities}
        for entity in snapshot.entities:
            for record_id in entity.source_record_ids or ["unknown"]:
                builder.add_entity(entity.name, entity.type, record_id, entity.confidence)
        for relation in snapshot.relations:
            subject = entity_by_id.get(relation.subject_id)
            obj = entity_by_id.get(relation.object_id)
            if subject and obj:
                merged_subject = builder.add_entity(subject.name, subject.type, relation.source_record_id, subject.confidence)
                merged_object = builder.add_entity(obj.name, obj.type, relation.source_record_id, obj.confidence)
                builder.add_relation(
                    merged_subject,
                    relation.predicate,
                    merged_object,
                    relation.evidence,
                    relation.source_record_id,
                    relation.confidence,
                )
    return builder.snapshot(sum(snapshot.source_record_count for snapshot in snapshots))
