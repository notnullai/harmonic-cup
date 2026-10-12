# Harmonic Knowledge Graph schema

The knowledge graph is what replaces historical failure data. The engine can say *what changed* — which sensors are surprised, in which regime, how fast the health index is moving. It cannot say *what that means* for this machine, because it has never seen this machine fail. The graph supplies that: the standing engineering knowledge that a bearing outer-race defect shows at BPFO, that a lubrication failure causes it, and that §4.2 of the manual says how to replace the bearing.

Frozen at the end of Phase 1 (2026-09-24). The loader and queries are built in Phase 4.

## Node types

| Label | Key properties | Notes |
|---|---|---|
| `Equipment` | `id`, `model`, `serial`, `nominal_rpm`, `iso_class` | One per physical machine. `id` matches the brain's machine profile. |
| `Component` | `id`, `kind`, `position` | `kind` ∈ {bearing, impeller, seal, motor, coupling, shaft}. `position` ∈ {drive_end, non_drive_end, …}. |
| `Sensor` | `id`, `kind`, `unit`, `sample_rate_hz` | `kind` is a `harmonic_feature_kind` name, which is the join back to the engine. |
| `FailureMode` | `id`, `iso_code`, `description`, `severity` | `severity` ∈ 1…5. |
| `Symptom` | `id`, `description`, `feature_kind`, `direction` | `direction` ∈ {up, down, oscillating}. |
| `Cause` | `id`, `description`, `category` | `category` ∈ {lubrication, installation, process, wear, contamination, electrical}. `Component.kind` includes `rotor` and `stator` (motors). |
| `Action` | `id`, `description`, `tool`, `duration_min`, `skill_level` | `skill_level` ∈ {operator, technician, specialist}. |
| `ManualSection` | `id`, `doc`, `section`, `page`, `title`, `text_chunk` | `id` is the citation token the GenAI layer must reference. |
| `Standard` | `id`, `name`, `clause` | e.g. ISO 10816-1. |

## Relations

```
(Equipment)     -[:HAS_COMPONENT]->     (Component)
(Component)     -[:MONITORED_BY]->      (Sensor)
(Sensor)        -[:DETECTS]->           (Symptom)
(Symptom)       -[:INDICATES {confidence}]-> (FailureMode)
(Cause)         -[:CAUSES]->            (FailureMode)
(FailureMode)   -[:MITIGATED_BY {priority}]-> (Action)
(Action)        -[:DOCUMENTED_IN]->     (ManualSection)
(FailureMode)   -[:GOVERNED_BY]->       (Standard)
(FailureMode)   -[:AFFECTS]->           (Component)
```

`confidence` on `INDICATES` is 0…1 and is what ranks candidate diagnoses. It is a property of the *relation*, not the failure mode: high BPFO indicates outer-race wear with high confidence, while a raised 1X indicates misalignment only weakly, because imbalance does the same thing.

## Constraints

```cypher
CREATE CONSTRAINT equipment_id     IF NOT EXISTS FOR (n:Equipment)     REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT component_id     IF NOT EXISTS FOR (n:Component)     REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT sensor_id        IF NOT EXISTS FOR (n:Sensor)        REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT failuremode_id   IF NOT EXISTS FOR (n:FailureMode)   REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT symptom_id       IF NOT EXISTS FOR (n:Symptom)       REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT cause_id         IF NOT EXISTS FOR (n:Cause)         REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT action_id        IF NOT EXISTS FOR (n:Action)        REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT manualsection_id IF NOT EXISTS FOR (n:ManualSection) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT standard_id      IF NOT EXISTS FOR (n:Standard)      REQUIRE n.id IS UNIQUE;

CREATE INDEX sensor_kind    IF NOT EXISTS FOR (n:Sensor)  ON (n.kind);
CREATE INDEX symptom_kind   IF NOT EXISTS FOR (n:Symptom) ON (n.feature_kind);
```

The uniqueness constraints are what make the loader idempotent: it `MERGE`s on `id`, so re-running it after re-extracting triples updates rather than duplicates.

## Joining the engine to the graph

An event carries the engine's view; the graph turns it into a diagnosis.

```
engine event                         graph entry point
------------------------------       -----------------------------------
top fault class                  ->  FailureMode.id            (direct, trained classes)
top surprised features           ->  Sensor.kind -> Symptom    (per-feature evidence)
regime                           ->  filters which Symptoms apply
health state, HHI, RUL           ->  carried through as context, not matched
```

Both paths are used. The trained class gives a strong prior; the surprised features give the *evidence* for it, which is what makes the answer explainable rather than an assertion. When they disagree, the query returns both and the confidence ordering decides — a disagreement is genuine information, not an error to hide.

Events can be multi-label (AGENTS.md constraint 5): two class neurons can fire at once, and the query must return candidates for both rather than forcing a single winner.

## Example: pump P-101 seed

```cypher
MERGE (p:Equipment {id: 'P-101'})
  SET p.model = 'centrifugal_pump', p.nominal_rpm = 2950, p.iso_class = 'III';

MERGE (b:Component {id: 'P-101.bearing.DE'})
  SET b.kind = 'bearing', b.position = 'drive_end';
MERGE (p)-[:HAS_COMPONENT]->(b);

MERGE (s:Sensor {id: 'P-101.vib.DE.radial'})
  SET s.kind = 'HARMONIC_FEAT_VIB_BPFO', s.unit = 'mm/s', s.sample_rate_hz = 25600;
MERGE (b)-[:MONITORED_BY]->(s);

MERGE (sym:Symptom {id: 'sym.bpfo.high'})
  SET sym.description = 'Elevated amplitude at the ball pass frequency, outer race',
      sym.feature_kind = 'HARMONIC_FEAT_VIB_BPFO', sym.direction = 'up';
MERGE (s)-[:DETECTS]->(sym);

MERGE (fm:FailureMode {id: 'fm.outer_race_wear'})
  SET fm.description = 'Outer race spalling', fm.severity = 4, fm.iso_code = 'ISO 15243 - 5.2';
MERGE (sym)-[:INDICATES {confidence: 0.90}]->(fm);
MERGE (fm)-[:AFFECTS]->(b);

MERGE (c:Cause {id: 'cause.lubrication_loss'})
  SET c.description = 'Loss or degradation of lubricant', c.category = 'lubrication';
MERGE (c)-[:CAUSES]->(fm);

MERGE (a:Action {id: 'act.replace_bearing'})
  SET a.description = 'Replace the drive-end bearing',
      a.tool = 'bearing puller, induction heater',
      a.duration_min = 180, a.skill_level = 'technician';
MERGE (fm)-[:MITIGATED_BY {priority: 1}]->(a);

MERGE (m:ManualSection {id: 'M-4.2'})
  SET m.doc = 'P-101 O&M Manual', m.section = '4.2', m.page = 41,
      m.title = 'Bearing replacement procedure';
MERGE (a)-[:DOCUMENTED_IN]->(m);

MERGE (std:Standard {id: 'iso-10816-1'})
  SET std.name = 'ISO 10816-1', std.clause = 'Class III, rigid foundation';
MERGE (fm)-[:GOVERNED_BY]->(std);
```

## Query shape

`kg/query.py::query_for_event(event) -> KGSubgraph` returns, for each candidate failure mode: its confidence, the symptoms that support it, its causes, its ranked actions, and the manual section IDs those actions cite. Those IDs become the *only* citations the GenAI layer may use — `rag/guard.py` rejects a response referencing anything outside them.

## Storage

Neo4j Community in Docker Compose, behind a `KGStore` interface. A NetworkX + JSON implementation of the same interface is the fallback, so the demo survives Docker being unavailable on the day. The interface exists from the start rather than being retrofitted, because a fallback added under time pressure is a fallback that does not work.
