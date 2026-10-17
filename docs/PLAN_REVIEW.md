# FaceGroup — Design Review

As of 2026-10-17 · Reviews [PLAN.md](../PLAN.md) v3 → v4 and [ARCHITECTURE.md](ARCHITECTURE.md) v1 → v3

**Verdict:** the plan now meets standard coding and design principles. Thirty findings came out of two review rounds; 29 are fixed in PLAN.md v4 and ARCHITECTURE.md v3, and one (DirectML vs CPU) is deferred to M1. The largest changes:
- a layered core with ports and adapters, replacing workers and the API touching storage directly;
- labels attached to face detections, so changing the model no longer wipes the user's work;
- a single database writer;
- filter-then-rank search;
- OneDrive safeguards;
- a simpler grouping design based on a k-NN vote over confirmed faces.

## How the review was done

1. **Round 1.** An independent reviewer agent read PLAN.md v3 and the prototype and judged them against the principles below. I reviewed them separately at the same time. The two reviews found nearly the same issues.
2. **Diagrams.** The single §3 diagram was replaced by 11 Mermaid diagrams in ARCHITECTURE.md. Each was rendered with mermaid-cli 11.4.2 and inspected as an image.
3. **Round 2.** The same reviewer checked every diagram against the text and the other diagrams, and re-checked each round-1 finding. Its new findings were fixed and the diagrams rendered and inspected again.

The scope is personal use on one Windows 11 laptop with 16 GB RAM and no graphics card.

## Principles checklist

| Principle | Before (v3) | After (v4 / arch v3) |
|---|---|---|
| Separation of concerns, layering | Workers and API read and wrote stores directly | Entry points → services → domain. Ports are owned by the core, adapters sit outside it |
| Dependency inversion | Not defined | Adapters implement core `Protocol`s and are wired only in `bootstrap.py` |
| Single responsibility | Assignment drawn inside the pipeline | IndexingService stores faces; PeopleService decides identity through an explicit hand-off |
| Single source of truth | Embedding stored in a BLOB *and* the vector index, no owner | BLOB is canonical. vec0 is a derived index that can be rebuilt |
| Data integrity | A model change re-created faces and lost every label | Labels sit on detections. Embeddings are per model. Detector changes carry labels over by IoU |
| KISS / YAGNI | 6 plug-in roles, cloud plug-ins, Tauri/React, per-centre prototypes, auto-tuned thresholds | One Protocol per role, no plug-in discovery. Captions deferred. k-NN vote instead of prototypes. One threshold with a slider |
| DRY | Every stage decoded the original again | Decode once into a working-image cache |
| Idempotency, fault tolerance | Undefined | Unique job keys, results written in one transaction, leases, expired leases count as attempts, transient vs permanent errors |
| Concurrency | Unspecified (prototype shared one connection across threads) | WAL, one writer thread with a command queue for indexing *and* user edits, read-only connections for reads |
| Resource limits | Unspecified | One indexing model at a time, explicit ONNX thread counts, RAM table, peak about 3–3.5 GB |
| Security | Unspecified | 127.0.0.1, Host + port allowlist, one-time launch URL → `HttpOnly SameSite=Strict` cookie, media served by id only |
| Robustness | Partly covered | Decompression-bomb limit, EXIF-rotate first, byte-based reads for Unicode paths, HEIC, video probe timeout |
| Testability | Not covered | Fake adapters, property tests on user constraints, 50-image integration set, evaluation scripts |
| Observability, config | Not covered | Rotating log, per-job errors in the UI, SSE progress, one typed TOML config |
| Consistency of the document | Open-source leftovers throughout | Removed; publishing rules confined to PLAN §7 |

## Findings and resolution

Severity: H = high, M = medium, L = low. R1 = round 1 (plan), R2 = round 2 (diagrams).

| # | Sev | Round | Finding | Resolution | Where |
|---|---|---|---|---|---|
| 1 | H | R1 | No service layer; API and workers touch storage | Core with services, ports, domain; adapters outside | Arch §2 |
| 2 | H | R1 | Face model change loses names, merges and rejections | `FACE` (detection) separated from `FACE_EMBEDDING` (per model); IoU carry-over; shadow re-index | Arch §3 rule 6, §4 |
| 3 | H | R1 | Leftovers from the open-source scope | Removed from §2, §4 criteria, §7, §8, M5 | PLAN |
| 4 | H | R1 | RAM and parallelism unspecified | One model at a time, decode pool → one inference thread, RAM table | Arch §3 rule 2, §6 |
| 5 | H | R1 | OneDrive: DB in a synced folder, online-only files downloaded by reads | DB in `%LOCALAPPDATA%` and refused under OneDrive; attribute check without reading; `online_only` state | Arch §1, §3 rule 5 |
| 6 | H | R1 | Rank first, filter after could return nothing | Filter first into a temp table, then vector + FTS5 | Arch §5c, PLAN §6 |
| 7 | H | R1 | Local API had no security design | Host + port allowlist, launch token → cookie, no CORS, media by id | Arch §8 |
| 8 | M | R1 | Decoding hazards | Bomb limit, EXIF first, HEIC, byte reads, timeouts | Arch §8 |
| 9 | M | R1 | Polymorphic refs, no migrations, merge chains, missing tables | `ITEM` table, `user_version` migrations, merges rewrite + `PEOPLE_EVENT`, `SETTINGS`, `SUPPRESSED_FACE` | Arch §4 |
| 10 | M | R1 | Embedding stored twice, no owner | BLOB canonical, vec0 derived, sqlite-vec pinned | Arch §4 |
| 11 | M | R1 | Stage dependencies wrong; repeated decoding | Ingest → derive → analysis graph; working-image cache | Arch §3 |
| 12 | M | R1 | No SQLite concurrency model | WAL + single writer thread | Arch §5a |
| 13 | M | R1 | Grouping under-specified; HDBSCAN unstable; four competing mechanisms | k-NN vote over confirmed faces; suggestions only for unassigned faces; reject-split; Chinese Whispers above about 50k | Arch §5b |
| 14 | M | R1 | Too much plug-in machinery | One Protocol per role; captions deferred; `insightface` package dropped in favour of ONNX + `norm_crop` | Arch §2, §7 |
| 15 | M | R1 | Open decisions blocked M1 | Interface marked "decide before M1", with HTMX recommended; date parsing replaced by filter chips | PLAN §9 |
| 16 | L | R1 | No testing, observability or config | Test layout, logs, SSE, TOML | Arch §7, §8 |
| 17 | L | R1 | "Forget this person" undone by the next scan | Suppression by box hash *and* embedding distance | Arch §4 |
| 18 | M | R2 | Layer diagram put ports below the domain (an upward dependency) | Ports moved inside the core; adapters outside with "implement" arrows; bootstrap shown | Arch §2 |
| 19 | M | R2 | Scheduler entry point missing; Web UI drawn inside the process | Scheduler added as an entry point; UI shown in the browser | Arch §2 |
| 20 | M | R2 | Pipeline: no change/move stage, results had no destination, Assign drawn as a stage | Change/move check added; outputs to tables; `PeopleService.assign_new` hand-off | Arch §3 |
| 21 | H | R2 | Expired leases did not count as attempts, so a crashing file would loop forever | Leasing increments attempts; dead after 3 | Arch §3 rule 4 |
| 22 | M | R2 | `done → pending` on a new model contradicted the job key | A new model_id creates new jobs; `cancelled` state added | Arch §3 |
| 23 | M | R2 | File lifecycle: missing transitions, no online-only state, move hashing could download files | Transitions and `online_only` added; hash only when both files are local; removal cascade defined | Arch §3 |
| 24 | H | R2 | `JOB` keyed by item, but derive creates items | `JOB.id`, `media_id` always, `item_id` nullable, unique key | Arch §4a |
| 25 | M | R2 | `track_group` on ITEM, though a frame can hold several people | `track_id` moved to FACE | Arch §4b |
| 26 | M | R2 | Missing PKs and unique keys, width/height, duration, `cover_face_id` link, `person_id` on events | Added; ER split into two readable diagrams | Arch §4 |
| 27 | M | R2 | Scan sequence returned 202 after the full walk; scheduler, derive and assign missing | 202 first; scheduler drives discover → derive → passes → assign | Arch §5a |
| 28 | H | R2 | UI edits had no path to the single writer | Edits go through the writer's command queue | Arch §5a, §6 |
| 29 | M | R2 | Query encoder residency unclear; peak RAM understated | Text side always loaded, outside `max_resident_models`; peak about 3–3.5 GB | Arch §6 |
| 30 | L | R2 | Captions half-included (Ollama in runtime only) | Deferred in all diagrams; extension point described | Arch intro |

## Diagrams in ARCHITECTURE.md

| # | Diagram | Type | Status |
|---|---|---|---|
| 1 | Context and file locations | flowchart | Reviewed, rendered |
| 2 | Layers: core, ports, adapters | flowchart | Redrawn after R2 |
| 3 | Indexing pipeline | flowchart | Redrawn after R2 |
| 4 | Job lifecycle | state | Fixed after R2 |
| 5 | File (media) lifecycle | state | Fixed after R2 |
| 6 | Data model: library and indexing | ER | Split and fixed after R2 |
| 7 | Data model: people | ER | Split and fixed after R2 |
| 8 | Scan and index | sequence | Reordered after R2 |
| 9 | Face assignment and suggestions | flowchart | Fixed after R2 |
| 10 | Search | sequence | Extended after R2 |
| 11 | Runtime view | flowchart | Fixed after R2 |

## Still open (not design defects)

1. **Interface choice** (PLAN §9.1): needed before M1.
2. **RAM and speed figures** are estimates from parameter counts. Measure them in M1.
3. **EmbeddingGemma 2 vs SigLIP 2:** settled by the M3 evaluation.
4. **DirectML vs CPU** ONNX Runtime package: pick one in M1, because the two conflict.
5. The runtime diagram is wide. It is readable at full width but small in narrow previews.
