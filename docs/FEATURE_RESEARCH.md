# FaceGroup — Feature research: similar apps

As of 2026-10-25. This survey covers 13 open-source and 12 commercial photo apps with face recognition or AI search, checked against official docs, feature pages and release notes (sources at the end). Items marked [U] were only seen in search snippets or third-party pages.

**Summary:** FaceGroup already matches the field on the core people loop (name, merge, split, reject, hide), semantic search and similar-photo search. It is ahead on three things few apps offer: undo for every people change (Google and Synology merges cannot be undone), learned per-person strictness, and fully local processing. The biggest gaps, in order of value for this library:

1. Auto-categories and declutter (screenshots, receipts, documents, memes).
2. Duplicates and similar-shot stacks.
3. OCR text search.
4. Richer people profiles: cover face, favourites, birthday and age.
5. A timeline with "on this day".
6. Drawing a box around a missed face.

## 1. What FaceGroup already has (and who else does)

| Feature | Also in | FaceGroup |
|---|---|---|
| Name, rename, merge, split people | all | ✓ |
| Remove wrong faces ("not this person"), remembered | Google, Apple, Tonfotos, Mylio | ✓ |
| Hide "not a person", with restore | Apple, Mylio, Ente, Synology | ✓ |
| Unknown-face review in groups, confirm or reject | LibrePhotos, Ente, Lightroom, Mylio | ✓ (parts, Quick check) |
| Focused yes/no question queue | Apple ("Review more photos"), Google | ✓ (Quick check) |
| Per-person sensitivity | Tonfotos ("offer fewer suggestions") | ✓ learned automatically |
| Undo for merges and all people changes | Ente (merges only) | ✓ every change |
| Corrections survive re-grouping | PhotoPrism, Immich | ✓ (property-tested) |
| Face-quality gate before grouping | digiKam (FIQA), Ente (blur), PhotoPrism | ✓ (size, score, blur) |
| Two faces in one photo are two people | — (not seen documented anywhere) | ✓ |
| Semantic search | Immich, Google, Apple, Lightroom, Excire | ✓ SigLIP 2 |
| Similar-photo search | LibrePhotos, ACDSee, QNAP, Damselfly | ✓ |
| Filters: person, date, folder | most | ✓ |
| Forget a person and suppress them on rescans | — | ✓ |
| Local and private, no cloud | Mylio, Excire, Tonfotos, digiKam, Ente | ✓ |
| OneDrive online-only files not downloaded | — | ✓ |

## 2. Gaps, ranked by value for FaceGroup

Value is for a single-user, local, privacy-first library that is (judging from the profiling aggregates) heavy on screenshots. Effort: S = hours, M = a day or two, L = several days.

### People

| # | Feature | Seen in | Value | Effort | Notes |
|---|---|---|---|---|---|
| P1 | **Choose a person's cover face** (with best-quality default) | Google, Apple, Mylio, Immich, PhotoPrism | High | S | Default is already the best confirmed face; add "Make cover" |
| P2 | **Favourite or pin people**, custom order | Apple, Immich | Medium | S | |
| P3 | **Birthday, then age shown on each photo; sort a person's faces by age; birthday memories** | Immich, Ente, Tonfotos | High | S–M | Also helps the matcher: a large age gap explains low similarity |
| P4 | **Profile: other names, relationship (Mom, Spouse), notes, "me"** | Google, Mylio, Tonfotos | Medium | S | "Me" enables "photos of me" |
| P5 | **People often together**: groups detected from co-occurrence | Apple | Medium | S | Data already exists (faces per photo) |
| P6 | **Keyboard-driven review**: Enter = yes, X = no, next | ACDSee, Apple | Medium | S | Faster Quick check |
| P7 | **Draw a box around a missed face** and assign it | Lightroom, PhotoPrism, LibrePhotos, digiKam, QNAP, Synology | Medium | M | Embeds the drawn crop |
| P8 | **Re-scan faces in chosen photos** (smaller-face retry) | PhotoPrism, LibrePhotos, Mylio | Low–Med | S | PhotoPrism retries at a smaller size when none found |
| P9 | **Pets** as people | Google, Apple (Immich forks only) | Medium | L | Needs a pet detector and re-ID model; check licences first |

### Search

| # | Feature | Seen in | Value | Effort | Notes |
|---|---|---|---|---|---|
| S1 | **OCR text search** (receipts, signs, screenshots) | Apple, Google, Mylio, Immich (PP-OCRv6) | **Very high** for a screenshot-heavy library | M | Planned M4. RapidOCR / PP-OCRv6, Apache-2.0 |
| S2 | **Person names in the query** ("Mom beach") and **several people at once** ("Mom & Dad") | Google, PhotoPrism (`people:a&b`) | High | S | |
| S3 | **More filters**: camera or lens, favourite or rating, media type, number of faces, "no people" | PhotoPrism, Synology, Immich | Medium | S | |
| S4 | **Places**: offline reverse geocoding (GeoNames), filter by city or country | nearly all | Medium (few GPS photos so far) | M | Offline dataset keeps it private |
| S5 | **Saved searches as smart albums** | PiGallery2, Synology, QNAP | Medium | S | |
| S6 | **Map view** | most | Low–Med | M | Online map tiles would break offline privacy; needs offline tiles or opt-in |
| S7 | **Plain English turned into filters** by a local LLM ("photos of Mom from last summer in Goa") | digiKam 9.2, Google Ask Photos | Medium | L | Later; fits the optional-VLM plug-in |

### Organising

| # | Feature | Seen in | Value | Effort | Notes |
|---|---|---|---|---|---|
| O1 | **Auto-categories**: screenshots, receipts, documents, memes, wallpapers; **declutter toggle** to hide them from browsing and grouping | Microsoft Photos (Copilot+), Google, OneDrive | **Very high** here | S–M | Zero-shot with the SigLIP 2 vectors already stored: no new model |
| O2 | **Duplicates** (exact and near-identical) with an **auto-picked keeper** | Immich, Mylio, Excire, ACDSee, LibrePhotos | High | M | Exact = content hash (stored); near = image embedding. FaceGroup never deletes; it can only list, or hide in the app |
| O3 | **Stacks**: similar shots taken close together, with a changeable top pick | Google, Synology, ACDSee, Lightroom, Immich | High | M | Time + embedding similarity |
| O4 | **Favourites, ratings, archive** (stored in FaceGroup, never in the files) | everywhere | Medium | S | |
| O5 | **Timeline** browsing by date, with a scrubber | Immich, Nextcloud Memories, LibrePhotos | Medium | M | |
| O6 | **Memories**: "on this day", best of year, birthdays | Immich, Amazon, Apple, Google | Medium | S (after O5) | |
| O7 | **Events or trips as automatic albums** (time + place clusters) | LibrePhotos, Apple | Medium | M | |
| O8 | **Best-shot picking** (sharpness, eyes open, expression) | Excire, Lightroom, ACDSee, digiKam | Medium | M–L | Partly free: face blur and quality already stored |

### Portability, privacy, background work

| # | Feature | Seen in | Value | Effort | Notes |
|---|---|---|---|---|---|
| X1 | **Export names and face regions** to XMP (MWG regions) in a separate export folder, never into originals | digiKam, LibrePhotos, Excire, ACDSee | Medium | M | Keeps your work portable. Writing sidecars next to photos would break the read-only promise, so opt-in only |
| X2 | **Import face tags** from Windows Photos, Lightroom, digiKam, Picasa as hints | PhotoPrism, ACDSee, Peakto | Low–Med | M | Rule from Mylio and Tonfotos: imported tags must not train recognition until confirmed |
| X3 | **Wipe all face data** with one switch | Google, Microsoft, Amazon | Medium | S | "Forget person" exists; add global |
| X4 | **Laptop-friendly indexing**: pause on battery or when hot, resume on mains | Ente | Medium | S | |
| X5 | **Nightly or automatic re-check** of unassigned faces | Immich, LibrePhotos | Low | S | Already happens on each refresh |

### Video (M5, unchanged)
Faces in sampled frames, frame-level search ("find the moment"), as in Apple and Peakto.

## 3. Patterns worth copying, and two to avoid

- **Local apps ask before naming anyone** (Mylio, Tonfotos, ACDSee) and make bulk confirm fast. FaceGroup already works this way.
- **Imported face tags never train recognition** until confirmed (Mylio, Tonfotos). Adopt this if X2 is built.
- **Google's "Remember list"** lets you see and delete what the assistant has learned about you. FaceGroup's Check page plays this role.
- **Avoid:** merges that cannot be undone (Google, Synology). Also avoid toggles that silently delete all face data without warning (Microsoft and Google delete groups when face grouping is switched off).

## 4. Suggested next batches

| Batch | Contents | Why |
|---|---|---|
| A. Declutter | O1 auto-categories + declutter toggle, O2 duplicates, O3 stacks | Biggest win for a screenshot-heavy library; uses the existing embeddings |
| B. Text in images | S1 OCR (M4), plus S2 names in search | Finds receipts, slides and chats by their words |
| C. People profiles | P1 cover, P2 favourites, P3 birthday and age, P4 profile and "me", P5 often together, P6 keyboard review | Small, visible improvements |
| D. Browse | O4 favourites and ratings, O5 timeline, O6 memories, O7 events, S5 saved searches | Turns the index into a pleasant photo browser |
| E. Missed faces | P7 draw a box, P8 re-scan | Fills recall gaps |
| F. Portability and privacy | X1 export, X3 wipe, X4 battery-aware indexing | Peace of mind |
| G. Later | S4 places and S6 map, S7 LLM search, P9 pets, O8 best shot, video (M5) | Larger or lower value now |

## Sources (opened by the research agents)

Open source:
- [Immich facial recognition](https://docs.immich.app/features/facial-recognition)
- [Immich searching](https://docs.immich.app/features/searching)
- [Immich duplicates](https://docs.immich.app/features/duplicates-utility)
- [Immich releases](https://github.com/immich-app/immich/releases)
- [PhotoPrism face recognition](https://docs.photoprism.app/user-guide/ai/face-recognition/)
- [PhotoPrism people](https://docs.photoprism.app/user-guide/organize/people/)
- [PhotoPrism search filters](https://docs.photoprism.app/user-guide/search/filters/)
- [PhotoPrism features](https://www.photoprism.app/features)
- [LibrePhotos](https://github.com/LibrePhotos/librephotos)
- [LibrePhotos face recognition](https://docs.librephotos.com/docs/user-guide/face-recognition)
- [digiKam 8.6](https://www.digikam.org/news/2025-03-15-8.6.0_release_announcement/)
- [digiKam 9.0](https://www.digikam.org/news/2026-03-08-9.0.0_release_announcement/)
- [digiKam LLM search](https://www.digikam.org/news/2026-08-20-advanced_search_improvements_with_llm/)
- [Ente ML](https://ente.com/ml)
- [Ente face recognition](https://ente.com/help/photos/features/search-and-discovery/face-recognition)
- [Nextcloud Memories](https://memories.gallery/)
- [Nextcloud Recognize](https://github.com/nextcloud/recognize)
- [Damselfly](https://damselfly.info/)
- [PhotoStructure 2026 notes](https://photostructure.com/about/2026-release-notes/)
- [PiGallery2 features](https://bpatrik.github.io/pigallery2/features/)
- [Lychee 7](https://lycheeorg.dev/2025-12-31-version-7/)
- [Piwigo AI](https://piwigo.org/blog/2025/05/21/piwigo-ai-artificial-intelligence-is-coming-to-piwigo/)

Commercial:
- [Google Photos face groups](https://support.google.com/photos/answer/6128838)
- [Google Ask Photos](https://support.google.com/photos/answer/15318661)
- [Google photo stacks](https://support.google.com/photos/answer/14169846)
- [Apple People & Pets](https://support.apple.com/en-us/108795)
- [Apple review faces (Mac)](https://support.apple.com/guide/photos/phtad9d981ab)
- [Apple group collections](https://support.apple.com/guide/photos/phtad4b69675/mac)
- [Apple search](https://support.apple.com/en-mk/guide/photos/pht64de33e5a/mac)
- [Apple Enhanced Visual Search](https://support.apple.com/122033)
- [Microsoft group by faces](https://support.microsoft.com/en-us/Windows/ai/ai-apps/group-photos-by-faces)
- [OneDrive Photos](https://support.microsoft.com/en-us/onedrive/windows/browse-your-photos-on-windows)
- [Microsoft Photos auto-categorization](https://blogs.windows.com/windows-insider/2025/09/25/ai-powered-auto-categorization-now-available-in-microsoft-photos/)
- [Synology Photos spec](https://www.synology.com/en-global/dsm/7.2/software_spec/synology_photos)
- [QNAP QuMagie guide](https://www.qnap.com/en/how-to/tutorial/article/qumagie-and-qumagie-mobile-quick-start-guide)
- [Mylio face tagging](https://manual.mylio.com/topic/tag-people-facial-recognition)
- [Mylio People view](https://manual.mylio.com/topic/browse-the-people-view)
- [Mylio features](https://mylio.com/business/top-10-features/)
- [Excire Foto](https://excire.com/en/excire-foto/)
- [Peakto](https://cyme.io/en/press/peakto-redefines-photo-management-with-state-of-the-art-features/)
- [Tonfotos FAQ](https://tonfotos.com/docs/faq/)
- [Tonfotos people info](https://tonfotos.com/docs/people/people-info)
- [ACDSee Photo Studio](https://www.acdsee.com/en/products/photo-studio-ultimate/features/)
- Lightroom: Adobe help returned 403, so these are [U]: [Lightroom Killer Tips](https://lightroomkillertips.com/finding-people-in-lightroom-classic/) and [PhotoshopCafe](https://photoshopcafe.com/big-lightroom-update-april-2026-new-features/)
