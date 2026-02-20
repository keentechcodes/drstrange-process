# VeriMed Textbook Ingestion Pipeline — Implementation Plan

## Agent Instructions: Read This First

Before writing any code, complete these steps in order:

1. **Survey the existing codebase** — run `find . -type f -name "*.py" | head -40` and `ls -la` from the project root. Read any existing `run_docstrange*.py`, `chunker*.py`, or `ingest*.py` files in full before touching anything. The task is to extend/refactor, not rewrite from scratch.

2. **Fetch library docs via context7 MCP** — required before implementing:
   - `resolve-library-id` for `pymupdf` then `query-docs` on "get_toc page extraction"
   - `resolve-library-id` for `pymilvus` then `query-docs` on "insert schema collection"
   - `resolve-library-id` for `beautifulsoup4` then `query-docs` on "html table parsing rowspan"

3. **Do not run DocStrange** during implementation — it takes hours on the full PDF. Test post-processor and chunker logic against the sample markdown file at `BATES_sample_306-324.md` (already in the repo).

---

## Project Context

**What VeriMed is:** A medical note validation system ("Grammarly for medical accuracy"). It validates student-uploaded clinical notes against authoritative textbook sources using NLI (Natural Language Inference). The textbook corpus is the ground truth.

**The stack:**
- Vector DB: Milvus (Docker Standalone locally, Zilliz Cloud in prod)
- Embeddings: PubMedBERT (768-dim, self-hosted)
- Sparse: Milvus BM25 built-in (auto-generated from `text` field)
- NLI: DeBERTa-SciFact
- PDF conversion: DocStrange (textbooks) — already runs, output is `.md`
- Current textbook: `BATES.pdf` — 481 PDF pages, book content pages 1–431 (printed), Index pages 445–464 (printed)

**The Milvus collection schema** (already defined, do not change field names):
```python
fields: id (VARCHAR PK), text (VARCHAR, BM25-enabled), dense_vector (FLOAT_VECTOR 768),
        sparse_vector (SPARSE_FLOAT_VECTOR), book (VARCHAR, partition_key),
        chapter_num (INT16), chapter_title (VARCHAR), section (VARCHAR),
        subsection (VARCHAR), page_start (INT16), page_end (INT16),
        chunk_index (INT16), content_type (VARCHAR)
```

---

## What DocStrange Produces (Critical Context)

DocStrange converts the PDF to a single `.md` file. Its specific output format and known issues:

**Format:**
- Each PDF page is wrapped: `## Page N` ... `---`
- Printed book page numbers appear as: `<page_number>289</page_number>`
- Images get descriptive alt-text: `![A diagram showing...](images/pageN_imgM.jpeg)` — this is valuable, preserve it
- Tables are rendered as HTML `<table>` blocks (not pipe tables)

**Known issues to fix in post-processing (confirmed from benchmark on pages 289–307):**
1. `## Page N` headers and `Bates' Pocket Guide...` running headers are boilerplate, not content
2. `Chapter 16 | The Musculoskeletal System` chapter reference lines repeat on many pages — strip
3. Two-column "EXAMINATION TECHNIQUES / POSSIBLE FINDINGS" layout is **not recoverable** — DocStrange flattens it into orphaned bold paragraphs. These pages must be tagged `content_type="exam_technique_flattened"` with `confidence="low"` rather than attempting repair.
4. Multi-page tables split at PDF page boundaries — the continuation page starts with `**Table 15-2 Abnormalities on Rectal Examination (continued)**`. These must be merged back into one HTML block.
5. Rowspan corruption occurs in merged tables — `rowspan="2"` on empty `<td>` cells causes row misalignment. Strip all `rowspan` attributes from empty cells after merge.
6. `(table continues on page N)` and `(continued)` / `(Continued)` standalone lines are layout artifacts — strip after using them for merge detection.

---

## Bates PDF Specifics

- **Total PDF pages:** 481
- **Page offset:** 17 (PDF page 18 = book page 1, confirmed visually)
- **Front matter to skip:** PDF pages 1–17 (cover, blanks, author, preface pages ix/xi/xiii, TOC pages xv/xvi)
- **Content pages:** PDF pages 18–461 (book pages 1–444, Chapters 1–20)
- **Back matter to skip:** PDF pages 462–481 (Index, book pages 445–464)
- **Chapter count:** 20 chapters
- **TOC (from book):**

| Chapter | Title | Book Page Start |
|---------|-------|----------------|
| 1 | Foundations for Clinical Proficiency | 1 |
| 2 | Evaluating Clinical Evidence | 29 |
| 3 | Interviewing and the Health History | 45 |
| 4 | Beginning the Physical Examination | 65 |
| 5 | Behavior and Mental Status | 87 |
| 6 | The Skin, Hair, and Nails | 99 |
| 7 | The Head and Neck | 129 |
| 8 | The Thorax and Lungs | 159 |
| 9 | The Cardiovascular System | 181 |
| 10 | The Breasts and Axillae | 203 |
| 11 | The Abdomen | 215 |
| 12 | The Peripheral Vascular System | 235 |
| 13 | Male Genitalia and Hernias | 249 |
| 14 | Female Genitalia | 265 |
| 15 | The Anus, Rectum, and Prostate | 283 |
| 16 | The Musculoskeletal System | 293 |
| 17 | The Nervous System | 331 |
| 18 | Assessing Children: Infancy through Adolescence | 369 |
| 19 | The Pregnant Woman | 403 |
| 20 | The Older Adult | 419 |

---

## Target Architecture

```
BATES.pdf
    ↓
[Step 0] extract_toc.py          → bates.toc.json + bates.meta.json
    ↓
[Step 1] DocStrange (existing)   → bates.md        (already exists / runs separately)
    ↓
[Step 2] postprocess_docstrange.py   → bates.pageblocks.json
    ↓
[Step 3] chunk_textbook.py           → list of Milvus-ready chunk dicts
    ↓
[Step 4] ingest_milvus.py (existing) → Milvus collection
```

**The PageBlock JSON is the contract between Step 2 and Step 3.** The chunker must not import or know about anything from the post-processor except this schema.

---

## Task List

### Task 0 — `extract_toc.py`

**Purpose:** Extract PyMuPDF TOC, compute page offset, emit sidecar files.

**Inputs:** `BATES.pdf`  
**Outputs:** `bates.toc.json`, `bates.meta.json`

**Logic:**

```
1. Open PDF with pymupdf (fitz)
2. Call doc.get_toc() → [[level, title, pdf_page_1based], ...]
   Note: PyMuPDF returns 1-based page numbers
3. Auto-detect page_offset:
   - Find the TOC entry for Chapter 1 (lowest pdf_page among level-1 entries
     that have a book page number explicitly in TOC, OR hardcode anchor:
     "Foundations for Clinical Proficiency" → known book_page=1)
   - offset = chapter1_pdf_page - 1   (since book_page=1)
   - For Bates this will be 17
4. For each TOC entry compute:
   - book_page_start = pdf_page - offset
   - pdf_page_end = next_entry.pdf_page - 1  (last entry uses total_pages)
   - book_page_end = pdf_page_end - offset
   - chapter_num: parse from title if "Chapter N" pattern, else null
   - entry_type: "chapter" | "front_matter" | "back_matter"
     - front_matter: book_page_start < 1 (roman numeral pages)
     - back_matter: title matches "Index" | "Appendix" | "References"
     - chapter: everything else
5. Write bates.toc.json (list of enriched TOC entry dicts)
6. Write bates.meta.json:
   {
     "book": "bates",
     "source_pdf": "BATES.pdf",
     "total_pdf_pages": 481,
     "page_offset": 17,
     "content_pdf_page_start": 18,   ← first chapter PDF page
     "content_pdf_page_end": 461,    ← last chapter PDF page
     "generated_at": "<ISO timestamp>"
   }
```

**bates.toc.json entry schema:**
```json
{
  "level": 1,
  "title": "The Musculoskeletal System",
  "chapter_num": 16,
  "pdf_page_start": 310,
  "pdf_page_end": 348,
  "book_page_start": 293,
  "book_page_end": 331,
  "entry_type": "chapter"
}
```

---

### Task 1 — Modify `run_docstrange_textbook.py` (minimal changes only)

**Only add these two things:**

1. Accept `--meta-file` argument pointing to `bates.meta.json`. If provided, automatically pass only the content page range to DocStrange (skip front matter and index). This avoids processing 37 pages of useless content.

2. If DocStrange supports page range arguments natively, use them. If not, post-filter: after conversion, strip `## Page N` blocks where N < `meta.content_pdf_page_start` or N > `meta.content_pdf_page_end`.

**Do not change the DocStrange call signature or output format.**

---

### Task 2 — `postprocess_docstrange.py`

This is the main new file. Six sequential passes, each function takes the output of the previous.

**Inputs:** `bates.md`, `bates.toc.json`, `bates.meta.json`  
**Output:** `bates.pageblocks.json` (list of PageBlock dicts)

#### PageBlock schema:
```python
{
    "pdf_page": int,                 # from ## Page N header
    "book_page_start": int,          # from <page_number> tag, offset-corrected
    "book_page_end": int,            # usually == book_page_start
    "chapter_num": int,              # from toc.json lookup
    "chapter_title": str,            # from toc.json lookup
    "section_path": list[str],       # ["Chapter Title", "Section", "Subsection"]
    "content_type": str,             # see types below
    "content": str,                  # cleaned content
    "confidence": str,               # "high" | "medium" | "low"
    "merged_from_pdf_pages": list[int]  # [N, N+1] if continuation merge happened
}
```

#### Content types (exactly these strings, they go into Milvus `content_type` field):
```
"prose"
"table"
"list"
"clinical_criteria"
"clinical_scale"
"exam_technique_flattened"
"image_desc"
"table_footnote"
```

#### Pass 1 — Page Splitter
```
- Split raw markdown at ## Page N boundaries
- Each block: {pdf_page: N, raw: <everything between ## Page N and next ## Page N>}
- Only include pages within meta["content_pdf_page_start"] to meta["content_pdf_page_end"]
- From each raw block:
    - Extract <page_number> tag values → book_page_start, book_page_end
    - If no tag found, interpolate from neighbors (some pages genuinely lack it)
    - Strip: the ## Page N line itself
    - Strip: "Bates' Pocket Guide to Physical Examination and History Taking" header lines
    - Strip: "Chapter N | <Title>" running header lines (regex: ^Chapter \d+ \| .+$)
    - Strip: "C H A P T E R" spaced-letter header (appears on chapter opener pages)
- Result: list of RawPageBlock {pdf_page, book_page_start, book_page_end, raw_cleaned}
```

#### Pass 2 — Chapter + Section Path Assignment
```
- Load toc.json
- For each RawPageBlock:
    - Look up which toc entry contains this pdf_page → get chapter_num, chapter_title
    - Parse H2/H3 headings from raw_cleaned to build section/subsection
    - Maintain running section_path state across pages (a heading on page N applies
      to all subsequent pages until a new heading of same/higher level appears)
    - section_path = [chapter_title, section, subsection]  (subsection may be empty)
- Result: each RawPageBlock gains {chapter_num, chapter_title, section_path}
```

#### Pass 3 — Exam Technique Page Detection
```
- For each block, check if raw_cleaned contains both:
    - Standalone line matching: ^EXAMINATION TECHNIQUES?$  (case-insensitive)
    - OR standalone line matching: ^POSSIBLE FINDINGS?$
- If either marker present → set is_exam_page = True on that block
- These pages are in Ch16 musculoskeletal, some Ch17 nervous system examination sections
- Do NOT attempt to repair the column structure. Just flag it.
```

#### Pass 4 — Table Continuation Merger
```
- Scan consecutive RawPageBlocks for this pattern:
    Page N:   contains <table>...</table> that ends mid-data (no structural close)
              OR simply contains a table + ends with "(table continues on page M)"
    Page N+1: raw_cleaned starts with **Table X-Y ... (continued)** or (Continued)
              OR first significant content is a partial <table> starting with <tbody>

- When pattern matches:
    1. Extract the <table> HTML from page N and page N+1
    2. Merge: take page N table, remove closing </tbody></table>,
       append page N+1's <tbody> rows, re-close
    3. Fix rowspan corruption on merged result:
       - Parse with BeautifulSoup
       - Find all <td rowspan="N"> where cell text is empty → strip the rowspan attribute
       - This fixes the Systemic Disorders table misalignment seen in benchmark
    4. Replace page N's table content with merged table
    5. Set page N's book_page_end = page N+1's book_page_end
    6. Set page N's merged_from_pdf_pages = [N, N+1]
    7. Mark page N+1 as "consumed" → exclude from output
    8. Strip "(table continues on page M)" and standalone "(continued)" lines

- Handle chains: a table may span 3+ pages (same pattern, iterate)
```

#### Pass 5 — Content Type Tagger + Block Splitter
```
For each block (after merge pass), split the raw_cleaned into sub-blocks, then tag each:

Splitting rules:
- HTML <table> blocks → always their own sub-block
- Image refs with non-empty alt-text → their own sub-block (type: image_desc)
  Pattern: ![<non-empty text>](images/...)
  Empty alt-text images: discard entirely (no retrieval value)
- Text between tables/images → prose/list/clinical sub-block

Tagging rules (in priority order):
1. is_exam_page=True → "exam_technique_flattened", confidence="low"
2. Starts with <table → "table", confidence="high"
3. Pipe table (|---|) → "table", confidence="high"
4. Has heading matching clinical box patterns AND numbered items:
   (Red Flag|Score|Steps for|BPH|AUA|CURB|HEART|NEXUS|WHO Bone) → "clinical_scale"
5. Has heading matching clinical box patterns AND bulleted items:
   (Red Flag|Signs of|Risk Factor|Criteria|Mnemonic|Four Signs) → "clinical_criteria"
6. Has 3+ bolded terms in definition pattern (anatomy glossary boxes like
   "Joint Anatomy — Important Terms") → "clinical_criteria"
7. Has bullet/numbered list markers → "list"
8. Is image alt-text → "image_desc"
9. Orphaned lines like "IBD, Inflammatory bowel disease; RA, Rheumatoid arthritis."
   appearing after a table → "table_footnote"
10. Everything else → "prose"
```

#### Pass 6 — BM25 Noise Cleanup
```
On every block's content string, apply these cleanups:
- Strip any surviving ## Page N tokens
- Strip standalone "(continued)" / "(Continued)" lines
- Strip standalone "(table continues on page N)" lines
- Strip page number artifacts: lines that are only a number (e.g., just "293")
- Normalize multiple blank lines to single blank line
- Strip leading/trailing whitespace from content

Do NOT strip:
- HTML table markup (chunker needs it intact)
- Image alt-text (it's BM25-retrievable content)
- Clinical abbreviation footnotes that are tagged as table_footnote
  (the chunker will decide whether to attach them to preceding table)
```

---

### Task 3 — `chunk_textbook.py`

**Inputs:** `bates.pageblocks.json`, `bates.meta.json`  
**Output:** list of dicts ready for Milvus insert (no vectors yet — embedding step is separate)

**The chunker reads PageBlock JSON only. It has no knowledge of DocStrange format.**

#### Chunking rules by content_type:

```
"table"                    → 1 atomic chunk (never split, regardless of token count)
"clinical_criteria"        → 1 atomic chunk
"clinical_scale"           → 1 atomic chunk
"image_desc"               → 1 atomic chunk
"exam_technique_flattened" → 1 chunk per PageBlock (already page-scoped)
"table_footnote"           → attach to the immediately preceding "table" chunk
                             by appending to its text field with a newline separator
                             (do not create a standalone chunk for footnotes)

"list"   → if ≤ 400 tokens: 1 atomic chunk
           if > 400 tokens: split at list item boundaries, NOT mid-item
           each split carries the same section_path

"prose"  → target 300–400 tokens, split at sentence boundaries
           1–2 sentence overlap between adjacent chunks (carry-forward)
           if a prose block is < 80 tokens: attempt merge with the next
           prose block in the same section before splitting
```

**Token counting:** Use PubMedBERT tokenizer (`dmis-lab/biobert-base-cased-v1.2` or whichever PubMedBERT variant is already in the project) — tokenizer must match the embedding model. Check existing code for which model is used.

#### Milvus chunk dict assembly:
```python
{
    "id": f"{book}_ch{chapter_num:02d}_p{page_start:04d}_c{chunk_index:04d}",
    # e.g., "bates_ch16_p0293_c0003"
    "text": chunk_text,              # cleaned content string
    "book": meta["book"],            # "bates"
    "chapter_num": block["chapter_num"],
    "chapter_title": block["chapter_title"],
    "section": block["section_path"][1] if len(section_path) > 1 else "",
    "subsection": block["section_path"][2] if len(section_path) > 2 else "",
    "page_start": block["book_page_start"],
    "page_end": block["book_page_end"],
    "chunk_index": chunk_index,       # sequential within this page block
    "content_type": block["content_type"],
    # NOTE: dense_vector and sparse_vector are NOT set here
    # They are populated in a separate embedding step (ingest_milvus.py)
}
```

#### ID collision guard:
If two chunks would produce the same ID (can happen when table footnote is merged into table), increment `chunk_index` until unique within the page. Log any collisions for inspection.

---

### Task 4 — Update `run_docstrange_textbook.py` (from Task 1 above)

Refer to Task 1 instructions. Keep changes minimal.

---

## File Structure After Implementation

```
verimed/
├── ingestion/
│   ├── extract_toc.py              ← NEW
│   ├── run_docstrange_textbook.py  ← MODIFIED (minor)
│   ├── postprocess_docstrange.py   ← NEW
│   ├── chunk_textbook.py           ← NEW
│   └── ingest_milvus.py            ← EXISTING (unchanged)
├── data/
│   ├── BATES.pdf
│   ├── bates.toc.json              ← generated by extract_toc.py
│   ├── bates.meta.json             ← generated by extract_toc.py
│   ├── bates.md                    ← generated by DocStrange (may already exist)
│   ├── bates.pageblocks.json       ← generated by postprocess_docstrange.py
│   └── samples/
│       └── BATES_sample_306-324.md ← use this for testing all new scripts
└── tests/
    └── test_postprocessor.py       ← NEW (unit tests for each pass)
```

---

## Testing Approach

**Do not test against full `bates.md` (doesn't exist yet / too slow).**

Test every pass against `BATES_sample_306-324.md`. This sample covers:
- Pages 289–307 (PDF pages ~306–324)
- Table 15-1 (BPH scoring scale) → should be tagged `clinical_scale`
- Table 15-2 spanning PDF pages 3–4 of sample → should merge into one table chunk
- "Joint Anatomy — Important Terms" box → should be `clinical_criteria`
- "Common Causes of Joint Pain by Age" two-column table → should be `table`
- Exam technique pages → should be `exam_technique_flattened`
- "Red Flag Signs for Low Back Pain" list → should be `clinical_criteria`
- DocStrange image alt-texts → should become `image_desc` chunks
- WHO Physical Activity Guidelines prose block → should be `prose`

For each pass, write an assertion that checks the specific known output for this sample. The Systemic Disorders table rowspan merge is the highest-priority test — disease-to-clue mapping must be correct after merge.

---

## Constraints / Do Nots

- Do not install new dependencies without checking existing `requirements.txt` first. `beautifulsoup4`, `lxml`, `pymupdf` should already be present.
- Do not change the Milvus schema field names — the collection may already have data.
- Do not run the full ingestion pipeline (DocStrange + Milvus insert) — just implement and unit test the new scripts.
- Do not hardcode page numbers beyond what is noted in "Bates PDF Specifics" above. The `extract_toc.py` script must compute these dynamically so the pipeline works for other books later.
- Keep each script independently runnable from the CLI with sensible `--input` / `--output` flags.
