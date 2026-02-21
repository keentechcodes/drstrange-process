# Code Review & DocStrange Optimization Report

**Date:** 2026-02-21
**Scope:** `drstrange-process` codebase — full review + docstrange model research
**Reviewer:** OpenCode (claude-opus-4-6)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Codebase Review](#codebase-review)
   - [extract_toc.py](#extract_tocpy)
   - [run_docstrange_textbook.py](#run_docstrange_textbookpy)
   - [setup_runpod.sh](#setup_runpodsh)
   - [Cross-cutting Concerns](#cross-cutting-concerns)
3. [DocStrange & Nanonets Model Research](#docstrange--nanonets-model-research)
   - [Model Landscape](#model-landscape)
   - [Nanonets-OCR-s vs Nanonets-OCR2-3B](#nanonets-ocr-s-vs-nanonets-ocr2-3b)
   - [Attention Implementation](#attention-implementation)
   - [DPI & Resolution](#dpi--resolution)
   - [Token Limits](#token-limits)
   - [System Prompt](#system-prompt)
   - [Official Usage Patterns](#official-usage-patterns)
4. [Changes Applied](#changes-applied)
   - [Bug Fixes](#bug-fixes)
   - [Model & Performance Optimizations](#model--performance-optimizations)
   - [Setup Script Improvements](#setup-script-improvements)
5. [Testing](#testing)

---

## Executive Summary

The review identified **3 high-priority bugs**, **4 medium-priority issues**, and **5 low-priority improvements** across the codebase. Additionally, research into the official docstrange source and Nanonets model ecosystem revealed the codebase was using an outdated model (`Nanonets-OCR-s`) with suboptimal configuration (SDPA attention, 200 DPI, 8192 token limit, custom system prompt). All issues have been fixed.

The single highest-impact change is the **model upgrade to Nanonets-OCR2-3B**, which wins 58% of head-to-head markdown extraction benchmarks against the old model at the same parameter count and VRAM cost.

---

## Codebase Review

### extract_toc.py

| Priority | Issue | Status |
|----------|-------|--------|
| Medium | `detect_page_offset` returned the first anchor match only — fragile if the first TOC entry is mislabeled | Fixed: now collects all offsets and returns the mode, with a warning for inconsistencies |
| Medium | Type hint `list[list]` was imprecise | Fixed: `list[tuple[int, str, int]]` |
| Low | `book_name` computed twice at the same scope level | Fixed: removed redundant assignment |
| Low | Silent `return [], {}` for unsupported PDFs with no embedded TOC | Fixed: prints actionable error message |

**Page number verification (from user):**
- TOC is on PDF pages 16-17
- First book page is PDF page 18
- Last chapter content ends at PDF page 460
- Page 461 is blank
- Index starts at PDF page 462 through end of file
- `BATES_CONTENT_END_PDF = 460` is confirmed correct

### run_docstrange_textbook.py

| Priority | Issue | Status |
|----------|-------|--------|
| **High** | `_normalize_headings` was a complete no-op — read heading levels and wrote them back unchanged | Fixed: stack-based algorithm that compresses level gaps (e.g., `# -> ###` becomes `# -> ##`) |
| **High** | `_clean_output` converted `<page_number>` tags to HTML comments, destroying tags the downstream post-processor needs | Fixed: `<page_number>` tags are now preserved; only watermarks and excess whitespace are cleaned |
| **High** | Monkey-patching 4 internal docstrange methods with no version pin or guard | Fixed: added version guard, compatibility comments, and `_DOCSTRANGE_TESTED_VERSION` constant |
| Medium | Error fallback `max_new_tokens or 4096` was dead code (variable always truthy) | Fixed: uses separate `_fallback_tokens = 4096` variable |
| Medium | `str` type annotations on `convert()` and `_extract_images_from_pdf()` but bodies immediately call `Path()` | Fixed: `str \| Path` annotations |
| Low | Duplicate `import torch` in two try/except blocks | Fixed: consolidated to single import |
| Low | Unused `chunk_pages` parameter accepted by constructor and CLI | Fixed: removed entirely |
| Low | Legacy `typing` imports (`Dict`, `List`, `Tuple`, `Optional`) while `extract_toc.py` uses modern syntax | Fixed: uses `dict`, `list`, `tuple`, `X \| None` (Python 3.11+) |
| Low | Unused `processor_self = self` in Optimization 2 block (closure didn't reference it) | Fixed: removed |
| Low | Stale CLI options in module docstring | Fixed: matches actual args |

### setup_runpod.sh

| Priority | Issue | Status |
|----------|-------|--------|
| Low | `curl \| sh` pipes installer directly to shell (unsafe if download fails mid-stream) | Fixed: downloads to temp file first |
| Low | `PATH` not persisted for future shell sessions | Fixed: appends to `.bashrc` |
| Low | Only caches deps for `extract_toc.py`, no mention of the other script | Fixed: documents why `run_docstrange_textbook.py` is skipped |

### Cross-cutting Concerns

| Issue | Notes |
|-------|-------|
| No `pyproject.toml` or shared dependency management | `extract_toc.py` uses PEP 723, `run_docstrange_textbook.py` doesn't. A shared `pyproject.toml` would be cleaner for the growing pipeline. |
| No tests | Implementation plan specifies `test_postprocessor.py` but nothing exists yet. |
| `.gitignore` excludes all generated outputs | `*.toc.json`, `*.meta.json`, and `*.md` are gitignored. Bootstrap process for new contributors is undocumented. |
| No logging framework | All scripts use `print()`. For long GPU jobs, Python's `logging` module with timestamps would help debugging. |

---

## DocStrange & Nanonets Model Research

### Model Landscape

Nanonets provides three OCR models as of Feb 2026:

| Model | Params | Release | Access | Base Model |
|-------|--------|---------|--------|------------|
| `Nanonets-OCR-s` | ~3B | June 2025 | HuggingFace (open) | Qwen2.5-VL-3B-Instruct |
| **`Nanonets-OCR2-3B`** | ~3B | Oct 2025 | HuggingFace (open) | Qwen2.5-VL-3B-Instruct |
| `Nanonets-OCR2-Plus` | Unknown | Oct 2025 | Cloud API only (via docstrange) | Unknown |

There is also an experimental `Nanonets-OCR2-1.5B-exp` (smaller, significantly worse — 79% lose rate vs OCR2-3B).

**Source:** [HuggingFace model pages](https://huggingface.co/nanonets), docstrange pyproject.toml (version 1.1.8), docstrange source code.

### Nanonets-OCR-s vs Nanonets-OCR2-3B

From the official evaluation tables on the Nanonets-OCR2-3B model card:

**Markdown extraction (head-to-head):**

| Matchup | Win Rate | Lose Rate | Both Correct |
|---------|----------|-----------|--------------|
| OCR2-3B vs OCR-s | **58.28%** | 30.61% | 11.12% |
| OCR2-3B vs GPT-5 | **72.87%** | 25.00% | 2.13% |
| OCR2-3B vs Gemini 2.5 Flash | 52.43% | 39.98% | 7.58% |

**Visual Question Answering (VQA):**

| Dataset | OCR2-3B | OCR2-Plus | Qwen2.5-VL-72B | Gemini 2.5 Flash |
|---------|---------|-----------|-----------------|------------------|
| ChartQA | 78.56 | 79.20 | 76.20 | 84.82 |
| DocVQA | **89.43** | 85.15 | 84.00 | 85.51 |

Key improvements in OCR2-3B over OCR-s:
- Multilingual support (English, Chinese, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Arabic, and more)
- Flowchart and org chart extraction (as Mermaid code)
- Improved handwritten document recognition
- Better complex table handling
- VQA capability

**Same VRAM footprint.** Both are 3B parameter Qwen2.5-VL fine-tunes. The upgrade is free in terms of compute cost.

### Attention Implementation

The docstrange source code (`nanonets_processor.py`) loads the model with:
```python
self.model = AutoModelForImageTextToText.from_pretrained(
    str(actual_model_path),
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True
)
```
No explicit attention implementation — defaults to the model's native attention.

The **official Nanonets-OCR2-3B usage example** on HuggingFace recommends:
```python
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
)
```

`flash_attention_2` is faster (~15-20%) and more memory-efficient than SDPA for the Qwen2.5-VL architecture. Requires the `flash-attn` package to be installed (`pip install flash-attn --no-build-isolation`).

**Our approach:** Try `flash_attention_2` first, fall back to `sdpa` if `flash-attn` isn't available.

### DPI & Resolution

From `docstrange/config.py`:
```python
class InternalConfig:
    pdf_image_dpi = 300
    pdf_image_scale = 2.0
```

The upstream default is **300 DPI**. Our code was overriding this to 200 DPI, which reduces OCR quality.

From the Nanonets-OCR2 model card "Tips to improve accuracy" section:
> "Increasing the image resolution will improve model's performance."

**Conclusion:** 300 DPI is the correct default for textbook processing. We removed the scale override (upstream 2.0 is appropriate).

### Token Limits

From `docstrange/pipeline/nanonets_processor.py`:
```python
def _extract_text_with_nanonets(self, image_path, max_new_tokens=4096):
```

The upstream default is 4096 tokens. Our code was using 8192.

From the **official Nanonets-OCR2-3B examples**:
```python
result = ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=15000)
```

And the vLLM example:
```python
response = client.chat.completions.create(
    ...
    max_tokens=15000
)
```

**15000 is the official recommendation.** Dense textbook pages with tables, lists, equations, and image descriptions can easily exceed 8K tokens. The old 8192 limit risked silent truncation.

### System Prompt

The model was fine-tuned with this system prompt:
```
You are a helpful assistant.
```

This is confirmed by:
1. `docstrange/pipeline/nanonets_processor.py` line in `_extract_text_with_nanonets`
2. The official usage examples on HuggingFace
3. The chat template in the model's tokenizer config

Our code was using a custom system prompt:
```
You are a helpful assistant specialized in extracting educational content from textbooks.
```

Adding domain-specific instructions in the system role can confuse fine-tuned VLMs because the model hasn't been trained with that system prompt. Custom instructions are better placed in the **user prompt** (which our `TEXTBOOK_PROMPT` and `SAFE_TEXTBOOK_PROMPT` already handle correctly).

### Official Usage Patterns

The official extraction prompt (used by both OCR-s and OCR2) is:

```
Extract the text from the above document as if you were reading it naturally.
Return the tables in html format. Return the equations in LaTeX representation.
If there is an image in the document and image caption is not present, add a
small description of the image inside the <img></img> tag; otherwise, add the
image caption inside <img></img>. Watermarks should be wrapped in brackets.
Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in
brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>.
Prefer using ☐ and ☑ for check boxes.
```

For **financial/table-heavy documents**, Nanonets recommends adding:
```
Only return HTML table within <table></table>.
```
And using `repetition_penalty=1` for complex tables.

Our `TEXTBOOK_PROMPT` and `SAFE_TEXTBOOK_PROMPT` extend this base prompt with textbook-specific instructions (headings hierarchy, figure captions, vocabulary formatting, etc.), which is the correct approach.

---

## Changes Applied

### Bug Fixes

1. **`_normalize_headings` no-op** — Replaced with stack-based heading level compression algorithm
2. **`_clean_output` destroying `<page_number>` tags** — Tags now preserved for downstream post-processor
3. **Dead `max_new_tokens or 4096` fallback** — Uses separate `_fallback_tokens` variable
4. **`detect_page_offset` fragility** — Collects all anchor offsets, returns mode with mismatch warnings
5. **Duplicate torch import** — Consolidated to single import block
6. **`str` vs `Path` type annotations** — Fixed to `str | Path`
7. **Unused `chunk_pages` parameter** — Removed from constructor, CLI, and main()
8. **Unused `processor_self` variable** — Removed from Optimization 2 block
9. **Silent failure for unsupported PDFs** — Added error message
10. **Redundant `book_name` computation** — Removed duplicate

### Model & Performance Optimizations

| Setting | Before | After | Rationale |
|---------|--------|-------|-----------|
| Model | `Nanonets-OCR-s` | `Nanonets-OCR2-3B` | 58% win rate on markdown benchmarks, same VRAM |
| Attention | `sdpa` | `flash_attention_2` (fallback: `sdpa`) | ~15-20% faster inference, officially recommended |
| DPI | 200 (overriding upstream 300) | 300 (upstream default) | "Increasing resolution improves performance" — Nanonets docs |
| `pdf_image_scale` | 2.5 (overriding upstream 2.0) | 2.0 (upstream default) | Upscaling after rasterization adds no real information |
| `max_new_tokens` | 8192 | 15000 | Official recommendation; prevents truncation on dense pages |
| System prompt | `"You are a helpful assistant specialized in extracting educational content from textbooks."` | `"You are a helpful assistant."` | Matches fine-tuning; domain instructions stay in user prompt |
| Metadata model string | `"nanonets/Nanonets-OCR-s (~4B VLM)"` | `"nanonets/Nanonets-OCR2-3B (Qwen2.5-VL-3B fine-tune)"` | Accurate |
| Version guard | None | Checks for `1.1.x`, warns on mismatch | Safety net for docstrange upgrades |
| GPU processor import | `docstrange.gpu_processor` only | Tries `docstrange.processors.gpu_processor` first (new path), falls back to old | Backwards compatible |

### Setup Script Improvements

1. **Safer uv install** — Downloads to temp file instead of piping to shell
2. **PATH persistence** — Appends to `.bashrc`
3. **Pre-downloads `Nanonets-OCR2-3B`** (~6 GB) via `huggingface_hub.snapshot_download`
4. **Installs `flash-attn`** if GPU is detected (for faster inference)
5. **Documents skipped scripts** and pre-cached models in setup summary

---

## Post-Review Validation

After the initial review and changes were applied, a manual validation pass was performed against the upstream docstrange source code. All claims were verified and two corrections were made.

### Validation Matrix

| Claim | Source | Verdict |
|-------|--------|---------|
| System prompt is `"You are a helpful assistant."` | `nanonets_processor.py` `_extract_text_with_nanonets` | Confirmed |
| `local_files_only=True` used upstream | `nanonets_processor.py` lines ~55-63 | Confirmed |
| Default `max_new_tokens=4096` upstream | `nanonets_processor.py` function signature | Confirmed |
| Official recommendation is `max_new_tokens=15000` | HuggingFace model card usage example | Confirmed |
| Official recommendation is `flash_attention_2` | HuggingFace model card usage example | Confirmed |
| `pdf_image_dpi=300`, `pdf_image_scale=2.0` upstream defaults | `config.py` `InternalConfig` class | Confirmed |
| `GPUProcessor` is at `docstrange.processors.gpu_processor` | Raw source fetched from GitHub | Confirmed |
| `_convert_pdf_to_images` uses `InternalConfig.pdf_image_dpi` | `gpu_processor.py` | Confirmed |
| docstrange version is 1.1.8 | `pyproject.toml` | Confirmed |
| docstrange now uses a 7B model by default (Aug 2025 README) | GitHub README "What's New" section | **New finding** |

### Corrections Applied Post-Validation

1. **GPU processor import was backwards** — The initial fix tried `docstrange.gpu_processor` first (old path, returns 404 on GitHub) and fell back to `docstrange.processors.gpu_processor` (correct path, confirmed from upstream source). Fixed: now imports from `docstrange.processors.gpu_processor` as the primary path.

2. **Missing `OSError` in `flash_attention_2` except clause** — If the model isn't cached locally but `flash-attn` IS installed, `from_pretrained(..., local_files_only=True)` raises `OSError`, not `ImportError` or `ValueError`. The except clause now catches `(ImportError, ValueError, OSError)` so it correctly falls through to the SDPA path.

### Notable Upstream Change: 7B Model Default

The docstrange README states the core model was upgraded to **7B parameters** in August 2025. This has the following implications:

- Our monkey-patch overrides the model to `Nanonets-OCR2-3B` (3B) regardless of what docstrange bundles, so this doesn't break anything.
- However, on newer docstrange installs we may be **downgrading** from the upstream 7B model to our 3B model.
- It is unclear whether the 7B model is the cloud-only `Nanonets-OCR2-Plus` or a new open-weight model.
- **Action item:** On next GPU run, check what model docstrange loads by default (without our patch) to determine if the 7B model is available for local inference. If it is, we should evaluate whether to use it instead of OCR2-3B, weighing quality improvement vs VRAM cost.

---

## Testing

To test the model output quality on the sample PDF:

```bash
# From the project root on your RunPod instance:

# Test with textbook prompt (default)
python run_docstrange_textbook.py BATES_sample_306-324.pdf results/sample_test

# Test with safe/anti-hallucination prompt
python run_docstrange_textbook.py BATES_sample_306-324.pdf results/sample_test_safe --safe
```

Compare `results/sample_test/BATES_sample_306-324.md` vs `results/sample_test_safe/BATES_sample_306-324.md` to evaluate prompt quality.

First run will download the model (~6 GB) unless `setup_runpod.sh` was run first.
