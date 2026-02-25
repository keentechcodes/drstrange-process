#!/usr/bin/env python3
"""run_chunked_ocr.py — Run DocStrange OCR on large PDFs in page-range chunks.

Splits a large PDF into smaller chunks (default 500 pages), runs the converter
on each chunk (optionally in parallel), then concatenates all markdown output
into a single file.  This works around DocStrange loading all page images into
RAM at once.

Usage:
    python run_chunked_ocr.py HARRISONS21ST.pdf results_harrisons --meta-file harrisons21st.meta.json
    python run_chunked_ocr.py HARRISONS21ST.pdf results_harrisons --meta-file harrisons21st.meta.json --chunk-size 300
    python run_chunked_ocr.py HARRISONS21ST.pdf results_harrisons --meta-file harrisons21st.meta.json --parallel 2

The --parallel flag processes multiple chunks concurrently. Useful on GPUs with
plenty of VRAM (each parallel worker loads the OCR model ~6GB).

The output structure matches what run_docstrange_textbook.py produces:
    results_harrisons/HARRISONS21ST.md          (concatenated markdown)
    results_harrisons/metadata.json              (combined metadata)
    results_harrisons/images/                    (all extracted images)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path

import fitz  # PyMuPDF


DEFAULT_CHUNK_SIZE = 500  # pages per chunk
DEFAULT_PARALLEL = 1  # number of chunks to process concurrently


def split_pdf(
    pdf_path: Path,
    output_dir: Path,
    start_page: int,
    end_page: int,
    chunk_size: int,
) -> list[dict]:
    """Split a PDF into page-range chunks.

    Args:
        pdf_path: Source PDF path.
        output_dir: Directory for chunk PDFs.
        start_page: First PDF page (1-indexed) to include.
        end_page: Last PDF page (1-indexed) to include.
        chunk_size: Max pages per chunk.

    Returns:
        List of dicts with chunk info: path, pdf_start, pdf_end, chunk_num.
    """
    doc = fitz.open(str(pdf_path))
    total = len(doc)

    # Clamp to valid range
    start_page = max(1, start_page)
    end_page = min(total, end_page)

    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    chunk_num = 0

    for chunk_start in range(start_page, end_page + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, end_page)
        chunk_num += 1

        chunk_filename = f"chunk_{chunk_num:03d}_p{chunk_start}-{chunk_end}.pdf"
        chunk_path = chunks_dir / chunk_filename

        # PyMuPDF uses 0-indexed pages
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=chunk_start - 1, to_page=chunk_end - 1)
        chunk_doc.save(str(chunk_path))
        chunk_doc.close()

        chunks.append(
            {
                "chunk_num": chunk_num,
                "path": str(chunk_path),
                "pdf_start": chunk_start,
                "pdf_end": chunk_end,
                "pages": chunk_end - chunk_start + 1,
            }
        )

        print(
            f"[chunked-ocr]   Chunk {chunk_num}: pages {chunk_start}-{chunk_end} ({chunks[-1]['pages']} pages) → {chunk_filename}"
        )

    doc.close()
    print(f"[chunked-ocr] Split into {len(chunks)} chunks")
    return chunks


def create_chunk_meta(
    original_meta: dict,
    chunk: dict,
) -> dict:
    """Create a meta.json for a single chunk.

    The converter uses content_pdf_page_start/end to filter pages and for
    its own page renumbering (offset = content_pdf_page_start - 1).  With
    content_pdf_page_start=1, the converter outputs ## Page 1, 2, 3 ...
    (chunk-relative).  The caller is responsible for shifting these to
    book-relative page numbers via ``renumber_chunk_pages()`` after the
    converter finishes.
    """
    chunk_pages = chunk["pages"]

    return {
        "book": original_meta.get("book", "unknown"),
        "source_pdf": Path(chunk["path"]).name,
        "total_pdf_pages": chunk_pages,
        "page_offset": 0,
        "content_pdf_page_start": 1,
        "content_pdf_page_end": chunk_pages,
        "generated_at": original_meta.get("generated_at", ""),
        # Track original page range for debugging
        "_original_pdf_start": chunk["pdf_start"],
        "_original_pdf_end": chunk["pdf_end"],
    }


def run_converter(
    chunk_pdf: Path,
    output_dir: Path,
    meta_path: Path,
    extra_args: list[str] | None = None,
) -> tuple[str | None, float]:
    """Run run_docstrange_textbook.py on a single chunk.

    Returns (markdown_text, elapsed_seconds).
    """
    cmd = [
        sys.executable,
        "run_docstrange_textbook.py",
        str(chunk_pdf),
        str(output_dir),
        f"--meta-file={meta_path}",
    ]
    if extra_args:
        cmd.extend(extra_args)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[chunked-ocr] WARNING: Converter exited with code {result.returncode}")
        return None, elapsed

    # Find the output markdown
    md_files = list(output_dir.glob("*.md"))
    if not md_files:
        return None, elapsed

    text = md_files[0].read_text(encoding="utf-8")
    return text, elapsed


def renumber_chunk_pages(
    md_text: str,
    original_pdf_start: int,
    original_offset: int,
) -> str:
    """Re-number ``## Page N`` headers from chunk-relative to book-relative.

    The converter produces chunk-relative page numbers (``## Page 1`` is the
    first page of the chunk).  This function shifts them so that
    ``## Page 1`` in a chunk starting at original PDF page *S* with book
    offset *O* becomes ``## Page (S - O)``.

    Example — Harrison's chunk 2, original PDF pages 542–1041, offset 41:
        ## Page 1  →  ## Page 501   (542 - 41 = 501)
        ## Page 2  →  ## Page 502
    """
    book_page_of_first_chunk_page = original_pdf_start - original_offset
    # chunk converter outputs ## Page 1 for the first page, so shift is
    # (book_page - 1).  new_num = old_num + shift.
    shift = book_page_of_first_chunk_page - 1

    if shift == 0:
        return md_text  # chunk 1 for content starting at page 1: no shift needed

    def _shift(match: re.Match) -> str:
        old_num = int(match.group(1))
        return f"## Page {old_num + shift}"

    return re.sub(r"^## Page (\d+)", _shift, md_text, flags=re.MULTILINE)


def concatenate_markdown(
    chunk_markdowns: list[tuple[int, str]],
) -> str:
    """Concatenate chunk markdowns into a single document.

    Each chunk's markdown has already been renumbered to book page numbers.
    """
    parts = []
    for chunk_num, md in chunk_markdowns:
        if md:
            parts.append(md.rstrip())

    return "\n\n".join(parts) + "\n"


def process_single_chunk(
    chunk: dict,
    output_dir: Path,
    original_meta: dict,
    original_offset: int,
    extra_args: list[str],
    start_chunk: int,
) -> tuple[int, str | None, float]:
    """Process a single chunk - designed to run in parallel.

    Returns (chunk_num, markdown_text or None, elapsed_seconds).
    """
    chunk_num = chunk["chunk_num"]

    if chunk_num < start_chunk:
        print(f"[chunked-ocr] Chunk {chunk_num}: skipping (before --start-chunk)")
        chunk_output_dir = output_dir / f"chunk_{chunk_num:03d}"
        md_files = (
            list(chunk_output_dir.glob("*.md")) if chunk_output_dir.exists() else []
        )
        if md_files:
            cached_md = md_files[0].read_text(encoding="utf-8")
            cached_md = renumber_chunk_pages(
                cached_md, chunk["pdf_start"], original_offset
            )
            return (chunk_num, cached_md, 0.0)
        return (chunk_num, None, 0.0)

    print(
        f"[chunked-ocr] Chunk {chunk_num}: starting (pages {chunk['pdf_start']}-{chunk['pdf_end']})"
    )

    chunk_output_dir = output_dir / f"chunk_{chunk_num:03d}"
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    chunk_meta = create_chunk_meta(original_meta, chunk)
    chunk_meta_path = chunk_output_dir / "meta.json"
    chunk_meta_path.write_text(json.dumps(chunk_meta, indent=2))

    t0 = time.time()
    md_text, elapsed = run_converter(
        Path(chunk["path"]),
        chunk_output_dir,
        chunk_meta_path,
        extra_args=extra_args,
    )
    total_elapsed = time.time() - t0

    if md_text:
        md_text = renumber_chunk_pages(md_text, chunk["pdf_start"], original_offset)
        print(f"[chunked-ocr] Chunk {chunk_num}: done in {total_elapsed:.0f}s")
        return (chunk_num, md_text, total_elapsed)
    else:
        print(f"[chunked-ocr] Chunk {chunk_num}: WARNING - no output")
        return (chunk_num, None, total_elapsed)


def main():
    parser = argparse.ArgumentParser(
        description="Run DocStrange OCR on large PDFs in page-range chunks.",
    )
    parser.add_argument("pdf", type=Path, help="Source PDF path")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--meta-file",
        type=Path,
        required=True,
        help="Path to meta.json from extract_toc.py",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Pages per chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--start-chunk",
        type=int,
        default=1,
        help="Resume from this chunk number (1-indexed, for crash recovery)",
    )
    parser.add_argument(
        "--converter-args",
        nargs="*",
        default=[],
        help="Extra args to pass to run_docstrange_textbook.py (e.g. --safe --dpi=200)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL,
        help=f"Number of chunks to process in parallel (default: {DEFAULT_PARALLEL})",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"[chunked-ocr] ERROR: PDF not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)
    if not args.meta_file.exists():
        print(
            f"[chunked-ocr] ERROR: Meta file not found: {args.meta_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load original meta
    original_meta = json.loads(args.meta_file.read_text())
    content_start = original_meta.get("content_pdf_page_start", 1)
    content_end = original_meta.get("content_pdf_page_end", 999999)
    original_offset = original_meta.get("page_offset", 0)

    print(f"[chunked-ocr] PDF: {args.pdf} ({args.pdf.stat().st_size / 1e6:.0f} MB)")
    print(f"[chunked-ocr] Content pages: PDF {content_start}-{content_end}")
    print(f"[chunked-ocr] Chunk size: {args.chunk_size} pages")

    # Phase 1: Split PDF
    print(f"\n[chunked-ocr] === Phase 1: Splitting PDF ===")
    t0 = time.time()
    chunks = split_pdf(
        args.pdf, args.output_dir, content_start, content_end, args.chunk_size
    )
    split_time = time.time() - t0
    print(f"[chunked-ocr] Split completed in {split_time:.1f}s")

    # Phase 2: Run converter on each chunk (potentially in parallel)
    print(
        f"\n[chunked-ocr] === Phase 2: Running OCR ({len(chunks)} chunks, parallel={args.parallel}) ==="
    )
    chunk_markdowns: list[tuple[int, str]] = []
    total_elapsed = 0.0

    # Filter chunks to process (respect --start-chunk)
    chunks_to_process = [c for c in chunks if c["chunk_num"] >= args.start_chunk]
    chunks_skipped = [c for c in chunks if c["chunk_num"] < args.start_chunk]

    # Load skipped chunks from cache
    for chunk in chunks_skipped:
        chunk_num = chunk["chunk_num"]
        print(f"[chunked-ocr] Chunk {chunk_num}: skipping (before --start-chunk)")
        chunk_output_dir = args.output_dir / f"chunk_{chunk_num:03d}"
        md_files = (
            list(chunk_output_dir.glob("*.md")) if chunk_output_dir.exists() else []
        )
        if md_files:
            cached_md = md_files[0].read_text(encoding="utf-8")
            cached_md = renumber_chunk_pages(
                cached_md, chunk["pdf_start"], original_offset
            )
            chunk_markdowns.append((chunk_num, cached_md))

    if args.parallel > 1:
        # Parallel processing using ProcessPoolExecutor
        # Each worker needs its own process, so we use spawn
        # Note: Each chunk loads the OCR model (~6GB VRAM), so parallel workers
        # compete for GPU memory. Start with --parallel=2 and monitor VRAM.
        max_workers = min(args.parallel, len(chunks_to_process))
        print(f"[chunked-ocr] Running {max_workers} chunks in parallel...")

        # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid CUDA fork issues
        # Threads share the same GPU context, which is fine for this use case
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_chunk,
                    chunk,
                    args.output_dir,
                    original_meta,
                    original_offset,
                    args.converter_args,
                    args.start_chunk,
                ): chunk
                for chunk in chunks_to_process
            }

            for future in as_completed(futures):
                chunk_num, md_text, elapsed = future.result()
                total_elapsed += elapsed
                if md_text:
                    chunk_markdowns.append((chunk_num, md_text))

                # Print progress
                done = len(chunk_markdowns)
                total = len(chunks)
                total_pages = sum(c["pages"] for c in chunks)
                pages_done = sum(
                    c["pages"] for c in chunks if c["chunk_num"] <= chunk_num
                )
                rate = pages_done / total_elapsed if total_elapsed > 0 else 0
                eta_remaining = (total_pages - pages_done) / rate if rate > 0 else 0
                print(
                    f"[chunked-ocr] Progress: {done}/{total} chunks, "
                    f"{pages_done}/{total_pages} pages, "
                    f"ETA: {eta_remaining / 3600:.1f}h"
                )
    else:
        # Sequential processing (original behavior)
        for chunk in chunks_to_process:
            chunk_num, md_text, elapsed = process_single_chunk(
                chunk,
                args.output_dir,
                original_meta,
                original_offset,
                args.converter_args,
                args.start_chunk,
            )
            total_elapsed += elapsed
            if md_text:
                chunk_markdowns.append((chunk_num, md_text))

            pages_done = sum(c["pages"] for c in chunks_to_process[:chunk_num])
            total_pages = sum(c["pages"] for c in chunks)
            rate = pages_done / total_elapsed if total_elapsed > 0 else 0
            eta_remaining = (total_pages - pages_done) / rate if rate > 0 else 0
            print(
                f"[chunked-ocr] Chunk {chunk_num} done in {elapsed:.0f}s "
                f"({pages_done}/{total_pages} pages, "
                f"ETA: {eta_remaining / 3600:.1f}h remaining)"
            )

    # Phase 3: Concatenate
    print(
        f"\n[chunked-ocr] === Phase 3: Concatenating {len(chunk_markdowns)} chunks ==="
    )
    if chunk_markdowns:
        final_md = concatenate_markdown(chunk_markdowns)
        stem = args.pdf.stem
        output_path = args.output_dir / f"{stem}.md"
        output_path.write_text(final_md, encoding="utf-8")
        print(f"[chunked-ocr] Output: {output_path} ({len(final_md):,} chars)")
    else:
        print("[chunked-ocr] ERROR: No markdown output from any chunk")
        sys.exit(1)

    # Phase 4: Combine images
    print(f"\n[chunked-ocr] === Phase 4: Collecting images ===")
    images_dir = args.output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    image_count = 0
    for chunk in chunks:
        chunk_num = chunk["chunk_num"]
        chunk_images_dir = args.output_dir / f"chunk_{chunk_num:03d}" / "images"
        if chunk_images_dir.exists():
            for img in chunk_images_dir.iterdir():
                if img.is_file():
                    # Prefix with chunk number to avoid collisions across chunks
                    target = images_dir / f"c{chunk_num:03d}_{img.name}"
                    shutil.copy2(str(img), str(target))
                    image_count += 1
    print(f"[chunked-ocr] Collected {image_count} images")

    # Summary
    total_pages = sum(c["pages"] for c in chunks)
    print(f"\n[chunked-ocr] === Summary ===")
    print(f"  Chunks: {len(chunks)} ({args.chunk_size} pages each)")
    print(f"  Pages processed: {total_pages}")
    print(f"  Total OCR time: {total_elapsed:.0f}s ({total_elapsed / 3600:.1f}h)")
    print(
        f"  Rate: {total_pages / total_elapsed:.1f} pages/s"
        if total_elapsed > 0
        else ""
    )
    print(f"  Output: {args.output_dir / f'{args.pdf.stem}.md'}")
    print(f"  Images: {image_count}")

    # Save combined metadata
    combined_meta = {
        "tool": "chunked-ocr",
        "source_pdf": args.pdf.name,
        "total_pages": total_pages,
        "chunks": len(chunks),
        "chunk_size": args.chunk_size,
        "total_ocr_time": round(total_elapsed, 2),
        "output_chars": len(final_md),
        "images_collected": image_count,
    }
    meta_out = args.output_dir / "metadata.json"
    meta_out.write_text(json.dumps(combined_meta, indent=2))
    print(f"[chunked-ocr] Metadata: {meta_out}")


if __name__ == "__main__":
    main()
