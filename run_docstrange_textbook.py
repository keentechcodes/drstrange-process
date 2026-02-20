#!/usr/bin/env python3
"""
DocStrange: Optimized PDF-to-Markdown converter for TEXTBOOKS
https://github.com/NanoNets/docstrange

Enhanced version for multimodal textbook processing with:
- Higher DPI for diagram/chart clarity
- Increased token limits for dense pages
- Custom OCR prompt for better hierarchy preservation
- Post-processing for vector DB optimization
- Multiple output formats (markdown, JSON, HTML)

Usage:
    .venvs/docstrange/bin/python run_docstrange_textbook.py <pdf> [output_dir] [options]

Options:
    --format=markdown|json|html   Output format (default: markdown)
    --dpi=200|300                 PDF rasterization DPI (default: 200)
    --chunk-pages=N               Split output every N pages (default: 50)
    --preserve-hierarchy          Normalize heading levels across document
"""

import sys
import time
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


class TextbookProcessor:
    """Enhanced processor for multimodal textbook conversion."""

    DEFAULT_PROMPT = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""

    TEXTBOOK_PROMPT = """Extract the text from this textbook page preserving the document hierarchy and structure:

1. HEADINGS: Use proper markdown heading levels (# for chapter titles, ## for sections, ### for subsections)
2. TABLES: Return in HTML <table> format with proper <thead> and <tbody>
3. EQUATIONS: Return in LaTeX format with $$ for display and $ for inline
4. IMAGES: Describe educational images/diagrams in <img>detailed description of what the image shows, including labels, diagrams, charts, and educational content</img>
5. FIGURES: Preserve figure captions and numbers (e.g., "Figure 15-1: Anatomy of...")
6. LISTS: Preserve numbered and bulleted lists with proper indentation
7. SIDEBARS/CALLOUTS: Wrap in <callout>...</callout> tags
8. PAGE NUMBERS: Wrap in <page_number>123</page_number>
9. VOCABULARY/DEFINITIONS: Format as **Term**: Definition
10. MAINTAIN reading order and logical flow of educational content"""

    SAFE_TEXTBOOK_PROMPT = """Extract the text from this textbook page exactly as written. Do not hallucinate or add content that is not visible in the image.

Guidelines:
- TABLES: If you can read the table cells, output them in HTML <table> format. If cells are unreadable, use empty cells rather than guessing content.
- HEADINGS: Use markdown heading levels (# ## ###) based on visual hierarchy
- LISTS: Preserve bullet and numbered lists exactly as formatted
- IMAGES: Briefly describe what is shown in <img>brief description</img> tags. Do not invent details.
- PAGE NUMBERS: Wrap in <page_number>N</page_number>
- Only output text that is actually visible and readable in the document."""

    def __init__(
        self,
        dpi: int = 200,
        max_tokens: int = 8192,
        use_textbook_prompt: bool = True,
        use_safe_prompt: bool = False,
        preserve_hierarchy: bool = True,
        output_format: str = "markdown",
        chunk_pages: int = 50,
    ):
        self.dpi = dpi
        self.max_tokens = max_tokens
        self.use_textbook_prompt = use_textbook_prompt
        self.use_safe_prompt = use_safe_prompt
        self.preserve_hierarchy = preserve_hierarchy
        self.output_format = output_format
        self.chunk_pages = chunk_pages
        self.extractor = None
        self._page_counter = {"current": 0, "total": 0}

    def _apply_optimizations(self) -> List[str]:
        """Apply textbook-specific optimizations to docstrange."""
        optimizations = []

        # Optimization 1: Higher DPI for textbook quality
        try:
            from docstrange.config import InternalConfig

            InternalConfig.pdf_image_dpi = self.dpi
            InternalConfig.pdf_image_scale = 2.5
            optimizations.append(f"DPI={self.dpi}")
        except (ImportError, AttributeError):
            pass

        # Optimization 2: SDPA attention + custom prompt + higher token limit
        try:
            import docstrange.pipeline.nanonets_processor as np_module

            _original_init = np_module.NanonetsDocumentProcessor._initialize_models
            processor_self = self

            def _patched_init(self, cache_dir=None):
                from transformers import (
                    AutoTokenizer,
                    AutoProcessor,
                    AutoModelForImageTextToText,
                )
                from docstrange.pipeline.model_downloader import ModelDownloader

                model_downloader = ModelDownloader(cache_dir)
                model_path = model_downloader.get_model_path("nanonets-ocr")
                actual_model_path = model_path / "Nanonets-OCR-ss"
                if not actual_model_path.exists():
                    actual_model_path = model_path

                self.model = AutoModelForImageTextToText.from_pretrained(
                    str(actual_model_path),
                    torch_dtype="auto",
                    device_map="auto",
                    local_files_only=True,
                    attn_implementation="sdpa",
                )
                self.model.eval()

                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(actual_model_path), local_files_only=True
                )
                self.processor = AutoProcessor.from_pretrained(
                    str(actual_model_path), local_files_only=True
                )

            np_module.NanonetsDocumentProcessor._initialize_models = _patched_init
            optimizations.append("SDPA attention")
        except (ImportError, AttributeError) as e:
            print(f"[textbook] SDPA patch failed: {e}")

        # Optimization 3: Custom textbook prompt + verbose logging
        try:
            import docstrange.pipeline.nanonets_processor as np_module

            _original_extract = (
                np_module.NanonetsDocumentProcessor._extract_text_with_nanonets
            )
            processor_self = self
            if self.use_safe_prompt:
                prompt_to_use = self.SAFE_TEXTBOOK_PROMPT
            elif self.use_textbook_prompt:
                prompt_to_use = self.TEXTBOOK_PROMPT
            else:
                prompt_to_use = self.DEFAULT_PROMPT
            max_tokens = self.max_tokens

            def _enhanced_extract(self, image_path, max_new_tokens=None):
                import time as _time

                max_new_tokens = max_tokens

                processor_self._page_counter["current"] += 1
                n = processor_self._page_counter["current"]
                total = processor_self._page_counter["total"]

                print(f"[textbook]   Page {n}/{total}...", end="", flush=True)
                t = _time.time()

                try:
                    from PIL import Image

                    image = Image.open(image_path)
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant specialized in extracting educational content from textbooks.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f"file://{image_path}"},
                                {"type": "text", "text": prompt_to_use},
                            ],
                        },
                    ]

                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self.processor(
                        text=[text], images=[image], padding=True, return_tensors="pt"
                    )
                    inputs = inputs.to(self.model.device)

                    output_ids = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False
                    )
                    generated_ids = [
                        output_ids[len(input_ids) :]
                        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                    ]

                    output_text = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    result = output_text[0]
                except Exception as e:
                    print(f" ERROR: {e}")
                    return _original_extract(self, image_path, max_new_tokens or 4096)

                elapsed = _time.time() - t
                chars = len(result) if result else 0
                print(f" done ({elapsed:.1f}s, {chars:,} chars)")
                return result

            np_module.NanonetsDocumentProcessor._extract_text_with_nanonets = (
                _enhanced_extract
            )
            optimizations.append(f"tokens={max_tokens}")
            if self.use_safe_prompt:
                optimizations.append("safe_prompt")
            elif self.use_textbook_prompt:
                optimizations.append("textbook_prompt")
            else:
                optimizations.append("default_prompt")
        except (ImportError, AttributeError) as e:
            print(f"[textbook] Enhanced extract patch failed: {e}")

        # Optimization 4: Page counter setup
        try:
            import docstrange.gpu_processor as gpu_mod

            _original_convert = gpu_mod.GPUProcessor._convert_pdf_to_images
            processor_self = self

            def _counting_convert(self, pdf_path):
                result = _original_convert(self, pdf_path)
                total = len(result) if result else 0
                processor_self._page_counter["total"] = total
                processor_self._page_counter["current"] = 0
                print(f"[textbook] Rasterized {total} pages (DPI={processor_self.dpi})")
                return result

            gpu_mod.GPUProcessor._convert_pdf_to_images = _counting_convert
        except (ImportError, AttributeError):
            pass

        return optimizations

    def _extract_images_from_pdf(
        self, pdf_path: str, images_dir: str, min_size: int = 100
    ) -> Dict[int, List[Dict]]:
        """Extract embedded images from PDF for vector DB multimodal support."""
        import fitz

        images_dir = Path(images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        page_images = {}
        seen_xrefs = set()

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            page_images[page_num + 1] = []

            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    base_image = doc.extract_image(xref)
                    width = base_image["width"]
                    height = base_image["height"]

                    if width < min_size or height < min_size:
                        continue

                    image_ext = base_image["ext"]
                    image_filename = f"page{page_num + 1}_img{img_idx + 1}.{image_ext}"
                    image_path = images_dir / image_filename

                    with open(image_path, "wb") as f:
                        f.write(base_image["image"])

                    page_images[page_num + 1].append(
                        {
                            "filename": image_filename,
                            "width": width,
                            "height": height,
                        }
                    )
                except Exception as e:
                    print(
                        f"  [warn] Image extraction failed (page {page_num + 1}): {e}"
                    )

        doc.close()
        return page_images

    def _replace_img_tags(
        self, markdown: str, page_images: Dict[int, List[Dict]]
    ) -> Tuple[str, int]:
        """Replace <img> tags with markdown image refs."""
        img_tag_pattern = re.compile(r"<img>(.*?)</img>", re.DOTALL)
        page_header_pattern = re.compile(r"^## Page (\d+)", re.MULTILINE)

        page_splits = list(page_header_pattern.finditer(markdown))
        if not page_splits:
            page_splits_ranges = [(1, 0, len(markdown))]
        else:
            page_splits_ranges = []
            for i, m in enumerate(page_splits):
                page_num = int(m.group(1))
                start = m.end()
                end = (
                    page_splits[i + 1].start()
                    if i + 1 < len(page_splits)
                    else len(markdown)
                )
                page_splits_ranges.append((page_num, start, end))

        replacements = []
        matched = 0

        for page_num, start, end in page_splits_ranges:
            page_content = markdown[start:end]
            extracted = page_images.get(page_num, [])

            for img_idx, img_match in enumerate(img_tag_pattern.finditer(page_content)):
                description = img_match.group(1).strip()
                abs_start = start + img_match.start()
                abs_end = start + img_match.end()

                if img_idx < len(extracted):
                    img_info = extracted[img_idx]
                    replacement = f"![{description}](images/{img_info['filename']})"
                    matched += 1
                else:
                    replacement = f"*[Image: {description}]*"

                replacements.append((abs_start, abs_end, replacement))

        result = markdown
        for abs_start, abs_end, replacement in reversed(replacements):
            result = result[:abs_start] + replacement + result[abs_end:]

        return result, matched

    def _normalize_headings(self, text: str) -> str:
        """Normalize heading hierarchy across the document."""
        if not self.preserve_hierarchy:
            return text

        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        lines = text.split("\n")
        result = []

        for line in lines:
            match = heading_pattern.match(line)
            if match:
                level = len(match.group(1))
                content = match.group(2)

                # Normalize: ensure no level jumps > 2 (e.g., # -> ### is okay, # -> #### isn't)
                result.append(f"{'#' * level} {content}")
            else:
                result.append(line)

        return "\n".join(result)

    def _clean_output(self, text: str) -> str:
        """Clean and normalize output for vector DB ingestion."""
        # Remove excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Clean up page number tags
        text = re.sub(
            r"<page_number>(\d+)</page_number>", r"\n<!-- Page \1 -->\n", text
        )

        # Clean watermark tags
        text = re.sub(r"<watermark>([^<]+)</watermark>", r"", text)

        return text.strip()

    def convert(
        self, pdf_path: str, output_dir: str
    ) -> Tuple[Optional[Path], Dict[str, Any]]:
        """Convert textbook PDF to optimized format for vector DB."""
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[textbook] Converting: {pdf_path.name}")
        print(f"[textbook] Output: {output_dir}")
        print(f"[textbook] Format: {self.output_format}, DPI: {self.dpi}")

        t0 = time.time()

        try:
            import torch

            print(f"[textbook] CUDA available: {torch.cuda.is_available()}")
        except ImportError:
            pass

        try:
            from docstrange import DocumentExtractor
        except ImportError as e:
            print(f"[textbook] ERROR: Failed to import docstrange: {e}")
            return None, {"error": str(e)}

        opts = self._apply_optimizations()
        if opts:
            print(f"[textbook] Optimizations: {', '.join(opts)}")

        use_gpu = False
        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except ImportError:
            pass

        mode = "GPU local" if use_gpu else "cloud API"
        print(f"[textbook] Mode: {mode}")
        print(f"[textbook] Loading extractor...")

        try:
            self.extractor = (
                DocumentExtractor(gpu=True) if use_gpu else DocumentExtractor()
            )
        except Exception as e:
            print(f"[textbook] ERROR: Failed to create extractor: {e}")
            return None, {"error": str(e)}

        model_time = time.time() - t0
        print(f"[textbook] Extractor ready in {model_time:.1f}s")

        print("[textbook] Processing PDF...")
        t1 = time.time()

        try:
            result = self.extractor.extract(str(pdf_path))

            if self.output_format == "json":
                text = json.dumps(result.extract_data(), indent=2, ensure_ascii=False)
                output_ext = ".json"
            elif self.output_format == "html":
                text = result.extract_html()
                output_ext = ".html"
            else:
                text = result.extract_markdown()
                output_ext = ".md"

        except Exception as e:
            print(f"[textbook] ERROR: Extraction failed: {e}")
            return None, {"error": str(e)}

        convert_time = time.time() - t1

        if not text:
            print("[textbook] WARNING: No text extracted")
            return None, {"error": "No text extracted"}

        # Post-processing
        if self.preserve_hierarchy and self.output_format == "markdown":
            text = self._normalize_headings(text)

        text = self._clean_output(text)

        # Extract and match images
        images_extracted = 0
        images_matched = 0
        try:
            images_dir = str(output_dir / "images")
            print("[textbook] Extracting images...")
            page_images = self._extract_images_from_pdf(str(pdf_path), images_dir)
            images_extracted = sum(len(imgs) for imgs in page_images.values())

            if images_extracted > 0 and self.output_format == "markdown":
                text, images_matched = self._replace_img_tags(text, page_images)
                print(f"[textbook] Matched {images_matched}/{images_extracted} images")
        except Exception as e:
            print(f"[textbook] Image extraction failed (non-fatal): {e}")

        # Save output
        output_path = output_dir / f"{pdf_path.stem}{output_ext}"
        output_path.write_text(text, encoding="utf-8")

        total_time = model_time + convert_time

        # Metadata
        meta = {
            "tool": "docstrange-textbook",
            "model": "nanonets/Nanonets-OCR-s (~4B VLM)",
            "mode": mode,
            "output_format": self.output_format,
            "dpi": self.dpi,
            "max_tokens": self.max_tokens,
            "preserve_hierarchy": self.preserve_hierarchy,
            "model_load_time": round(model_time, 2),
            "conversion_time": round(convert_time, 2),
            "total_time": round(total_time, 2),
            "output_chars": len(text),
            "output_words": len(text.split()) if text else 0,
            "images_extracted": images_extracted,
            "images_matched": images_matched,
            "pages_processed": self._page_counter["total"],
        }

        (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        print(f"[textbook] Output: {output_path}")
        print(
            f"[textbook] {len(text):,} chars | {len(text.split()):,} words | {total_time:.1f}s"
        )

        return output_path, meta


def main():
    parser = argparse.ArgumentParser(
        description="Convert textbook PDF to optimized format for vector DB"
    )
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument(
        "output_dir", nargs="?", default="results/textbook", help="Output directory"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json", "html"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        choices=[150, 200, 300],
        default=200,
        help="PDF rasterization DPI (default: 200)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=8192,
        help="Max output tokens per page (default: 8192)",
    )
    parser.add_argument(
        "--no-hierarchy", action="store_true", help="Disable heading normalization"
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Use safe prompt with anti-hallucination guardrails (recommended for tables)",
    )
    parser.add_argument(
        "--chunk-pages",
        type=int,
        default=50,
        help="Split output every N pages (future feature)",
    )

    args = parser.parse_args()

    processor = TextbookProcessor(
        dpi=args.dpi,
        max_tokens=args.tokens,
        use_textbook_prompt=not args.safe,
        use_safe_prompt=args.safe,
        preserve_hierarchy=not args.no_hierarchy,
        output_format=args.format,
        chunk_pages=args.chunk_pages,
    )

    output_path, meta = processor.convert(args.pdf, args.output_dir)

    if output_path:
        print(f"\n[textbook] SUCCESS: {output_path}")
        sys.exit(0)
    else:
        print(f"\n[textbook] FAILED: {meta.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
