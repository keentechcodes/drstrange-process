#!/usr/bin/env python3
"""
Analyze BATES.md file to extract per-page statistics.
"""

import re
import statistics
from collections import defaultdict


def parse_pages(filepath):
    """Parse the markdown file and extract pages with their content."""
    pages = []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by page headers: both "# Page N" and "## Page N"
    # Pattern matches both formats
    pattern = r"(?:^|\n)(#{1,2}\s+Page\s+(\d+))\s*\n"

    matches = list(re.finditer(pattern, content))

    for i, match in enumerate(matches):
        page_num = match.group(2)
        start_pos = match.end()

        # Get content until next page header or end of file
        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)

        page_content = content[start_pos:end_pos]

        # Calculate stats (excluding the page header line itself)
        char_count = len(page_content)
        line_count = page_content.count("\n") + (
            1 if page_content and not page_content.endswith("\n") else 0
        )

        # Get first 80 characters of content after the header (excluding newlines at start)
        clean_content = page_content.lstrip("\n")
        first_80 = clean_content[:80].replace("\n", " ")

        pages.append(
            {
                "page_num": page_num,
                "char_count": char_count,
                "line_count": line_count,
                "first_80": first_80,
                "content": page_content,
            }
        )

    return pages


def calculate_percentile(data, percentile):
    """Calculate the given percentile for a list of numbers."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    index = (percentile / 100) * (n - 1)

    if index.is_integer():
        return sorted_data[int(index)]
    else:
        lower = int(index)
        upper = lower + 1
        fraction = index - lower
        return sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])


def analyze_pages(pages):
    """Analyze pages and generate statistics."""
    char_counts = [p["char_count"] for p in pages]
    line_counts = [p["line_count"] for p in pages]

    # Calculate distribution stats for characters
    char_stats = {
        "mean": statistics.mean(char_counts),
        "median": statistics.median(char_counts),
        "p25": calculate_percentile(char_counts, 25),
        "p75": calculate_percentile(char_counts, 75),
        "p90": calculate_percentile(char_counts, 90),
        "p95": calculate_percentile(char_counts, 95),
        "p99": calculate_percentile(char_counts, 99),
        "max": max(char_counts),
    }

    # Calculate distribution stats for lines
    line_stats = {
        "mean": statistics.mean(line_counts),
        "median": statistics.median(line_counts),
        "p25": calculate_percentile(line_counts, 25),
        "p75": calculate_percentile(line_counts, 75),
        "p90": calculate_percentile(line_counts, 90),
        "p95": calculate_percentile(line_counts, 95),
        "p99": calculate_percentile(line_counts, 99),
        "max": max(line_counts),
    }

    # Find pages above 95th percentile for character count
    p95_chars = char_stats["p95"]
    above_p95 = [
        (p["page_num"], p["char_count"]) for p in pages if p["char_count"] > p95_chars
    ]

    # Find pages below 100 characters
    below_100 = [
        (p["page_num"], p["char_count"]) for p in pages if p["char_count"] < 100
    ]

    # Get top 20 largest pages
    top_20 = sorted(pages, key=lambda x: x["char_count"], reverse=True)[:20]

    # Bucket counts for character ranges
    buckets = {
        "0-500": 0,
        "500-1000": 0,
        "1000-2000": 0,
        "2000-5000": 0,
        "5000-10000": 0,
        "10000+": 0,
    }

    for p in pages:
        chars = p["char_count"]
        if chars < 500:
            buckets["0-500"] += 1
        elif chars < 1000:
            buckets["500-1000"] += 1
        elif chars < 2000:
            buckets["1000-2000"] += 1
        elif chars < 5000:
            buckets["2000-5000"] += 1
        elif chars < 10000:
            buckets["5000-10000"] += 1
        else:
            buckets["10000+"] += 1

    # Find pages with <mermaid> tags
    mermaid_pages = [
        (p["page_num"], p["char_count"]) for p in pages if "<mermaid>" in p["content"]
    ]

    # Find pages with "Computer Vision"
    cv_pages = [
        (p["page_num"], p["char_count"])
        for p in pages
        if "Computer Vision" in p["content"]
    ]

    return {
        "char_stats": char_stats,
        "line_stats": line_stats,
        "above_p95": above_p95,
        "below_100": below_100,
        "top_20": top_20,
        "buckets": buckets,
        "mermaid_pages": mermaid_pages,
        "cv_pages": cv_pages,
        "total_pages": len(pages),
    }


def main():
    filepath = (
        "/home/keenora/Documents/MedifFact/drstrange-process/results_book/BATES.md"
    )

    print("=" * 80)
    print("BATES.md PAGE STATISTICS ANALYSIS")
    print("=" * 80)

    pages = parse_pages(filepath)
    results = analyze_pages(pages)

    # 1. Distribution stats
    print("\n## 1. DISTRIBUTION STATISTICS\n")
    print("### Characters")
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Mean | {results['char_stats']['mean']:.1f} |")
    print(f"| Median | {results['char_stats']['median']:.1f} |")
    print(f"| P25 | {results['char_stats']['p25']:.1f} |")
    print(f"| P75 | {results['char_stats']['p75']:.1f} |")
    print(f"| P90 | {results['char_stats']['p90']:.1f} |")
    print(f"| P95 | {results['char_stats']['p95']:.1f} |")
    print(f"| P99 | {results['char_stats']['p99']:.1f} |")
    print(f"| Max | {results['char_stats']['max']} |")

    print("\n### Lines")
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Mean | {results['line_stats']['mean']:.1f} |")
    print(f"| Median | {results['line_stats']['median']:.1f} |")
    print(f"| P25 | {results['line_stats']['p25']:.1f} |")
    print(f"| P75 | {results['line_stats']['p75']:.1f} |")
    print(f"| P90 | {results['line_stats']['p90']:.1f} |")
    print(f"| P95 | {results['line_stats']['p95']:.1f} |")
    print(f"| P99 | {results['line_stats']['p99']:.1f} |")
    print(f"| Max | {results['line_stats']['max']} |")

    # 2. Pages above 95th percentile (potential hallucination candidates)
    print("\n## 2. PAGES ABOVE 95TH PERCENTILE FOR CHARACTER COUNT")
    print(f"(Threshold: {results['char_stats']['p95']:.1f} characters)\n")
    if results["above_p95"]:
        print("| Page | Character Count |")
        print("|------|-----------------|")
        for page_num, char_count in sorted(
            results["above_p95"], key=lambda x: x[1], reverse=True
        ):
            print(f"| {page_num} | {char_count} |")
        print(f"\n**Total: {len(results['above_p95'])} pages**")
    else:
        print("No pages above 95th percentile.")

    # 3. Pages below 100 characters
    print("\n## 3. PAGES BELOW 100 CHARACTERS (Potential blank/stub pages)\n")
    if results["below_100"]:
        print("| Page | Character Count |")
        print("|------|-----------------|")
        for page_num, char_count in sorted(results["below_100"], key=lambda x: x[1]):
            print(f"| {page_num} | {char_count} |")
        print(f"\n**Total: {len(results['below_100'])} pages**")
    else:
        print("No pages below 100 characters.")

    # 4. Top 20 largest pages
    print("\n## 4. TOP 20 LARGEST PAGES\n")
    print("| Page | Characters | Lines | First 80 Characters |")
    print("|------|------------|-------|---------------------|")
    for p in results["top_20"]:
        page_num = p["page_num"]
        chars = p["char_count"]
        lines = p["line_count"]
        first_80 = p["first_80"].replace("|", "\\|")
        print(f"| {page_num} | {chars} | {lines} | {first_80} |")

    # 5. Character count buckets
    print("\n## 5. CHARACTER COUNT BUCKETS\n")
    print("| Bucket | Count | Percentage |")
    print("|--------|-------|------------|")
    total = results["total_pages"]
    for bucket, count in results["buckets"].items():
        pct = (count / total) * 100 if total > 0 else 0
        print(f"| {bucket} chars | {count} | {pct:.1f}% |")
    print(f"| **Total** | **{total}** | **100%** |")

    # 6. Pages with <mermaid> tags
    print("\n## 6. PAGES WITH <MERMAID> TAGS\n")
    if results["mermaid_pages"]:
        print("| Page | Character Count |")
        print("|------|-----------------|")
        for page_num, char_count in results["mermaid_pages"]:
            print(f"| {page_num} | {char_count} |")
        print(f"\n**Total: {len(results['mermaid_pages'])} pages**")
    else:
        print("No pages with <mermaid> tags found.")

    # 7. Pages with "Computer Vision"
    print("\n## 7. PAGES WITH 'COMPUTER VISION'\n")
    if results["cv_pages"]:
        print("| Page | Character Count |")
        print("|------|-----------------|")
        for page_num, char_count in results["cv_pages"]:
            print(f"| {page_num} | {char_count} |")
        print(f"\n**Total: {len(results['cv_pages'])} pages**")
    else:
        print("No pages with 'Computer Vision' found.")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
