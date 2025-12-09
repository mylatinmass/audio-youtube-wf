#!/usr/bin/env python3
import os
import re
import json
import argparse
import textwrap

# We intentionally avoid any functions that write files.
# We reuse your mdx_generator.py (which should return a string and not write).
# And we reuse your find_homily to guarantee we’re isolating HOMILY only.

def parse_yaml_front_matter(mdx_text: str) -> dict:
    m = re.search(r"^---\s*(.*?)\s*---", mdx_text, re.DOTALL | re.MULTILINE)
    if not m:
        return {}
    block = m.group(1)
    # simple YAML-ish parse for quoted scalars + shorts array preview
    fm = {}
    for line in block.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if v.startswith('"') and v.endswith('"'):
            fm[k] = v[1:-1]
        elif k == "shorts:":
            fm["__shorts_present__"] = True
        # ignore nested/arrays here; we’ll preview shorts later from body
    return fm

def count_shorts_yaml(mdx_text: str) -> int:
    # crude but effective: count YAML list items under "shorts:"
    m = re.search(r"^shorts:\s*(.*?)(?:^---$|^\S)", mdx_text, re.DOTALL | re.MULTILINE)
    if not m:
        return 0
    block = m.group(1)
    return len(re.findall(r'^\s*-\s*\{', block, re.MULTILINE))

def extract_shorts_preview(mdx_text: str, limit=3):
    m = re.search(r"^shorts:\s*(.*?)(?:^---$|^\S)", mdx_text, re.DOTALL | re.MULTILINE)
    if not m:
        return []
    block = m.group(1)
    items = re.findall(r'^\s*-\s*\{(.*?)\}\s*$', block, re.MULTILINE)
    return items[:limit]

def extract_toc(mdx_text: str):
    # look in body after second '---'
    parts = re.split(r"^---\s*.*?^---\s*", mdx_text, flags=re.DOTALL | re.MULTILINE)
    if len(parts) < 2:
        return []
    body = parts[1]
    m = re.search(r"^## Summary of Headings\s*(.*?)\n\n", body, re.DOTALL | re.MULTILINE)
    if not m:
        return []
    toc_block = m.group(1)
    titles = re.findall(r"- \[(.*?)\]\(#", toc_block)
    return titles

def snippet(s, n=240):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"

def main():
    ap = argparse.ArgumentParser(description="Dry-run workflow: console logs only, no files created.")
    ap.add_argument("--transcript-json", required=True, help="Path to transcript JSON (the same one you use in working/).")
    ap.add_argument("--show-mdx", action="store_true", help="Also dump the full MDX to stdout (careful: long).")
    args = ap.parse_args()

    # 1) Load transcript JSON
    if not os.path.exists(args.transcript_json):
        raise SystemExit(f"Missing transcript JSON: {args.transcript_json}")
    with open(args.transcript_json, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # 2) Homily-only extraction (no writes)
    #    Use your existing find_homily to guarantee HOMILY isolation
    from text_find import find_homily
    start, end, homily_text, segments = find_homily(transcript)

    print("=== DRY RUN: INPUT SUMMARY ===")
    print(f"Transcript file: {args.transcript_json}")
    print(f"Homily start: {start:.2f}s | end: {end:.2f}s | dur: {max(0.0, end - start):.2f}s")
    print(f"Homily text: {len(homily_text)} chars, {len(segments)} segments")
    print(f"Homily preview: {snippet(homily_text, 320)}\n")

    # 3) Generate MDX via your mdx_generator WITHOUT writing
    import mdx_generator  # your module
    mdx_str = mdx_generator.mdx_generator(homily_text, segments=segments)

    # 4) Parse & print diagnostics (front matter, TOC, shorts)
    fm = parse_yaml_front_matter(mdx_str)
    shorts_count = count_shorts_yaml(mdx_str)
    toc_titles = extract_toc(mdx_str)
    shorts_preview = extract_shorts_preview(mdx_str, limit=3)

    print("=== FRONT MATTER (key fields) ===")
    for k in ["title", "description", "keywords", "youtube_description", "youtube_hash", "slug", "mdx_file"]:
        val = fm.get(k, "")
        if k == "youtube_description":
            print(f"{k}:")
            print(textwrap.indent(snippet(val, 800), "  "))
        else:
            print(f"{k}: {snippet(val, 180)}")
    print()

    print("=== TOC (Summary of Headings) ===")
    if toc_titles:
        for i, t in enumerate(toc_titles, 1):
            print(f"{i}. {t}")
    else:
        print("(none)")
    print()

    print("=== SHORTS ===")
    print(f"Total shorts found in YAML: {shorts_count}")
    if shorts_preview:
        print("Preview (first few YAML objects raw):")
        for i, raw in enumerate(shorts_preview, 1):
            print(f"  {i}) {{ {snippet(raw, 240)} }}")
    print()

    # 5) Liturgical block presence
    #    Look for a block quoted line with **(1962 Missal)** right after the H1
    body = re.split(r"^---\s*.*?^---\s*", mdx_str, flags=re.DOTALL | re.MULTILINE)
    liturgical_found = False
    if len(body) >= 2:
        btxt = body[1]
        if re.search(r"^> \*\*.*\(1962 Missal\)\*\*", btxt, re.MULTILINE):
            liturgical_found = True
    print("=== LITURGICAL BLOCK ===")
    print("Present (one reading from 1962 Missal):", "YES" if liturgical_found else "NO")
    print()

    # Optional: dump full MDX (no writes)
    if args.show_mdx:
        print("=== FULL MDX (BEGIN) ===")
        print(mdx_str)
        print("=== FULL MDX (END) ===")

    print("=== DRY RUN COMPLETE ===")

if __name__ == "__main__":
    main()
