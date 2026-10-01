#!/usr/bin/env python3
"""
Preprocess HTML files for EPUB generation.
Makes all anchor IDs globally unique by prefixing with chapter identifier,
then updates all internal links to use the prefixed anchors.
"""

import re
import sys
from pathlib import Path

# Import from sibling module
sys.path.insert(0, str(Path(__file__).parent))


def file_to_prefix(filepath):
    """Generate a unique prefix from a file path."""
    # Convert path like 'network/benchmarks/README.html' to 'network-benchmarks-readme'
    path = Path(filepath)
    parts = list(path.parent.parts) + [path.stem.lower()]
    # Filter out empty parts and join with hyphen
    prefix = '-'.join(p for p in parts if p and p != '.')
    return prefix if prefix else 'root'


def prefix_anchors_in_file(html_file, prefix):
    """Add prefix to all id attributes in an HTML file."""
    path = Path(html_file)
    if not path.exists():
        return {}
    
    content = path.read_text(encoding='utf-8')
    original = content
    anchor_map = {}  # old_anchor -> new_anchor
    
    def replace_id(match):
        quote_start = match.group(1)  # 'id="' or "id='"
        old_id = match.group(2)
        quote_end = match.group(3)
        
        new_id = f"{prefix}--{old_id}"
        anchor_map[old_id] = new_id
        return f"{quote_start}{new_id}{quote_end}"
    
    content = re.sub(r'(id=["\'])([^"\']+)(["\'])', replace_id, content)
    
    if content != original:
        path.write_text(content, encoding='utf-8')
    
    return anchor_map


def update_links_in_file(html_file, file_prefix_map, file_anchor_maps):
    """Update all internal links to use prefixed anchors."""
    path = Path(html_file)
    if not path.exists():
        return
    
    content = path.read_text(encoding='utf-8')
    original = content
    current_dir = path.parent
    current_prefix = file_prefix_map.get(str(path), file_prefix_map.get(html_file, ''))
    
    # Build set of valid HTML targets
    html_file_set = set(file_prefix_map.keys())
    
    def convert_href(match):
        prefix = match.group(1)  # 'href="' or "href='"
        href = match.group(2)
        quote = match.group(3)
        
        # Skip external links and mailto
        if href.startswith(('http://', 'https://', 'mailto:')):
            return match.group(0)
        
        # Handle anchor-only links (within same file)
        if href.startswith('#'):
            old_anchor = href[1:]
            # Look up the prefixed anchor for this file
            anchor_map = file_anchor_maps.get(str(path), file_anchor_maps.get(html_file, {}))
            if old_anchor in anchor_map:
                return f'{prefix}#{anchor_map[old_anchor]}{quote}'
            return match.group(0)
        
        # Parse href into path and anchor
        if '#' in href:
            href_path, anchor = href.split('#', 1)
        else:
            href_path = href
            anchor = None
        
        # Skip non-HTML links
        if not href_path.endswith('.html'):
            return match.group(0)
        
        # Resolve relative path
        try:
            resolved = (current_dir / href_path).resolve()
            rel_path = str(resolved.relative_to(Path.cwd()))
        except (ValueError, OSError):
            return match.group(0)
        
        # Check if target is in our HTML file list
        if rel_path in html_file_set:
            target_prefix = file_prefix_map[rel_path]
            target_anchor_map = file_anchor_maps.get(rel_path, {})
            
            if anchor and anchor in target_anchor_map:
                # Link to specific anchor in target file
                return f'{prefix}#{target_anchor_map[anchor]}{quote}'
            else:
                # Link to file without anchor - find first heading anchor
                if target_anchor_map:
                    # Get the first anchor (usually the main heading)
                    # We'll use a naming convention: look for the file's main id
                    first_anchor = next(iter(target_anchor_map.values()), None)
                    if first_anchor:
                        return f'{prefix}#{first_anchor}{quote}'
        
        return match.group(0)
    
    # Match href="..." or href='...'
    content = re.sub(r'(href=["\'])([^"\']+)(["\'])', convert_href, content)
    
    if content != original:
        path.write_text(content, encoding='utf-8')
        print(f"Updated: {html_file}")


def main():
    chapters_file = Path("chapters-html.txt")
    if not chapters_file.exists():
        print("Error: chapters-html.txt not found")
        sys.exit(1)
    
    html_files = [line.strip() for line in chapters_file.read_text().split('\n') if line.strip()]
    
    if not html_files:
        print("Error: No HTML files found")
        sys.exit(1)
    
    print(f"Processing {len(html_files)} HTML files for EPUB...")
    
    # Step 1: Generate unique prefix for each file
    file_prefix_map = {}
    for html_file in html_files:
        prefix = file_to_prefix(html_file)
        file_prefix_map[html_file] = prefix
        file_prefix_map[str(Path(html_file))] = prefix
    
    # Step 2: Prefix all anchors in each file and build anchor maps
    file_anchor_maps = {}
    for html_file in html_files:
        prefix = file_prefix_map[html_file]
        anchor_map = prefix_anchors_in_file(html_file, prefix)
        file_anchor_maps[html_file] = anchor_map
        file_anchor_maps[str(Path(html_file))] = anchor_map
    
    # Step 3: Update all links to use prefixed anchors
    for html_file in html_files:
        update_links_in_file(html_file, file_prefix_map, file_anchor_maps)
    
    print("Done preprocessing for EPUB")


if __name__ == "__main__":
    main()
