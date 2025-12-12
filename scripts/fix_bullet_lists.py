#!/usr/bin/env python3
"""
Fix bullet list formatting in markdown files.
Adds blank lines between consecutive bullet list items and ensures 
lists start on a new line.
"""

import re
from pathlib import Path
import sys


def fix_bullet_lists(content: str) -> tuple[str, int]:
    """
    Fix bullet list formatting by adding blank lines between consecutive items
    and ensuring the list starts on a new line (separating it from previous text).
    
    Returns:
        tuple of (fixed_content, number_of_fixes)
    """
    lines = content.split('\n')
    fixed_lines = []
    fixes = 0
    i = 0
    
    # Define regex for a bullet item (dash, space, not followed by [)
    bullet_pattern = r'^- (?!\[).+'
    
    while i < len(lines):
        current_line = lines[i]
        
        # Check if current line is a bullet
        is_current_bullet = bool(re.match(bullet_pattern, current_line))
        
        # --- FIX 1: Add blank line BEFORE the list starts ---
        if is_current_bullet and i > 0:
            prev_line = lines[i - 1]
            # If previous line is not blank and is NOT a bullet itself
            # (meaning we are transitioning from text -> list)
            if prev_line.strip() and not re.match(bullet_pattern, prev_line):
                fixed_lines.append('')
                fixes += 1

        fixed_lines.append(current_line)
        
        # --- FIX 2: Add blank line BETWEEN list items (Existing logic) ---
        if is_current_bullet:
            # Look ahead to see if next line is also a bullet
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.match(bullet_pattern, next_line):
                    fixed_lines.append('')
                    fixes += 1
        
        i += 1
    
    return '\n'.join(fixed_lines), fixes


def process_file(file_path: Path) -> bool:
    """
    Process a single markdown file.
    
    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        fixed_content, num_fixes = fix_bullet_lists(original_content)
        
        if num_fixes > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"✓ {file_path.relative_to(Path.cwd())}: {num_fixes} fixes")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    """Main function to process all markdown files in docs/ directory."""
    docs_dir = Path('docs').resolve()
    base_dir = Path.cwd().resolve()
    
    if not docs_dir.exists():
        print("Error: docs/ directory not found", file=sys.stderr)
        sys.exit(1)
    
    # Find all markdown files
    md_files = list(docs_dir.rglob('*.md'))
    
    if not md_files:
        print("No markdown files found in docs/", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(md_files)} markdown files to process\n")
    
    modified_count = 0
    total_fixes = 0
    
    for md_file in sorted(md_files):
        # Using read_text/write_text for cleaner file handling
        try:
            original_content = md_file.read_text(encoding='utf-8')
            fixed_content, num_fixes = fix_bullet_lists(original_content)
            
            if num_fixes > 0:
                md_file.write_text(fixed_content, encoding='utf-8')
                try:
                    rel_path = md_file.relative_to(base_dir)
                except ValueError:
                    rel_path = md_file
                print(f"✓ {rel_path}: {num_fixes} fixes")
                modified_count += 1
                total_fixes += num_fixes
        except Exception as e:
             print(f"✗ Error processing {md_file}: {e}", file=sys.stderr)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files processed: {len(md_files)}")
    print(f"  Files modified: {modified_count}")
    print(f"  Total fixes: {total_fixes}")
    print(f"{'='*60}")
    
    if modified_count > 0:
        print("\n✓ All bullet lists have been fixed!")
        print("  Run 'mkdocs build' to verify the changes.")
    else:
        print("\n✓ No fixes needed - all bullet lists are properly formatted.")


if __name__ == '__main__':
    main()