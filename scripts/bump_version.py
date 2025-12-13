#!/usr/bin/env python3
"""
Version bumping utility for ALchemist.

Updates version in all required files and optionally creates a git tag.

Usage:
    python bump_version.py 0.3.1
    python bump_version.py 0.3.1 --tag
    python bump_version.py 0.3.1 --tag --push
"""

import sys
import re
import subprocess
from pathlib import Path


def update_pyproject_toml(root: Path, new_version: str) -> bool:
    """Update version in pyproject.toml."""
    file_path = root / 'pyproject.toml'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Match version = "x.y.z" or version = "x.y.zbN"
    pattern = r'(version\s*=\s*")[^"]+(")'
    new_content = re.sub(pattern, rf'\g<1>{new_version}\g<2>', content)
    
    if content == new_content:
        print(f"✗ No version found in {file_path}")
        return False
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"✓ Updated {file_path}")
    return True


def update_package_json(root: Path, new_version: str) -> bool:
    """Update version in alchemist-web/package.json."""
    file_path = root / 'alchemist-web' / 'package.json'
    
    if not file_path.exists():
        print(f"✗ File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Match "version": "x.y.z"
    pattern = r'("version"\s*:\s*")[^"]+(")'
    new_content = re.sub(pattern, rf'\g<1>{new_version}\g<2>', content)
    
    if content == new_content:
        print(f"✗ No version found in {file_path}")
        return False
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"✓ Updated {file_path}")
    return True


def create_git_tag(version: str, push: bool = False) -> bool:
    """Create and optionally push a git tag."""
    tag_name = f"v{version}"
    
    # Check if tag already exists
    result = subprocess.run(
        ['git', 'tag', '-l', tag_name],
        capture_output=True,
        text=True
    )
    
    if result.stdout.strip():
        print(f"✗ Tag {tag_name} already exists")
        return False
    
    # Create tag
    result = subprocess.run(
        ['git', 'tag', '-a', tag_name, '-m', f'Release {tag_name}'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"✗ Failed to create tag: {result.stderr}")
        return False
    
    print(f"✓ Created tag {tag_name}")
    
    if push:
        result = subprocess.run(
            ['git', 'push', 'origin', tag_name],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"✗ Failed to push tag: {result.stderr}")
            return False
        
        print(f"✓ Pushed tag {tag_name}")
    
    return True


def validate_version(version: str) -> bool:
    """Validate version format."""
    # Allow: x.y.z, x.y.za, x.y.zbN (e.g., 0.3.0, 0.3.0a1, 0.3.0b1)
    pattern = r'^\d+\.\d+\.\d+([ab]\d+)?$'
    if not re.match(pattern, version):
        print(f"✗ Invalid version format: {version}")
        print(f"  Expected: x.y.z or x.y.z[ab]N (e.g., 0.3.1 or 0.3.0b1)")
        return False
    return True


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python bump_version.py VERSION [--tag] [--push]")
        print("Example: python bump_version.py 0.3.1 --tag --push")
        sys.exit(1)
    
    new_version = sys.argv[1]
    create_tag = '--tag' in sys.argv
    push_tag = '--push' in sys.argv
    
    if not validate_version(new_version):
        sys.exit(1)
    
    print("\n" + "="*60)
    print(f"Bumping version to {new_version}")
    print("="*60 + "\n")
    
    # Resolve to repository root (script lives in scripts/)
    root = Path(__file__).resolve().parent.parent
    
    # Update files
    success = True
    success &= update_pyproject_toml(root, new_version)
    success &= update_package_json(root, new_version)
    
    if not success:
        print("\n✗ Version bump failed!")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Version updated successfully!")
    print("="*60)
    
    # Create git tag if requested
    if create_tag:
        print()
        if not create_git_tag(new_version, push_tag):
            sys.exit(1)
    
    # Print next steps
    print("\nNext steps:")
    if not create_tag:
        print(f"  1. Review changes: git diff")
        print(f"  2. Commit: git add -A && git commit -m 'Bump version to {new_version}'")
        print(f"  3. Tag: git tag v{new_version}")
        print(f"  4. Push: git push origin main && git push origin v{new_version}")
    elif not push_tag:
        print(f"  1. Review changes: git diff")
        print(f"  2. Commit: git add -A && git commit -m 'Bump version to {new_version}'")
        print(f"  3. Push: git push origin main && git push origin v{new_version}")
    else:
        print(f"  1. Review changes: git diff")
        print(f"  2. Commit: git add -A && git commit -m 'Bump version to {new_version}'")
        print(f"  3. Push commits: git push origin main")
        print(f"  4. GitHub Actions will build and release automatically!")
    
    print(f"\nTo test the build locally:")
    print(f"  python test_build.py")


if __name__ == '__main__':
    main()
