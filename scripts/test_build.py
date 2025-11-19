#!/usr/bin/env python3
"""
Test script to verify build configuration and wheel contents.

Usage:
    python test_build.py
"""

import sys
import subprocess
import zipfile
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a shell command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with code {e.returncode}")
        print(e.stdout)
        print(e.stderr)
        return False


def check_wheel_contents(wheel_path):
    """Verify wheel contains required files."""
    print(f"\n{'='*60}")
    print(f"Checking wheel contents: {wheel_path.name}")
    print('='*60)
    
    required_files = [
        'api/static/index.html',
        'api/main.py',
        'alchemist_core/session.py',
    ]
    
    with zipfile.ZipFile(wheel_path, 'r') as zf:
        file_list = zf.namelist()
        
        # Check for required files
        missing = []
        for req in required_files:
            matches = [f for f in file_list if req in f]
            if matches:
                print(f"✓ Found: {req}")
                for match in matches[:3]:  # Show first 3 matches
                    print(f"  - {match}")
            else:
                print(f"✗ Missing: {req}")
                missing.append(req)
        
        # Count static files
        static_files = [f for f in file_list if 'api/static/' in f]
        print(f"\n✓ Total static files: {len(static_files)}")
        
        # Show sample static files
        print("\nSample static files:")
        for f in static_files[:10]:
            print(f"  - {f}")
        
        if missing:
            print(f"\n✗ ERROR: Missing required files: {missing}")
            return False
        
        if len(static_files) < 5:
            print("\n✗ WARNING: Very few static files found. React build may have failed.")
            return False
        
        print("\n✓ Wheel contents verified successfully!")
        return True


def main():
    """Run build verification tests."""
    root = Path(__file__).parent
    
    print("\n" + "="*60)
    print("ALchemist Build Verification Test")
    print("="*60)
    
    # Clean previous builds
    print("\n1. Cleaning previous build artifacts...")
    for path in ['dist', 'build', 'api/static', 'alchemist-web/dist']:
        full_path = root / path
        if full_path.exists():
            print(f"  Removing {path}/")
            subprocess.run(['rm', '-rf', str(full_path)])
    
    # Check if build_hooks.py exists
    build_hooks = root / 'build_hooks.py'
    if not build_hooks.exists():
        print(f"\n✗ ERROR: build_hooks.py not found at {build_hooks}")
        return False
    print(f"✓ build_hooks.py found")
    
    # Check if frontend exists
    frontend_dir = root / 'alchemist-web'
    if not frontend_dir.exists():
        print(f"\n✗ ERROR: Frontend directory not found at {frontend_dir}")
        return False
    print(f"✓ Frontend directory found")
    
    package_json = frontend_dir / 'package.json'
    if not package_json.exists():
        print(f"\n✗ ERROR: package.json not found")
        return False
    print(f"✓ package.json found")
    
    # Build package
    print("\n2. Building package...")
    if not run_command([sys.executable, '-m', 'build'], cwd=root):
        print("\n✗ Build failed!")
        return False
    
    # Find wheel
    dist_dir = root / 'dist'
    wheels = list(dist_dir.glob('*.whl'))
    if not wheels:
        print("\n✗ ERROR: No wheel found in dist/")
        return False
    
    wheel_path = wheels[0]
    print(f"\n✓ Wheel created: {wheel_path.name}")
    
    # Verify wheel contents
    if not check_wheel_contents(wheel_path):
        print("\n✗ Wheel verification failed!")
        return False
    
    # Success!
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print(f"\nWheel ready: {wheel_path}")
    print("\nNext steps:")
    print("  1. Test install: pip install dist/*.whl")
    print("  2. Test command: alchemist-web --production")
    print("  3. Open browser: http://localhost:8000")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
