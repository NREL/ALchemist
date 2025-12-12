"""
Custom build hooks for setuptools to bundle the React frontend.

This script runs during `python -m build` to compile the React UI
and include it in the wheel distribution.

For editable installs (pip install -e .), the frontend build is skipped.
"""

import subprocess
import shutil
import os
from pathlib import Path
from setuptools.command.build_py import build_py as _build_py


class BuildWithFrontend(_build_py):
    """Custom build command that compiles React frontend before packaging."""
    
    def run(self):
        """Build frontend, copy to api/static/, then run normal build."""
        
        # Skip frontend build for editable installs
        if os.environ.get('SKIP_FRONTEND_BUILD') == '1':
            print("\nSkipping frontend build (editable install)")
            super().run()
            return
        
        print("\n" + "="*60)
        print("CUSTOM BUILD: Compiling React frontend...")
        print("="*60 + "\n")
        
        # Get project root
        root = Path(__file__).resolve().parent
        frontend_dir = root / "alchemist-web"
        api_static = root / "api" / "static"
        
        # Check if frontend directory exists
        if not frontend_dir.exists():
            print(f"WARNING: Frontend directory not found at {frontend_dir}")
            print("Skipping frontend build. Web UI will not be available.")
            super().run()
            return
        
        # Check if package.json exists
        package_json = frontend_dir / "package.json"
        if not package_json.exists():
            print(f"WARNING: package.json not found at {package_json}")
            print("Skipping frontend build.")
            super().run()
            return
        
        try:
            # Clean old build artifacts
            print("Cleaning old build artifacts...")
            dist_dir = frontend_dir / "dist"
            if dist_dir.exists():
                shutil.rmtree(dist_dir)
            if api_static.exists():
                shutil.rmtree(api_static)
            
            # Install npm dependencies
            print("\nInstalling npm dependencies...")
            subprocess.check_call(
                ["npm", "ci" if (frontend_dir / "package-lock.json").exists() else "install"],
                cwd=frontend_dir
            )
            
            # Build frontend
            print("\nBuilding React application...")
            subprocess.check_call(
                ["npm", "run", "build"],
                cwd=frontend_dir
            )
            
            # Verify build output exists
            if not dist_dir.exists():
                raise RuntimeError(
                    f"Frontend build failed: {dist_dir} was not created. "
                    "Check that 'npm run build' produces a dist/ folder."
                )
            
            # Copy build output to api/static/
            print(f"\nCopying build output to {api_static}...")
            shutil.copytree(dist_dir, api_static)
            
            # Verify static files were copied
            index_html = api_static / "index.html"
            if not index_html.exists():
                raise RuntimeError(
                    f"Build verification failed: {index_html} not found. "
                    "The frontend build may be incomplete."
                )
            
            print("\n" + "="*60)
            print("✓ Frontend build complete!")
            print(f"  Static files: {api_static}")
            print("="*60 + "\n")
            
        except subprocess.CalledProcessError as e:
            print("\n" + "!"*60)
            print(f"ERROR: Frontend build failed!")
            print(f"  Command: {e.cmd}")
            print(f"  Return code: {e.returncode}")
            print("!"*60 + "\n")
            raise RuntimeError(
                "Frontend build failed. Ensure Node.js and npm are installed "
                "and package.json is properly configured."
            ) from e
        
        except Exception as e:
            print("\n" + "!"*60)
            print(f"ERROR: Unexpected error during frontend build: {e}")
            print("!"*60 + "\n")
            raise
        
        # Run the standard build_py command
        super().run()
