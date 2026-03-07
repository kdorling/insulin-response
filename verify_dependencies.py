#!/usr/bin/env python3
"""
Dependency Verification Script

This script checks that all required packages are installed and importable.
Run this after installing requirements.txt to verify the setup.
"""

import sys
from typing import List, Tuple


def verify_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Verify that a package is installed and importable.
    
    Args:
        package_name: Display name of the package
        import_name: Name to use for import (defaults to package_name)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, f"✓ {package_name}"
    except ImportError as e:
        return False, f"✗ {package_name}: {str(e)}"
    except Exception as e:
        return False, f"✗ {package_name}: Unexpected error - {str(e)}"


def main():
    """Main verification function."""
    print("=" * 60)
    print("Dependency Verification for Insulin Response Modeling System")
    print("=" * 60)
    print()
    
    # List of packages to verify: (display_name, import_name)
    packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("statsmodels", "statsmodels"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("torch (PyTorch)", "torch"),
        ("jupyter", "jupyter"),
        ("hypothesis", "hypothesis"),
        ("pytest", "pytest"),
        ("pytest-cov", "pytest_cov"),
    ]
    
    results: List[Tuple[bool, str]] = []
    
    print("Checking packages...")
    print()
    
    for display_name, import_name in packages:
        success, message = verify_package(display_name, import_name)
        results.append((success, message))
        print(message)
    
    print()
    print("=" * 60)
    
    # Summary
    successful = sum(1 for success, _ in results if success)
    total = len(results)
    
    if successful == total:
        print(f"SUCCESS: All {total} packages are installed and importable!")
        print()
        print("You can now proceed with using the system.")
        return 0
    else:
        failed = total - successful
        print(f"FAILURE: {failed} out of {total} packages failed verification.")
        print()
        print("Please install missing packages using:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
