#!/usr/bin/env python3

try:
    import cosmology_amilly

    print("✅ Successfully imported cosmology_amilly")

    from cosmology_amilly import linear_growth, sigma8

    print("✅ Successfully imported submodules")

    print("Package file location:", cosmology_amilly.__file__)

except ImportError as e:
    print("❌ Import error:", e)
    import sys

    print("Python executable:", sys.executable)
    print("Python path:", sys.path[:5])
