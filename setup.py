"""Build configuration for native C extensions.

This file exists solely to declare ext_modules, which pyproject.toml
cannot express.  All other metadata lives in pyproject.toml.
"""

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "brileta.util._native",
            sources=[
                "brileta/util/_native.c",
                "brileta/util/_native_pathfinding.c",
                "brileta/util/_native_fov.c",
                "brileta/util/_native_wfc.c",
            ],
        ),
    ],
)
