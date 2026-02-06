"""Build configuration for native C extensions.

This file exists solely to declare ext_modules, which pyproject.toml
cannot express.  All other metadata lives in pyproject.toml.
"""

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "catley.util._native",
            sources=[
                "catley/util/_native.c",
                "catley/util/_native_pathfinding.c",
                "catley/util/_native_fov.c",
            ],
        ),
    ],
)
