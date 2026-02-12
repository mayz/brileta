"""Build configuration for native C extensions.

This file exists solely to declare ext_modules, which pyproject.toml
cannot express.  All other metadata lives in pyproject.toml.
"""

from pathlib import Path

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "brileta.util._native",
            sources=list(Path("brileta/util/native").glob("_native*.c")),
        ),
    ],
)
