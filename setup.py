"""Build configuration for native C extensions.

This file exists solely to declare ext_modules, which pyproject.toml
cannot express.  All other metadata lives in pyproject.toml.
"""

import sys
from pathlib import Path

from setuptools import Extension, setup

# MSVC compiles .c in a legacy dialect where C11's _Static_assert isn't
# recognized; /std:c11 enables it. Clang/GCC (macOS/Linux) already default to a
# new-enough standard, so Unix builds need no extra flags.
# ponytail: assumes MSVC on Windows (CI + standard toolchain); mingw would want -std=c11
extra_compile_args = ["/std:c11"] if sys.platform == "win32" else []

setup(
    ext_modules=[
        Extension(
            "brileta.util._native",
            sources=list(Path("brileta/util/native").glob("_native*.c")),
            extra_compile_args=extra_compile_args,
        ),
    ],
)
