"""PyInstaller entry point.

`python -m brileta` runs the package's __main__, but PyInstaller needs a plain
script as its entry. This just calls the same main().
"""

import multiprocessing

if __name__ == "__main__":
    # The game uses a ProcessPoolExecutor. In a frozen build, worker processes
    # re-exec this executable with multiprocessing bootstrap args; freeze_support
    # intercepts them and runs the worker instead of falling through to main()
    # (where argparse would reject the args). Must be the first call.
    multiprocessing.freeze_support()

    from brileta.__main__ import main

    main()
