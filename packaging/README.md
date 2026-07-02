# Packaging Brileta for distribution (itch.io)

Produces a standalone, double-clickable build. Players need no uv, Python, or
toolchain. Everything - the compiled C extension, wgpu's native backend, glfw's
native lib, and all runtime assets - is bundled by PyInstaller.

## Files

- `brileta.spec` (repo root) - the PyInstaller build recipe.
- `packaging/brileta_launcher.py` - frozen entry point. Calls
  `multiprocessing.freeze_support()` (the game uses a process pool) then the
  normal `main()`.
- `.github/workflows/build-distributables.yml` - CI matrix that builds macOS,
  Windows, and Linux artifacts.

## Build locally (macOS, this machine)

```bash
make dist
```

Output:
- `dist/Brileta.app` - the double-clickable app.
- `dist/Brileta-macos.zip` - the app zipped for itch.io (via `ditto`, which
  preserves the bundle's symlinks and metadata).

On Windows/Linux the same target produces `dist/Brileta/` (a folder with the
executable inside) and `dist/Brileta-<OS>.zip`.

Under the hood `make dist` just runs `uv run pyinstaller brileta.spec`.

## Cross-platform builds

PyInstaller does not cross-compile - each OS must build on its own machine. Use
the CI workflow: push a `v*` tag or trigger **Build distributables** manually in
the Actions tab. It runs the same PyInstaller build on macOS, Windows, and Linux
runners and uploads one artifact per platform. Download them from the run's
Artifacts section (each is already a zip).

## Uploading to itch.io

Two options.

**Web uploader:** on your game's Edit page, upload the platform zip, tick the
matching platform (e.g. "This file will be played in the browser" stays off;
check macOS/Windows/Linux), and set it as the executable download.

**butler CLI (recommended for repeat pushes):**

```bash
# once: install and log in
butler login

# push each platform build to its own channel
butler push dist/Brileta-macos.zip   YOUR_USER/brileta:osx
butler push dist/Brileta-windows.zip YOUR_USER/brileta:windows
butler push dist/Brileta-linux.zip   YOUR_USER/brileta:linux
```

butler diffs against the previous push, so updates upload only what changed.

## Notes

- The macOS `.app` is unsigned and not notarized, so Gatekeeper blocks it on
  first launch ("Apple could not verify..."). Players who install via the itch
  app are fine (it strips quarantine); direct-download users need this note.
  Put it on the itch page / GitHub Release for the macOS build:

  > macOS: this build is unsigned. After unzipping, open Terminal and run
  > `xattr -dr com.apple.quarantine Brileta.app`, then open the app. (Or:
  > System Settings → Privacy & Security → "Open Anyway".)

  Sign/notarize with an Apple Developer account later if you want to remove
  this step.
- Assets resolve at runtime relative to the `brileta` package's parent
  directory, which is why the spec bundles `assets/` at the bundle root - no
  code change needed. If you add a new asset directory, it's already covered
  (the whole `assets/` tree is collected).
