"""
Patch gfootball's GLSL shaders for better visual quality.
Run this once in WSL:  python3 patch_gfootball_shaders.py

Changes:
  postprocess.frag  - brighter, more saturated, blue sky background, less fog
  lighting.frag     - stronger stadium lights
  ambient.frag      - warmer ambient (stadium floodlight feel, not cold blue)
"""
import importlib.util, pathlib, shutil, sys, textwrap

# ── Find gfootball install path ───────────────────────────────────────────────
def find_shader_dir() -> pathlib.Path:
    spec = importlib.util.find_spec("gfootball")
    if spec is None:
        sys.exit("gfootball not found — activate the correct virtualenv first.")
    pkg = pathlib.Path(spec.origin).parent          # .../gfootball/__init__.py → gfootball/
    # Shaders live in the sibling third_party directory baked into the package
    candidates = [
        pkg.parent / "third_party" / "gfootball_engine" / "data" / "media" / "shaders",
        pkg / "data" / "third_party" / "gfootball_engine" / "data" / "media" / "shaders",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: walk up and search
    for root in [pkg.parent, pkg.parent.parent]:
        hits = list(root.rglob("postprocess.frag"))
        if hits:
            return hits[0].parent
    sys.exit(f"Cannot locate shader directory. Searched: {candidates}")


shader_dir = find_shader_dir()
print(f"Shader directory: {shader_dir}")


# ── Backup helper ─────────────────────────────────────────────────────────────
def patch(filename: str, replacements: list[tuple[str, str]]):
    path = shader_dir / filename
    if not path.exists():
        print(f"  SKIP (not found): {filename}")
        return
    bak = path.with_suffix(path.suffix + ".orig")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  backed up → {bak.name}")
    src = path.read_text()
    for old, new in replacements:
        if old not in src:
            print(f"  WARNING: pattern not found in {filename}: {old!r}")
            continue
        src = src.replace(old, new, 1)
    path.write_text(src)
    print(f"  patched: {filename}")


# ── postprocess.frag ──────────────────────────────────────────────────────────
# Goals: brighter (1.0→1.35), warmer saturation, blue sky, slightly less fog
patch("postprocess.frag", [
    # Sky/background color: cold grey → bright stadium-day blue
    (
        "vec3 fogColor = vec3(0.85, 0.85, 0.9);",
        "vec3 fogColor = vec3(0.42, 0.65, 0.95);",
    ),
    # Max fog: 0.25 → 0.15  (crisper distant view)
    (
        "float fogFactor = clamp(fragDepth * 0.01f * (1.0f - fogScale) - 0.16f * fogScale, 0.0f, 0.25f);",
        "float fogFactor = clamp(fragDepth * 0.01f * (1.0f - fogScale) - 0.16f * fogScale, 0.0f, 0.15f);",
    ),
    # Brightness: 1.0 → 1.35
    (
        "float brightness = 1.0f;",
        "float brightness = 1.35f;",
    ),
    # Saturation base lifted: 0.4 → 0.58, SSAO weight eased
    (
        "float saturation = 0.95f * (0.4f + SSAO * 0.6f);",
        "float saturation = 1.10f * (0.58f + SSAO * 0.42f);",
    ),
    # Contrast bias: soften very slightly (0.3 → 0.22)
    (
        "float contrastBias = 0.3f;//0.1f; // 0 == normal .. 1 == 'fake hdri'",
        "float contrastBias = 0.22f; // 0 == normal .. 1 == 'fake hdri'",
    ),
])


# ── lighting.frag ─────────────────────────────────────────────────────────────
# Goal: stronger directional lights (2.0 → 3.0), shadows slightly softer
patch("lighting.frag", [
    # Light brightness: 2.0 → 3.0  (stadium floodlights are very bright)
    (
        "float brightness = 2.0f;//1.5f;",
        "float brightness = 3.0f;",
    ),
    # Shadow: keep soft (0.75/0.25 is fine, but let's open shadows a touch)
    (
        "    shaded *= 0.75;\n    shaded += 0.25;",
        "    shaded *= 0.65;\n    shaded += 0.35;",
    ),
])


# ── ambient.frag ──────────────────────────────────────────────────────────────
# Goal: warmer fill light (cold blue → warm white), brighter ambient
patch("ambient.frag", [
    # Ambient brightness: 0.15 → 0.28
    (
        "float brightness = 0.15f;//0.25f;",
        "float brightness = 0.28f;",
    ),
    # Color tint: cold blue (0.9, 1.0, 1.2) → warm stadium white (1.05, 1.0, 0.88)
    (
        "base *= vec3(0.9f, 1.0f, 1.2f) * brightness;",
        "base *= vec3(1.05f, 1.0f, 0.88f) * brightness;",
    ),
    # SSAO intensity: soften slightly so shadows aren't too crushed
    (
        "SSAO = SSAO * 1.5f - 0.5f; // exaggerate effect",
        "SSAO = SSAO * 1.2f - 0.2f; // slightly softened",
    ),
])


print("\nDone. Restart gfootball to see changes.")
print("To revert: rename .orig files back to their original names.")
