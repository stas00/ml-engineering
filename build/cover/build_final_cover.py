#!/usr/bin/env python3
"""Build the 'Machine Learning Engineering' book cover into <repo>/images/.

The cover canvas is US Letter aspect (8.5:11) so the PDF is a true Letter page
with no cropping.

Outputs:
  - Machine-Learning-Engineering-book-cover-1275x1650.png : full-resolution flat cover
  - Machine-Learning-Engineering-book-cover.png           : optimized display image
  - Machine-Learning-Engineering-book-cover-<wxh>.png     : small thumbnail (size in name)
  - Machine-Learning-Engineering-book-cover.pdf           : vector PDF at US Letter size
                                                           (612x792pt, 8.5x11in)
  - Machine-Learning-Engineering-book-cover.svg           : self-contained, each text
                                                           section its own Inkscape layer
  - Machine-Learning-Engineering-book-cover.ora           : OpenRaster; opens in GIMP/Krita
                                                           with background + text layers

This directory is self-contained: the only input is
sources/Machine-Learning-Engineering-art.png (the robot art with no text, flattened
from the GIMP source). The title/author are re-typeset as editable text layers.

Requires: ImageMagick (magick) + rsvg-convert (librsvg). zopflipng optional
(used to further shrink the PNGs when present). No GIMP needed.
"""
import base64
import pathlib
import shutil
import subprocess
import tempfile
import zipfile

HERE = pathlib.Path(__file__).parent
IMAGES = HERE.parent.parent / "images"            # <repo>/images
ART = HERE / "sources" / "Machine-Learning-Engineering-art.png"
STEM = "Machine-Learning-Engineering-book-cover"

# --- canvas / layout geometry -------------------------------------------------
W, H = 1275, 1650          # US Letter aspect (8.5:11) at 150 DPI
CX = W // 2
DARK = "#1e2226"           # uniform art background -> seamless side extension
MONO = "DejaVu Sans Mono, Menlo, Consolas, monospace"
TEXT = "#d8dace"           # light warm-gray title color (sampled from original)

# Title geometry, scaled ~2.19x up from the original 548px artwork.
TX = 90                    # left margin (x)
FS = 62                    # font size
L1_Y = 125                 # baseline, line 1 (title)
L2_Y = 226                 # baseline, line 2 (author)

# Each text section: label -> inner SVG markup on the full WxH canvas.
LAYERS = {
    "title": (
        f'<text x="{TX}" y="{L1_Y}" font-family="{MONO}" '
        f'font-size="{FS}" fill="{TEXT}">Machine Learning Engineering</text>'
    ),
    "author": (
        f'<text x="{TX}" y="{L2_Y}" font-family="{MONO}" '
        f'font-size="{FS}" fill="{TEXT}">by Stas Bekman</text>'
    ),
}
ORDER = ["background", "title", "author"]


def run(*args):
    subprocess.run([str(a) for a in args], check=True)


def b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def compose_art(out: pathlib.Path):
    """Scale the art to fill the canvas height, then pad the left side with the
    art's own background color so the canvas reaches US Letter aspect seamlessly
    (the robot stays flush to the right, text sits over the dark left area)."""
    run("magick", ART, "-resize", f"x{H}", "-background", DARK,
        "-gravity", "East", "-extent", f"{W}x{H}", "-depth", "8", "-strip", out)


def render(svg_text: str, out_png: pathlib.Path):
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as f:
        f.write(svg_text)
        tmp = f.name
    subprocess.run(["rsvg-convert", "-o", str(out_png), tmp], check=True)
    pathlib.Path(tmp).unlink()


# US Letter page in px so rsvg-convert (96dpi: 1px=0.75pt) yields 612x792pt.
LETTER_W, LETTER_H = 816, 1056   # -> 612 x 792 pt (8.5 x 11 in)


def letter_pdf(svg_path: pathlib.Path, out_pdf: pathlib.Path):
    """Vector PDF at exact US Letter size (612x792pt). The canvas is already
    Letter aspect, so this is a clean uniform downscale with no cropping; text
    stays vector and the art is embedded as raster."""
    subprocess.run(
        ["rsvg-convert", "-f", "pdf", "-w", str(LETTER_W), "-h", str(LETTER_H),
         "-o", str(out_pdf), str(svg_path)],
        check=True,
    )


def svg_open() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
    )


def bg_layer_markup(bg_png: pathlib.Path) -> str:
    return (
        f'<image x="0" y="0" width="{W}" height="{H}" '
        f'xlink:href="data:image/png;base64,{b64(bg_png)}"/>'
    )


def write_master_svg(bg_png: pathlib.Path) -> pathlib.Path:
    parts = [svg_open()]
    parts.append(
        '  <g inkscape:groupmode="layer" inkscape:label="background">\n'
        f'    {bg_layer_markup(bg_png)}\n  </g>\n'
    )
    for label, inner in LAYERS.items():
        parts.append(
            f'  <g inkscape:groupmode="layer" inkscape:label="{label}">\n'
            f'    {inner}\n  </g>\n'
        )
    parts.append("</svg>\n")
    out = IMAGES / f"{STEM}.svg"
    out.write_text("".join(parts))
    return out


def layer_svg(inner: str) -> str:
    return svg_open() + "  " + inner + "\n</svg>\n"


def build_ora(bg_png: pathlib.Path, flat_png: pathlib.Path) -> pathlib.Path:
    tmp = pathlib.Path(tempfile.mkdtemp())
    data = tmp / "data"
    data.mkdir()

    render(svg_open() + "  " + bg_layer_markup(bg_png) + "\n</svg>\n",
           data / "background.png")
    for label, inner in LAYERS.items():
        render(layer_svg(inner), data / f"{label}.png")

    lines = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        f'<image version="0.0.3" w="{W}" h="{H}" xres="96" yres="96">',
        "  <stack>",
    ]
    for label in reversed(ORDER):  # first <layer> = topmost
        lines.append(
            f'    <layer name="{label}" src="data/{label}.png" '
            'x="0" y="0" opacity="1.0" visibility="visible" composite-op="svg:src-over"/>'
        )
    lines += ["  </stack>", "</image>", ""]
    (tmp / "stack.xml").write_text("\n".join(lines))

    shutil.copy(flat_png, tmp / "mergedimage.png")
    (tmp / "Thumbnails").mkdir()
    run("magick", flat_png, "-resize", "256x340", tmp / "Thumbnails" / "thumbnail.png")

    out = IMAGES / f"{STEM}.ora"
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("mimetype", "image/openraster", compress_type=zipfile.ZIP_STORED)
        for p in sorted(tmp.rglob("*")):
            if p.is_file() and p.name != "mimetype":
                z.write(p, p.relative_to(tmp), compress_type=zipfile.ZIP_DEFLATED)
    shutil.rmtree(tmp)
    return out


def optimize_small(full_png: pathlib.Path, out_png: pathlib.Path, box: str):
    """Resize within `box` (WxH, aspect preserved) + strip + zopfli-optimize."""
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d) / "small.png"
        run("magick", full_png, "-resize", box, "-strip", tmp)
        if out_png.exists():
            out_png.unlink()
        if shutil.which("zopflipng"):
            run("zopflipng", "-y", tmp, out_png)
        else:
            shutil.copy(tmp, out_png)


def main():
    IMAGES.mkdir(exist_ok=True)
    # drop stale dimensioned PNGs from previous (differently sized) builds
    for stale in IMAGES.glob(f"{STEM}-*x*.png"):
        stale.unlink()
    with tempfile.TemporaryDirectory() as d:
        bg = pathlib.Path(d) / "bg.png"
        compose_art(bg)
        svg = write_master_svg(bg)
        png_full = IMAGES / f"{STEM}-{W}x{H}.png"
        render(svg.read_text(), png_full)
        pdf = IMAGES / f"{STEM}.pdf"
        letter_pdf(svg, pdf)
        ora = build_ora(bg, png_full)
        png_small = IMAGES / f"{STEM}.png"
        optimize_small(png_full, png_small, "548x754")

        # much smaller thumbnail, with its size in the filename
        thumb_tmp = IMAGES / f"{STEM}-thumb.png"
        optimize_small(png_full, thumb_tmp, "200x300")
        dims = subprocess.check_output(
            ["magick", "identify", "-format", "%wx%h", str(thumb_tmp)]
        ).decode().strip()
        png_thumb = IMAGES / f"{STEM}-{dims}.png"
        if png_thumb.exists():
            png_thumb.unlink()
        thumb_tmp.rename(png_thumb)

    for p in (png_small, png_thumb, png_full, pdf, svg, ora):
        print(f"wrote images/{p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
