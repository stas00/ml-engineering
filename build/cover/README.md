To rebuild:

```
python3 build_final_cover.py
```

Requirements: ImageMagick (magick) + rsvg-convert; zopflipng optional (used to shrink PNGs further if present).

The only input is `sources/Machine-Learning-Engineering-art.png` (the robot art with no
text, flattened from the GIMP source). The title/author are re-typeset as editable text
layers, the dark background is extended to US Letter aspect, and the script writes the
full-resolution PNG, an optimized display PNG, a small thumbnail, a true US Letter PDF
(612x792pt), a layered SVG, and a layered ORA into `../../images/`.
