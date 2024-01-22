from pathlib import Path

def get_markdown_files(md_chapters_file):
    return [Path(l) for l in md_chapters_file.read_text().splitlines() if len(l)>0]
