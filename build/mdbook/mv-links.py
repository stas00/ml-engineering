"""

when chapters are moved around this script rewrites local relative links

python build/mdbook/mv-links.py slurm orchestration/slurm

"""

import datetime
import re
import sys
from pathlib import Path

from utils.build_utils import get_markdown_files
from utils.github_md_utils import md_rename_relative_links, md_process_local_links


def rewrite_links(markdown_path, src, dst):
    md_content = markdown_path.read_text()

    cwd_rel_path = markdown_path.parent
    md_content = md_process_local_links(md_content, md_rename_relative_links, cwd_rel_path=cwd_rel_path, src=src, dst=dst)

    markdown_path.write_text(md_content)


if __name__ == "__main__":

    src, dst = sys.argv[1:3]

    print(f"Renaming {src} => {dst}")

    md_chapters_file = Path("chapters-md.txt")
    markdown_files = get_markdown_files(md_chapters_file)

    for markdown_file in markdown_files:
        rewrite_links(markdown_file, src=src, dst=dst)
