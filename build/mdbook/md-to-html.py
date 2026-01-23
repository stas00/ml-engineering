import argparse
import datetime
import re

from functools import partial
from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin
from pathlib import Path

from utils.github_md_utils import md_header_to_anchor, md_process_local_links, md_expand_links, md_convert_md_target_to_html
from utils.build_utils import get_markdown_files

mdit = (
    MarkdownIt('commonmark', {'breaks':True, 'html':True})
    .use(anchors_plugin, max_level=7, permalink=False, slug_func=md_header_to_anchor)
    .enable('table')
)

my_repo_url = "https://github.com/stas00/ml-engineering/blob/master"

def convert_markdown_to_html(markdown_path, args):
    md_content = markdown_path.read_text()

    cwd_rel_path = markdown_path.parent

    repo_url = my_repo_url if not args.local else ""
    md_content = md_process_local_links(md_content, md_expand_links, cwd_rel_path=cwd_rel_path, repo_url=repo_url)
    md_content = md_process_local_links(md_content, md_convert_md_target_to_html)

    #tokens = mdit.parse(md_content)
    html_content = mdit.render(md_content)
    # we don't want <br />, since github doesn't use it in its md presentation
    html_content = re.sub('<br />', '', html_content)

    html_file = markdown_path.with_suffix(".html")
    html_file.write_text(html_content)

def make_cover_page_file(cover_md_file, date):
    with open(cover_md_file, "w") as f:
        f.write(f"""
![](images/Machine-Learning-Engineering-book-cover.png)

## Machine Learning Engineering Open Book

This is an ebook version of [Machine Learning Engineering Open Book by Stas Bekman](https://github.com/stas00/ml-engineering/) generated on {date}.

As this book is constantly being updated, if you downloaded it as a pdf or an epub file and the date isn't recent, chances are that it's already outdated - make sure to check the latest version at [https://github.com/stas00/ml-engineering](https://github.com/stas00/ml-engineering/).
""")
    return Path(cover_md_file)

def write_html_index(html_chapters_file, markdown_files):
    html_chapters = [str(l.with_suffix(".html")) for l in markdown_files]
    html_chapters_file.write_text("\n".join(html_chapters))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--local',  action="store_true", help="all local files remain local")
    args = parser.parse_args()

    date = datetime.datetime.now().strftime("%Y-%m-%d")

    cover_md_file = "book-front.md"

    md_chapters_file = Path("chapters-md.txt")
    html_chapters_file = Path("chapters-html.txt")

    pdf_file = f"Stas Bekman - Machine Learning Engineering ({date}).pdf"

    markdown_files = [make_cover_page_file(cover_md_file, date)] + get_markdown_files(md_chapters_file)

    pdf_files = []
    for markdown_file in markdown_files:
        convert_markdown_to_html(markdown_file, args)

    write_html_index(html_chapters_file, markdown_files)
