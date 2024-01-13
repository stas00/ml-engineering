

from github_md_utils import md_header_to_anchor, md_process_local_links, md_expand_links, md_convert_md_target_to_html
from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin
from pathlib import Path
import datetime

mdit = (
    MarkdownIt('commonmark', {'breaks':True, 'html':True})
    .use(anchors_plugin, max_level=7, permalink=False, slug_func=md_header_to_anchor)
    .enable('table')
)

def convert_markdown_to_html(markdown_path):
    text = markdown_path.read_text()

    cwd_path = markdown_path.parent
    text = md_process_local_links(text, cwd_path, md_expand_links)
    text = md_process_local_links(text, cwd_path, md_convert_md_target_to_html)

    tokens = mdit.parse(text)
    html_content = mdit.render(text)

    html_file = markdown_path.with_suffix(".html")
    html_file.write_text(html_content)

def make_cover_page_file(cover_md_file, date):
    with open(cover_md_file, "w") as f:
        f.write(f"""
## Machine Learning Engineering

This is a PDF version of [Machine Learning Engineering by Stas Bekman](https://github.com/stas00/ml-engineering).

As this book is an early work in progress that gets updated frequently, if you downloaded it as a pdf file, chances are that it's already outdated - make sure to check the latest version at [https://github.com/stas00/ml-engineering](https://github.com/stas00/ml-engineering).

This PDF was generated on {date}.
""")
    return Path(cover_md_file)

def get_markdown_files(md_chapters_file, html_chapters_file):
    return [Path(l) for l in md_chapters_file.read_text().splitlines() if len(l)>0]

def write_html_index(html_chapters_file, markdown_files):
    html_chapters = [str(l.with_suffix(".html")) for l in markdown_files]
    html_chapters_file.write_text("\n".join(html_chapters))


if __name__ == "__main__":

    date = datetime.datetime.now().strftime("%Y-%m-%d")

    cover_md_file = "book-front.md"

    md_chapters_file = Path("chapters-md.txt")
    html_chapters_file = Path("chapters-html.txt")

    pdf_file = f"Stas Bekman - Machine Learning Engineering ({date}).pdf"

    markdown_files = [make_cover_page_file(cover_md_file, date)] + get_markdown_files(md_chapters_file, html_chapters_file)

    pdf_files = []
    for markdown_file in markdown_files:
        convert_markdown_to_html(markdown_file)

    write_html_index(html_chapters_file, markdown_files)
