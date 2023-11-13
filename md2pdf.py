import PyPDF2
import datetime
import markdown
import os

from markdown_it import MarkdownIt
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.front_matter import front_matter_plugin
from weasyprint import HTML

def convert_markdown_to_pdf(markdown_path, pdf_path):
    md = (
        MarkdownIt('commonmark' ,{'breaks':True,'html':True})
        .use(front_matter_plugin)
        .use(footnote_plugin)
        .enable('table')
    )
    try:
        with open(markdown_path, 'r', encoding='utf-8') as md_file:
            text = md_file.read()
        tokens = md.parse(text)
        html_content = md.render(text)
        # html_content = markdown_to_html(markdown_path)
        html_to_pdf(html_content, pdf_path)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def markdown_to_html(markdown_path):
    with open(markdown_path, 'r', encoding='utf-8') as md_file:
        markdown_text = md_file.read()
        html = markdown.markdown(markdown_text)
    return html

def html_to_pdf(html_content, pdf_path):
    HTML(string=html_content).write_pdf(pdf_path)

def concatenate_pdfs(pdf_files, output_pdf):
    pdf_merger = PyPDF2.PdfMerger()
    for pdf_file in pdf_files:
        pdf_merger.append(pdf_file)
    pdf_merger.write(output_pdf)
    pdf_merger.close()

def make_cover_page_file(cover_md_file, date):
    with open(cover_md_file, "w") as f:
        f.write(f"""
## Machine Learning Engineering

This is a PDF version of [Machine Learning Engineering by Stas Bekman](https://github.com/stas00/ml-engineering).

As this book is an early work in progress that gets updated frequently, if you downloaded it as a pdf file, chances are that it's already outdated - make sure to check the latest version at [https://github.com/stas00/ml-engineering](https://github.com/stas00/ml-engineering).

This PDF was generated on {date}.
""")
    return cover_md_file

def get_markdown_files(chapters_file):
    with open(chapters_file) as f:
        chapters = [l for l in f.read().splitlines() if len(l)>0]
    return chapters

if __name__ == "__main__":

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    cover_md_file = "book-front.md"
    chapters_file = "chapters.txt"
    pdf_file = f"Stas Bekman - Machine Learning Engineering ({date}).pdf"

    markdown_files = [make_cover_page_file(cover_md_file, date)] + get_markdown_files(chapters_file)

    cleanup_files = [cover_md_file]

    pdf_files = []
    for markdown_file in markdown_files:
        pdf_file = markdown_file.replace(".md", ".pdf")
        if convert_markdown_to_pdf(markdown_file, pdf_file):
            pdf_files.append(pdf_file)

    cleanup_files += pdf_files

    if pdf_files:
        concatenate_pdfs(pdf_files, pdf_file)
        print(f"PDFs successfully concatenated into {pdf_file}")
    else:
        print(f"No PDFs generated, check {chapters_file}")

    for f in cleanup_files:
        os.unlink(f)
