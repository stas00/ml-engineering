import os
import markdown
from weasyprint import HTML
import PyPDF2

from markdown_it import MarkdownIt
from mdit_py_plugins.front_matter import front_matter_plugin
from mdit_py_plugins.footnote import footnote_plugin

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

def find_markdown_files(root_dir):
    markdown_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):
                markdown_files.append(os.path.join(root, file))
    return markdown_files

if __name__ == "__main__":
    root_directory = "." 
    out_pdf_path = 'book.pdf'
    markdown_files = find_markdown_files(root_directory)

    if not markdown_files:
        print("No Markdown files found.")
    else:
        pdf_files = []
        for markdown_file in markdown_files:
            pdf_file = markdown_file.replace(".md", ".pdf")
            if convert_markdown_to_pdf(markdown_file, pdf_file):
                pdf_files.append(pdf_file)

        if pdf_files:
            concatenate_pdfs(pdf_files, out_pdf_path)
            print(f"PDFs successfully concatenated into {out_pdf_path}")
        else:
            print("No PDFs generated.")
