# usage: make help

.PHONY: help spell prep-html-files html html-local pdf epub upload check-links-local check-links-all clean
.DEFAULT_GOAL := help

help: ## this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

# pip install codespell
spell: ## spellcheck
	@codespell --write-changes --skip "*.pdf" --skip "*.json"

prep-html-files: ## prepare html-files
	echo book-front.html > chapters-html.txt
	perl -ne 's|\.md|.html|; print' chapters-md.txt >> chapters-html.txt

html: prep-html-files ## make html version w/ scripts linking to their url at my github repo
	python build/mdbook/md-to-html.py

html-local: prep-html-files ## make html version w/ scripts remaining local
	python build/mdbook/md-to-html.py --local

pdf: html ## make pdf version (from html files)
	prince --no-author-style -s build/prince_style.css --pdf-title="Stas Bekman - Machine Learning Engineering ($$(date))" -o "Stas Bekman - Machine Learning Engineering.pdf" $$(cat chapters-html.txt | tr "\n" " ")

epub: html ## make epub version (from html files)
	python build/mdbook/preprocess-html-for-epub.py && \
	pandoc --from html --to epub3 \
		--output "Stas Bekman - Machine Learning Engineering.epub" \
		--metadata title="Machine Learning Engineering" \
		--metadata author="Stas Bekman" \
		--metadata date="$$(date +%Y-%m-%d)" \
		--metadata language="en" \
		--epub-cover-image=images/Machine-Learning-Engineering-book-cover.png \
		--resource-path=.:$$(cat chapters-html.txt | xargs -n1 dirname | awk '!seen[$$0]++' | tr "\n" ":") \
		$$(cat chapters-html.txt | tr "\n" " ")

upload: pdf epub ## upload pdf to the hub
	cp "Stas Bekman - Machine Learning Engineering.pdf" ml-engineering-book/
	cp "Stas Bekman - Machine Learning Engineering.epub" ml-engineering-book/
	cd ml-engineering-book/ && git commit -m "new version" "Stas Bekman - Machine Learning Engineering.pdf" "Stas Bekman - Machine Learning Engineering.epub" && git push

check-links-local: html-local ## check local links
	linkchecker --config build/linkcheckerrc $$(cat chapters-html.txt | tr "\n" " ") | tee linkchecker-local.txt

check-links-all: html ## check all links including external ones
	linkchecker --config build/linkcheckerrc $$(cat chapters-html.txt | tr "\n" " ") --check-extern --user-agent="Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0" | tee linkchecker-all.txt

clean: ## remove build files
	find . -name "*html" -exec rm {} \;
