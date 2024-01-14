# usage: make help

.PHONY: help spell html pdf checklinks clean
.DEFAULT_GOAL := help

help: ## this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

# pip install codespell
spell: ## spellcheck
	@codespell --write-changes --skip "*.pdf" .

html: ## make html version
	python build/utils/md-to-html.py
	echo book-front.html > chapters-html.txt
	perl -ne 's|\.md|.html|; print' chapters-md.txt >> chapters-html.txt

pdf: html ## make pdf version (from html files)
	prince --no-author-style -s build/prince_style.css --pdf-title="Stas Bekman - Machine Learning Engineering ($$(date))" -o "Stas Bekman - Machine Learning Engineering.pdf" $$(cat chapters-html.txt | tr "\n" " ")

check-links-local: html ## check local links
	linkchecker --config build/linkcheckerrc $$(cat chapters-html.txt | tr "\n" " ") | tee -a linkchecker-local.txt

check-links-all: html ## check all links including external ones
	linkchecker --config build/linkcheckerrc $$(cat chapters-html.txt | tr "\n" " ") |  --check-extern --user-agent="Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0" | tee -a linkchecker-all.txt

clean: ## remove build files
	find . -name "*html" -exec rm {} \;
