# usage: make help

.PHONY: help spell html pdf checklinks clean
.DEFAULT_GOAL := help

help: ## this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

# pip install codespell
spell: ## spellcheck
	@codespell --write-changes .

html: ## make html version
	python utils/md-to-html.py

pdf: html ## make pdf version (from html files)
	prince --no-author-style -s build/prince_style.css --pdf-title="Stas Bekman - Machine Learning Engineering ($$(date))" -o "Stas Bekman - Machine Learning Engineering.pdf" $$(cat chapters-html.txt | tr "\n" " ")

checklinks: html ## check links
	linkchecker --ignore-url=index --file-output=html --config build/linkcheckerrc $$(cat chapters-html.txt | tr "\n" " ")

clean: ## remove build files
	find . -name "*html" -exec rm {} \;
