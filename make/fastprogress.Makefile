# usage: make help

.PHONY: clean clean-test clean-pyc clean-build docs help clean-pypi clean-build-pypi clean-pyc-pypi clean-test-pypi dist-pypi upload-pypi clean-conda clean-build-conda clean-pyc-conda clean-test-conda dist-conda upload-conda test tag bump bump-minor bump-major bump-dev bump-minor-dev bump-major-dev bump-post-release commit-tag git-pull git-not-dirty test-install upload release

version_file = fastprogress/version.py
version = $(shell python setup.py --version)

.DEFAULT_GOAL := help


define WAIT_TILL_PIP_VER_IS_AVAILABLE_BASH =
# note that when:
# bash -c "command" arg1
# is called, the first argument is actually $0 and not $1 as it's inside bash!
#
# is_pip_ver_available "1.0.14"
# returns (echo's) 1 if yes, 0 otherwise
#
# since pip doesn't have a way to check whether a certain version is available,
# here we are using a hack, calling:
# pip install fastprogress==
# which doesn't find the unspecified version and returns all available
# versions instead, which is what we search
function is_pip_ver_available() {
    local ver="$$0"
    local out="$$(pip install fastprogress== |& grep $$ver)"
    if [[ -n "$$out" ]]; then
        echo 1
    else
        echo 0
    fi
}

function wait_till_pip_ver_is_available() {
    local ver="$$1"
    if [[ $$(is_pip_ver_available $$ver) == "1" ]]; then
        echo "fastprogress-$$ver is available on pypi"
        return 0
    fi

    COUNTER=0
    echo "waiting for fastprogress-$$ver package to become visible on pypi:"
    while [[ $$(is_pip_ver_available $$ver) != "1" ]]; do
        echo -en "\\rwaiting: $$COUNTER secs"
        COUNTER=$$[$$COUNTER +5]
	    sleep 5
    done
    echo -e "\rwaited: $$COUNTER secs    "
    echo -e "fastprogress-$$ver is now available on pypi"
}

echo "checking version $$0"
wait_till_pip_ver_is_available "$$0"
endef
export WAIT_TILL_PIP_VER_IS_AVAILABLE_BASH

help: ## this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


##@ PyPI

clean-pypi: clean-build-pypi clean-pyc-pypi clean-test-pypi ## remove all build, test, coverage and python artifacts

clean-build-pypi: ## remove pypi build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc-pypi: ## remove python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test-pypi: ## remove pypi test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

dist-pypi: clean-pypi ## build pypi source and wheel package
	@echo "\n\n*** Building pypi packages"
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

upload-pypi: dist-pypi ## release pypi package
	@echo "\n\n*** Uploading" dist/* "to pypi\n"
	twine upload dist/*


##@ Conda

clean-conda: clean-build-conda clean-pyc-conda clean-test-conda ## remove all build, test, coverage and python artifacts

clean-build-conda: ## remove conda build artifacts
	@echo "\n\n*** conda build purge"
	conda build purge-all
	@echo "\n\n*** rm -fr conda-dist/"
	rm -fr conda-dist/

clean-pyc-conda: ## remove conda python file artifacts

clean-test-conda: ## remove conda test and coverage artifacts

dist-conda: clean-conda ## build conda package
	@echo "\n\n*** Building conda package"
	mkdir "conda-dist"
	conda-build ./conda/ --output-folder conda-dist
	ls -l conda-dist/noarch/*tar.bz2

upload-conda: ## release conda package
	@echo "\n\n*** Uploading" conda-dist/noarch/*tar.bz2 "to fastai@anaconda.org\n"
	anaconda upload conda-dist/noarch/*tar.bz2 -u fastai



##@ Combined (pip and conda)

## package and upload a release

clean: clean-pypi clean-conda ## clean pip && conda package

dist: clean dist-pypi dist-conda ## build pip && conda package

upload: upload-pypi upload-conda ## release pip && conda package

install: clean ## install the package to the active python's site-packages
	python setup.py install

test: ## run tests with the default python
	python setup.py --quiet test

tools-update: ## install/update build tools
	@echo "\n\n*** Updating build tools"
	conda install -y conda-verify conda-build anaconda-client
	pip install -U twine

update-fastai: ## reminder to update fastai deps
	@echo "\n\n*** Reminder"
	@echo "If this was a bug fix or a change of API, now update the following 3 'fastai' dependency files:\n  * conda/meta.yaml\n  * imports/core.py\n  * setup.py\nwith this release's 'fastprogress' version number."

release: ## do it all (other than testing)
	${MAKE} tools-update
	${MAKE} git-pull
	${MAKE} test
	${MAKE} git-not-dirty
	${MAKE} bump
	${MAKE} commit-tag
	${MAKE} dist
	${MAKE} upload
	${MAKE} test-install
	${MAKE} update-fastai

##@ git helpers

git-pull: ## git pull
	@echo "\n\n*** Making sure we have the latest checkout"
	git pull
	git status

git-not-dirty:
	@echo "*** Checking that everything is committed"
	@if [ -n "$(shell git status -s)" ]; then\
		echo "git status is not clean. You have uncommitted git files";\
		exit 1;\
	else\
		echo "git status is clean";\
    fi

commit-tag: ## commit and tag the release
	@echo "\n\n*** Commit $(version) version"
	git commit -m "version $(version) release" $(version_file)

	@echo "\n\n*** Tag $(version) version"
	git tag -a $(version) -m "$(version)" && git push --tags

	@echo "\n\n*** Push all changes"
	git push


##@ Testing new package installation

test-install: ## test conda/pip package by installing that version them
	@echo "\n\n*** Install/uninstall $(version) pip version"
	@pip uninstall -y fastprogress

	@echo "\n\n*** waiting for $(version) pip version to become visible"
	bash -c "$$WAIT_TILL_PIP_VER_IS_AVAILABLE_BASH" $(version)

	pip install fastprogress==$(version)
	pip uninstall -y fastprogress

	@echo "\n\n*** Install/uninstall $(version) conda version"
	@# skip, throws error when uninstalled @conda uninstall -y fastprogress

	@echo "\n\n*** waiting for $(version) conda version to become visible"
	@perl -e '$$v=shift; $$p="fastprogress"; $$|++; sub ok {`conda search -c fastai $$p==$$v >/dev/null 2>&1`; return $$? ? 0 : 1}; print "waiting for $$p-$$v to become available on conda\n"; $$c=0; while (not ok()) { print "\rwaiting: $$c secs"; $$c+=5;sleep 5; }; print "\n$$p-$$v is now available on conda\n"' $(version)

	conda install -y -c fastai fastprogress==$(version)
	@# leave conda package installed: conda uninstall -y fastprogress


##@ Version bumping

# Support semver, but using python's .dev0/.post0 instead of -dev0/-post0

bump-major: ## bump major level; remove .devX if any
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2+1, 0, 0); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-minor: ## bump minor level; remove .devX if any
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2, $$3+1, 0); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-patch: ## bump patch level unless has .devX, then don't bump, but remove .devX
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=$$5 ? join(".", $$2, $$3, $$4) :join(".", $$2, $$3, $$4+1); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump: bump-patch ## alias to bump-patch (as it's used often)

bump-post-release: ## add .post1 or bump post-release level .post2, .post3, ...
	@perl -pi -e 's{((\d+\.\d+\.\d+)(\.\w+\d+)?)}{do { $$o=$$1; $$b=$$2; $$l=$$3||".post0"}; $$l=~s/(\d+)$$/$$1+1/e; $$n="$$b$$l"; print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n}e' $(version_file)

bump-major-dev: ## bump major level and add .dev0
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2+1, 0, 0, "dev0"); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-minor-dev: ## bump minor level and add .dev0
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2, $$3+1, 0, "dev0"); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-patch-dev: ## bump patch level and add .dev0
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2, $$3, $$4+1, "dev0"); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-dev: bump-patch-dev ## alias to bump-patch-dev (as it's used often)



###@ Coverage
# coverage: ## check code coverage quickly with the default python
# 	coverage run --source fastprogress -m pytest
# 	coverage report -m
# 	coverage html
# 	$(BROWSER) htmlcov/index.html
