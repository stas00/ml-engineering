# Misc recipes


## check only files that are under git and are .py files and are under specific top-level dirs

```
check_dirs := examples tests src utils

# get modified files since the branch was made
fork_point_sha := $(shell git merge-base --fork-point master)
joined_dirs := $(shell echo $(check_dirs) | tr " " "|")
modified_py_files := $(shell git diff --name-only $(fork_point_sha) | egrep '^($(joined_dirs))' | egrep '\.py$$')
#$(info modified files are: $(modified_py_files))

modified_only_fixup:
	@if [ -n "$(modified_py_files)" ]; then \
		echo "Checking/fixing $(modified_py_files)"; \
		black $(modified_py_files); \
		isort $(modified_py_files); \
		flake8 $(modified_py_files); \
	else \
		echo "No library .py files were modified"; \
	fi
```
also rewrote it in python due to some windows setup not being able to handle that [https://github.com/huggingface/transformers/blob/517eaf460b06936f41c0d3c5c92c2c7feaf61fc7/utils/get_modified_files.py](https://github.com/huggingface/transformers/blob/517eaf460b06936f41c0d3c5c92c2c7feaf61fc7/utils/get_modified_files.py).
