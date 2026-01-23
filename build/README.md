# Book Building

Important: this is still a WIP - it mostly works, but stylesheets need some work to make the pdf really nice. Should be complete in a few weeks.

This document assumes you're working from the root of the repo.

## Installation requirements

1. Install python packages used during book build
```
pip install -r build/requirements.txt
```

2. Download the free version of [Prince XML](https://www.princexml.com/download/). It's used to build the pdf version of this book.


## Build html

```
make html
```

## Build pdf

```
make pdf
```

It will first build the html target and then will use it to build the pdf version.

## Build epub

```
make epub
```

It will first build the html target and then will use it to build the epub version.


## Check links and anchors

To validate that all local links and anchored links are valid run:
```
make check-links-local
```

To additionally also check external links
```
make check-links-all
```
use the latter sparingly to avoid being banned for hammering servers.


## Move md files/dirs and adjust relative links


e.g. `slurm` => `orchestration/slurm`
```
src=slurm
dst=orchestration/slurm

mkdir -p orchestration
git mv $src $dst
perl -pi -e "s|$src|$dst|" chapters-md.txt
python build/mdbook/mv-links.py $src $dst
git checkout $dst
make check-links-local

```

## Resize images

When included images are too large, make them smaller a bit:

```
mogrify -format png -resize 1024x1024\> *png
```
