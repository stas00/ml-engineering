# Makefiles

Makefiles have been around since 1976! They are extremely useful almost 50 years later. Wow!

## Current Makefiles

I'm lazy, so when I need to release a new package that I maintain, I like to just type:

```
make release
```
and have it bump up the version, update `CHANGES.md`, tag the release, start a new dev branch, commit all that, build the `pip`/`conda` packages, upload them to pypi/conda servers, wait till they become available, install them and test that they work correctly. Speaking of lazy...

You can see an example of such powerful [Makefile](https://github.com/stas00/ipyexperiments/blob/master/Makefile) for my [ipyexperiments](https://github.com/stas00/ipyexperiments) package.

Here are some Makefiles from projects I worked on in the past and that were worth saving, so that parts of those could be re-used in new projects.

## fastai Makefiles

I did work on fastai a few years back and created a pretty complex `Makefile` that made a complicated release process with just `make release`. Of course, it has a bunch of other useful targets in it. After I left they were removed, so you can still find them here:

* fastai [Makefile](./fastai.Makefile) + [Usage doc](./fastai.release.md).
* fastprogress [Makefile](./fastprogress.Makefile) + [Usage doc](./fastprogress.release.md).

`fastprogress` is another of fastai projects which is much less complex and had a much simpler, yet still powerful `Makefile`.

