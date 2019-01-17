# Profiling various Algorithm X implementations

This directory is used for profiling grf against various implementations of Algorithm X from
elsewhere. The implementations considered are:

* `grf` (this repository): pure Python
* [`libdlx` by tdons](https://github.com/tdons/dlx): C with Python bindings
* [`dlx-cpp` by jlaire](https://github.com/jlaire/dlx-cpp): C++
* [`exact-cover` by Arthur Lee](https://hackage.haskell.org/package/exact-cover): Java

## Instructions to set up

Run in this directory from a clean checkout.

	sudo apt install cmake cabal-install

	# Set up grf
	ln -s ../grf.py .

	# Set up libdlx
	git clone https://github.com/tdons/dlx.git
	pushd dlx
	make
	popd
	
	# Set up dlx-cpp
	git clone https://github.com/jlaire/dlx-cpp.git
	pushd dlx-cpp
	make examples
	popd
	# Executable exists in: dlx-cpp/build/dlx -pvs	

	# Set up exact-cover
	cabal update
	curl -O https://hackage.haskell.org/package/exact-cover-0.1.0.0/exact-cover-0.1.0.0.tar.gz
	tar xzf exact-cover-0.1.0.0.tar.gz
	rm exact-cover-0.1.0.0.tar.gz
	pushd exact-cover-0.1.0.0
	cabal install -fbuildExamples
	...?
	ghc -XViewPatterns ...hs?
	popd

TODO: learn enough Haskell to make that one work.

## To run the profiler

