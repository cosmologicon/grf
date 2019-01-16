# Profiling various Algorithm X implementations

This directory is used for profiling grf against various implementations of Algorithm X from
elsewhere. The implementations considered are:

* `grf` (this repository): pure Python
* [`libdlx` by tdons](https://github.com/tdons/dlx): C with Python bindings
* [`dlx-cpp` by jlaire](https://github.com/jlaire/dlx-cpp): C++

## Instructions to set up

Run in this directory from a clean checkout.

	sudo apt install cmake

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

## To run the profiler

