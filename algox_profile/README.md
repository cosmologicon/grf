# Profiling various Algorithm X implementations

This directory is used for profiling grf against various implementations of Algorithm X from
elsewhere. The implementations considered are:

* `grf` (this repository): pure Python
* `libdlx`: C with Python bindings

## Instructions to run

Run in this directory from a clean checkout.

	sudo apt install cmake

	# Set up grf
	ln -s ../grf.py .

	# Set up libdlx
	git clone https://github.com/tdons/dlx.git
	pushd dlx
	make
	popd
	


