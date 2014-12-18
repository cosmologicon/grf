grf
===

Simple tools for solving graph problems in python. Currently implemented:

  * Hamiltonian path and cycle
  * Exact cover

To install
----------

Download `grf.py` and put it in your source directory.

Representing data
-----------------

Data in grf is as simple and general as possible, to let you use whatever you're comfortable with. Examples of graphs that grf accepts are:

    [[0, 1], [0, 2], [0, 3], [2, 3]]
    set(("n0", "n1"), ("n0", "n2"), ("n0", "n3"), ("n2", "n3"))
    "AB AC AD CD".split()

Graphs are represented as collections of edges. Each edge is a length-2 collection of the two nodes it connects. Nodes can be any hashable, sortable data type (e.g. strings, numbers, tuples).

Exact cover takes a collection of collections of nodes. Again, any python collection should be fine.

    [(1, 4, 7), (1, 4), (4, 5, 7), (3, 5, 6), (2, 3, 6, 7), (2, 7)]
