# grf

Simple tools for solving graph problems in python. Currently implemented:

  * Hamiltonian path and cycle
  * Exact cover
  * A* path finding

## To install

Download `grf.py` and put it in your source directory.

## Representing data

Data structures in grf are as simple and general as possible, to let you use whatever you're comfortable with. Examples of graphs that grf accepts are:

    [[0, 1], [0, 2], [0, 3], [2, 3]]
    set(("n0", "n1"), ("n0", "n2"), ("n0", "n3"), ("n2", "n3"))
    "AB AC AD CD".split()

Graphs are represented as collections of edges. Each edge is a length-2 collection of the two nodes it connects. Nodes can be any hashable data type (e.g. strings, numbers, tuples, frozensets).

Exact cover takes a collection of collections of nodes. Again, any python collection should be fine.

    [(1, 4, 7), (1, 4), (4, 5, 7), (3, 5, 6), (2, 3, 6, 7), (2, 7)]

## API

### Exact cover

If you have a set of nodes, and a set of subsets of those nodes, the exact cover problem is finding a set of those subsets such that every node appears in exactly one subset.

The name "exact cover" refers to the problem of tiling a chessboard with shapes, each of which covers multiple squares. In that case, a node is a square on the chessboard, and a subset of nodes is the subset of squares covered by a certain piece in a certain place with a certain orientation. Exact cover problems, in general, do not need to resemble this specific case. Logic puzzles that can be expressed in terms of constraints, such as Sudoku, can often be implemented in terms of exact cover.

You will pass in a collection of subsets of nodes, and receive back a set of those subsets. For example:

	input: [(1, 4, 7), (1, 4), (4, 5, 7), (3, 5, 6), (2, 3, 6, 7), (2, 7)]
	output: [(1, 4), (3, 5, 6), (2, 7)]

If you pass in a dict as input, it will be treated as a mapping from subset names to the subsets themselves, and the output will be a collection of subset names instead:

	input: {"A": (1, 4, 7), "B": (1, 4), "C": (4, 5, 7),
		"D": (3, 5, 6), "E": (2, 3, 6, 7), "F": (2, 7)}
	output: ["B", "D", "F"]

The `exact_cover` function will return a solution if one exists, or `None` if not:

	solution = grf.exact_cover(subsets)

The `exact_covers` function will return a list of all solutions. The list will be empty if no solution exists:

	solutions = grf.exact_covers(subsets)

You can optionally specify a `max_solutions` argument. The solver will stop at this many solutions if it reaches them. For the purpose of determining whether solutions are distinct, subsets that appear multiple times in the input are counted as distinct.

	solutions = grf.exact_covers(subsets, max_solutions = 10)

The `can_exact_cover` function returns `True` or `False` depending on whether a solution exists:

	solution_exists = grf.can_exact_cover(subsets)

The `unique_exact_cover` returns `True` if exactly one solution exists, or `False` if 0 or multiple solutions exist:

	solution_is_unique = grf.unique_exact_cover(subsets)

The `solve_unique_exact_cover` function returns a length-2 tuple. The first element is a solution if one exists, or `None` otherwise. The second element is a bool signifying whether exactly one solution exists:

	solution, solution_is_unique = grf.solve_unique_exact_cover(subsets)

For each of these functions, you can also optionally specify a `nodes` argument, which is a collection of all nodes. This is generally unnecessary, as the set of all nodes can be derived from the subsets. Specifying this just means no solution will be found if any node from the set of all nodes does not appear in any subset.

	grf.exact_cover([(1, 2), (3, 4)], nodes = [1, 2, 3, 4, 5]) => None
	
It is a `ValueError` for the set of all nodes to have a node appear more than once:

	grf.exact_cover([(1, 2), (1, 3)], nodes = [1, 1, 2, 3]) => ValueError

If a subset contains a node more than once, that subset cannot appear in a solution, although the
repeated node is still required to be in the output:

	grf.exact_cover([(1, 1), (2,)]) => None

See the next section to allow for nodes to appear more than once.

### Generalized exact cover

The generalized problem allows nodes to appear more than once in the result, or not at all. Every
node has a minimum number of times it must appear in the output, and a maximum number of times it
must appear.




### Partial cover

Partial cover is similar to exact cover, but not every node needs to appear in the solution. It takes two arguments, a collection of subsets of nodes, and a collection of nodes that must be covered.

	solution = grf.partial_cover(subsets, required_nodes)

Every node in `required_nodes` will appear exactly once in the solution. Every other node will appear either zero or one times.

Other variations are available, as with `exact_cover`:

	solutions = grf.partial_covers(subsets, nodes)
	solutions = grf.partial_covers(subsets, nodes, max_solutions = 10)
	solution_exists = grf.can_partial_cover(subsets, nodes)
	solution_is_unique = grf.unique_partial_cover(subsets, nodes)
	solution, solution_is_unique = grf.solve_unique_partial_cover(subsets, nodes)

