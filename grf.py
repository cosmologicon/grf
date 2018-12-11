import math, random, string
from collections import defaultdict, Counter

def nodes(graph):
	return sorted(set.union(*map(set, graph))) if graph else []

def is_connected(graph):
	if not graph:
		return True
	edges = list(set(edge) for edge in graph)
	connected_nodes = edges.pop(0)
	while edges:
		connected_edges = [edge for edge in edges if connected_nodes & edge]
		if not connected_edges:
			return False
		edges = [edge for edge in edges if not connected_nodes & edge]
		connected_nodes |= set.union(*connected_edges)
	return True

def graph_to_adjacency(graph):
	adjacency = defaultdict(set)
	for node0, node1 in graph:
		adjacency[node0].add(node1)
		adjacency[node1].add(node0)
	return dict((node, sorted(anodes)) for node, anodes in adjacency.items())

# http://www.cs.berkeley.edu/~sinclair/cs271/n14.pdf
def hamiltonian_path(graph, require_cycle=False):
	if not is_connected(graph):
		return False
	all_nodes = nodes(graph)
	n = len(all_nodes)
	if n <= 2:
		return list(all_nodes)
	max_steps = int(math.ceil(4 * (n - 1) * math.log(n - 1)))
	adj = graph_to_adjacency(graph)
	path = [random.choice(all_nodes)]
	for step in range(max_steps):
		if len(path) == n and (not require_cycle or path[0] in adj[path[-1]]):
			return path
		y = random.choice(adj[path[-1]])
		if y in path:
			z = path.index(y)
			path = path[z+1:] + path[z::-1]
		else:
			path.append(y)
	return None

def hamiltonian_cycle(graph):
	return hamiltonian_path(graph, require_cycle=True)

# Exact cover using Algorithm X
# http://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X

# Consider a sparse matrix where the jnode'th column corresponds to a node on the graph to be
# covered, and the jsubset'th row corresponds to a subset of these nodes. The matrix is populated at
# (jnode, jsubset) when the corresponding subset contains the corresponding node.

# Exact cover Algorithm X inner loop.
# jnodes: frozenset of jnodes that still have to be covered (i.e. the columns still in the matrix).
#   This one needs to be a frozenset because it's used as a key for dead_cache.
# jsubsets: set of jsubsets that are still available (i.e. the rows still in the matrix)
# node_counts: list mapping jnode to number of available subsets the node appears in, i.e. the total
#   of the jnode'th column. (This is redundant with jsubsets, but tracked for optimization.)
# subsets: list mapping jsubset to the set of jnodes in that subset, i.e. the set of populated
#   columns in the jsubset'th row.
# containers: list mapping jnode to the set of jsubsets that contain it, i.e. the set of populated
#   rows in the jnode'th column.
# overlaps: list mapping jsubset to the set of jsubsets whose subsets intersect this one, i.e. the
#   set of rows that share at least one column with the jsubset'th row.
# dead_cache: set containing inputs (jsubsets) known to produce no output.
def _algox_inner(jnodes, jsubsets, node_counts, subsets, containers, overlaps, dead_cache):
	# Base case: no nodes left to be covered - solution found.
	if not jnodes:
		yield []
		return
	# jnodes is a sufficient cache key - node_counts and jsubsets are redundant within a given problem.
	if dead_cache is not None and jnodes in dead_cache:
		return
	dead = True  # If still true at the end, this input produces no outputs.
	# Select the node that appears in the fewest subsets, i.e. the column with the fewest 1's.
	selected_jnode = min(jnodes, key = node_counts.__getitem__)
	# The subsets that contain the selected node, i.e. the rows in which the selected column has a 1.
	# These are candidates for the solution: exactly one of these subsets must appear in each solution.
	selected_jsubsets = list(containers[selected_jnode] & jsubsets)
	random.shuffle(selected_jsubsets)
	for selected_jsubset in selected_jsubsets:
		# Remaining nodes to be covered, i.e. the columns still in the matrix.
		sub_jnodes = jnodes - subsets[selected_jsubset]
		# Determine which rows will be removed when this subset is used.
		removed_jsubsets = jsubsets & overlaps[selected_jsubset]
		# Remaining subsets available to cover, i.e. the rows still in the matrix.
		sub_jsubsets = jsubsets - removed_jsubsets
		# The node counts remaining when the given rows are removed.
		sub_node_counts = list(node_counts)
		for removed_jsubset in removed_jsubsets:
			for jnode in subsets[removed_jsubset]:
				sub_node_counts[jnode] -= 1
		for solution in _algox_inner(sub_jnodes, sub_jsubsets, sub_node_counts, subsets, containers, overlaps, dead_cache):
			solution.append(selected_jsubset)
			yield solution
			dead = False
	if dead_cache is not None and dead:
		dead_cache.add(jnodes)

# nodes: sequence of nodes.
# subsets: sequence of collections of nodes.
# subset_names: sequence of subset names, in same order as subsets.
def _algox_outer(nodes, subsets, subset_names):
	jnodes = frozenset(range(len(nodes)))
	node_jnodes = { node: jnode for jnode, node in enumerate(nodes) }
	node_set = set(node_jnodes)
	subsetjs = []
	jsubsets = set()
	for jsubset, subset in enumerate(subsets):
		len0 = len(subset)
		subset = set(subset)
		if subset - node_set:
			raise ValueError("Subset contains nodes not in set of all nodes: {}".format(list(subset - node_set)))
		subsetj = set(node_jnodes[node] for node in subset)
		subsetjs.append(subsetj)
		if len(subsetj) == len0:
			jsubsets.add(jsubset)
	containers = [set() for _ in nodes]
	for jsubset in jsubsets:
		for jnode in subsetjs[jsubset]:
			containers[jnode].add(jsubset)
	node_counts = [len(container) for container in containers]
	overlaps = [set() for _ in subsets]
	for jnode, container in enumerate(containers):
		for jsubset in container:
			overlaps[jsubset] |= container
	for solution in _algox_inner(jnodes, jsubsets, node_counts, subsetjs, containers, overlaps, set()):
		yield [subset_names[jsubset] for jsubset in solution]

def _get_subset_names(subsets):
	if isinstance(subsets, dict):
		subset_names, subsets = map(list, zip(*subsets.items()))
	else:
		subset_names = subsets = list(subsets)
	return subset_names, subsets

def _pull_items(generator, max_items):
	ret = []
	for value in generator:
		ret.append(value)
		if max_items is not None and len(ret) >= max_items:
			break
	return ret

def exact_covers(subsets, nodes = None, max_solutions = None):
	if max_solutions is not None and max_solutions <= 0:
		raise ValueError
	subset_names, subsets = _get_subset_names(subsets)
	if nodes is None:
		nodes = list(set(node for subset in subsets for node in subset))
	else:
		nodes = list(nodes)
		if len(nodes) != len(set(nodes)):
			multinodes = [node for node in set(nodes) if nodes.count(node) > 1]
			raise ValueError("Invalid multiple nodes: {}".format(multinodes))
	return _pull_items(_algox_outer(nodes, subsets, subset_names), max_solutions)

def _algox(subsets, nodes, shuffle):
	if not nodes:
		yield []
		return
	node_counts = Counter(node for subset in subsets for node in subset)
	if len(node_counts) < len(nodes):
		return
	selected_node = min(nodes, key = node_counts.get)
	subset_choices = [subset for subset in subsets if selected_node in subset]
	if shuffle:
		random.shuffle(subset_choices)
	for selected_subset in subset_choices:
		new_nodes = [node for node in nodes if node not in selected_subset]
		isvalid = set(selected_subset).isdisjoint
		new_subsets = [subset for subset in subsets if isvalid(set(subset))]
		for subcover in _algox(new_subsets, new_nodes, shuffle):
			yield subcover + [selected_subset]

def partial_covers(subsets, nodes, max_solutions = None):
	subset_names, subsets = _get_subset_names(subsets)
	all_nodes = set.union(*map(set, subsets))
	if not set(nodes) <= all_nodes:
		return []
	index_to_node = list(all_nodes)
	node_to_index = {node: j for j, node in enumerate(index_to_node)}
	index_subsets = [tuple(map(node_to_index.get, subset)) for subset in subsets]
	index_nodes = [node_to_index[node] for node in nodes]
	solutions = []
	shuffle = max_solutions is not None
	for index_solution in _algox(index_subsets, index_nodes, shuffle):
		solution = []
		for index_subset in index_solution:
			subset = subset_names[index_subsets.index(index_subset)]
			solution.append(subset)
		solutions.append(solution)
		if max_solutions is not None and len(solutions) == max_solutions:
			return solutions
	return solutions

def exact_cover(subsets, nodes = None):
	solutions = exact_covers(subsets, nodes = nodes, max_solutions = 1)
	return solutions[0] if solutions else None

def can_exact_cover(subsets, nodes = None):
	return bool(exact_covers(subsets, nodes = nodes, max_solutions = 1))

def unique_exact_cover(subsets, nodes = None):
	return len(exact_covers(subsets, nodes = nodes, max_solutions = 2)) == 1

def can_unique_exact_cover(subsets, nodes = None):
	solutions = exact_covers(subsets, nodes = nodes, max_solutions = 2)
	return (solutions[0] if solutions else None), len(solutions) == 1

def partial_cover(subsets, nodes):
	solutions = partial_covers(subsets, nodes, max_solutions = 1)
	return solutions[0] if solutions else None

def can_partial_cover(subsets, nodes):
	return bool(partial_covers(subsets, nodes, max_solutions = 1))

def unique_partial_cover(subsets, nodes):
	return len(partial_covers(subsets, nodes, max_solutions = 2)) == 1

def can_unique_partial_cover(subsets, nodes):
	solutions = partial_covers(subsets, nodes, max_solutions = 2)
	return (solutions[0] if solutions else None), len(solutions) == 1

# Polyominoes are sets of grid coordinates (which are ordered pairs of integers).
#   Example: ((0, 0), (1, 0), (1, 1), (2, 1))
# Polyominoes may optionally have an additional non-tuple element, which is the piece's name.
#   Example: ("Z", (0, 0), (1, 0), (1, 1), (2, 1))
def _split_poly(poly):
	labels, cells = [], []
	for x in poly:
		if type(x) is tuple:
			cells.append(x)
		else:
			labels.append(x)
	if len(labels) > 1:
		raise ValueError
	if len(labels) == 1:
		return labels[0], tuple(cells)
	return None, tuple(cells)
def _join_poly(label, cells):
	return cells if label is None else (label,) + cells
			
# Parse polyominoes from a string
# Polyominoes are defined with any non-whitespace character
def parse_polyominoes(spec, annotate = False, align = True, allow_disconnected = False):
	spots = defaultdict(list)
	for y, line in enumerate(spec.splitlines()):
		for x, char in enumerate(line):
			if not char.strip():
				continue
			spots[char].append((x, y))
	polys = []
	def is_adjacent(node0, node1):
		(x0, y0), (x1, y1) = node0, node1
		return abs(x0 - x1) + abs(y0 - y1) == 1
	def get_connected(nodes):
		nodesets = []
		for node in nodes:
			thisset = set([node])
			othersets = []
			for nodeset in nodesets:
				if any(is_adjacent(node, n) for n in nodeset):
					thisset |= nodeset
				else:
					othersets.append(nodeset)
			nodesets = othersets + [thisset]
		return nodesets
	for char, nodes in spots.items():
		if allow_disconnected:
			polys.append((char, nodes))
		else:
			for nodeset in get_connected(nodes):
				polys.append((char, nodeset))
	polys = [(char, tuple(sorted(poly))) for char, poly in polys]
	polys.sort(key = lambda cpoly: cpoly[1])
	if align:
		polys = [(char, align_polyomino(poly)) for char, poly in polys]
	if annotate:
		ks, vs = zip(*polys)
		if len(set(ks)) == 1 and ks[0] in ".*#":
			polys = list(zip(_generic_labels(), vs))
	return [(_join_poly(char, poly) if annotate else poly) for char, poly in polys]
		
# A B C ... Z AA AB AC ... AZ BA BB BC ...
def _generic_labels():
	def powerset(sets):
		if not sets:
			yield ""
			return
		for value in sets[0]:
			for other in powerset(sets[1:]):
				yield value + other
	for n in range(1, 100):
		for label in powerset([string.ascii_uppercase] * n):
			yield label

def align_polyomino(poly):
	label, poly = _split_poly(poly)
	xs, ys = zip(*poly)
	x0, y0 = min(xs), min(ys)
	poly = [(x - x0, y - y0) for x, y in sorted(poly)]
	return _join_poly(label, poly)

def polyomino_rotations(poly, flip = False):
	label, poly = _split_poly(poly)
	def rotated(poly):
		return tuple((y, -x) for x, y in poly)
	def flipped(poly):
		return tuple((-x, y) for x, y in poly)
	rotations = set()
	for _ in range(4):
		rotations.add(align_polyomino(poly))
		if flip:
			rotations.add(align_polyomino(flipped(poly)))
		poly = rotated(poly)
	return [(_join_poly(label, poly) for poly in sorted(rotations))]


# https://en.wikipedia.org/wiki/A*_search_algorithm
# A* search where every edge has uniform weight.
# start: the starting state
# goal: the goal state
# neighbors(state): iterate over neighboring states
# h(state): consistent heuristic for number of steps to goal
def astar_uniform(start, goal, neighbors, h):
	import heapq
	checked = set()
	tocheck = [(h(start), start)]
	g = {start: 0}
	previous = {start: None}
	while tocheck:
		_, state = heapq.heappop(tocheck)
		if state == goal:
			path = [goal]
			while path[-1] != start:
				path.append(previous[path[-1]])
			return list(reversed(path))
		checked.add(state)
		newg = g[state] + 1
		for newstate in neighbors(state):
			if newstate in checked:
				continue
			if newstate not in g:
				item = newg + h(state), newstate
				heapq.heappush(tocheck, item)
			elif newg > g[newstate]:
				continue
			g[newstate] = newg
			previous[newstate] = state
	return None

