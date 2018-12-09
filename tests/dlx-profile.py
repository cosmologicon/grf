# Profile various implementations of the inner loop of Algorithm X.

# Need to download this repository:
# https://github.com/DavideCanton/DancingLinksX

# Compared with the baseline implementation, the following improvements *are* worthwhile:
#   * j-indexing subsets and working with jsubsets
#   * auxiliary node count list
#   * lazy evaluation of overlapping subsets (single solutions only)
#   * caching subcalls (multiple solutions only)
# The following make essentially no difference:
#   * mutable vs immutable data structures
#   * in-place updates vs creating new data structures


from __future__ import division
import time, random
from collections import Counter, defaultdict
from DancingLinksX import dlmatrix, alg_x

def get_inputs():
	# Minimal example
	yield [(1, 4, 7), (1, 4), (4, 5, 7), (3, 5, 6), (2, 3, 6, 7), (2, 7)], 1

	# Sudoku
	grid = list("..24..59..6...1.47.....5......3.....3.45.6.2...6.7...44..7....8..9...1...2.....5.")
	def get_constraints(row, column, digit):
		yield "f", row, column
		yield "r", row, digit
		yield "c", column, digit
		yield "b", int(row // 3), int(column // 3), digit
	constraints = { (row * 9 + column, digit) : list(get_constraints(row, column, digit))
		for row in range(9) for column in range(9) for digit in "123456789" }
	used_constraints = set().union(*[cons for (j, digit), cons in constraints.items() if grid[j] == digit])
	available = {name: cons for name, cons in constraints.items() if used_constraints.isdisjoint(cons)}
	yield list(available.values()), 1

	# Dominos on a NxN grid
	N = 6
	tiles = [((x, y), (x + 1, y)) for x in range(N - 1) for y in range(N)]
	tiles += [((x, y), (x, y + 1)) for x in range(N) for y in range(N - 1)]
	yield [[x + N * y for x, y in tile] for tile in tiles], 6728

def profile(algox, algox_args):
	ret = []
	for input, noutput in get_inputs():
		input = canonicalize(input)
		args = algox_args(input)
#		print(len(list(algox(*args))))
#		for sol in algox(*args):
#			print(sol)
		assert len(list(algox(*args))) == noutput
		t0 = time.time()
		n = 0
		targ = 0
		while time.time() - t0 < 1:
			for _ in range(max(n, 1)):
				targ0 = time.time()
				args = algox_args(input)
				targ += time.time() - targ0
				list(algox(*args))
				n += 1
		ret.append((n, (time.time() - t0) / n, targ / n))
	return ret

def canonicalize(subsets):
	nodes = sorted(set(node for subset in subsets for node in subset))
	return sorted(sorted(nodes.index(node) for node in subset) for subset in subsets)

# Baseline, straightforward implementation with immutable data structures.
def algox0_args(subsets):
	subsets = frozenset(frozenset(subset) for subset in subsets)
	nodes = frozenset(node for subset in subsets for node in subset)
	return nodes, subsets
def algox0(nodes, subsets):
	if not nodes:
		yield []
		return
	node_counts = Counter(node for subset in subsets for node in subset)
	if len(node_counts) < len(nodes):
		return
	selected_node = min(nodes, key = lambda node: (node_counts[node], node))
	subset_choices = sorted(subset for subset in subsets if selected_node in subset)
	for selected_subset in subset_choices:
		new_nodes = nodes - selected_subset
		new_subsets = frozenset(subset for subset in subsets if selected_subset.isdisjoint(subset))
		for subcover in algox0(new_nodes, new_subsets):
			yield subcover + [selected_subset]

# Mutable data structures that get updated when passed down.
def algox1_args(subsets):
	subsets = set(frozenset(subset) for subset in subsets)
	nodes = set(node for subset in subsets for node in subset)
	return nodes, subsets
def algox1(nodes, subsets):
	if not nodes:
		yield []
		return
	node_counts = Counter(node for subset in subsets for node in subset)
	if len(node_counts) < len(nodes):
		return
	selected_node = min(nodes, key = lambda node: (node_counts[node], node))
	subset_choices = sorted(subset for subset in subsets if selected_node in subset)
	for selected_subset in subset_choices:
		removed_subsets = set(subset for subset in subsets if not selected_subset.isdisjoint(subset))
		nodes -= selected_subset
		subsets -= removed_subsets
		for subcover in algox1(nodes, subsets):
			yield subcover + [selected_subset]
		nodes |= selected_subset
		subsets |= removed_subsets

# Like algox0 but with an auxiliary data structure to do subset inclusion rather than using the
# subsets themselves.
# Only jnodes and jsubsets are updated with each iteration.
def algox2_args(subsets):
	subsets = sorted(sorted(subset) for subset in subsets)
	subsets = [set(subset) for subset in subsets]
	nodes = sorted(set(node for subset in subsets for node in subset))
	# containers[jnode] = the set of jsubsets that contain the given node
	containers = [set(jsubset for jsubset, subset in enumerate(subsets) if node in subset) for node in nodes]
	# overlappers[jsubset] = the set of jsubsets that overlap the given subset
	overlappers = [set(ksubset for ksubset, s1 in enumerate(subsets) if s0 & s1) for s0 in subsets]
	jnodes, jsubsets = set(range(len(nodes))), set(range(len(subsets)))
	return jnodes, jsubsets, subsets, containers, overlappers
def algox2(jnodes, jsubsets, subsets, containers, overlappers):
#	print(len(jnodes))
	if not jnodes:
		yield []
		return
	node_counts = {jnode: len(containers[jnode] & jsubsets) for jnode in jnodes}
	selected_jnode = min(jnodes, key = lambda jnode: (node_counts[jnode], jnode))
	if node_counts[selected_jnode] == 0:
#		print(selected_jnode)
		return
	jsubset_choices = sorted(containers[selected_jnode] & jsubsets)
	for selected_jsubset in jsubset_choices:
		new_jnodes = jnodes - subsets[selected_jsubset]
		new_jsubsets = jsubsets - overlappers[selected_jsubset]
		for subcover in algox2(new_jnodes, new_jsubsets, subsets, containers, overlappers):
			yield subcover + [selected_jsubset]

# Like algox2 but lazily evaluate overlaps.
def algox2l_args(subsets):
	subsets = sorted(sorted(subset) for subset in subsets)
	subsets = [set(subset) for subset in subsets]
	nodes = sorted(set(node for subset in subsets for node in subset))
	# containers[jnode] = the set of jsubsets that contain the given node
	containers = [set(jsubset for jsubset, subset in enumerate(subsets) if node in subset) for node in nodes]
	# overlappers[jsubset] = the set of jsubsets that overlap the given subset
	overlappers = [None] * len(subsets)
	jnodes, jsubsets = set(range(len(nodes))), set(range(len(subsets)))
	return jnodes, jsubsets, subsets, containers, overlappers
def algox2l(jnodes, jsubsets, subsets, containers, overlappers):
	if not jnodes:
		yield []
		return
	node_counts = {jnode: len(containers[jnode] & jsubsets) for jnode in jnodes}
	selected_jnode = min(jnodes, key = lambda jnode: (node_counts[jnode], jnode))
	if node_counts[selected_jnode] == 0:
		return
	jsubset_choices = sorted(containers[selected_jnode] & jsubsets)
	for selected_jsubset in jsubset_choices:
		new_jnodes = jnodes - subsets[selected_jsubset]
#		if overlappers[selected_jsubset] is None:
#			s0 = subsets[selected_jsubset]
#			overlappers[selected_jsubset] = set(ksubset for ksubset, s1 in enumerate(subsets) if s0 & s1)
#		new_jsubsets = jsubsets - overlappers[selected_jsubset]
		ok = subsets[selected_jsubset].isdisjoint
		new_jsubsets = set(jsubset for jsubset in jsubsets if ok(subsets[jsubset]))
		for subcover in algox2l(new_jnodes, new_jsubsets, subsets, containers, overlappers):
			yield subcover + [selected_jsubset]

# Like algox2l but cache subcalls.
def algox2c_args(subsets):
	subsets = sorted(sorted(subset) for subset in subsets)
	subsets = [set(subset) for subset in subsets]
	nodes = sorted(set(node for subset in subsets for node in subset))
	# containers[jnode] = the set of jsubsets that contain the given node
	containers = [set(jsubset for jsubset, subset in enumerate(subsets) if node in subset) for node in nodes]
	# overlappers[jsubset] = the set of jsubsets that overlap the given subset
	overlappers = [None] * len(subsets)
	jnodes, jsubsets = set(range(len(nodes))), set(range(len(subsets)))
	return jnodes, jsubsets, subsets, containers, overlappers, {}
def algox2c(jnodes, jsubsets, subsets, containers, overlappers, subcalls):
	if not jnodes:
		yield []
		return
	key = frozenset(jnodes), frozenset(jsubsets)
	if key in subcalls:
		yield from subcalls[key]
		return
	subcalls[key] = scalls = []
	node_counts = {jnode: len(containers[jnode] & jsubsets) for jnode in jnodes}
	selected_jnode = min(jnodes, key = lambda jnode: (node_counts[jnode], jnode))
	if node_counts[selected_jnode] == 0:
		return
	jsubset_choices = sorted(containers[selected_jnode] & jsubsets)
	for selected_jsubset in jsubset_choices:
		new_jnodes = jnodes - subsets[selected_jsubset]
		if overlappers[selected_jsubset] is None:
			s0 = subsets[selected_jsubset]
			overlappers[selected_jsubset] = set(ksubset for ksubset, s1 in enumerate(subsets) if s0 & s1)
		new_jsubsets = jsubsets - overlappers[selected_jsubset]
		for subcover in algox2c(new_jnodes, new_jsubsets, subsets, containers, overlappers, subcalls):
			scalls.append(subcover + [selected_jsubset])
			yield subcover + [selected_jsubset]


# Like algox2 but random node choice
algox2r_args = algox2_args
def algox2r(jnodes, jsubsets, subsets, containers, overlappers):
	if not jnodes:
		yield []
		return
	node_counts = {jnode: len(containers[jnode] & jsubsets) for jnode in jnodes}
	selected_jnode = min(jnodes, key = lambda jnode: (node_counts[jnode], random.random()))
	if node_counts[selected_jnode] == 0:
		return
	jsubset_choices = list(containers[selected_jnode] & jsubsets)
	random.shuffle(jsubset_choices)
	for selected_jsubset in jsubset_choices:
		new_jnodes = jnodes - subsets[selected_jsubset]
		new_jsubsets = jsubsets - overlappers[selected_jsubset]
		for subcover in algox2(new_jnodes, new_jsubsets, subsets, containers, overlappers):
			yield subcover + [selected_jsubset]


# Like algox2 but subsets are sorted by size so that largest is chosen first.
def algox2o_args(subsets):
	subsets = sorted(sorted(subset) for subset in subsets)
	subsets.sort(key = len, reverse = True)
	subsets = [set(subset) for subset in subsets]
	nodes = sorted(set(node for subset in subsets for node in subset))
	# containers[jnode] = the set of jsubsets that contain the given node
	containers = [set(jsubset for jsubset, subset in enumerate(subsets) if node in subset) for node in nodes]
	# overlappers[jsubset] = the set of jsubsets that overlap the given subset
	overlappers = [set(ksubset for ksubset, s1 in enumerate(subsets) if s0 & s1) for s0 in subsets]
	jnodes, jsubsets = set(range(len(nodes))), set(range(len(subsets)))
	return jnodes, jsubsets, subsets, containers, overlappers
algox2o = algox2


# Like algox2 but keep track of node counts rather than computing them dynamically
def algox2n_args(subsets):
	subsets = sorted(sorted(subset) for subset in subsets)
	subsets = [set(subset) for subset in subsets]
	nodes = sorted(set(node for subset in subsets for node in subset))
	containers = [set(jsubset for jsubset, subset in enumerate(subsets) if node in subset) for node in nodes]
	overlappers = [set(ksubset for ksubset, s1 in enumerate(subsets) if s0 & s1) for s0 in subsets]
	jnodes, jsubsets = set(range(len(nodes))), set(range(len(subsets)))
	node_counts = [0] * len(nodes)
	for subset in subsets:
		for node in subset:
			node_counts[node] += 1
	return jnodes, jsubsets, subsets, containers, overlappers, node_counts
def algox2n(jnodes, jsubsets, subsets, containers, overlappers, node_counts):
	if not jnodes:
		yield []
		return
	selected_jnode = min(jnodes, key = lambda jnode: (node_counts[jnode], jnode))
	if node_counts[selected_jnode] == 0:
		return
	jsubset_choices = sorted(containers[selected_jnode] & jsubsets)
	for selected_jsubset in jsubset_choices:
		new_jnodes = jnodes - subsets[selected_jsubset]
		new_jsubsets = jsubsets - overlappers[selected_jsubset]
		removed_subsets = jsubsets & overlappers[selected_jsubset]
		new_node_counts = list(node_counts)
		for jsubset in removed_subsets:
			for node in subsets[jsubset]:
				new_node_counts[node] -= 1
		for subcover in algox2n(new_jnodes, new_jsubsets, subsets, containers, overlappers, new_node_counts):
			yield subcover + [selected_jsubset]

# Like algox2n but node_counts is updated in place.
algox2ni_args = algox2n_args
def algox2ni(jnodes, jsubsets, subsets, containers, overlappers, node_counts):
	if not jnodes:
		yield []
		return
	selected_jnode = min(jnodes, key = lambda jnode: (node_counts[jnode], jnode))
	if node_counts[selected_jnode] == 0:
		return
	jsubset_choices = sorted(containers[selected_jnode] & jsubsets)
	for selected_jsubset in jsubset_choices:
		new_jnodes = jnodes - subsets[selected_jsubset]
		new_jsubsets = jsubsets - overlappers[selected_jsubset]
		removed_subsets = jsubsets & overlappers[selected_jsubset]
		for jsubset in removed_subsets:
			for node in subsets[jsubset]:
				node_counts[node] -= 1
		for subcover in algox2n(new_jnodes, new_jsubsets, subsets, containers, overlappers, node_counts):
			yield subcover + [selected_jsubset]
		for jsubset in removed_subsets:
			for node in subsets[jsubset]:
				node_counts[node] += 1

# Like algox2ni but track min_jnode
def algox2nj_args(subsets):
	jnodes, jsubsets, subsets, containers, overlappers, node_counts = algox2n_args(subsets)
	min_jnode = min(jnodes, key = lambda node: node_counts[node])
	return jnodes, jsubsets, subsets, containers, overlappers, node_counts, min_jnode
def algox2nj(jnodes, jsubsets, subsets, containers, overlappers, node_counts, min_jnode):
	if not jnodes:
		yield []
		return
	if node_counts[min_jnode] == 0:
		return
	jsubset_choices = sorted(containers[min_jnode] & jsubsets)
	for selected_jsubset in jsubset_choices:
		new_jnodes = jnodes - subsets[selected_jsubset]
		new_jsubsets = jsubsets - overlappers[selected_jsubset]
		new_min_jnode = min_jnode
		removed_subsets = jsubsets & overlappers[selected_jsubset]
		for jsubset in removed_subsets:
			for node in subsets[jsubset]:
				node_counts[node] -= 1
				if node not in subsets[selected_jsubset] and node_counts[node] < node_counts[new_min_jnode]:
					new_min_jnode = node
		if new_min_jnode in subsets[selected_jsubset] and new_jnodes:
			new_min_jnode = min(new_jnodes, key = lambda node: node_counts[node])
		for subcover in algox2nj(new_jnodes, new_jsubsets, subsets, containers, overlappers, node_counts, new_min_jnode):
			yield subcover + [selected_jsubset]
		for jsubset in removed_subsets:
			for node in subsets[jsubset]:
				node_counts[node] += 1

# Like algox2ni but track min_jnode in a complicated setup
def algox2ns_args(subsets):
	jnodes, jsubsets, subsets, containers, overlappers, node_counts = algox2n_args(subsets)
	nodes_by_counts = defaultdict(set)
	for jnode, node_count in enumerate(node_counts):
		nodes_by_counts[node_count].add(jnode)
	return jnodes, jsubsets, subsets, containers, overlappers, node_counts, nodes_by_counts
def algox2ns(jnodes, jsubsets, subsets, containers, overlappers, node_counts, nodes_by_counts):
#	print(jnodes, jsubsets, node_counts, nodes_by_counts)
	if not jnodes:
		yield []
		return
	for count, nodes_by_count in sorted(nodes_by_counts.items()):
		toselect = nodes_by_count & jnodes
		if toselect and count == 0:
			return
		if toselect:
			min_jnode = next(iter(toselect))
			break
	jsubset_choices = sorted(containers[min_jnode] & jsubsets)
	for selected_jsubset in jsubset_choices:
		new_jnodes = jnodes - subsets[selected_jsubset]
		new_jsubsets = jsubsets - overlappers[selected_jsubset]
		new_min_jnode = min_jnode
		removed_subsets = jsubsets & overlappers[selected_jsubset]
		for jsubset in removed_subsets:
			for node in subsets[jsubset]:
				nodes_by_counts[node_counts[node]].remove(node)
				node_counts[node] -= 1
				nodes_by_counts[node_counts[node]].add(node)
		for subcover in algox2ns(new_jnodes, new_jsubsets, subsets, containers, overlappers, node_counts, nodes_by_counts):
			yield subcover + [selected_jsubset]
		for jsubset in removed_subsets:
			for node in subsets[jsubset]:
				nodes_by_counts[node_counts[node]].remove(node)
				node_counts[node] += 1
				nodes_by_counts[node_counts[node]].add(node)

# Precompute overlappers.
def algox3_args(subsets):
	subsets = frozenset(frozenset(subset) for subset in subsets)
	nodes = frozenset(node for subset in subsets for node in subset)
	overlappers = { s0: set(s1 for s1 in subsets if s0 & s1) for s0 in subsets }
	return nodes, subsets, overlappers
def algox3(nodes, subsets, overlappers):
	if not nodes:
		yield []
		return
	node_counts = Counter(node for subset in subsets for node in subset)
	if len(node_counts) < len(nodes):
		return
	selected_node = min(nodes, key = lambda node: (node_counts[node], node))
	subset_choices = sorted(subset for subset in subsets if selected_node in subset)
	for selected_subset in subset_choices:
		new_nodes = nodes - selected_subset
		new_subsets = subsets - overlappers[selected_subset]
		for subcover in algox3(new_nodes, new_subsets, overlappers):
			yield subcover + [selected_subset]

# Use jsubsets without precomputation.
def algox4_args(subsets):
	subsets = sorted(sorted(subset) for subset in subsets)
	subsets = [set(subset) for subset in subsets]
	nodes = sorted(set(node for subset in subsets for node in subset))
	jnodes, jsubsets = set(range(len(nodes))), set(range(len(subsets)))
	return jnodes, jsubsets, subsets
def algox4(jnodes, jsubsets, subsets):
	if not jnodes:
		yield []
		return
	node_counts = Counter(node for jsubset in jsubsets for node in subsets[jsubset])
	selected_jnode = min(jnodes, key = lambda jnode: (node_counts[jnode], jnode))
	if node_counts[selected_jnode] == 0:
		return
	jsubset_choices = sorted(jsubset for jsubset in jsubsets if selected_jnode in subsets[jsubset])
	for selected_jsubset in jsubset_choices:
		selected_subset = subsets[selected_jsubset]
		new_jnodes = jnodes - subsets[selected_jsubset]
		new_jsubsets = set(jsubset for jsubset in jsubsets if selected_subset.isdisjoint(subsets[jsubset]))
		for subcover in algox4(new_jnodes, new_jsubsets, subsets):
			yield subcover + [selected_jsubset]

# Precompute containers.
def algox5_args(subsets):
	subsets = frozenset(frozenset(subset) for subset in subsets)
	nodes = frozenset(node for subset in subsets for node in subset)
	containers = { node: set(subset for subset in subsets if node in subset) for node in nodes }
	return nodes, subsets, containers
def algox5(nodes, subsets, containers):
	if not nodes:
		yield []
		return
	node_counts = Counter(node for subset in subsets for node in subset)
	if len(node_counts) < len(nodes):
		return
	selected_node = min(nodes, key = lambda node: (node_counts[node], node))
	subset_choices = sorted(containers[selected_node] & subsets)
	for selected_subset in subset_choices:
		new_nodes = nodes - selected_subset
		new_subsets = frozenset(subset for subset in subsets if selected_subset.isdisjoint(subset))
		for subcover in algox5(new_nodes, new_subsets, containers):
			yield subcover + [selected_subset]

# algox2 without precomputing overlappers
def algox6_args(subsets):
	subsets = sorted(sorted(subset) for subset in subsets)
	subsets = [set(subset) for subset in subsets]
	nodes = sorted(set(node for subset in subsets for node in subset))
	# containers[jnode] = the set of jsubsets that contain the given node
	containers = [set(jsubset for jsubset, subset in enumerate(subsets) if node in subset) for node in nodes]
	jnodes, jsubsets = set(range(len(nodes))), set(range(len(subsets)))
	return jnodes, jsubsets, subsets, containers
def algox6(jnodes, jsubsets, subsets, containers):
	if not jnodes:
		yield []
		return
	node_counts = {jnode: len(containers[jnode] & jsubsets) for jnode in jnodes}
	selected_jnode = min(jnodes, key = lambda jnode: (node_counts[jnode], jnode))
	if node_counts[selected_jnode] == 0:
		return
	jsubset_choices = sorted(containers[selected_jnode] & jsubsets)
	for selected_jsubset in jsubset_choices:
		selected_subset = subsets[selected_jsubset]
		new_jnodes = jnodes - selected_subset
		new_jsubsets = set(jsubset for jsubset in jsubsets if selected_subset.isdisjoint(subsets[jsubset]))
		for subcover in algox6(new_jnodes, new_jsubsets, subsets, containers):
			yield subcover + [selected_jsubset]

def algox7_args(subsets):
	subsets = sorted(sorted(subset) for subset in subsets)
	nodes = sorted(set(node for subset in subsets for node in subset))
	matrix = dlmatrix.DancingLinksMatrix(len(nodes))
	for row in subsets:
		matrix.add_sparse_row(row, already_sorted=True)
	matrix.end_add()
	return matrix,
def algox7(matrix):
	solutions = []
	alg = alg_x.AlgorithmX(matrix, solutions.append)
	alg()
	return solutions


if False:
	input, noutput = list(get_inputs())[1]
	input = canonicalize(input)
	args = algox2ni_args(input)
	print(list(algox2ni(*args)))
	exit()
print("2ns", *profile(algox2ns, algox2ns_args))
print("2nj", *profile(algox2nj, algox2nj_args))

print("2n", *profile(algox2n, algox2n_args))
print("2ni", *profile(algox2ni, algox2ni_args))
print("2c", *profile(algox2c, algox2c_args))
print(2, *profile(algox2, algox2_args))
print("2l", *profile(algox2l, algox2l_args))
print("2r", *profile(algox2r, algox2r_args))
print("2o", *profile(algox2o, algox2o_args))

print(0, *profile(algox0, algox0_args))
print(1, *profile(algox1, algox1_args))
print(3, *profile(algox3, algox3_args))
print(4, *profile(algox4, algox4_args))
print(5, *profile(algox5, algox5_args))
print(6, *profile(algox6, algox6_args))
print(7, *profile(algox7, algox7_args))





