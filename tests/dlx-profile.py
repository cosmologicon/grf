# Profile various implementations of the inner loop of Algorithm X.

# Need to download this repository:
# https://github.com/DavideCanton/DancingLinksX

# Compared with the baseline implementation, the following improvements *are* worthwhile:
#   * j-indexing subsets and working with jsubsets
#   * auxiliary node count list
#   * lazy evaluation of overlapping subsets (single solutions only)
#     (actually this is no longer true with the improved overlap formula)
#   * caching subcalls (multiple solutions only)
#   * appending to subcall list rather than creating copy (multiple solutions only)
# The following makes it worse:
#   * creating a subfunction for the inner loop
# The following make essentially no difference, or a very minor difference that can go either way
# depending on the problem:
#   * mutable vs immutable data structures
#   * in-place updates vs creating new data structures
#   * forgoing the final subcall when jnodes is empty
#   * ordering of the subset iteration
#   * caching more than just the dead branches
#   * caching on the set of remaining nodes vs the set of remaining subsets

from __future__ import division
import time, random, math
from collections import Counter, defaultdict
from DancingLinksX import dlmatrix, alg_x

Ttest = 1

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

	# 16x16 Sudoku
	grid = list(
		"6..8......A.E.......A.....57.0.B..52.F8......7.3.C..2....D.38..."
		"9..B7.3....F.2....7F86......14.....0F4.CB.E1.......A1B...2..9..5"
		"...1..27.9....D..6.5.9..A.........D..1...4..2F7..8..5C.F..B....."
		"..8...4B.50A.E3F.39...6...1B..0.20.E.5A..7..4.......32.........A"
	)
	def get_constraints(row, column, digit):
		yield "f", row, column
		yield "r", row, digit
		yield "c", column, digit
		yield "b", int(row // 4), int(column // 4), digit
	constraints = { (row * 16 + column, digit) : list(get_constraints(row, column, digit))
		for row in range(16) for column in range(16) for digit in "0123456789ABCDEF" }
	used_constraints = set().union(*[cons for (j, digit), cons in constraints.items() if grid[j] == digit])
	available = {name: cons for name, cons in constraints.items() if used_constraints.isdisjoint(cons)}
	yield list(available.values()), 1
	

	# Dominos on a NxN grid
	N = 6
	tiles = [((x, y), (x + 1, y)) for x in range(N - 1) for y in range(N)]
	tiles += [((x, y), (x, y + 1)) for x in range(N) for y in range(N - 1)]
	yield [[x + N * y for x, y in tile] for tile in tiles], 6728

	# N queens
	N, Nsol = 10, 724
	qs = []
	for x in range(N):
		for y in range(N):
			qs.append([("x", x), ("y", y), ("d", x+y), ("b", x-y)])
	qs += [[("d", d)] for d in range(2*N-1)]
	qs += [[("b", b)] for b in range(-N+1, N)]
	yield qs, Nsol

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
		while time.time() - t0 < Ttest:
			for _ in range(max(n, 1)):
				targ0 = time.time()
				args = algox_args(input)
				targ += time.time() - targ0
				list(algox(*args))
				n += 1
		ret.append(n)
		ret.append("%.3g" % ((time.time() - t0) / n))
		ret.append("%.3g" % (targ / n))
		ret.append("")
	return ret


def profilecompare(alg0, alg1):
	algs = [alg0, alg1]
	for input, noutput in get_inputs():
		input = canonicalize(input)
		n = 0
		s = 0  # Number of times alg1 was faster
		# p = s/n
		# abs(p - 0.5) > z * sqrt(1/4 / n)
		tend = time.time() + 1000
		while abs(s - n/2) ** 2 <= 25/4 * n and time.time() < tend:
			totest = [0, 1]
			random.shuffle(totest)
			ts = [None, None]
			for j in totest:
				algox, algox_args = algs[j]
				t0 = time.time()
				list(algox(*algox_args(input)))
				ts[j] = time.time() - t0
			s += ts[1] < ts[0]
			n += ts[1] != ts[0]
#			print(n, s, s/n, abs(s/n - 0.5) / math.sqrt(1/4 / n))
		print(n, s, s/n, abs(s/n - 0.5) / math.sqrt(1/4 / n))
	

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


# All known improvements
def algoxZ_args(subsets):
	subsets = sorted(sorted(subset) for subset in subsets)
	subsets = [set(subset) for subset in subsets]
	nodes = sorted(set(node for subset in subsets for node in subset))
	containers = [set() for node in nodes]
	for jsubset, subset in enumerate(subsets):
		for node in subset:
			containers[node].add(jsubset)
	overlappers = [set() for subset in subsets]
	for jnode, container in enumerate(containers):
		for jsubset in container:
			overlappers[jsubset] |= container
	jnodes, jsubsets = frozenset(range(len(nodes))), frozenset(range(len(subsets)))
	node_counts = [0] * len(nodes)
	for subset in subsets:
		for node in subset:
			node_counts[node] += 1
	return jnodes, jsubsets, subsets, containers, overlappers, node_counts, set()
def algoxZ(jnodes, jsubsets, subsets, containers, overlappers, node_counts, dead_input):
	if not jnodes:
		yield []
		return
	if jsubsets in dead_input:
		return
	dead = True
	min_jnode = min(jnodes, key = node_counts.__getitem__)
	if node_counts[min_jnode] == 0:
		return
	jsubset_choices = sorted(containers[min_jnode] & jsubsets)
	for selected_jsubset in jsubset_choices:
		removed_subsets = jsubsets & overlappers[selected_jsubset]
		new_jsubsets = jsubsets - removed_subsets
		new_jnodes = jnodes - subsets[selected_jsubset]
		new_node_counts = list(node_counts)
		for jsubset in removed_subsets:
			for node in subsets[jsubset]:
				new_node_counts[node] -= 1
		for subcover in algoxZ(new_jnodes, new_jsubsets, subsets, containers, overlappers, new_node_counts, dead_input):
			yield subcover + [selected_jsubset]
			dead = False
	if dead:
		dead_input.add(jsubsets)

def algoxZs(jnodes, jsubsets, subsets, containers, overlappers, node_counts, subcalls):
	def _algo(jnodes, jsubsets, node_counts):
		if not jnodes:
			yield []
			return
		key = jsubsets
		if key in subcalls:
			yield from subcalls[key]
			return
		subcalls[key] = scalls = []
		min_jnode = min(jnodes, key = node_counts.__getitem__)
		if node_counts[min_jnode] == 0:
			return
		jsubset_choices = sorted(containers[min_jnode] & jsubsets)
		for selected_jsubset in jsubset_choices:
			removed_subsets = jsubsets & overlappers[selected_jsubset]
			new_jsubsets = jsubsets - removed_subsets
			new_jnodes = jnodes - subsets[selected_jsubset]
			new_node_counts = list(node_counts)
			for jsubset in removed_subsets:
				for node in subsets[jsubset]:
					new_node_counts[node] -= 1
			for subcover in _algo(new_jnodes, new_jsubsets, new_node_counts):
				scalls.append(subcover + [selected_jsubset])
				yield subcover + [selected_jsubset]
	yield from _algo(jnodes, jsubsets, node_counts)

def algoxZv(jnodes, jsubsets, subsets, containers, overlappers, node_counts, dead_input):
	if not jnodes:
		yield []
		return
	if jnodes in dead_input:
		return
	dead = True
	min_jnode = min(jnodes, key = node_counts.__getitem__)
	if node_counts[min_jnode] == 0:
		return
	jsubset_choices = sorted(containers[min_jnode] & jsubsets)
	for selected_jsubset in jsubset_choices:
		removed_subsets = jsubsets & overlappers[selected_jsubset]
		new_jsubsets = jsubsets - removed_subsets
		new_jnodes = jnodes - subsets[selected_jsubset]
		new_node_counts = list(node_counts)
		for jsubset in removed_subsets:
			for node in subsets[jsubset]:
				new_node_counts[node] -= 1
		for subcover in algoxZv(new_jnodes, new_jsubsets, subsets, containers, overlappers, new_node_counts, dead_input):
			yield subcover + [selected_jsubset]
			dead = False
	if dead:
		dead_input.add(jnodes)


if False:
#	profilecompare((algoxZ, algoxZ_args), (algoxZs, algoxZ_args))
#	profilecompare((algoxZ, algoxZ_args), (algox7, algox7_args))
	profilecompare((algoxZ, algoxZ_args), (algoxZv, algoxZ_args))
#	profilecompare((algoxZv, algoxZv2_args), (algoxZv, algoxZv_args))
	exit()


if False:
	input, noutput = list(get_inputs())[2]
	input = canonicalize(input)
	args = algoxZ_args(input)
	print(list(algoxZ(*args)))
	exit()
print("Z", *profile(algoxZ, algoxZ_args))
print("Zv", *profile(algoxZv, algoxZ_args))
#print("Zs", *profile(algoxZs, algoxZ_args))

print(2, *profile(algox2, algox2_args))
if False:
	print("2ns", *profile(algox2ns, algox2ns_args))
	print("2nj", *profile(algox2nj, algox2nj_args))
	print("2n", *profile(algox2n, algox2n_args))
	print("2ni", *profile(algox2ni, algox2ni_args))
	print("2c", *profile(algox2c, algox2c_args))
	print("2l", *profile(algox2l, algox2l_args))
	print("2r", *profile(algox2r, algox2r_args))
	print("2o", *profile(algox2o, algox2o_args))

if False:
	print(0, *profile(algox0, algox0_args))
	print(1, *profile(algox1, algox1_args))
	print(3, *profile(algox3, algox3_args))
	print(4, *profile(algox4, algox4_args))
	print(5, *profile(algox5, algox5_args))
	print(6, *profile(algox6, algox6_args))
print(7, *profile(algox7, algox7_args))





