import math, random
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
def exact_covers(subsets, nodes = None, max_solutions = None):
	def algox(subsets, nodes):
		if not nodes:
			yield []
		node_counts = Counter(node for subset in subsets for node in subset)
		if len(node_counts) < len(nodes):
			return
		selected_node = min(nodes, key = node_counts.get)
		subset_choices = [subset for subset in subsets if selected_node in subset]
		if max_solutions is not None:
			random.shuffle(subset_choices)
		for selected_subset in subset_choices:
			new_nodes = [node for node in nodes if node not in selected_subset]
			isvalid = set(selected_subset).isdisjoint
			new_subsets = [subset for subset in subsets if isvalid(set(subset))]
			for subcover in algox(new_subsets, new_nodes):
				yield subcover + [selected_subset]
	if isinstance(subsets, dict):
		subset_names, subsets = map(list, zip(*subsets.items()))
	else:
		subset_names = subsets = list(subsets)
	all_nodes = set.union(*map(set, subsets))
	if nodes is not None:
		if not set(nodes) <= all_nodes:
			return []
	index_to_node = list(all_nodes)
	node_to_index = {node: j for j, node in enumerate(all_nodes)}
	index_subsets = [tuple(map(node_to_index.get, subset)) for subset in subsets]
	index_nodes = list(range(len(index_to_node)))
	solutions = []
	for index_solution in algox(index_subsets, index_nodes):
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

