# Profiling for partial cover implementation.

import time

def exact_args(subsets):
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
def exact(jnodes, jsubsets, subsets, containers, overlappers, node_counts, dead_input):
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
		for subcover in exact(new_jnodes, new_jsubsets, subsets, containers, overlappers, new_node_counts, dead_input):
			yield subcover + [selected_jsubset]
			dead = False
	if dead:
		dead_input.add(jnodes)


def partial0_args(subsets, nodes):
	subsets = list(subsets)
	all_nodes = set.union(*map(set, subsets))
	subsets += [[node] for node in all_nodes - set(nodes)]
	return exact_args(subsets)
partial0 = exact


def partial1_args(subsets, nodes):
	subsets = [set(subset) for subset in subsets]
	all_nodes = sorted(set.union(*map(set, subsets)))
	node_set = set(nodes)
	nodes = sorted(nodes)
	containers = [set() for node in all_nodes]
	for jsubset, subset in enumerate(subsets):
		for node in subset:
			containers[node].add(jsubset)
	overlappers = [set() for subset in subsets]
	for jnode, container in enumerate(containers):
		for jsubset in container:
			overlappers[jsubset] |= container
	node_counts = [0] * len(all_nodes)
	for subset in subsets:
		for node in subset:
			node_counts[node] += 1
#	jnodes = frozenset(node_set)
	jnodes = frozenset(all_nodes)
	jsubsets = frozenset(range(len(subsets)))
#	subsets = [subset & node_set for subset in subsets]
#	containers = [containers[node] for node in nodes]
#	node_counts = [node_counts[node] for node in nodes]
	return jnodes, jsubsets, subsets, node_set, containers, overlappers, node_counts, set()

def partial1(jnodes, jsubsets, subsets, required_jnodes, containers, overlappers, node_counts, dead_input):
	selectable_jnodes = jnodes & required_jnodes
	if not selectable_jnodes:
		yield []
		return
	if jnodes in dead_input:
		return
	dead = True
	min_jnode = min(selectable_jnodes, key = node_counts.__getitem__)
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
		for subcover in partial1(new_jnodes, new_jsubsets, subsets, required_jnodes, containers, overlappers, new_node_counts, dead_input):
			yield subcover + [selected_jsubset]
			dead = False
	if dead:
		dead_input.add(jnodes)



def canonicalize(subsets, nodes):
	all_nodes = sorted(set(node for subset in subsets for node in subset))
	subsets = sorted(sorted(all_nodes.index(node) for node in subset) for subset in subsets)
	nodes = [all_nodes.index(node) for node in nodes]
	return subsets, nodes


# N queens
N, Nsol = 12, 724
subsets = []
for x in range(N):
	for y in range(N):
		subsets.append([("x", x), ("y", y), ("d", x+y), ("b", x-y)])
nodes = [(p, k) for p in "xy" for k in range(N)]
subsets, nodes = canonicalize(subsets, nodes)

for algo, algo_args in [(partial0, partial0_args), (partial1, partial1_args)]:
	args = algo_args(subsets, nodes)
	t0 = time.time()
	nsol = len(list(algo(*args)))
	print(nsol, time.time() - t0)

