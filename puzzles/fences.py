# Solution to Good Fences Make Sad and Disgusted Neighbors
# Using Algorithm X with multi cover (generalized exact cover)
# 2018 MIT Mystery Hunt

# Runs in 2-3 minutes.

import grf
from collections import Counter

layouts = """
3222310000320343141036633541146451103533135534530423523533101
3431536632403430342000124433012553521345204152330313453102344
5510012101103300000363000010330012110000000000012123000032011
2134513013355343554336634254534353334200142521342343533350544
1234520123431015454233230143554310145534243355314445432234425
1233330363234033004440001453322010202223225000020412103400015
5453542050451011133544421053256415541055103413635502333354445
5014142011232443234245342215232412534215200330133002036310033
"""
layouts = [line.strip() for line in layouts.splitlines() if line.strip()]


# Cells have (x, y) coordinates in a tilted coordinate system.
# Top left corner is (0, 0). Top right corner is (n, 0). Left corner is (0, n).
linelengths = 5, 6, 7, 8, 9, 8, 7, 6, 5
linestarts = 0, 0, 0, 0, 0, 1, 2, 3, 4
cells = [(x, y) for y, (l, s) in enumerate(zip(linelengths, linestarts)) for x in range(s, s + l)]

# Two cells are adjacent if their coordinate differences are in this list.
eways = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)]
def isneighbor(cell0, cell1):
	x0, y0 = cell0
	x1, y1 = cell1
	return (x1 - x0, y1 - y0) in eways
def allneighbors(cell):
	x, y = cell
	return [(x + dx, y + dy) for dx, dy in eways]
# An edge is defined by the pair of cells it's between. For edges on the outside of the grid, one
# of these cells will not be in the cells list.
edges = set(
	tuple(sorted([cell0, cell1]))
	for cell0 in cells
	for cell1 in allneighbors(cell0)
)
# A vertex is defined by the three cells it's between.
vertices = set(
	tuple(sorted([cell0, cell1, cell2]))
	for cell0 in cells
	for cell1 in allneighbors(cell0)
	for cell2 in allneighbors(cell0)
	if cell1 < cell2 and isneighbor(cell1, cell2)
)
def vertexon(vertex, edge):
	return all(cell in vertex for cell in edge)

# Can't think of any way to express the constraint that all edges must be connected. Just filter
# solutions based on this check.
def singlefence(solution):
	edges = [edge for ctype, edge in solution if ctype == "fenceat"]
	fconnect = lambda edge0, edge1: any(vertexon(vertex, edge0) and vertexon(vertex, edge1) for vertex in vertices)
	graph = grf.fconnect_to_graph(edges, fconnect)
	return grf.is_connected(graph)
def printsolution(solution):
	layout = { (x, y): " " for x in range(40) for y in range(40) }
	def cellpos(cell):
		x, y = cell
		return 4 * x - 2 * y + 12, 3 + 3 * y
	def edgeposes(edge):
		xs, ys = zip(*[cellpos(cell) for cell in edge])
		yield sum(xs) // len(xs), sum(ys) // len(ys)
		yield (sum(xs) + 1) // len(xs), (sum(ys) + 1) // len(ys)
		for vertex in vertices:
			if vertexon(vertex, edge):
				yield vertexpos(vertex)
	def vertexpos(vertex):
		xs, ys = zip(*[cellpos(cell) for cell in vertex])
		return sum(xs) // len(xs), sum(ys) // len(ys)
	for ctype, value in solution:
		if ctype == "sad":
			layout[cellpos(value)] = "S"
		if ctype == "disgusted":
			layout[cellpos(value)] = "D"
		if ctype == "fenceat":
			for pos in edgeposes(value):
				layout[pos] = "#"
	for y in range(40):
		print(*[layout[(x, y)] for x in range(40)])


def solve(layout, title):
	grid = { cell: int(c) for cell, c in zip(cells, layout) }
	node_ranges = {}
	subsets = {}
	for cell, n in grid.items():
		# Each cell must have a mood assigned.
		node_ranges[("mood", cell)] = 1
		# This is a trick to express the disgust condition. A cell gets +1 lofence and +1 hifence
		# for every fence that borders it. If the cell's mood is disgust, then these must both total
		# to n. In that case, the cell gives itself +(6-n) of each, bringing the total lofence and
		# total hifence both to n. If the cell's mood is sad, then these can be anything. In that
		# case the cell gives itself +6 hifence and +0 lofence, so no matter how many fences it has
		# they both fall into the right range.
		node_ranges[("lofence", cell)] = 0, 6
		node_ranges[("hifence", cell)] = 6, 12
		node_ranges[("losad", cell)] = 0, 6
		node_ranges[("hisad", cell)] = 6, 12

		subsets[("sad", cell)] = (
			[("mood", cell)] + 
			[("hifence", cell)] * 6 + 
			[("losad", cell)] * (6 - n) + 
			[("hisad", cell)] * (6 - n) + 
			[("losad", cell1) for cell1 in grid if isneighbor(cell, cell1)] +
			[("hisad", cell1) for cell1 in grid if isneighbor(cell, cell1)]
		)
		subsets[("disgusted", cell)] = (
			[("mood", cell)] +
			[("hisad", cell)] * 6 +
			[("lofence", cell)] * (6 - n) +
			[("hifence", cell)] * (6 - n)
		)

	for edge in edges:
		subsets[("fenceat", edge)] = [("lofence", cell) for cell in edge if cell in grid] + [("hifence", cell) for cell in edge if cell in grid]
	for vertex in vertices:
		# We require every vertex to have fencesto = 2. For vertices that are on the fence, this
		# means that exactly two of the three edges that connect at the vertex must be on the fence.
		# For vertices that are not on the fence, the nofence condition means that none of the edges
		# may be on the fence.
		node_ranges[("fencesto", vertex)] = 2
		subsets[("nofence", vertex)] = [("fencesto", vertex)] * 2
		for edge in edges:
			if vertexon(vertex, edge):
				subsets[("fenceat", edge)].append(("fencesto", vertex))
	print(title)
	for solution in grf.multi_covers(subsets, node_ranges=node_ranges):
		if singlefence(solution):
			printsolution(solution)

for jlayout, layout in enumerate(layouts, 1):
	solve(layout, "Puzzle #{}".format(jlayout))

