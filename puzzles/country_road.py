# https://en.wikipedia.org/wiki/Country_Road_(puzzle)

import grf

# https://www.nikoli.co.jp/ja/puzzles/country_road/
# Completes in 0.1s
grid = """
AABBBBC
AADDBEF
GHHDBEF
IHHJKKF
ILMNKKO
ILMNNPP
QMMMMPP
"""
counts = dict(zip("AEFHIK", (4, 1, 3, 2, 1, 4)))


# https://mellowmelon.wordpress.com/2010/04/04/puzzle-235/
# Unable to complete
if False:
	grid = """
	AABBBCDDDEFFFFFFFFGGGGHH
	AAIBBCDDJEFKKKKKKFLMMGNN
	PIIIQCDRJSSTTUUUULLMMMOO
	PPIQQCVRWWXTTXYYZLaabbOO
	PIIIQVVRWWXTTXYYZcaadeee
	ffIgghhiXXXTTXXXZcaadeee
	fjjjghhikkXXXXllZcccdeee
	ffjmgnnikkXXXXllopqqqqqr
	sssmmtuuXXXvvXXXopwwwxrr
	yyymmtzu00XvvX11op23xxx4
	5566mtzz00XvvX1177222x88
	556699z!!!@$%%^^&&*(xxx8
	))``~,z<<<@$$%%^^**((x==
	))``~,,>>>@/$\%??[[]]]==
	"""
	counts = dict(zip("TXaehmz06^", (8, 14, 4, 5, 4, 4, 2, 4, 3, 3)))


grid = [line.strip() for line in grid.splitlines() if line.strip()]
grid = { (x, y): c for y, line in enumerate(grid) for x, c in enumerate(line) }
rooms = sorted(set(grid.values()))
def isadjacent(cell0, cell1):
	(x0, y0), (x1, y1) = cell0, cell1
	return abs(x0 - x1) + abs(y0 - y1) == 1
edges = sorted((cell0, cell1) for cell0 in grid for cell1 in grid if cell0 < cell1 and isadjacent(cell0, cell1))
def iswall(edge):
	cell0, cell1 = edge
	return grid[cell0] != grid[cell1]
walls = [edge for edge in edges if iswall(edge)]

reqs = {}
covers = { ("edge", edge): [] for edge in edges }
# Every cell must be entered exactly 0 or 2 times.
reqs.update({ ("cell", cell): 2 for cell in grid })
covers.update({ ("nocell", cell): [("cell", cell)] * 2 for cell in grid })
for edge in edges:
	for cell in edge:
		covers[("edge", edge)].append(("cell", cell))
# The wall of every room must be crossed exactly 0 or 2 times.
reqs.update({ ("cross", room): 2 for room in rooms })
covers.update({ ("nocross", room): [("cross", room)] * 2 for room in rooms })
for wall in walls:
	for cell in wall:
		covers[("edge", wall)].append(("cross", grid[cell]))
# Every numbered room must have exactly 1 more edge covering it than the number in the room.
reqs.update({ ("count", room): count + 1 for room, count in counts.items() })
for edge in edges:
	for room in set(grid[cell] for cell in edge):
		if room in counts:
			covers[("edge", edge)].append(("count", room))
# For every wall, at least one of the two cells in the wall must be entered.
for wall in walls:
	reqs[("wall", wall)] = 2, 4
	for edge in edges:
		for cell in edge:
			if cell in wall:
				covers[("edge", edge)].append(("wall", wall))

def printsolution(solution):
	layout = [[" " for x in range(100)] for y in range(40)]
	def cellpos(cell):
		x, y = cell
		return 3 + 3 * x, 2 + 2 * y
	def edgeposes(edge):
		(x0, y0), (x1, y1) = edge
		px, py = cellpos((x0, y0))
		dx, dy = (x1 - x0) * 3, (y1 - y0) * 2
		for x in range(dx + 1):
			for y in range(dy + 1):
				yield px + x, py + y
	for ctype, value in solution:
		if ctype == "edge":
			for px, py in edgeposes(value):
				layout[py][px] = "#"
	for cell, room in grid.items():
		px, py = cellpos(cell)
		layout[py][px] = room
	for line in layout:
		print("".join(line))

grf.DEBUG = True
for solution in grf.multi_covers(covers, node_ranges = reqs):
	printsolution(solution)

