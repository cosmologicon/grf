# http://web.mit.edu/puzzle/www/2018/full/puzzle/laratel_wf_15_fusion_reactor.html
# This doesn't work. Possibly because the puzzle as given is underspecified.


import grf

layout = """
aaabbbbbbbbbb
aaBbbbcDdddeE
Affffbcddddhe
gfffbbccccChe
gfffiiiiccche
gffFiiiiihhhe
gffiiiiiihhhe
gffiiiiiihhhe
Gfffiiiiihhhe
gggffiIiihhhe
gjjjjjjjjhhhe
ggjjjjjJjhhHe
ggggggggghhhe
"""

layout = {
	(x, y): c
	for y, line in enumerate([line.strip() for line in layout.splitlines() if line.strip()])
	for x, c in enumerate(line)
}
cells = sorted(layout.keys())
def isadjacent(cell0, cell1):
	(x0, y0), (x1, y1) = cell0, cell1
	return abs(x0 - x1) + abs(y0 - y1) == 1
edges = [(cell0, cell1) for cell0 in cells for cell1 in cells if cell0 < cell1 and isadjacent(cell0, cell1)]
compartments = { (x, y): c.upper() for (x, y), c in layout.items() }
sightings = set(cell for cell, c in layout.items() if c.isupper())

reqs = {}
covers = {}
# Every cell must have exactly zero or two edges covered.
# Sighting cells must have exactly two.
for cell in cells:
	reqs[("cell", cell)] = 2
for edge in edges:
	covers[("edge", edge)] = [("cell", cell) for cell in edge]
for cell in cells:
	if cell not in sightings:
		covers[("empty", cell)] = [("cell", cell)] * 2

# The wall of every compartment must be crossed exactly twice.
for c in set(compartments.values()):
	reqs[("cross", c)] = 2
for edge in edges:
	cs = [compartments[cell] for cell in edge]
	if cs[0] != cs[1]:
		covers[("edge", edge)] += [("cross", c) for c in cs]

# If an edge crosses between two compartments, then at least one of the two cells connected by
# that edge must be entered.
alongs = []
for edge in edges:
	cs = [compartments[cell] for cell in edge]
	if cs[0] != cs[1]:
		reqs[("along", edge)] = 2, 4
		alongs += [(cell, edge) for cell in edge]
for edge in edges:
	for cell, aedge in alongs:
		if cell in edge:
			covers[("edge", edge)].append(("along", aedge))

def isconnected(solution):
	sedges = [edge for ctype, edge in solution if ctype == "edge"]
	return grf.is_connected(sedges)

grf.DEBUG = True
print(grf.multi_covers(covers, node_ranges=reqs))	




