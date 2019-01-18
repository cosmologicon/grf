import unittest, collections
import grf

class GrfTest(unittest.TestCase):

	def testNodes(self):
		self.assertEqual([], grf.nodes([]))
		self.assertEqual(list("ABD"), grf.nodes(["AB", "AD"]))
		self.assertEqual(list("ABCD"), grf.nodes(["AB", "CD"]))

	def testIsConnected(self):
		self.assertTrue(grf.is_connected([]))
		self.assertTrue(grf.is_connected(["AB"]))
		self.assertFalse(grf.is_connected(["AB", "CD"]))
		self.assertFalse(grf.is_connected(["AB", "CD", "DF"]))
		self.assertTrue(grf.is_connected(["AB", "CD", "DB"]))

	# TODO: deterministic test
	def testHamiltonianPath(self):
		def checkHamiltonian(graph):
			path = grf.hamiltonian_path(graph)
			self.assertEqual(sorted(path), sorted(grf.nodes(graph)))
			graph = list(map(tuple, graph))
			for node0, node1 in zip(path, path[1:]):
				self.assertTrue((node0, node1) in graph or (node1, node0) in graph)
		checkHamiltonian(["AB"])
		checkHamiltonian("AB BC CD DE".split())
		checkHamiltonian("AB BC CD AD".split())
		self.assertFalse(grf.hamiltonian_path("AB CD".split()))
		self.assertFalse(grf.hamiltonian_path("AB AC AD".split()))

	def testExactCoverAPI(self):
		self.assertEqual([], grf.exact_cover([]))
		self.assertEqual([], grf.exact_cover({}))
		self.assertEqual([], grf.exact_cover(""))
		self.assertEqual([], grf.exact_cover(set()))
		self.assertIsNotNone(grf.exact_cover([[]]))
		self.assertEqual(["AB", "CD"], grf.exact_cover("AB BC CD".split()))
		self.assertEqual([0, 2], grf.exact_cover({0: "AB", 1: "BC", 2: "CD"}))
		# The nodes argument is generally unnecessary.
		self.assertEqual(["AB", "CD"], grf.exact_cover("AB BC CD".split(), "ABCD"))
		# No solution will be found if any node does not appear in any subset.
		self.assertIsNone(grf.exact_cover("AB BC CD".split(), "ABCDE"))
		# It is a ValueError for a subset to contain a node not in the set of all nodes.
		self.assertRaises(ValueError, grf.exact_cover, "AB BC CD".split(), "ABC")
		# It is a ValueError for the set of all nodes to have a node appear more than once.
		self.assertRaises(ValueError, grf.exact_cover, "AB BC CD".split(), "ABCDD")
		# If a subset contains a node more than once, that subset cannot appear in a solution.
		self.assertIsNone(grf.exact_cover("AA B".split()))

	def testExactCover(self):
		def checkExactCover(pieces):
			cover = grf.exact_cover(pieces)
			self.assertTrue(all(piece in pieces for piece in cover))
			nodes = [node for piece in cover for node in piece]
			all_nodes = set(node for piece in pieces for node in piece)
			self.assertEqual(sorted(nodes), sorted(all_nodes))
		checkExactCover("AB CD".split())
		checkExactCover("AB BC CD".split())
		checkExactCover([(1, 4, 7), (1, 4), (4, 5, 7), (3, 5, 6), (2, 3, 6, 7), (2, 7)])
		self.assertIsNone(grf.exact_cover("AB BC".split()))

	def testExactCoversAPI(self):
		# exact_covers
		self.assertEqual(1, len(grf.exact_covers([])))
		self.assertEqual(2, len(grf.exact_covers([[]])))
		self.assertEqual(8, len(grf.exact_covers([[], [], []])))
		self.assertEqual(2, len(grf.exact_covers([[0], []])))
		self.assertEqual(0, len(grf.exact_covers([[0, 0], []])))
		self.assertEqual(1, len(grf.exact_covers("AB CD".split())))
		self.assertEqual(0, len(grf.exact_covers("AB BC".split())))
		self.assertEqual(6, len(grf.exact_covers("A A A B B".split())))
		self.assertEqual(6, len(grf.exact_covers("A A A B B".split(), max_solutions=10)))
		self.assertEqual(2, len(grf.exact_covers("A A A B B".split(), max_solutions=2)))
		# can_exact_cover
		self.assertTrue(grf.can_exact_cover("AB CD".split()))
		self.assertFalse(grf.can_exact_cover("AB BC".split()))
		# unique_exact_cover
		self.assertTrue(grf.unique_exact_cover("AB CD".split()))
		self.assertFalse(grf.unique_exact_cover("AB BC".split()))
		self.assertFalse(grf.unique_exact_cover("AB CD AD BC".split()))
		# can_unique_exact_cover
		solution, solution_is_unique = grf.can_unique_exact_cover("AB CD AD BC".split())
		self.assertIsNotNone(solution)
		self.assertFalse(solution_is_unique)
		solution, solution_is_unique = grf.can_unique_exact_cover("AB CD AD".split())
		self.assertIsNotNone(solution)
		self.assertTrue(solution_is_unique)
		solution, solution_is_unique = grf.can_unique_exact_cover("AB BC".split())
		self.assertIsNone(solution)
		self.assertFalse(solution_is_unique)

	def testPartialCoverAPI(self):
		self.assertEqual([], grf.partial_cover([], []))
		self.assertEqual([], grf.partial_cover({}, set()))
		self.assertEqual([], grf.partial_cover("", ""))
		self.assertIsNotNone(grf.partial_cover([[]], []))
		self.assertEqual(["AB", "CD"], grf.partial_cover("AB BC CD".split(), "ABD"))
		self.assertEqual([0, 2], grf.partial_cover({0: "AB", 1: "BC", 2: "CD"}, "ABD"))
		# No solution will be found if any node does not appear in any subset.
		self.assertIsNone(grf.partial_cover("AB BC CD".split(), "ABCDE"))
		# It is a ValueError for the set of all nodes to have a node appear more than once.
		self.assertRaises(ValueError, grf.partial_cover, "AB BC CD".split(), "ABCDD")
		# If a subset contains a node more than once, that subset cannot appear in a solution.
		self.assertIsNone(grf.partial_cover("AA B".split(), "AB"))

	def testPartialCover(self):
		def checkPartialCover(pieces, nodes):
			cover = grf.partial_cover(pieces, nodes)
			self.assertTrue(all(piece in pieces for piece in cover))
			for node in nodes:
				self.assertTrue(any(node in piece for piece in cover))
			counts = collections.Counter([node for piece in cover for node in piece])
			(_, count), = counts.most_common(1)
			self.assertEqual(count, 1)
		checkPartialCover("AB CD".split(), "AD")
		checkPartialCover("AB BC CD".split(), "ABCD")
		checkPartialCover("AB CD ACE".split(), "AD")
		self.assertFalse(grf.partial_cover("AB BC".split(), "AC"))

	def testPartialCoversAPI(self):
		# Empty and optional subsets.
		self.assertEqual(1, len(grf.partial_covers([], [])))
		self.assertEqual(2, len(grf.partial_covers([[]], [])))
		self.assertEqual(8, len(grf.partial_covers([[], [], []], [])))
		self.assertEqual(8, len(grf.partial_covers([[0], [1], [2]], [])))
		self.assertEqual(5, len(grf.partial_covers([[0, 1], [1, 2], [2, 3]], [])))

	def testMultiCoversAPI(self):
		self.assertEqual(1, len(grf.multi_covers([], [], [])))
		self.assertEqual(2, len(grf.multi_covers([[]], [], [])))
		self.assertEqual(8, len(grf.multi_covers([[], [], []], [], [])))
		self.assertEqual(1, len(grf.multi_covers([[0]], [(0, 1)], [(0, 1)])))
		self.assertEqual(6, len(grf.multi_covers([[0], [0], [0]], [(0, 1)], [(0, 2)])))
		self.assertEqual(4, len(grf.multi_covers([[0], {0: 1}, {0: 2}], [(0, 1)], [(0, 2)])))
		self.assertEqual(1, len(grf.multi_covers([[0], [0]], [(0, 0)], [(0, 0)])))
		self.assertEqual(3, len(grf.multi_covers([[0], [0]], [(0, 1)], [(0, 3)])))
		
		# exact_covers
		self.assertEqual(1, len(grf.exact_covers([])))
		self.assertEqual(2, len(grf.exact_covers([[]])))
		self.assertEqual(8, len(grf.exact_covers([[], [], []])))
		self.assertEqual(2, len(grf.exact_covers([[0], []])))
		self.assertEqual(0, len(grf.exact_covers([[0, 0], []])))
		self.assertEqual(1, len(grf.exact_covers("AB CD".split())))
		self.assertEqual(0, len(grf.exact_covers("AB BC".split())))
		self.assertEqual(6, len(grf.exact_covers("A A A B B".split())))
		self.assertEqual(6, len(grf.exact_covers("A A A B B".split(), max_solutions=10)))
		self.assertEqual(2, len(grf.exact_covers("A A A B B".split(), max_solutions=2)))
		# can_exact_cover
		self.assertTrue(grf.can_exact_cover("AB CD".split()))
		self.assertFalse(grf.can_exact_cover("AB BC".split()))
		# unique_exact_cover
		self.assertTrue(grf.unique_exact_cover("AB CD".split()))
		self.assertFalse(grf.unique_exact_cover("AB BC".split()))
		self.assertFalse(grf.unique_exact_cover("AB CD AD BC".split()))
		# can_unique_exact_cover
		solution, solution_is_unique = grf.can_unique_exact_cover("AB CD AD BC".split())
		self.assertIsNotNone(solution)
		self.assertFalse(solution_is_unique)
		solution, solution_is_unique = grf.can_unique_exact_cover("AB CD AD".split())
		self.assertIsNotNone(solution)
		self.assertTrue(solution_is_unique)
		solution, solution_is_unique = grf.can_unique_exact_cover("AB BC".split())
		self.assertIsNone(solution)
		self.assertFalse(solution_is_unique)

#def parse_polys(spec, annotate = False, align = True, allow_disconnected = False):


	def testParsePolys(self):
		self.assertEqual([((0, 0),)], grf.parse_polys("#"))
		self.assertEqual([((0, 0),)], grf.parse_polys(" #"))
		self.assertEqual([((1, 0),)], grf.parse_polys(" #", align=False))
		self.assertEqual([((0, 0),), ((0, 0),)], grf.parse_polys("# #"))
		self.assertEqual([((0, 0), (2, 0),)], grf.parse_polys("# #", allow_disconnected=True))
		self.assertEqual([((0, 0), (0, 1), (1, 1), (2, 0), (2, 1))], grf.parse_polys("# #\n###"))
		self.assertEqual([((0, 0), (1, 0))], grf.parse_polys("AA"))
		self.assertEqual([((0, 0),), ((0, 0),)], grf.parse_polys("AB"))
		self.assertEqual([((0, 0),), ((0, 0),), ((0, 0),)], grf.parse_polys("ABA"))
		self.assertEqual([((0, 0), (2, 0)), ((0, 0),)], grf.parse_polys("ABA", allow_disconnected=True))
		self.assertEqual([("A", (0, 0))], grf.parse_polys("#", annotate=True))
		self.assertEqual([("A", (0, 0), (1, 0))], grf.parse_polys("##", annotate=True))
		self.assertEqual([("A", (0, 0)), ("B", (0, 0))], grf.parse_polys("# #", annotate=True))
		self.assertEqual([("X", (0, 0)), ("Y", (0, 0))], grf.parse_polys("X Y", annotate=True))
		self.assertEqual([("X", (0, 0)), ("Y", (0, 0))], grf.parse_polys("XY", annotate=True))
		self.assertEqual([("X", (0, 0)), ("X", (0, 0)), ("Y", (0, 0))], grf.parse_polys("XYX", annotate=True))
		self.assertEqual([("X", (0, 0), (2, 0)), ("Y", (0, 0))], grf.parse_polys("XYX", annotate=True, allow_disconnected=True))

	def testParseGrid(self):
		self.assertEqual({}, grf.parse_grid(""))
		self.assertEqual({(0, 0): "A"}, grf.parse_grid("A"))
		self.assertEqual({(0, 0): "A", (1, 0): "B"}, grf.parse_grid("AB"))
		self.assertEqual({(0, 0): "A", (1, 1): "A"}, grf.parse_grid("A\n A"))

	def testAstarUniform(self):
		neighbors = dict(zip("ABCDEFG", "BC ACF ABDF CFE DG BCD E".split())).get
		h = dict(zip("ABCDEFG", (2,2,2,1,0,1,0))).get
		self.assertEqual("".join(grf.astar_uniform("A", "G", neighbors, h)), "ACDEG")

if __name__ == '__main__':
	unittest.main()

