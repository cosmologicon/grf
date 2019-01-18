# http://web.mit.edu/puzzle/www/2012/puzzles/betsy_johnson/caterpillars/solution/

import grf

pieces = """
                        ##
                         ##
                          #
                          ######
                               #
                               ##
                                ####
                                   #
                      ######  #    #
                           ## #    #
                            ###    #
                                ####
                      #####
                          #   #####
                          #       #
                        ###      ##
                       ##        #
                                 #
                                 #
                                 #
                       #     #  ## ##
     ####         ###  ##    #  #   #
        ####        #   #   ##  #   #
           ###      #   #   #       #
             ###    #   #   # ##### #
        ##          #  ##  ## #   ###
###      ##       ###  #   #  ##
  ##      ##     ##    ##  #   #
   ###     #  ## #      #  #    
     ####  #  #  #     ##  #     ##
        ## #  ####         #      #
         ###               ########

"""

polys = grf.parse_polys(pieces, annotate=True)
grid = grf.rect_grid(16, 16)
grid0 = grid
covers = [cover for poly in polys for cover in grf.poly_within_grid(poly, grid, flip=True)]
piece_names = [poly[0] for poly in polys]

solution = grf.partial_cover(covers, piece_names)
for cover in solution:
	label = cover[0]
	for cell in cover[1:]:
		grid[cell] = label
for y in range(20):
	print(*[grid.get((x, y), " ") for x in range(20)])

grid = { cell: ("#" if char == "." else " ") for cell, char in grid.items() }
for y in range(20):
	print(*[grid.get((x, y), " ") * 2 for x in range(20)], sep = "")



