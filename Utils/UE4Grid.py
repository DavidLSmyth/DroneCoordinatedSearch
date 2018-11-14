# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:03:14 2018

@author: 13383861
"""

from Utils.UE4Coord import UE4Coord


class UE4GridFactory:
    def __init__(self, lng_spacing, lat_spacing, origin, x_lim=None, y_lim=None, no_x=None, no_y=None):
        self.lat_spacing = lat_spacing
        self.lng_spacing = lng_spacing
        self.origin = origin
        if x_lim and y_lim:
            self.x_lim, self.y_lim = x_lim, y_lim
            self.create_grid_with_limits()
        if no_x and no_y: 
            self.no_x, self.no_y = no_x, no_y
            self.create_grid_with_no_points()
        if not all([x_lim, y_lim]) and not all([no_x, no_y]):
            raise Exception('Either give a limit to the grid or an x and y spacing')
            
        
    def create_grid_with_limits(self):
        self.no_x = int(self.x_lim/self.lng_spacing)
        self.no_y = int(self.y_lim/self.lat_spacing)
        self.create_grid_with_no_points()
        
    def create_grid_with_no_points(self):
        self.grid = []
        backtrack = False
        for x_counter in range(self.no_x):
            if backtrack:
                for y_counter in range(self.no_y):
                    self.grid.append(self.origin + UE4Coord(x_counter * self.lng_spacing, y_counter * self.lat_spacing))
                backtrack = not backtrack
            else:
                for y_counter in range(self.no_y-1, -1, -1):
                    self.grid.append(self.origin + UE4Coord(x_counter * self.lng_spacing, y_counter * self.lat_spacing))
                backtrack = not backtrack
            
    def get_grid_points(self):
        return self.grid
    
class UE4Grid:
    def __init__(self, lat_spacing, lng_spacing, origin, x_lim=None, y_lim=None, no_x=None, no_y=None):
        if not all([x_lim, y_lim]) and not all([no_x, no_y]):
            raise Exception('Either give a limit to the grid or an x and y spacing')
        self.origin = origin
        self.grid_points = UE4GridFactory(lat_spacing, lng_spacing, origin, x_lim, y_lim, no_x, no_y).get_grid_points()
        self.lat_spacing = lat_spacing
        self.lng_spacing = lng_spacing
        self.no_x = no_x if no_x else int(x_lim/lng_spacing)
        self.no_y = no_y if no_y else int(y_lim/lat_spacing)
    
    def get_grid_points(self):
        return self.grid_points
    
    def get_lat_spacing(self):
        return self.lat_spacing
    
    def get_lng_spacing(self):
        return self.lng_spacing
    
    def get_no_points_x(self):
        return self.no_x
    
    def get_no_points_y(self):
        return self.no_y
    
    def get_neighbors(self, grid_loc, radius):
        '''Gets neighbors of grid_loc within radius.'''
        return list(filter(lambda alt_grid_loc: alt_grid_loc.get_dist_to_other(grid_loc) <= radius and alt_grid_loc != grid_loc, self.get_grid_points()))
    
    
if __name__ == "__main__":
    test_grid = UE4Grid(1, 1, UE4Coord(0,0), 10, 6)
    assert set(test_grid.get_neighbors(UE4Coord(2,2), 1.9)) == set([UE4Coord(1,2), UE4Coord(2,1), UE4Coord(2,3), UE4Coord(3,2), UE4Coord(3,3), UE4Coord(1,3), UE4Coord(1,1), UE4Coord(3,1)])
    