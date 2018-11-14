# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:02:10 2018

@author: 13383861
"""

class UE4Coord:
    '''A coordinate which represents an objects location in an unreal engine environment'''
    def __init__(self, x_val, y_val, z_val = 0):
        self.x_val, self.y_val, self.z_val = x_val, y_val, z_val
        if not isinstance(self.x_val, float):
            try:
                self.x_val = float(self.x_val)
            except Exception as e:
                raise(e)
        
        if not isinstance(self.y_val, float):
            try:
                self.y_val = float(self.y_val)
            except Exception as e:
                raise(e)
                
        if not isinstance(self.z_val, float):
            try:
                self.z_val = float(self.z_val)
            except Exception as e:
                raise(e)
                
    def to_vector3r(self):
        return Vector3r(self.x_val, self.y_val, self.z_val)
        
    def __add__(self, other):
        return UE4Coord(self.x_val + other.x_val, self.y_val + other.y_val, self.z_val + other.z_val)
        
    def __sub__(self, other):
        return UE4Coord(self.x_val - other.x_val, self.y_val - other.y_val, self.z_val - other.z_val)
        
    def mul(self, int):
        pass
    
    def get_dist_to_other(self, other):
        return ((self.x_val - other.x_val)**2 + (self.y_val - other.y_val)**2 + (self.z_val - other.z_val)**2)**0.5
    
    def __eq__(self, other):
        if self.x_val == other.x_val and self.y_val == other.y_val and self.z_val == other.z_val:
            return True
        else:
            return False
    
    def __str__(self):
        return 'UE4Coord({x_val},{y_val},{z_val})'.format(x_val = self.x_val, y_val = self.y_val, z_val = self.z_val)
    
    def __repr__(self):
        return 'UE4Coord({x_val},{y_val},{z_val})'.format(x_val = self.x_val, y_val = self.y_val, z_val = self.z_val)
    
    def __hash__(self):
         return hash(repr(self))