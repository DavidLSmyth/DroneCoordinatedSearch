# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:55:04 2018

@author: 13383861
"""

import random

from Utils.AgentObservation import AgentObservation
from Utils.ObservationSetManager import ObservationSetManager
from Utils.UE4Coord import UE4Coord
from Utils.UE4Grid import UE4Grid
from Utils.BeliefMap import create_belief_map, BeliefMap
       
def get_move_from_belief_map_epsilon_greedy(belief_map: BeliefMap, current_grid_loc: UE4Coord, epsilon: float, eff_radius = None) -> UE4Coord:
    '''Epsilon greedy move selection based on neighbors in belief map'''
    #assume grid is regular, get all neighbors that are within max(lat_spacing, long_spacing)7
    #assuming that lat_spacing < 2* lng_spacing and visa versa
    if not eff_radius:
        eff_radius = max(belief_map.get_grid().get_lat_spacing(), belief_map.get_grid().get_lng_spacing())
    #a list of UE4Coord
    neighbors = belief_map.get_grid().get_neighbors(current_grid_loc, eff_radius)
    #don't move to new position if can't find any neighbors to move to
    if not neighbors:
        return current_grid_loc
    #neighbors = list(filter(lambda grid_loc: grid_loc.get_dist_to_other(current_grid_loc) <= eff_radius and grid_loc!=current_grid_loc, bel_map.keys()))
    if random.random() < epsilon:
        #epsilon random
        return_move = random.choice(neighbors)
    else:
        #otherwise choose move that has highest value
        max_move_value = 0        
        for neighbor in neighbors:
            if belief_map.get_belief_map_component(neighbor).likelihood > max_move_value:
                max_move_value = belief_map.get_belief_map_component(neighbor).likelihood
                return_move = neighbor
#        move = max(map(lambda neighbor: bel_map[neighbor].likelihood, neighbors))
    return return_move

if __name__ == "__main__":
    test_grid = UE4Grid(1, 1, UE4Coord(0,0), 6, 5)
    
    obs1 = AgentObservation(UE4Coord(0,0),0.5, 1, 1234, 'agent2')
    obs2 = AgentObservation(UE4Coord(0,0),0.7, 2, 1235, 'agent2')
    obs3 = AgentObservation(UE4Coord(0,1),0.95, 3, 1237, 'agent2')
    #(grid, agent_name, prior = {})
    obs_man = ObservationSetManager("agent1")
    obs_man.update_rav_obs_set('agent2', [obs1, obs2, obs3])
    belief_map = obs_man.get_discrete_belief_map_from_observations(test_grid)
    
    assert get_move_from_belief_map_epsilon_greedy(belief_map, UE4Coord(1,1), 0.0, 1.8) == UE4Coord(0,1)