# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:11:39 2018

@author: 13383861
"""

import sys
import enum
sys.path.append('.')
sys.path.append('..')
sys.path.append("D:\ReinforcementLearning\DetectSourceAgent")
import os
import time
from collections import namedtuple
import random
import typing
import functools
import json
import threading
import pathlib

import AirSimInterface.client as airsim
from AirSimInterface.types import *

from Utils.UE4Coord import UE4Coord
from Utils.UE4Grid import UE4Grid
from Utils.ImageAnalysis import sensor_reading
from Utils.UE4Coord import UE4Coord
from Utils.AgentObservation import (AgentObservation, AgentObservations, 
                                    _init_observations_file, _write_to_obserations_file)
from Utils.ObservationSetManager import ObservationSetManager
from Utils.BeliefMap import (BeliefMap, create_belief_map, 
                             create_belief_map_from_observations, 
                             BeliefMapComponent, calc_posterior)

from Utils.AgentAnalysis import (AgentAnalysisState, AgentAnalysisMetadata,
                                 get_agent_state_for_analysis, _get_agent_state_for_analysis,
                                 _init_state_for_analysis_file, _update_state_for_analysis_file,
                                 _init_agent_metadata_file,_write_to_agent_metadata_file)

from Utils.ActionSelection import get_move_from_belief_map_epsilon_greedy



import numpy as np


#%%           

def calc_likelihood(observations: typing.List[float]):
    return functools.reduce(lambda x, y: x*y, observations)

assert calc_likelihood([0.1,0.1,0.2,0.4]) == 0.1*0.1*0.2*0.4



    #grid, agent_name, prior = {}

#update_bel_map(update_bel_map(test_map, 0.5, 3), 0.5,3)
    

                
    
if __name__ != '__main__':
    from ClientMock import Vector3r
    grid = UE4Grid(20, 15, UE4Coord(0,0), 120, 150)
    
    #grid, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, performance_csv_path: "file path that agent can write performance to", prior = []
    occupancy_grid_agent = OccupancyGridAgent(grid, get_move_from_belief_map_epsilon_greedy, -12, 0.2, MockRavForTesting(), 'agent1')
    #write some tests for agent here
    occupancy_grid_agent.current_pos_intended = UE4Coord(0,0)
    occupancy_grid_agent.current_pos_measured = None
    occupancy_grid_agent.current_reading = 0.1
    occupancy_grid_agent.get_agent_state_for_analysis()
    occupancy_grid_agent.explore_timestep()
    
    #####################  Functions that can deal with the initialization of RAVs  ####################
    #%%       
def create_rav(client, rav_name):
    client.confirmConnection()
    client.enableApiControl(True, rav_name)
    client.armDisarm(True, rav_name)
    client.takeoffAsync(vehicle_name = rav_name).join()


def destroy_rav(client, rav_name):
    client.enableApiControl(False, rav_name)
    client.landAsync(vehicle_name = rav_name).join()
    client.armDisarm(False, rav_name)
    

def run_t_timesteps(occupancy_grid_agent):
    print('type(occupancy_grid_agent)', type(occupancy_grid_agent))
    occupancy_grid_agent.explore_t_timesteps(2)
        
def show_grid(grid):
    for grid_coord_index in range(1,len(grid.get_grid_points())):
        client.showPlannedWaypoints(grid.get_grid_points()[grid_coord_index-1].x_val, 
                                 grid.get_grid_points()[grid_coord_index-1].y_val,
                                 grid.get_grid_points()[grid_coord_index-1].z_val,
                                 grid.get_grid_points()[grid_coord_index].x_val, 
                                 grid.get_grid_points()[grid_coord_index].y_val, 
                                 grid.get_grid_points()[grid_coord_index].z_val,
                                 lifetime = 200)

#%%    

if __name__ == '__main__':
    
    grid = UE4Grid(20, 15, UE4Coord(0,0), 120, 150)
    rav_names = ["Drone2"]
                 #, "Drone2"]
    client = airsim.MultirotorClient()
    for rav_name in rav_names:
        create_rav(client, rav_name)
    #assert client.getVehiclesInRange("Drone1", ["Drone2"],1000000) == ["Drone2"]
    #print('vehicles in range: ', client.getVehiclesInRange("Drone1", ["Drone2"] ,1000000))
    #rav1.simShowPawnPath(False, 1200, 20)
    #grid shared between rav
    
    
    #grid, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, performance_csv_path: "file path that agent can write performance to", prior = []
    
    

    
    occupancy_grid_agent1 = OccupancyGridAgent(grid, UE4Coord(0,0), get_move_from_belief_map_epsilon_greedy, -12, 0.3, client, "Drone2")
    occupancy_grid_agent1.explore_t_timesteps(20)
    #occupancy_grid_agent2 = OccupancyGridAgent(grid, UE4Coord(20,15),get_move_from_belief_map_epsilon_greedy, -12, 0.3, client, "Drone2")
    #occupancy_grid_agent1.explore_t_timesteps(10)
    #p1 = threading.Thread(target = run_t_timesteps, args = (occupancy_grid_agent1,))
    #p2 = threading.Thread(target = run_t_timesteps, args = (occupancy_grid_agent2,))
    #p1.start()
    #p2.start()
    #p1.join()
    #p2.join()

    
    # showPlannedWaypoints(self, x1, y1, z1, x2, y2, z2, thickness=50, lifetime=10, debug_line_color='red', vehicle_name = '')
    
    
    destroy_rav(client, "Drone2")
    #destroy_rav(client, "Drone2")
    #for grid_loc in grid_locs:
    ##rav.moveOnPathAsync(list(map(lambda x: x.to_vector3r(),grid_locs)), 8)
    #rav.moveToPositionAsync(0,0, -20, 5).join()
    #print('rav position: {}'.format(rav.getMultirotorState().kinematics_estimated.position))
    #responses = rav.simGetImages([ImageRequest("3", ImageType.Scene)])
    #response = responses[0]
    #filename = OccupancyGridAgent.ImageDir + "/photo_" + str(1)
    #airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    # grid, move_from_bel_map_callable, height, epsilon, multirotor_client, prior = []
    #pos, likelihood = OccupancyGridAgent(grid, get_move_from_bel_map, -12, 0.3, rav, "Drone1").explore_t_timesteps(125)
    #print('determined {} as source with likelihood {}'.format(pos, likelihood))
    #rav.moveToPositionAsync(pos.x_val, pos.y_val, -5, 3).join()
        
        
        
        
    



