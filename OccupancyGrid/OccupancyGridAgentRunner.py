# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:12:56 2018

@author: 13383861
"""


import argparse
import sys

from OccupancyGridAgent import OccupancyGridAgent
from Utils.UE4Grid import UE4Grid
from Utils.UE4Coord import UE4Coord
from Utils.ActionSelection import get_move_from_belief_map_epsilon_greedy
import AirSimInterface.client as airsim


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
    occupancy_grid_agent.explore_t_timesteps(2)
        
    
def show_grid(grid, client):
    for grid_coord_index in range(1,len(grid.get_grid_points())):
        client.showPlannedWaypoints(grid.get_grid_points()[grid_coord_index-1].x_val, 
                                 grid.get_grid_points()[grid_coord_index-1].y_val,
                                 grid.get_grid_points()[grid_coord_index-1].z_val,
                                 grid.get_grid_points()[grid_coord_index].x_val, 
                                 grid.get_grid_points()[grid_coord_index].y_val, 
                                 grid.get_grid_points()[grid_coord_index].z_val,
                                 lifetime = 200)
        
def parse_args(args):
    parser = argparse.ArgumentParser(description='''Run an Occupancy Grid Agent in a UE4 world with the AirSim plugin running''')
    parser.add_argument('agent_name', type=str, help='The uniquely identifiable name of the agent to run', action='store')
    parser.add_argument('no_timesteps', type=int, help='The number of timesteps for the agent to explore', action='store')
    parser.add_argument('-other_agent_names', "--names-list", nargs="+", help='The uniquely identifiable names of other agents active in the environment', action='store', default = [])
    return parser.parse_args(args)



if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    agent_name = args.agent_name
    print('args: ',args)
    if "names_list" in args:
        other_active_agents = args.names_list
    else:
        other_active_agents = []
    

    #hard code grid for now
    grid = UE4Grid(20, 15, UE4Coord(0,0), 120, 150)    
    client = airsim.MultirotorClient()    
    create_rav(client, agent_name)

    print('Running with other active agents: {}'.format(other_active_agents))
    #grid, initial_pos, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, prior = {}
    OccupancyGridAgent(grid, UE4Coord(0,0), get_move_from_belief_map_epsilon_greedy, -12, 0.3, client, agent_name, other_active_agents).explore_t_timesteps(args.no_timesteps)
    
    
    destroy_rav(client, agent_name)
























