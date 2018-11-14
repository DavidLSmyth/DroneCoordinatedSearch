# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:17:07 2018

@author: 13383861
"""

import sys
sys.path.append('..')
sys.path.append("D:\ReinforcementLearning\DetectSourceAgent")
import os
import time
import json
from collections import namedtuple
import logging

import numpy as np

import AirSimInterface.client as airsim
from AirSimInterface.types import *

from Utils.UE4Coord import UE4Coord
from Utils.UE4Grid import UE4Grid
from Utils.ImageAnalysis import sensor_reading
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


class BaseROCSAFEAgent:
    '''Base class for all agents related to the ROCSAFE project, contains minimal functionality. Designed with goal main in mind
    to be able to compare and measure agent performance in a consistent way'''
    pass
    
class BaseGridAgent:
    '''Base class for all agents that use a grid representation of the environment, contains minimal functionality. Designed with goal main in mind
    to be able to compare and measure agent performance in a consistent way'''
    def __init__(self):
        pass
    
    def actuate(self):
        pass
    
    def perceive(self):
        pass
    
    def get_state(self):
        pass
        
def setup_logger(name, log_file, formatter, level=logging.INFO):
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
def logger(msg, level, logfile): 
    log = logging.getLogger(logfile)
    if level == 'info'    : log.info(msg) 
    if level == 'warning' : log.warning(msg)
    if level == 'error'   : log.error(msg)


#logging.basicConfig(datefmt='%Y/%m/%d %I:%M:%S %p',
#                    level=autologging.TRACE,
#                    filename='D:/ReinforcementLearning/DetectSourceAgent/Logging/.log', 
#                    format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")
    
root_logger = logging.getLogger('')
log_directory = "D:/ReinforcementLearning/DetectSourceAgent/Logging/"
general_formatter = logging.Formatter("%(levelname)s:%(name)s:%(funcName)s:%(message)s")
csv_formatter = logging.Formatter("%(message)s")

#create a base agent class

class OccupancyGridAgent():
    '''agent that moves around an occupancy grid in order to locate a source of radiation. Uses a rav agent'''
    ImageDir = 'D:/ReinforcementLearning/DetectSourceAgent/Data/SensorData'
    #stores analysis csvs. Each csv contains agent state at each timestep
    AgentStateDir = "D:/ReinforcementLearning/DetectSourceAgent/Analysis"
    #stores observation json
    ObservationDir = "D:/ReinforcementLearning/DetectSourceAgent/Observations"
    MockedImageDir = 'D:/ReinforcementLearning/DetectSource/Data/MockData'
    #break apart this into components, one which manages actuation/sensing, one which manages/represents state, etc.
    def __init__(self, grid, initial_pos, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, other_active_agents = [], prior = {}, comms_radius = 100000):
        #list expected types of everything here   
        self.rav = multirotor_client
        self.grid = grid
        self.current_pos_intended = initial_pos
        self.grid_locs = grid.get_grid_points()
        self.timestep = 0
        self.rav_operational_height = height
        self.move_from_bel_map_callable = move_from_bel_map_callable
        self.epsilon = epsilon
        self.agent_name = agent_name
        self.agent_states_for_analysis = []
        self.other_active_agents = other_active_agents
        self.total_dist_travelled = 0
        self.distance_covered_this_timestep = 0
        self.prop_battery_cap_used = 0
        self.current_battery_cap = 1
        self.comms_radius = comms_radius
        self.others_coordinated_this_timestep = []
        #manages observations of this agent and other agents
        self.observation_manager = ObservationSetManager(self.agent_name)
        
        self.setup_logs()
        
        self.agent_state_file_loc = OccupancyGridAgent.AgentStateDir + f"/{self.agent_name}.csv"
        self.observations_file_loc = OccupancyGridAgent.ObservationDir + f"/{self.agent_name}.csv"
        
        self.current_belief_map = create_belief_map(self.grid, self.agent_name, prior)
        self.init_state_for_analysis_file(self.agent_state_file_loc)
        self.init_observations_file(self.observations_file_loc)  
        
        
    def setup_logs(self):
        setup_logger("move",log_directory+self.agent_name+"move_log.log", general_formatter)
        setup_logger("state",log_directory+self.agent_name+"state_log.log",csv_formatter)
        setup_logger("comms",log_directory+self.agent_name+"comms_log.log", general_formatter)
        setup_logger("observations",log_directory+self.agent_name+"observations_log.log", csv_formatter)
        
                       
    def __eq__(self, other):
        '''This agent is the same as another agent if names are the same. Refine this later'''
        return self.agent_name == other.agent_name
        
    def get_available_actions(self, state):
        '''Returns actions available to RAV based on its current state'''
        #currently just move, eventually: go_to_charge, keep_charging, stop_charging, etc.
        pass
        
    def get_belief_map_after_t_timesteps(self, t):
        '''Calculates what the agent's belief map would be after t timesteps'''
        pass
    
    def charge_battery(self):
        #request from api if battery can be charged. While charging, capacity goes up. Charging can be cancelled at any point to explore some more.
        pass
        
    def move_agent(self, destination_intended: UE4Coord):
        self.rav.moveToPositionAsync(destination_intended.x_val, destination_intended.y_val, self.rav_operational_height, 3, vehicle_name = self.agent_name).join()
        logger(str(destination_intended) + ' ' + str(self.timestep), "info", "move")
        self.distance_covered_this_timestep = self.current_pos_intended.get_dist_to_other(destination_intended)
        self.total_dist_travelled += self.current_pos_intended.get_dist_to_other(destination_intended)
        
        self.prop_battery_cap_used += self.current_battery_cap - self.rav.getRemainingBatteryCap()
        self.current_battery_cap = self.rav.getRemainingBatteryCap()
        
    def get_agent_name(self):
        return self.agent_name
    
    def get_agent(self):
        return self.rav
        
    def get_agent_state_for_analysis(self):
        '''AgentAnalysisState = namedtuple('AgentAnalysisState', ['timestep','timestamp','rav_name',
                                                 'position_intended','position_measured',
                                                 'total_dist_travelled','remaining_batt_cap',
                                                 'prop_battery_cap_used',
                                                 'sensor_reading',
                                                 #which other agents did the agent coordinate with on this timestep
                                                 'coordinated_with_other_names'])'''
        return AgentAnalysisState(self.timestep, time.time(), self.get_agent_name(), 
                                  self.current_pos_intended, self.current_pos_measured,
                                  self.total_dist_travelled, self.rav.getRemainingBatteryCap(),
                                  self.prop_battery_cap_used,
                                  self.current_reading, 
                                  self.others_coordinated_this_timestep)
        
    def init_state_for_analysis_file(self, file_path):
        _init_state_for_analysis_file(file_path)
            
    def update_state_for_analysis_file(self, file_path, agent_state):
        _update_state_for_analysis_file(file_path, agent_state)
        
    def init_observations_file(self, file_path):
        _init_observations_file(file_path)
        
    def update_observations_file(self, file_path, agent_observation):
        _write_to_obserations_file(file_path, agent_observation)
        
        
    #coordination strategy:
    #agent will write all measurements in its possession to a file at each timestep. When communication requested,
    #other agent will read all measurements from the file.
        
    def can_coord_with_other(self, other_rav_name, range_m):
        #check if any other ravs in comm radius. if so, return which ravs can be communicated with
        return self.rav.can_coord_with_other(self.get_agent_name(), other_rav_name, range_m)
    
    def coord_with_other(self, other_rav_name):
        '''coordinate with other rav by requesting their measurement list and sending our own measurement list first write own measurement list to file'''
        logger(str(other_rav_name) + ' ' + str(self.timestep), "info", "comms")

        observations_from_other_agents = self._read_observations(OccupancyGridAgent.ObservationDir + f"/{other_rav_name}.csv")
        #update observation manager and also observation file
        for observation_from_other_agent in observations_from_other_agents:
            new_observation = self.observation_manager.update_with_obseravation(observation_from_other_agent)
            if new_observation:
                #update observations not already observed
                self.update_observations_file(self.observations_file_loc, new_observation)
                logger(str(observation_from_other_agent), "info", "observations")

        self.current_belief_map = create_belief_map_from_observations(self.grid, self.agent_name, self.current_belief_map.get_prior(), self.observation_manager.get_all_observations())
        self.others_coordinated_this_timestep.append(other_rav_name)       

    
    def _read_observations(self, file_loc):
        with open(file_loc, 'r') as f:
            lines = f.read().split('\n')
            observations = set([AgentObservation(eval(line.split(', ')[0]), float(line.split(', ')[1]), int(line.split(', ')[2]), float(line.split(', ')[3]), str(line.split(', ')[4])) for line in lines[1:]])
            return observations
    
    def record_image(self):
        responses = self.rav.simGetImages([ImageRequest("3", ImageType.Scene)], vehicle_name = self.agent_name)
        response = responses.pop()
        # get numpy array
        filename = OccupancyGridAgent.ImageDir + "/photo_" + str(self.timestep)
        airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8) 

    def update_agent_pos_measured(self):
        self.current_pos_measured = self.rav.getMultirotorState().kinematics_estimated.position
    
    def explore_timestep(self):
        '''Gets rav to explore next timestep'''
        
        next_pos = self.move_from_bel_map_callable(self.current_belief_map, self.current_pos_intended, self.epsilon)
        print("self.current_pos_intended: {}".format(self.current_pos_intended ))
        self.move_agent(next_pos)      
        self.current_pos_intended = next_pos
        
        self.current_pos_measured = self.rav.getMultirotorState(vehicle_name = self.agent_name).kinematics_estimated.position
        
        self.update_agent_pos_measured()
        #record image at location
        self.record_image()
        
        #get sensor reading, can be done on separate thread        
        self.current_reading = float(sensor_reading(OccupancyGridAgent.ImageDir + "/photo_" + str(self.timestep) + '.png')[0])

        self.current_belief_map.update_from_prob(self.current_pos_intended, self.current_reading)
                
        newest_observation = AgentObservation(self.current_pos_intended, self.current_reading, self.timestep, time.time(), self.agent_name)
        self.observation_manager.update_rav_obs_set(self.agent_name, [AgentObservation(self.current_pos_intended, self.current_reading, self.timestep, time.time(), self.agent_name)])
        
        #coordinate with other agents if possible
        for other_active_agent in self.other_active_agents:
            if self.can_coord_with_other(other_active_agent, self.comms_radius):
                self.coord_with_other(other_active_agent)

        #self._write_observations(self.observations_file_loc)
        self.update_state_for_analysis_file(self.agent_state_file_loc, self.get_agent_state_for_analysis())
        logger(str(self.get_agent_state_for_analysis()), "info", "state")
        self.update_observations_file(self.observations_file_loc, newest_observation)
        logger(str(newest_observation), "info", "observations")
        #if agent is in range, communicate
        
    def explore_t_timesteps(self, t: int):
        for i in range(t):
            self.explore_timestep()
            self.timestep += 1
        #print("current belief map: {}".format(self.current_belief_map))
        return self.current_belief_map
    
if __name__ == "__main__":
    from Utils.ClientMock import KinematicsState, MockRavForTesting, ImageType, Vector3r
    grid = UE4Grid(15, 20, UE4Coord(0,0), 60, 45)
    #grid, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, performance_csv_path: "file path that agent can write performance to", prior = []
    #grid, initial_pos, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, prior = {}
    occupancy_grid_agent = OccupancyGridAgent(grid, UE4Coord(0,0), get_move_from_belief_map_epsilon_greedy, -12, 0.2, MockRavForTesting(), 'agent1')
    #write some tests for agent here
    occupancy_grid_agent.current_pos_intended = UE4Coord(0,0)
    occupancy_grid_agent.current_pos_measured = None
    occupancy_grid_agent.current_reading = 0.1
    occupancy_grid_agent.get_agent_state_for_analysis()
    occupancy_grid_agent.explore_timestep()
        
    #belief_map: BeliefMap, current_grid_loc: UE4Coord, epsilon: float, eff_radius = None) -> UE4Coord:
    dont_move = lambda belief_map, current_grid_loc, epsilon:  UE4Coord(15,20)
    print(grid.get_grid_points())
    ###Check that agents can communicate with each other
    occupancy_grid_agent1 = OccupancyGridAgent(grid, UE4Coord(0,0), get_move_from_belief_map_epsilon_greedy, -12, 0.2, MockRavForTesting(), 'agent1', ['agent2'])
    occupancy_grid_agent2 = OccupancyGridAgent(grid, UE4Coord(15,20), dont_move, -12, 0.2, MockRavForTesting(), 'agent2', ['agent1'])
    
    print(occupancy_grid_agent2.can_coord_with_other(occupancy_grid_agent1.agent_name))
    occupancy_grid_agent1.explore_timestep()
    occupancy_grid_agent2.explore_timestep()
    occupancy_grid_agent1.coord_with_other("agent2")
    assert occupancy_grid_agent1.current_belief_map.get_belief_map_component(UE4Coord(15,20)).likelihood < 0.1
    #assert occupancy_grid_agent1.current_belief_map.get_belief_map_component(UE4Coord(20,15)).likelihood < 0.1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    