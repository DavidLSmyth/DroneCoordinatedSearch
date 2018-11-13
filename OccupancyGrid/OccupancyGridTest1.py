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
from utils import UE4Coord

from Utils.UE4Grid import UE4Grid
from Utils.ImageAnalysis import sensor_reading
from Utils.UE4Coord import UE4Coord
from Utils.AgentObservation import AgentObservation, AgentObservations



import numpy as np


#%%
class Vector3r:
    x_val = 0.0
    y_val = 0.0
    z_val = 0.0

    def __init__(self, x_val = 0.0, y_val = 0.0, z_val = 0.0):
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val

    @staticmethod
    def nanVector3r():
        return Vector3r(np.nan, np.nan, np.nan)
    
    def __str__(self):
        return f"Vector3r({self.x_val}, {self.y_val}, {self.z_val})"

    def __add__(self, other):
        return Vector3r(self.x_val + other.x_val, self.y_val + other.y_val, self.z_val + other.z_val)

    def __sub__(self, other):
        return Vector3r(self.x_val - other.x_val, self.y_val - other.y_val, self.z_val - other.z_val)

    def __truediv__(self, other):
        if type(other) in [int, float] + np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float']:
            return Vector3r( self.x_val / other, self.y_val / other, self.z_val / other)
        else: 
            raise TypeError('unsupported operand type(s) for /: %s and %s' % ( str(type(self)), str(type(other))) )

    def __mul__(self, other):
        if type(other) in [int, float] + np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float']:
            return Vector3r(self.x_val*other, self.y_val*other, self.z_val)
        else: 
            raise TypeError('unsupported operand type(s) for *: %s and %s' % ( str(type(self)), str(type(other))) )

    def dot(self, other):
        if type(self) == type(other):
            return self.x_val*other.x_val + self.y_val*other.y_val + self.z_val*other.z_val
        else:
            raise TypeError('unsupported operand type(s) for \'dot\': %s and %s' % ( str(type(self)), str(type(other))) )

    def cross(self, other):
        if type(self) == type(other):
            cross_product = np.cross(self.to_numpy_array(), other.to_numpy_array)
            return Vector3r(cross_product[0], cross_product[1], cross_product[2])
        else:
            raise TypeError('unsupported operand type(s) for \'cross\': %s and %s' % ( str(type(self)), str(type(other))) )

    def get_length(self):
        return ( self.x_val**2 + self.y_val**2 + self.z_val**2 )**0.5

    def distance_to(self, other):
        return ( (self.x_val-other.x_val)**2 + (self.y_val-other.y_val)**2 + (self.z_val-other.z_val)**2 )**0.5

    def to_Quaternionr(self):
        return Quaternionr(self.x_val, self.y_val, self.z_val, 0)

    def to_numpy_array(self):
        return np.array([self.x_val, self.y_val, self.z_val], dtype=np.float32)

#%%

#import UE4Coord
        
#import UE4Grid
        
    
        



#test





#%%






#%%
#Calculation of posterior given prior and observations



#%%


#######################  Belief map and tests  #######################


#%%
#######################  Observation Set Manager and tests  #######################



#%%
#########################  Action selection strategies  #########################
       
def get_move_from_belief_map_epsilon_greedy(belief_map: BeliefMap, current_grid_loc: UE4Coord, epsilon: float, eff_radius = None) -> UE4Coord:
    '''Epsilon greedy move selection'''
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

test_grid = UE4Grid(1, 1, UE4Coord(0,0), 6, 5)

obs1 = AgentObservation(UE4Coord(0,0),0.5, 1, 1234, 'agent2')
obs2 = AgentObservation(UE4Coord(0,0),0.7, 2, 1235, 'agent2')
obs3 = AgentObservation(UE4Coord(0,1),0.95, 3, 1237, 'agent2')
#(grid, agent_name, prior = {})
obs_man = ObservationSetManager("agent1")
obs_man.update_rav_obs_set('agent2', [obs1, obs2, obs3])
belief_map = obs_man.get_discrete_belief_map_from_observations(test_grid)

assert get_move_from_belief_map_epsilon_greedy(belief_map, UE4Coord(1,1), 0.0, 1.8) == UE4Coord(0,1)

#%%
#everything that could be important for measuring agent performance/progress
AgentAnalysisState = namedtuple('AgentAnalysisState', ['timestep',
                                                       'timestamp',
                                                       'rav_name',
                                                     'position_intended',
                                                     'position_measured',
                                                     #maybe add distance travelled for current timestep
                                                     'total_dist_travelled',
                                                     'remaining_batt_cap',
                                                     'prop_battery_cap_used',
                                                     'sensor_reading',
                                                     #is it necessary to record the grid along with the likelihoods in case want the grid to 
                                                     #dynamically change? For now assume grid is fixed and in 1-1 correspondance with likelihoods
                                                     #'occ_grid_likelihoods',
                                                     #which other agents did the agent coordinate with on this timestep
                                                     'coordinated_with_other_names'])
    

#metadata related to the agent - details about grid its operating in, prior that was worked with, to be updated...
AgentAnalysisMetadata= namedtuple("MissionAnalysisData", ["agents_used", "grid_origin", 'grid_lat_spacing', 
                                                         'grid_lng_spacing','lng_lim', 'lat_lim',
                                                         'no_lat_points', 'no_lng_points', 'prior'])


    



def get_agent_state_for_analysis(agent_analysis_state: AgentAnalysisState):
    '''Returns elements of agent state that are important for analysis that can be written to csv. Position, battery cap., total_dist_travelled, battery_consumed, occ_grid'''
    #csv_headers = ['timestep', 'timestamp', 'rav_name', 'position_intended', 'position_measured', 'total_dist_travelled', 'remaining_batt_cap', 'prop_battery_cap_used', 'sensor_reading', 'occ_grid_locs', 'occ_grid_likelihoods', 'coordinated_with_other_bool', 'coordinated_with_other_names']
    #return str(agent_analysis_state._fields).replace(')','').replace('(','').replace("'", '')
    return _get_agent_state_for_analysis(**agent_analysis_state._asdict())


def _get_agent_state_for_analysis(timestep, timestamp, rav_name, position_intended, position_measured, total_dist_travelled, remaining_batt_cap, prop_battery_cap_used, sensor_reading, coordinated_with_other_names):
    return f"{timestep},{timestamp},{rav_name},{position_intended},{position_measured},{total_dist_travelled},{remaining_batt_cap},{prop_battery_cap_used},{sensor_reading},{coordinated_with_other_names}"
 

def _get_agent_observation(grid_loc,probability,timestep,timestamp,observer_name):
    return f"{grid_loc},{probability},{timestep},{timestamp},{observer_name}"
     
def get_agent_observation(agent_observation: AgentObservation):
    '''Returns elements of agent state that are important for analysis that can be written to csv. Position, battery cap., total_dist_travelled, battery_consumed, occ_grid'''
    #csv_headers = ['timestep', 'timestamp', 'rav_name', 'position_intended', 'position_measured', 'total_dist_travelled', 'remaining_batt_cap', 'prop_battery_cap_used', 'sensor_reading', 'occ_grid_locs', 'occ_grid_likelihoods', 'coordinated_with_other_bool', 'coordinated_with_other_names']
    #return str(agent_analysis_state._fields).replace(')','').replace('(','').replace("'", '')
    return _get_agent_observation(**agent_observation._asdict())

def _init_state_for_analysis_file(file_path):
    with open(file_path, 'w+') as f:
        f.write(','.join(AgentAnalysisState._fields) + '\n')

def _update_state_for_analysis_file(file_path, csv_row):
    with open(file_path, 'a') as f:
        f.write(csv_row + '\n')

def _init_agent_metadata_file(file_path):
    with open(file_path, 'w+') as f:
        f.write(','.join(AgentAnalysisMetadata._fields) + '\n')

def _write_to_agent_metadata_file(file_path, csv_row):
    with open(file_path, 'a') as f:
        f.write(csv_row + '\n')
    
#AgentObservation = namedtuple('obs_location', ['grid_loc','probability','timestep', 'timestamp', 'observer_name'])
def _init_observations_file(file_path):
    with open(file_path, 'w+') as f:
        f.write(','.join(AgentObservation._fields) + '\n')
        
def _write_to_obserations_file(file_path, agent_observation):
    with open(file_path, 'a') as f:
        f.write(get_agent_observation(agent_observation) + '\n')
    

testAgentAnalysisState = AgentAnalysisState(2, 100, 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test') 
testAgentAnalysisState._asdict()
assert get_agent_state_for_analysis(testAgentAnalysisState) == "2,100,test,test,test,test,test,test,test,test"

def write_agent_performance_to_csv(csv_header, csv_rows, file_name):
    with open(file_name, 'w') as csv_write_f:
        csv_write_f.write(csv_header)
        csv_write_f.write('\n')
        for csv_row in csv_rows:
            csv_write_f.write(csv_row)
            csv_write_f.write('\n')
            

def calc_likelihood(observations: typing.List[float]):
    return functools.reduce(lambda x, y: x*y, observations)

assert calc_likelihood([0.1,0.1,0.2,0.4]) == 0.1*0.1*0.2*0.4


def create_belief_map_from_observations(grid: UE4Grid, agent_name: str, agent_belief_map_prior: typing.Dict[UE4Coord, float], agent_observations: typing.Set[AgentObservations]):
    '''Since the calculation of posterior likelihood is based only on prior and observations (independent of order), updating a belief map component from measurements can be done 
    by the following update formula: 
                                prior * product(over all i observations) observation_i
        ----------------------------------------------------------------------------------------------------------------------
        prior * product(over all i observations) observation_i + (1-prior) * product(over all i observations) (1-observation_i)
    '''

    return_bel_map = create_belief_map(grid.get_grid(), agent_name, agent_belief_map_prior)
    #update belief map based on all observations...
    return_bel_map.update_from_observations(agent_observations)
    #grid, agent_name, prior = {}

#update_bel_map(update_bel_map(test_map, 0.5, 3), 0.5,3)
    

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
    def __init__(self, grid, initial_pos, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, prior = {}):
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
        self.total_dist_travelled = 0
        self.distance_covered_this_timestep = 0
        self.prop_battery_cap_used = 0
        self.current_battery_cap = 1
        self.coordinated_this_timestep = False
        self.others_coordinated_this_timestep = []
        #manages observations of this agent and other agents
        self.observation_manager = ObservationSetManager(self.agent_name)
        
        self.agent_state_file_loc = OccupancyGridAgent.AgentStateDir + f"/{self.agent_name}.csv"
        self.observations_file_loc = OccupancyGridAgent.ObservationDir + f"/{self.agent_name}.csv"
        
        self.current_belief_map = create_belief_map(self.grid, self.agent_name, prior)
        self.init_state_for_analysis_file(self.agent_state_file_loc)
        self.init_observations_file(self.observations_file_loc)  
        
                       
    def __eq__(self, other):
        '''This agent is the same as another agent if names are the same. Refine this later'''
        return self.agent_name == other.agent_name
        
    def get_available_actions(self, state):
        '''Returns actions available to RAV based on its current state'''
        pass
        
    def get_belief_map_after_t_timesteps(self, t):
        '''Calculates what the agent's belief map would be after t timesteps'''
        pass
    
    def charge_battery(self):
        #request from api if battery can be charged. While charging, capacity goes up. Charging can be cancelled at any point to explore some more.
        pass
        
    def move_agent(self, destination_intended: UE4Coord):
        self.rav.moveToPositionAsync(destination_intended.x_val, destination_intended.y_val, self.rav_operational_height, 3, vehicle_name = self.agent_name).join()

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
        return get_agent_state_for_analysis(AgentAnalysisState(self.timestep, time.time(), self.get_agent_name(), 
                                  self.current_pos_intended, self.current_pos_measured,
                                  self.total_dist_travelled, self.rav.getRemainingBatteryCap(),
                                  self.prop_battery_cap_used,
                                  self.current_reading, 
                                  #'[' + ','.join(map(lambda loc: loc.likelihood, self.current_belief_map.get_belief_map_components())) + ']',
                                  #self.get_grid_locs_likelihoods_lists()[1], 
                                  self.others_coordinated_this_timestep))
        
    def init_state_for_analysis_file(self, file_path):
        _init_state_for_analysis_file(file_path)
            
    def update_state_for_analysis_file(self, file_path, csv_row):
        _update_state_for_analysis_file(file_path, csv_row)
        
    def init_observations_file(self, file_path):
        _init_observations_file(file_path)
        
    def update_observations_file(self, file_path, agent_observation):
        _write_to_obserations_file(file_path, agent_observation)
        
        
    #coordination strategy:
    #agent will write all measurements in its possession to a file at each timestep. When communication requested,
    #other agent will read all measurements from the file.
        
    def can_coord_with_other(self, other_rav_name):
        #check if any other ravs in comm radius. if so, return which ravs can be communicated with
        return self.rav.can_coord_with_other(other_rav_name)
    
    def coord_with_other(self, other_rav_name):
        '''coordinate with other rav by requesting their measurement list and sending our own measurement list first write own measurement list to file'''
        if self.can_coord_with_other(other_rav_name):
            observations_from_other_agents = self._read_observations(self)
            print('read observations from other agents: {}'.format(observations_from_other_agents))
            for observations_from_other_agent in observations_from_other_agents.values():
                self.observation_manager.update_rav_obs_set(observations_from_other_agent)
            #this only updates observations not seen previously since a set is maintained of all seen observations
            self.current_belief_map.update_from_observations(self.observation_manager.get_all_observations())
            self.others_coordinated_this_timestep.append(other_rav_name)
            self.coordinated_this_timestep = True
       
    
    def _write_observations(self, file_loc):
        '''writes agent measurements to file to be read by other agent'''
        with open(file_loc, 'a') as f:
            json.dump(str(self.observation_manager.observation_sets), f)
    
    def _read_observations(self, file_loc):
        with open(file_loc, 'r') as f:
            return json.loads(f)
    
    def record_image(self):
        responses = self.rav.simGetImages([ImageRequest("3", ImageType.Scene)], vehicle_name = self.agent_name)
        response = responses[0]
        # get numpy array
        filename = OccupancyGridAgent.ImageDir + "/photo_" + str(self.timestep)
        airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8) 
        print("saved image here: {}".format(os.path.normpath(filename + '.png')))

    def update_agent_pos_measured(self):
        self.current_pos_measured = self.rav.getMultirotorState().kinematics_estimated.position
    
    def explore_timestep(self):
        '''Gets rav to explore next timestep'''
        #grid: UE4Grid, agent_name: str, agent_belief_map_prior: typing.Dict[UE4Coord, float], agent_observations: typing.List[AgentObservations]
        
        next_pos = self.move_from_bel_map_callable(self.current_belief_map, self.current_pos_intended, self.epsilon)
        print("self.current_pos_intended: {}".format(self.current_pos_intended ))
        self.move_agent(next_pos)      
        self.current_pos_intended = next_pos
        
        self.current_pos_measured = self.rav.getMultirotorState(vehicle_name = self.agent_name).kinematics_estimated.position
        
        self.update_agent_pos_measured()
        #record image at location
        self.record_image()
        
        #get sensor reading, can be done on separate thread
        print('getting sensor reading for {}'.format(OccupancyGridAgent.ImageDir + "/photo_" + str(self.timestep) + '.png'))
        
        self.current_reading = float(sensor_reading(OccupancyGridAgent.ImageDir + "/photo_" + str(self.timestep) + '.png')[0])
        #mocked sensor reading
        #self.current_reading = float(sensor_reading("D:/ReinforcementLearning/DetectSourceAgent/Data/MockData/test_train.jpg")[0])
        print('sensro reading: {}'.format(self.current_reading))
    
        print("updating belief map position {} from {}".format(self.current_pos_intended, self.current_belief_map.get_belief_map_component(self.current_pos_intended)))
        
        self.current_belief_map.update_from_prob(self.current_pos_intended, self.current_reading)
        
        print(" to {}".format(self.current_belief_map.get_belief_map_component(self.current_pos_intended)))
        
        #['grid_loc','probability','timestep', 'timestamp', 'observer_name'])
        newest_observation = AgentObservation(self.current_pos_intended, self.current_reading, self.timestep, time.time(), self.agent_name)
        self.observation_manager.update_rav_obs_set(self.agent_name, [AgentObservation(self.current_pos_intended, self.current_reading, self.timestep, time.time(), self.agent_name)])
        
        #self._write_observations(self.observations_file_loc)
        self.update_state_for_analysis_file(self.agent_state_file_loc, self.get_agent_state_for_analysis())
        print("Observation made: {}".format(newest_observation))
        self.update_observations_file(self.observations_file_loc, newest_observation)
        #if agent is in range, communicate
        
    def explore_t_timesteps(self, t: int):
        for i in range(t):
            self.explore_timestep()
            self.timestep += 1
        #print("current belief map: {}".format(self.current_belief_map))
        return self.current_belief_map
                
    
if __name__ != '__main__':
    grid = UE4Grid(20, 15, UE4Coord(0,0), 120, 150)
    class KinematicsState():
        position = Vector3r()
        linear_velocity = Vector3r()
        angular_velocity = Vector3r()
        linear_acceleration = Vector3r()
        angular_acceleration = Vector3r()
    
    class MockRavForTesting:
        def __init__(self):
            self.kinematics_estimated = KinematicsState()
        
        def getRemainingBatteryCap(self):
            return 0.9
        
        def moveToPositionAsync(self, destination_intendedx, destination_intendedy, rav_operational_height, velocity):
            import threading
            t = threading.Thread(target = lambda : None)
            t.start()
            return t
        
        def can_coord_with_other(self,other_rav_name):
            pass
        
        def simGetImages(self, l):
            pass
        
        def getMultirotorState(self):
            return self
        
    def ImageRequest(*args, **kwargs):
        return 
    
    class ImageType(enum.Enum):
        Scene = ''
    
    
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
    
    #for grid_coord_index in range(1,len(grid.get_grid_points())):
    #    client.showPlannedWaypoints(grid.get_grid_points()[grid_coord_index-1].x_val, 
    #                             grid.get_grid_points()[grid_coord_index-1].y_val,
    #                             grid.get_grid_points()[grid_coord_index-1].z_val,
    #                             grid.get_grid_points()[grid_coord_index].x_val, 
    #                             grid.get_grid_points()[grid_coord_index].y_val, 
    #                             grid.get_grid_points()[grid_coord_index].z_val,
    #                             lifetime = 200)

    
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
        
        
        
        
    



