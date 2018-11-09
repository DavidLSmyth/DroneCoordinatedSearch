# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:11:39 2018

@author: 13383861
"""

import sys
sys.path.append('.')
sys.path.append('..')
import requests
import os
import time
from collections import namedtuple
import copy
import random
import typing
import functools
import json
import multiprocessing
import pathlib
import AirSimInterface.client as airsim
from AirSimInterface.types import *
import numpy as np



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
        return 'UE4Coord({x_val}, {y_val}, {z_val})'.format(x_val = self.x_val, y_val = self.y_val, z_val = self.z_val)
    
    def __repr__(self):
        return 'UE4Coord({x_val}, {y_val}, {z_val})'.format(x_val = self.x_val, y_val = self.y_val, z_val = self.z_val)
    
    def __hash__(self):
         return hash(repr(self))
        
class UE4GridFactory:
    def __init__(self, lat_spacing, lng_spacing, origin, x_lim=None, y_lim=None, no_x=None, no_y=None):
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
        return list(filter(lambda alt_grid_loc: alt_grid_loc.get_dist_to_other(grid_loc) <= radius and grid_loc!=current_grid_loc, self.get_grid_points()))
        
test_grid = UE4Grid(20, 15, UE4Coord(0,0), 100, 60)
    
def get_image_response(image_loc: str):
    headers = {'Prediction-Key': "fdc828690c3843fe8dc65e532d506d7e", "Content-type": "application/octet-stream", "Content-Length": "1000"}
    with open(image_loc,'rb') as f:
        response =requests.post('https://southcentralus.api.cognitive.microsoft.com/customvision/v2.0/Prediction/287a5a82-272d-45f3-be6a-98bdeba3454c/image?iterationId=3d1fd99c-0b93-432f-b275-77260edc46d3', data=f, headers=headers)
    return response.json()


def get_highest_pred(image_json):
    max_pred = 0
    max_pred_details = ''
    for pred in image_json['predictions']:
        if pred['probability'] > max_pred:
            max_pred = pred['probability']
            max_pred_details = pred
    return max_pred, max_pred_details
        
sensor_reading = lambda image_loc: get_highest_pred(get_image_response(image_loc))

assert get_highest_pred(get_image_response('C:/Users/13383861/Downloads/test_train.jpg'))[0] > 0.6
#test



#an agent precept consists of a grid location, a detection probability, a timestep, a timestamp and the observer name
AgentObservation = namedtuple('obs_location', ['grid_loc','probability','timestep', 'timestamp', 'observer_name'])

#A belief map component consists of a grid location and a likelihood
BeliefMapComponent = namedtuple('belief_map_component', ['grid_loc','likelihood'])

class AgentObservations():
    '''A class which records agent observations in a UE4Grid'''
    def __init__(self, grid: UE4Grid):
        self.grid = grid
        #observations consist of agent percepts
        self.observations = []
        
    def record_agent_observation(self, new_observation: AgentObservation):
        self.observations.append(new_observation)
        
    def get_most_recent_observation(self, observations = []):
        '''Returns the most recent observation in a list of observations'''
        if not observations:
            observations = self.observations
        return sorted(observations, key = lambda observation: observation.timestamp, reverse = True)[0]
    
    def get_most_recent_observation_at_position(self, grid_loc: UE4Coord):
        return self.get_most_recent_observation(self.get_all_observations_at_position(grid_loc))
    
    def get_all_observations_at_position(self, grid_loc: UE4Coord):
        return list(filter(lambda observation: observation.grid_loc == grid_loc, self.observations))
    
test_agent_observations = AgentObservations(test_grid)
obs1 = AgentObservation(UE4Coord(0,0),0.5, 1, 1234, 'agent1')
obs2 = AgentObservation(UE4Coord(0,0),0.7, 2, 1235, 'agent1')
obs3 = AgentObservation(UE4Coord(0,1),0.9, 3, 1237, 'agent1')

test_agent_observations.record_agent_observation(obs1)
test_agent_observations.record_agent_observation(obs2)
test_agent_observations.record_agent_observation(obs3)

assert test_agent_observations.get_most_recent_observation() == obs3
assert test_agent_observations.get_most_recent_observation_at_position(UE4Coord(0,0)) == obs2
assert test_agent_observations.get_all_observations_at_position(UE4Coord(0,1)) == [obs3]


calc_posterior = lambda observation, prior: (prior * observation) / ((prior * observation) + (1-prior)*(1-observation))
assert abs(calc_posterior(0.5, 0.2) - 0.2) <= 0.001
assert abs(calc_posterior(0.8, 0.2) - 0.5) <= 0.001

def get_posterior_given_obs(observations:list, prior):
    '''For a sequence of observations calculates the posterior probability given a prior.'''
    for observation in observations:
        prior = calc_posterior(observation, prior)
    return prior
        
assert abs(get_posterior_given_obs([0.5,0.2,0.8], 0.5) - 0.5) <= 0.001


#A belief map has an agent name (beliefs belong to an agent) consists of belief map components
#Leave this as namedtuple if don't need to define methods
class BeliefMap(typing.NamedTuple):
    agent_name: str
    grid: UE4Grid
    belief_map_components: typing.List[BeliefMapComponent]
    prior: typing.Dict[UE4Coord, float]
    
    def get_prior(self)->typing.Dict[UE4Coord, float]:
        return self.prior
    
    def get_grid(self)->UE4Grid:
        return self.grid
            
    def get_belief_map_components(self):
        return self.belief_map_components
    
    def update_from_prob(self, grid_loc, obs_prob):
        prior_val = self.get_belief_map_component(grid_loc).likelihood
        self.belief_map_components[self._get_observation_grid_index(grid_loc)] = BeliefMapComponent(grid_loc, get_posterior_given_obs([obs_prob], prior_val))
        
    def update_from_observation(self, agent_observation: AgentObservation):
        self.update_from_prob(agent_observation.grid_loc, agent_observation.probability)
        
    def update_from_observations(self, agent_observations: typing.List[AgentObservation]):
        for observation in sorted(agent_observations, key = lambda x: x.timestamp, reverse = True):
            self.update_from_prob(observation.probability)
    
    def _get_observation_grid_index(self, grid_loc: UE4Coord):
        return self.belief_map_components.index(self.get_belief_map_component(grid_loc))
    
    def get_belief_map_component(self, grid_loc):
        return next(filter(lambda belief_map_component: belief_map_component.grid_loc == grid_loc, self.belief_map_components))
 
#maybe this should go in contsructor and make regular class
def create_belief_map(grid, agent_name, prior = {}):
    '''Creates an occupancy belief map for a given observer and a set of grid locations.
    Prior is a mapping of grid_points to probabilities'''
    if not prior:
        #use uniform uninformative prior
        prior = {grid_point: 1/len(grid.get_grid_points()) for grid_point in grid.get_grid_points()}
    return BeliefMap(agent_name, grid, [BeliefMapComponent(grid_point, prior[grid_point]) for grid_point in grid.get_grid_points()], prior)
    #return {grid_locs[i]: ObsLocation(grid_locs[i],prior[i], 0, time.time(), observer_name) for i in range(len(grid_locs))}

test_map = create_belief_map(test_grid, "agent1")
assert test_map.get_belief_map_component(UE4Coord(0,0)) == BeliefMapComponent(UE4Coord(0,0), 1/len(test_grid.get_grid_points()))
assert test_map._get_observation_grid_index(UE4Coord(0,0)) == 2
test_map.update_from_prob(UE4Coord(0,0), 0.9)
assert 0.34<test_map.get_belief_map_component(UE4Coord(0,0)).likelihood < 0.35


#prove order in which observations come in doesn't matter
obs1 = AgentObservation(UE4Coord(0,0),0.4, 1, 1234, 'agent1')
obs2 = AgentObservation(UE4Coord(0,0),0.7, 2, 1235, 'agent1')
obs3 = AgentObservation(UE4Coord(0,0),0.93, 3, 1237, 'agent1')
test_map = create_belief_map(test_grid, "agent1")
test_map.update_from_observation(obs1)
test_map.update_from_observation(obs2)
test_map.update_from_observation(obs3)
assert 0.54 < test_map.get_belief_map_component(UE4Coord(0,0)).likelihood < 0.55 

#now check observing in a different order gives same result
test_map = create_belief_map(test_grid, "agent1")
test_map.update_from_observation(obs2)
test_map.update_from_observation(obs1)
test_map.update_from_observation(obs3)
assert 0.54 < test_map.get_belief_map_component(UE4Coord(0,0)).likelihood < 0.55 

#now check observing in a different order gives same result
test_map = create_belief_map(test_grid, "agent1")
test_map.update_from_observation(obs3)
test_map.update_from_observation(obs2)
test_map.update_from_observation(obs1)
assert 0.54 < test_map.get_belief_map_component(UE4Coord(0,0)).likelihood < 0.55 



class ObservationListManager:
    '''Manages the sensor measurements of other agents'''
    
    def __init__(self, self_agent_name: 'name of the agent that owns this observation list manager'):
        #self.self_meas_list = []
        #key value pairs of form agent_name: list of AgentObservations
        self.observation_sets = {}
        self.init_rav_observation_set(self_agent_name)
    
    def init_rav_observation_set(self, rav_name, observations: typing.Set[AgentObservation]=set()):
        '''initialise a new list of observations for a RAV'''
        self.observation_sets[rav_name] = observations
    
    def get_observation_set(self, rav_name) -> typing.Set[AgentObservation]:
        '''Get list of observations from a RAV'''
        return self.observation_sets[rav_name]
    
    def update_rav_obs_set(self, rav_name, observations: typing.Set[AgentObservation]):
        #check if rav is present before updating
        if rav_name not in self.observation_sets:
            self.init_rav_observation_set(rav_name)
        #this avoids recording duplicate observations
        self.observation_sets[rav_name].update(observations)
        
    def update_from_other_obs_list_man(self, other):
        '''Might need to check that the timestamps must be different...'''
        for rav_name, observation_set in other.observation_sets.items():
            self.update_rav_obs_set(rav_name, observation_set)
        
   
test_ObservationListManager = ObservationListManager('agent1')

obs1 = AgentObservation(UE4Coord(0,0),0.5, 1, 1234, 'agent2')
obs2 = AgentObservation(UE4Coord(0,0),0.7, 2, 1235, 'agent2')
obs3 = AgentObservation(UE4Coord(0,1),0.9, 3, 1237, 'agent2')

test_ObservationListManager.init_rav_observation_set('agent2', set([obs1, obs2]))
test_ObservationListManager.update_rav_obs_set('agent2', set([obs3]))

assert test_ObservationListManager.get_observation_set('agent2') == set([obs1, obs2, obs3])
assert test_ObservationListManager.get_observation_set('agent1') == set()
   
#test that duplicate measurements won't occur
obs4 = AgentObservation(UE4Coord(0,1),0.9, 3, 1237, 'agent2')
test_ObservationListManager.update_rav_obs_set('agent2', set([obs4]))
assert test_ObservationListManager.get_observation_set('agent2') == set([obs1, obs2, obs3])

#%%

       
def get_move_from_belief_map(bel_map, current_grid_loc, lat_spacing, lng_spacing, epsilon):
    '''take and epsilon greedy approach'''
    #assume grid is regular, get all neighbors that are within max(lat_spacing, long_spacing)7
    #assuming that lat_spacing < 2* lng_spacing and visa versa
    eff_radius = max(lat_spacing, lng_spacing)
    neighbors = bel_map.get_grid().get_neighbors(current_grid_loc, eff_radius)
    #neighbors = list(filter(lambda grid_loc: grid_loc.get_dist_to_other(current_grid_loc) <= eff_radius and grid_loc!=current_grid_loc, bel_map.keys()))
    print('neighbors: {}'.format(neighbors))
    if random.random() < epsilon:
        #epsilon random
        move = random.choice(neighbors)
        print('returning random move: {}'.format(move))
    else:
        #otherwise choose move that has highest value
        move = neighbors[0]
        max_move_value = bel_map[neighbors[0]].likelihood
        for neighbor in neighbors:
            if bel_map[neighbor].likelihood > max_move_value:
                max_move_value = bel_map[neighbor].likelihood
                move = neighbor
#        move = max(map(lambda neighbor: bel_map[neighbor].likelihood, neighbors))
        print('returning greedy move: {}'.format(move))
    return move

#everything that could be important for measuring agent performance/progress
AgentAnalysisState = namedtuple('AgentAnalysisState', ['timestep','timestamp','rav_name',
                                                 'position_intended','position_measured',
                                                 'total_dist_travelled','remaining_batt_cap',
                                                 'prop_battery_cap_used',
                                                 'sensor_reading','occ_grid_locs',
                                                 'occ_grid_likelihoods','coordinated_with_other_bool',
                                                 'coordinated_with_other_names'])

testAgentAnalysisState = AgentAnalysisState(2, 100, 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test')
    
def _init_state_for_analysis_file(file_path, csv_header):
    with open(file_path, 'w') as f:
        f.write(csv_header + '\n')
        
        
def _update_state_for_analysis_file(file_path, csv_row):
    with open(file_path, 'w') as f:
        f.write(csv_row + '\n')
        


def _get_agent_state_for_analysis(timestep, timestamp, rav_name, position_intended, position_measured, total_dist_travelled, remaining_batt_cap, prop_battery_cap_used, sensor_reading, occ_grid_locs, occ_grid_likelihoods, coordinated_with_other_bool, coordinated_with_other_names):
    return f"{timestep},{timestamp},{rav_name},{position_intended},{position_measured},{total_dist_travelled},{remaining_batt_cap},{prop_battery_cap_used},{sensor_reading},{occ_grid_locs},{occ_grid_likelihoods},{coordinated_with_other_bool},{coordinated_with_other_names}"
    
    
def get_agent_state_for_analysis(agent_analysis_state: AgentAnalysisState):
    '''Returns elements of agent state that are important for analysis that can be written to csv. Position, battery cap., total_dist_travelled, battery_consumed, occ_grid'''
    #csv_headers = ['timestep', 'timestamp', 'rav_name', 'position_intended', 'position_measured', 'total_dist_travelled', 'remaining_batt_cap', 'prop_battery_cap_used', 'sensor_reading', 'occ_grid_locs', 'occ_grid_likelihoods', 'coordinated_with_other_bool', 'coordinated_with_other_names']
    return _get_agent_state_for_analysis(**agent_analysis_state._asdict())
    
assert get_agent_state_for_analysis(testAgentAnalysisState) == "2,100,test,test,test,test,test,test,test,test,test,test,test"

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


def create_belief_map_from_observations(grid: UE4Grid, agent_name: str, agent_belief_map_prior: typing.Dict[UE4Coord, float], agent_observations: typing.List[AgentObservations]):
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
    


    
class BaseAgent:
    '''Base class for all agents, contains minimal functionality. Designed with goal main in mind
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
    ImageDir = 'D:/ReinforcementLearning/Data/SensorData'
    #break apart this into components, one which manages actuation/sensing, one which manages/represents state, etc.
    def __init__(self, grid, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, performance_csv_path: "file path that agent can write performance to", prior = []):
        #grid, agent_name, prior = {}
        #self.curr_belief_map = create_belief_map(grid.get_grid(), agent_name, prior)
        self.rav = multirotor_client
        self.grid = grid
        self.grid_locs = grid.get_grid()
        self.current_pos_measured = self.rav.getMultirotorState().kinematics_estimated.position
        self.timestep = 0
        self.rav_operational_height = height
        self.move_from_bel_map_callable = move_from_bel_map_callable
        self.epsilon = epsilon
        self.agent_name = agent_name
        self.agent_states_for_analysis = []
        self.total_dist_travelled = 0
        self.prop_battery_cap_used = 0
        self.current_battery_cap = 1
        self.coordinated_this_timestep = False
        self.others_coordinated_this_timestep = []
        #manages observations of this agent and other agents
        self.observation_manager = ObservationListManager(self.agent_name)
        
    def __eq__(self, other):
        '''This agent is the same as another agent if names are the same'''
        return self.agent_name == other.agent_name
        
    def get_available_actions(self, state):
        '''Returns actions available to RAV based on its current state'''
        pass
        
    def get_belief_map_after_t_timesteps(self, t):
        '''Calculates what the agent's belief map would be after t timesteps'''
        pass
    
    def charge_battery(self):
        #request from api if battery can be charged. While charging, cap goes up. Charging can be cancelled at any point.
        pass
        
    def move_agent(self, destination_intended: UE4Coord):
        self.rav.moveToPositionAsync(destination_intended.x_val, destination_intended.y_val, self.rav_operational_height, 3).join()
        #update battery capacity
        self.total_dist_travelled += self.current_pos_indended.get_dist_to_other(destination_intended)
        self.prop_battery_cap_used += self.current_battery_cap - self.rav.getRemainingBatteryCap()
        self.current_battery_cap = self.rav.getRemainingBatteryCap()
        
    def get_agent_name(self):
        return self.agent_name
    
    def get_agent(self):
        return self.rav
        
    def get_agent_state_for_analysis(self):
        return AgentAnalysisState(self.timestep, time.time(), self.get_agent_name(), 
                                  self.current_pos_intended, self.current_pos_measured,
                                  self.total_dist_travelled, self.rav.getRemainingBatteryCap(),
                                  self.current_reading, self.grid_locs, 
                                  self.get_grid_locs_likelihoods_lists()[1],
                                  self.coordinated_this_timestep, 
                                  self.others_coordinated_this_timestep)
        
    def init_state_for_analysis_file(self, file_path, csv_header):
        _init_state_for_analysis_file(file_path, csv_header)
            
    def update_state_for_analysis_file(self, file_path, csv_row):
        _update_state_for_analysis_file(file_path, csv_row)
        
        
    #coordination strategy:
    #agent will write all measurements in its possession to a file at each timestep. When communication requested,
    #other agent will read all measurements from the file.
        
    def can_coord_with_other(self, other_rav_name):
        #check if any other ravs in comm radius. if so, return which ravs can be communicated with
        return self.rav.can_coord_with_other(other_rav_name)
    
    def coord_with_other(self, other_rav_name):
        '''coordinate with other rav by requesting their measurement list and sending our own measurement list first write own measurement list to file'''
        if self.can_coord_with_other(other_rav_name):
            observations_from_other = self._read_observations(self)
            self.observation_manager.update_rav_obs_set(observations_from_other)
       
    
    def _write_measurements(self, file_loc):
        '''writes agent measurements to file to be read by other agent'''
        with open(file_loc, 'w') as f:
            json.dump(self.observation_manager.observation_lists, f)
    
    def _read_measurements(self, file_loc):
        with open(file_loc, 'r') as f:
            return json.loads(f)
    
    def record_image(self):
        responses = self.rav.simGetImages([ImageRequest("3", ImageType.Scene)])
        response = responses[0]
        # get numpy array
        filename = OccupancyGridAgent.ImageDir + "/photo_" + str(self.timestep)
        airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8) 
        print("saved image here: {}".format(os.path.normpath(filename + '.png')))

    def update_agent_pos_measured(self):
        self.current_pos_measured = self.rav.getMultirotorState().kinematics_estimated.position
    
    def explore_timestep(self):
        '''Gets rav to explore next timestep'''
        #move rav to next location (nearest to rav)
        #bel_map, current_grid_loc, lat_spacing, lng_spacing, epsilon
        next_pos = self.move_from_bel_map_callable(self.curr_belief_map, self.current_pos_intended, self.grid.lat_spacing, self.grid.lng_spacing, self.epsilon)
        self.move_agent(next_pos)
        
        self.current_pos_intended = next_pos
        self.update_agent_pos_measured()
        #record image at location
        self.record_image()
        #get sensor reading, can be done on separate thread
        print('getting sensor reading for {}'.format(OccupancyGridAgent.ImageDir + "/photo_" + str(self.timestep) + '.png'))
        
        self.current_reading = float(sensor_reading(OccupancyGridAgent.ImageDir + "/photo_" + str(self.timestep) + '.png')[0])
        
        #['grid_loc','probability','timestep', 'timestamp', 'observer_name'])
        self.observation_manager.update_rav_meas_list(self.agent_name, [AgentObservation(self.current_pos_intended, self.current_reading, self.timestep, time.time(), self.agent_name)])
        
        
    def explore_t_timesteps(self, t: int):
        for i in range(t):
            self.explore_timestep()
            self.timestep += 1
        suspected_pos_likelihood = 0
        for position in self.curr_belief_map.keys():
            if self.curr_belief_map[position].likelihood > suspected_pos_likelihood:
                suspected_position = position
                suspected_pos_likelihood = self.curr_belief_map[position].likelihood
                
        return (suspected_position, suspected_pos_likelihood)
                
        
       
def create_rav(port):
    rav = airsim.MultirotorClient(port)
    rav.confirmConnection()
    rav.enableApiControl(True)
    rav.armDisarm(True)
    return rav

def run_t_timesteps(occupancy_grid_agent):
    occupancy_grid_agent.expore_t_timesteps(2)
        
if __name__ == '__main__':
    
    rav1 = create_rav(41451)
    rav2 = create_rav(41452)
    
    #rav1.simShowPawnPath(False, 1200, 20)

    #grid shared between rav
    grid =  UE4Grid(20, 15, UE4Coord(0,0), 120, 150)
    
    #grid, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, performance_csv_path: "file path that agent can write performance to", prior = []
    occupancy_grid_agent1 = OccupancyGridAgent(grid, get_move_from_belief_map, -12, 0.3, rav1, "Drone1", pathlib.Path("D:\ReinforcementLearning\DetectSourceAgent\Analysis"))
    occupancy_grid_agent2 = OccupancyGridAgent(grid, get_move_from_belief_map, -12, 0.3, rav2, "Drone2")
    p1 = multiprocessing.Process(target = run_t_timesteps, args = (occupancy_grid_agent1))
    p2 = multiprocessing.Process(target = run_t_timesteps, args = (occupancy_grid_agent2))
    p1.start()
    p1.join()
    p2.start()
    p2.join()
    
    # showPlannedWaypoints(self, x1, y1, z1, x2, y2, z2, thickness=50, lifetime=10, debug_line_color='red', vehicle_name = '')
    #for grid_coord_index in range(1,len(grid.get_grid())):
    #    rav.showPlannedWaypoints(grid.get_grid()[grid_coord_index-1].x_val, 
    #                             grid.get_grid()[grid_coord_index-1].y_val,
    #                             grid.get_grid()[grid_coord_index-1].z_val,
    #                             grid.get_grid()[grid_coord_index].x_val, 
    #                             grid.get_grid()[grid_coord_index].y_val, 
    #                             grid.get_grid()[grid_coord_index].z_val,
    #                             lifetime = 30)
    
    
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
        
        
        
        
    



