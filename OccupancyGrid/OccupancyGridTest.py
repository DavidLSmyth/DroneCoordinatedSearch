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
import client as airsim
from DetectSourceAgent.types import *
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
            
    def get_grid(self):
        return self.grid
    
class UE4Grid:
    def __init__(self, lat_spacing, lng_spacing, origin, x_lim=None, y_lim=None, no_x=None, no_y=None):
        if not all([x_lim, y_lim]) and not all([no_x, no_y]):
            raise Exception('Either give a limit to the grid or an x and y spacing')
        self.grid = UE4GridFactory(lat_spacing, lng_spacing, origin, x_lim, y_lim, no_x, no_y).get_grid()
        self.lat_spacing = lat_spacing
        self.lng_spacing = lng_spacing
        self.no_x = no_x if no_x else int(x_lim/lng_spacing)
        self.no_y = no_y if no_y else int(y_lim/lat_spacing)
    
    def get_grid(self):
        return self.grid
    
    def get_lat_spacing(self):
        return self.lat_spacing
    
    def get_lng_spacing(self):
        return self.lng_spacing
    
    def get_no_points_x(self):
        return self.no_x
    
    def get_no_points_y(self):
        return self.no_y
        
grid = UE4Grid(20, 15, UE4Coord(0,0), 100, 60).get_grid()
    

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
AgentPercept = namedtuple('obs_location', ['grid_loc','probability','timestep', 'timestamp', 'observer_name'])

#A belief map component consists of a grid location and a likelihood
BeliefMapComponent = namedtuple('belief_map_component', ['grid_loc','likelihood'])

#A belief map consists of belief map components
#Leave this as namedtuple if don't need to define methods
class BeliefMap(typing.NamedTuple):
    
    grid: UE4Grid
    BeliefMapComponents: typing.List[BeliefMapComponent]
    
    def get_grid():
        return grid
    
    def get_belief_map_components():
        return BeliefMapComponents
    
    
BeliefMap(grid, []) 

#test_obs = Obs(0.1, 2.3)
def create_bel_map(grid_locs, observer_name, prior = []):
    '''Creates an occupancy belief map for a given observer and a set of grid locations'''
    if not prior:
        #use uniform uninformative prior
        prior = [1/len(grid_locs) for i in grid_locs]
    return    
    #return {grid_locs[i]: ObsLocation(grid_locs[i],prior[i], 0, time.time(), observer_name) for i in range(len(grid_locs))}

test_map = create_bel_map(UE4GridFactory(10, 12, UE4Coord(0,0), 50, 60).get_grid())
#%%
def update_bel_map(bel_map, obs: 'sernsor reading of prob. found target', grid_loc, timestep):
    '''Updates an agents belief map with new observations'''
    print('updating with prob: {}'.format(obs))
    print('previous likelihood: {}'.format(bel_map[grid_loc].likelihood))
    print(bel_map[grid_loc].likelihood)
    new_prob = (bel_map[grid_loc].likelihood*obs)/((obs*bel_map[grid_loc].likelihood) + ((1-obs)*(1-bel_map[grid_loc].likelihood)))
    new_bel_map = copy.deepcopy(bel_map)
    print('updating probability of target being in location {} to {}'.format(grid_loc, new_prob))
    new_bel_map[grid_loc] = ObsLocation(grid_loc, new_prob, timestep, time.time())
    return new_bel_map

def get_dist(grid_loc1, grid_loc2):
    '''assuming that grid_loc1, grid_loc2 have an x and y component'''
    return (((grid_loc1.x_val - grid_loc2.x_val )**2) + ((grid_loc1.y_val - grid_loc2.y_val)**2))**0.5

assert abs(get_dist(UE4Coord(0,1), UE4Coord(1,4)) - 10**0.5) <= 0.001
                
                
def get_move_from_bel_map(bel_map, current_grid_loc, lat_spacing, lng_spacing, epsilon):
    '''take and epsilon greedy approach'''
    #assume grid is regular, get all neighbors that are within max(lat_spacing, long_spacing)7
    #assuming that lat_spacing < 2* lng_spacing and visa versa
    eff_radius = max(lat_spacing, lng_spacing)
    neighbors = list(filter(lambda grid_loc: get_dist(grid_loc, current_grid_loc) <= eff_radius and grid_loc!=current_grid_loc, bel_map.keys()))
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
        
calc_posterior = lambda observation, prior: (prior * observation) / ((prior * observation) + (1-prior)*(1-observation))

assert abs(calc_posterior(0.5, 0.2) - 0.2) <= 0.001
assert abs(calc_posterior(0.8, 0.2) - 0.5) <= 0.001

def get_posterior_given_obs(observations:list, prior):
    for observation in observations:
        prior = calc_posterior(observation, prior)
    return prior
        
assert abs(get_posterior_given_obs([0.5,0.2,0.8], 0.5) - 0.5) <= 0.001

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

def merge_occ_grids(agent1_observation_grid, agent1_observations, agent2_observations):
    '''Merges occ_grid2 observations with occ_grid1 observations. If occ_grid2 observations more recent than occ_grid1s observations, update occ_grid1s observations.
    More recent observations from other agents are treated as new observations for current agent'''
    #Obs = namedtuple('obs', ['likelihood','timestep', 'timestamp'])
    for observation_loc in obs_grid1:
        if obs_grid2[observation_loc].timestamp > obs_grid1[observation_loc].timestamp:
            print('updating obs_grid1 likelihood at location {}'.format(observation_loc))
            obs_grid1[observation_loc] = ObsLocation(observation_loc, obs_grid2[observation_loc].likelihood, obs_grid2[observation_loc].timestep, obs_grid2[observation_loc].timestamp)
            
    return obs_grid1


obs_grid1 = create_bel_map(grid)
obs_grid2 = create_bel_map(grid)
obs_grid1
obs_grid2
obs_grid2[grid[0]] = ObsLocation(grid[0],0.5,0, time.time())
obs_grid2[grid[5]] = ObsLocation(grid[5],0.6,0, time.time())

print(obs_grid1[grid[0]])
print(obs_grid1[grid[5]])

print('\n\n')
print(obs_grid2[grid[0]])
print(obs_grid2[grid[5]])

obs_grid1 = merge_occ_grids(obs_grid1, obs_grid2)

print('\n\n',obs_grid1[grid[0]])
print(obs_grid1[grid[5]])


assert obs_grid1 == obs_grid2

get_move_from_bel_map(test_map, UE4Coord(0,0), 10, 12, 0.2)
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
        self.curr_belief_map = create_bel_map(grid.get_grid(), prior)
        self.belief_maps = [self.curr_belief_map]
        self.rav = multirotor_client
        self.grid = grid
        self.grid_locs = grid.get_grid()
        self.current_pos_measured = self.rav.getMultirotorState().kinematics_estimated.position
        #starts uninitialized
        #self.current_pos_intended = sel
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
        
        #add a list of measurements that RAV has taken
        #measurement is a grid location, probability, timestep, timestamp
        self.measurement_list = []
        
    def __eq__(self, other):
        '''This agent is the same as another agent if names are the same'''
        return self.agent_name == other.agent_name
        
    def get_available_actions(self, state):
        '''Returns actions available to RAV based on its current state'''
        pass
        
    def charge_battery(self):
        #request from api if battery can be charged. While charging, cap goes up. Charging can be cancelled at any point.
        pass
        
    def move_agent(self, destination_intended: UE4Coord):
        self.rav.moveToPositionAsync(destination_intended.x_val, destination_intended.y_val, self.rav_operational_height, 3).join()
        #update battery capacity
        self.total_dist_travelled += get_dist(self.current_pos_indended, destination_intended)
        self.prop_battery_cap_used += self.current_battery_cap - self.rav.getRemainingBatteryCap()
        self.current_battery_cap = self.rav.getRemainingBatteryCap()
        
    def get_agent_name(self):
        return self.agent_name
    
    def get_agent(self):
        return self.rav
    
    
    def get_grid_locs_likelihoods_map(self):
        return {loc: obs.likelihood for loc, obs in self.grid.items()}
    
    def get_grid_locs_likelihoods_lists(self):
        keys = self.get_grid_locs_likelihoods_map.keys()
        return ([keys],[self.get_grid_locs_likelihoods_map[key] for key in keys])
    
    #'timestep','timestamp','rav_name',
    #                                             'position_intended','position_measured',
    #                                             'total_dist_travelled','remaining_batt_cap',
    #                                             'sensor_reading','occ_grid_locs',
    #                                             'occ_grid_likelihoods','coordinated_with_other_bool',
    #                                             'coordinated_with_other_names'
    
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
        
    def can_coord_with_other(self, other_rav_name):
        #check if any other ravs in comm radius. if so, return which ravs can be communicated with
        return self.rav.can_coord_with_other(other_rav_name)
    
    def coord_with_other(self, other_rav_name):
        #coordinate with other rav by requesting their occupancy grid and sending own occupancy grid
       if self.can_coord_with_other(other_rav_name):
           pass
    
    def merge_occ_grid(self, other_agent_occ_grid):
        '''Assuming another agent also has occupancy grid, fuse theirs with mine'''
        pass
    
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

        #Obs = namedtuple('obs', ['likelihood','timestep', 'timestamp'])
        #ObsLocation = namedtuple('obslocation', ['grid_loc','likelihood','timestep', 'timestamp'])
        #['likelihood','timestep', 'timestamp'])
        measurement_list.append(ObsLocation(self.current_pos_intended, self.current_reading, self.timestep, time.time())
        
        #update belief map based on sensor reading
        self.curr_belief_map = update_bel_map(self.curr_belief_map, self.current_reading, self.current_pos_intended, self.timestep)
        self.belief_maps.append(self.curr_belief_map)
        with open("D:/ReinforcementLearning/Data/BeliefMapData/beliefMap" + str(self.timestep) + ".json", 'w') as f:
            f.write(str(self.curr_belief_map))
        #
    
        
    def explore_t_timesteps(self, t: int):
        for i in range(t):
            self.explore_timestep()
            self.timestep += 1
        suspected_position = self.suspected_pos = grid.get_grid()[0]
        suspected_pos_likelihood = 0
        for position in self.curr_belief_map.keys():
            if self.curr_belief_map[position].likelihood > suspected_pos_likelihood:
                suspected_position = position
                suspected_pos_likelihood = self.curr_belief_map[position].likelihood
                
        return (suspected_position, suspected_pos_likelihood)
                
        
        
        
if __name__ == '__main__':
    rav = airsim.MultirotorClient()
    rav.confirmConnection()
    rav.enableApiControl(True)
    rav.armDisarm(True)
    rav.simShowPawnPath(False, 1200, 20)
    rav.takeoffAsync().join()
    print(rav.getMultirotorState().position)
    rav.moveToPositionAsync(100, 100, -10, 9).join()
    print(rav.getMultirotorState().position)
    print(rav.battery.get_battery_cap())
    rav.moveToPositionAsync(0, 0, -10, 9).join()
    print(rav.getMultirotorState().position)
    print(rav.battery.get_battery_cap())
    rav.moveToPositionAsync(100, 100, -10, 9).join()
    print(rav.battery.get_battery_cap())
    rav.moveToPositionAsync(0, 0, -10, 9).join()
    print(rav.battery.get_battery_cap())
    print(rav.getMultirotorState().position)
    rav.moveToPositionAsync(100, 100, -10, 9).join()
    print(rav.battery.get_battery_cap())
    rav.moveToPositionAsync(0, 0, -10, 9).join()
    print(rav.battery.get_battery_cap())
    
    print('landing')
    #rav.landAsync().join()
    rav.armDisarm(False)
    rav.enableApiControl(False)
    #grid =  UE4Grid(20, 15, UE4Coord(0,0), 120, 150)
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
        
        
        
        
    



