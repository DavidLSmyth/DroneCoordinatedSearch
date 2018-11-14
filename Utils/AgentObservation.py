# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:30:11 2018

@author: 13383861
"""

from collections import namedtuple

from Utils.UE4Coord import UE4Coord
from Utils.UE4Grid import UE4Grid


#an agent precept consists of a grid location, a detection probability, a timestep, a timestamp and the observer name
AgentObservation = namedtuple('AgentObservation', ['grid_loc','probability','timestep', 'timestamp', 'observer_name'])

#test = {'grid_loc': 1,'probability': 2,'timestep': 2, 'timestamp': 4, 'observer_name': 8}
#test1 = [1,2,4,8, 9]
#AgentObservation(**test)
#AgentObservation(*test1)

#Code and test for class which manages agent observations in a set grid
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
    
    
def _get_agent_observation_csv(grid_loc,probability,timestep,timestamp,observer_name):
    return f"{grid_loc}, {probability}, {timestep}, {timestamp}, {observer_name}"
     
def get_agent_observation_csv(agent_observation: AgentObservation):
    '''Returns elements of agent state that are important for analysis that can be written to csv. Position, battery cap., total_dist_travelled, battery_consumed, occ_grid'''
    #csv_headers = ['timestep', 'timestamp', 'rav_name', 'position_intended', 'position_measured', 'total_dist_travelled', 'remaining_batt_cap', 'prop_battery_cap_used', 'sensor_reading', 'occ_grid_locs', 'occ_grid_likelihoods', 'coordinated_with_other_bool', 'coordinated_with_other_names']
    #return str(agent_analysis_state._fields).replace(')','').replace('(','').replace("'", '')
    return _get_agent_observation_csv(**agent_observation._asdict())

#AgentObservation = namedtuple('obs_location', ['grid_loc','probability','timestep', 'timestamp', 'observer_name'])
def _init_observations_file(file_path):
    with open(file_path, 'w+') as f:
        f.write(','.join(AgentObservation._fields))
        
def _write_to_obserations_file(file_path, agent_observation):
    with open(file_path, 'a') as f:
        f.write('\n' + get_agent_observation_csv(agent_observation))
        
    
if __name__ == "__main__":
    test_grid = UE4Grid(1, 1, UE4Coord(0,0), 10, 6)
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