# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:45:44 2018

@author: 13383861
"""
import typing
import functools
from Utils.AgentObservation import AgentObservation
from Utils.UE4Coord import UE4Coord
from Utils.UE4Grid import UE4Grid
from Utils.BeliefMap import create_belief_map

class ObservationSetManager:
    
    '''
    Manages the sensor measurements of other agents. Observations don't have to be taken at disrete locations - 
    the continuous position can be recorded and the grid location inferred from this.
    Calculating a belief map from these sets of observations requires a grid so that each recorded observation can
    be 
    '''
    
    def __init__(self, agent_name: 'name of the agent that owns this observation list manager'):

        #key value pairs of form agent_name: set of AgentObservations
        self.observation_sets = dict()
        self.agent_name = agent_name
        #agent should initialize its own observations
        self.init_rav_observation_set(self.agent_name)    
    
    #really strange behaviour: using this initialises the class with observations that don't exist... self.observation_sets[rav_name] = set()
    def init_rav_observation_set(self, rav_name, observations = None):
        '''initialise a new list of observations for a RAV'''
        if not observations:
            self.observation_sets[rav_name] = set()
        else:
            self.observation_sets[rav_name] = observations

    def get_all_observations(self):
        return functools.reduce(lambda x,y: x.union(y) if x else y, self.observation_sets.values(), set())
        
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
            
    def get_discrete_belief_map_from_observations(self, grid):
        '''Given a descrete grid, returns a belief map containing the likelihood of the source
        being contained in each grid segment'''
        #ToDo:
        #Currently observations must be made at grid locations - instead compute which observations are made 
        #in each grid location and then compute the belief map
        return_belief_map = create_belief_map(grid, self.agent_name)
        return_belief_map.update_from_observations(self.get_all_observations())
        return return_belief_map
    
    def get_continuous_belief_map_from_observations(self, grid_bounds):
        '''Given grid bounds, returns a function which returns the likelihood given the 
        continuous position of the RAV. I.E. transform the discrete PDF as above to a 
        continuous one.'''
        pass


if __name__ == "__main__":
    test_grid = UE4Grid(1, 1, UE4Coord(0,0), 6, 5)
    
    test_ObservationSetManager = ObservationSetManager('agent1')
    test_ObservationSetManager.observation_sets
    
    obs1 = AgentObservation(UE4Coord(0,0),0.5, 1, 1234, 'agent2')
    obs2 = AgentObservation(UE4Coord(0,0),0.7, 2, 1235, 'agent2')
    obs3 = AgentObservation(UE4Coord(0,1),0.95, 3, 1237, 'agent2')
    obs4 = AgentObservation(UE4Coord(0,1),0.9, 3, 1238, 'agent1')
    
    test_ObservationSetManager.init_rav_observation_set('agent2', set([obs1, obs2]))
    test_ObservationSetManager.observation_sets
    test_ObservationSetManager.update_rav_obs_set('agent2', set([obs3]))
    
    test_ObservationSetManager.get_all_observations()
    
    assert test_ObservationSetManager.get_observation_set('agent2') == set([obs1, obs2, obs3]), "agent2 observations should have been added to set"
    assert test_ObservationSetManager.get_observation_set('agent1') == set([]), "agent1 observations should be empty"
       
    test_ObservationSetManager.update_rav_obs_set('agent1', set([obs4]))
    
    assert not test_ObservationSetManager.get_all_observations().difference(set([obs1, obs2, obs3, obs4]))

    
    ###################################################
    # Check that duplicate observations aren't added
    test_grid = UE4Grid(1, 1, UE4Coord(0,0), 6, 5)
    test1_ObservationSetManager = ObservationSetManager('agent1')
    
    obs1 = AgentObservation(UE4Coord(0,0),0.5, 1, 1234, 'agent2')
    obs2 = AgentObservation(UE4Coord(0,0),0.7, 2, 1235, 'agent2')
    obs3 = AgentObservation(UE4Coord(0,1),0.95, 3, 1237, 'agent2')
    
    test1_ObservationSetManager.update_rav_obs_set('agent2',[obs1, obs2, obs3])
    test1_ObservationSetManager.observation_sets
    #test that duplicate measurements won't occur
    obs4 = AgentObservation(UE4Coord(0,1),0.95, 3, 1237, 'agent2')
    test1_ObservationSetManager.update_rav_obs_set('agent2', set([obs4]))
    
    assert test1_ObservationSetManager.get_observation_set('agent2') == set([obs1, obs2, obs3]), "Duplicated observations should be ignored"
    
    assert abs(test1_ObservationSetManager.get_discrete_belief_map_from_observations(test_grid).get_belief_map_component(UE4Coord(0,0)).likelihood - 0.074468) < 0.0001
    assert abs(test1_ObservationSetManager.get_discrete_belief_map_from_observations(test_grid).get_belief_map_component(UE4Coord(0,1)).likelihood - 0.395833) < 0.0001
    
    
    
    