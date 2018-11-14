# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:40:27 2018

@author: 13383861
"""

import sys
sys.path.append('..')
sys.path.append('.')
from collections import namedtuple
import typing
from Utils.UE4Grid import UE4Grid
from Utils.UE4Coord import UE4Coord
from Utils.AgentObservation import AgentObservation

#calculation of posterior distribution is bayes likelihood updated formula
calc_posterior = lambda observation, prior: (prior * observation) / ((prior * observation) + (1-prior)*(1-observation))

#A belief map component consists of a grid location and a likelihood
BeliefMapComponent = namedtuple('belief_map_component', ['grid_loc','likelihood'])

def get_posterior_given_obs(observations:list, prior):
    '''For a sequence of observations calculates the posterior probability given a prior.'''
    for observation in observations:
        prior = calc_posterior(observation, prior)
    return prior
        


#######################  Belief map and tests  #######################

#A belief map has an agent name (beliefs belong to an agent) consists of belief map components
#Leave this as namedtuple if don't need to define methods
class BeliefMap:
    
    def __init__(self, agent_name: str, grid: UE4Grid, belief_map_components: typing.List[BeliefMapComponent], prior: typing.Dict[UE4Coord, float]):
        self.agent_name = agent_name
        self.grid = grid
        self.belief_map_components = belief_map_components
        self.prior = prior
    
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
        
    def update_from_observations(self, agent_observations: typing.Set[AgentObservation]):
        for observation in agent_observations:
            self.update_from_observation(observation)
    
    def _get_observation_grid_index(self, grid_loc: UE4Coord):
        return self.belief_map_components.index(self.get_belief_map_component(grid_loc))
    
    def get_belief_map_component(self, grid_loc):
        if grid_loc in map(lambda belief_map_component: belief_map_component.grid_loc ,self.belief_map_components):
            return next(filter(lambda belief_map_component: belief_map_component.grid_loc == grid_loc, self.belief_map_components))
        else:
            raise Exception("{} is not in the belief map".format(grid_loc))
    
    def __eq__(self, other):
        #check if grids are the same an componenents are the same
        pass
    
    def __str__(self):
        return str({"grid": self.grid,"prior": self.prior,"agent_name": self.agent_name,"components": self.belief_map_components})
 
#maybe this should go in contsructor and make regular class
def create_belief_map(grid, agent_name, prior = {}):
    '''Creates an occupancy belief map for a given observer and a set of grid locations.
    Prior is a mapping of grid_points to probabilities'''
    if not prior:
        #use uniform uninformative prior
        prior = {grid_point: 1/len(grid.get_grid_points()) for grid_point in grid.get_grid_points()}
    return BeliefMap(agent_name, grid, [BeliefMapComponent(grid_point, prior[grid_point]) for grid_point in grid.get_grid_points()], prior)
    #return {grid_locs[i]: ObsLocation(grid_locs[i],prior[i], 0, time.time(), observer_name) for i in range(len(grid_locs))}

 
def create_belief_map_from_observations(grid: UE4Grid, agent_name: str, agent_belief_map_prior: typing.Dict[UE4Coord, float], agent_observations: typing.Set[AgentObservation]):
    '''Since the calculation of posterior likelihood is based only on prior and observations (independent of order), updating a belief map component from measurements can be done 
    by the following update formula: 
                                prior * product(over all i observations) observation_i
        ----------------------------------------------------------------------------------------------------------------------
        prior * product(over all i observations) observation_i + (1-prior) * product(over all i observations) (1-observation_i)
    '''

    return_bel_map = create_belief_map(grid, agent_name, agent_belief_map_prior)
    #update belief map based on all observations...
    return_bel_map.update_from_observations(agent_observations)
    return return_bel_map

if __name__ == "__main__":
    
    #tests for calc_posterior
    assert abs(calc_posterior(0.5, 0.2) - 0.2) <= 0.001
    assert abs(calc_posterior(0.8, 0.2) - 0.5) <= 0.001
    
    #tests for get_posterior_given_obs
    assert abs(get_posterior_given_obs([0.5,0.2,0.8], 0.5) - 0.5) <= 0.001
    
    #tests for belief map
    test_grid = UE4Grid(1, 1, UE4Coord(0,0), 10, 6)
    test_map = create_belief_map(test_grid, "agent1")
    assert test_map.get_belief_map_component(UE4Coord(0,0)) == BeliefMapComponent(UE4Coord(0,0), 1/len(test_grid.get_grid_points()))
    assert test_map._get_observation_grid_index(UE4Coord(0,0)) == 5
    
    test_map.update_from_prob(UE4Coord(0,0), 0.9)
    assert 0.132<test_map.get_belief_map_component(UE4Coord(0,0)).likelihood < 0.133
    
    #prove order in which observations come in doesn't matter
    obs1 = AgentObservation(UE4Coord(0,0),0.4, 1, 1234, 'agent1')
    obs2 = AgentObservation(UE4Coord(0,0),0.7, 2, 1235, 'agent1')
    obs3 = AgentObservation(UE4Coord(0,0),0.93, 3, 1237, 'agent1')
    test_map = create_belief_map(test_grid, "agent1")
    test_map.update_from_observation(obs1)
    assert 0.0111 < test_map.get_belief_map_component(UE4Coord(0,0)).likelihood < 0.0112 
    test_map.update_from_observation(obs2)
    assert 0.025688 < test_map.get_belief_map_component(UE4Coord(0,0)).likelihood < 0.0256881 
    test_map.update_from_observation(obs3)
    assert 0.2594 < test_map.get_belief_map_component(UE4Coord(0,0)).likelihood < 0.2595
    
    #now check observing in a different order gives same result
    test_map = create_belief_map(test_grid, "agent1")
    test_map.update_from_observation(obs2)
    test_map.update_from_observation(obs1)
    test_map.update_from_observation(obs3)
    assert 0.2594 < test_map.get_belief_map_component(UE4Coord(0,0)).likelihood < 0.2595 
    
    #now check observing in a different order gives same result
    test_map = create_belief_map(test_grid, "agent1")
    test_map.update_from_observation(obs3)
    test_map.update_from_observation(obs2)
    test_map.update_from_observation(obs1)
    assert 0.2594 < test_map.get_belief_map_component(UE4Coord(0,0)).likelihood < 0.2595



    test_map = create_belief_map_from_observations(test_grid, "agent1", {grid_point: 1/len(test_grid.get_grid_points()) for grid_point in test_grid.get_grid_points()}, set([obs1, obs2, obs3]))
    assert 0.2594 < test_map.get_belief_map_component(UE4Coord(0,0)).likelihood < 0.2595





