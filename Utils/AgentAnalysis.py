# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:00:30 2018

@author: 13383861
"""

from collections import namedtuple

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
    print(agent_analysis_state)
    return _get_agent_state_for_analysis(**agent_analysis_state._asdict())

def _get_agent_state_for_analysis(timestep, timestamp, rav_name, position_intended, position_measured, total_dist_travelled, remaining_batt_cap, prop_battery_cap_used, sensor_reading, coordinated_with_other_names):
    return f"{timestep},{timestamp},{rav_name},{position_intended},{position_measured},{total_dist_travelled},{remaining_batt_cap},{prop_battery_cap_used},{sensor_reading},{coordinated_with_other_names}"
 
def _init_state_for_analysis_file(file_path):
    with open(file_path, 'w+') as f:
        f.write(','.join(AgentAnalysisState._fields) + '\n')

def _update_state_for_analysis_file(file_path, agent_analysis_state: AgentAnalysisState):
    with open(file_path, 'a') as f:
        f.write(get_agent_state_for_analysis(agent_analysis_state) + '\n')
        
        
def _init_agent_metadata_file(file_path):
    with open(file_path, 'w+') as f:
        f.write(','.join(AgentAnalysisMetadata._fields) + '\n')

def _write_to_agent_metadata_file(file_path, meta_data: AgentAnalysisMetadata):
    with open(file_path, 'a') as f:
        f.write(''.join()  + '\n')
        
if __name__ == "__main__":
    testAgentAnalysisState = AgentAnalysisState(2, 100, 'test', 'test', 'test', 'test', 'test', 'test', 'test', 'test') 
    testAgentAnalysisState._asdict()
    assert get_agent_state_for_analysis(testAgentAnalysisState) == "2,100,test,test,test,test,test,test,test,test"