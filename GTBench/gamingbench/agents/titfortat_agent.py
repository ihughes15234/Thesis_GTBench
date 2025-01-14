
import numpy as np

from gamingbench.agents.base_agent import BaseAgent
from gamingbench.agents.base_agent import BaseAgent
from gamingbench.prompts.observation_prompts import construct_observation_prompt
from gamingbench.prompts.system_prompts import construct_system_prompt
from gamingbench.prompts.step_prompts.prompt_agent import construct_step_prompt
import sqlite3 #custom add 


class TitForTatAgent(BaseAgent):

    def __init__(self, config, **kwargs):
        super(TitForTatAgent, self).__init__(config)

    def step(self, observations):
        assert observations, print('Tit-for-Tat Agent only works for Iterated Prisoner\'s Dilemma')
        
        opponent_moves = observations['opponent_moves']
        
        if len(opponent_moves) > 0:
            move = '<Silent>' if opponent_moves[-1] == 'C' else '<Testify>'
        else:
            move = '<Silent>'
        
        agent_action_list = observations['legal_moves']
        openspiel_action_list = observations['openspiel_legal_actions']
        
        #state = observations['state']
        #action = self.bot.step(state)
        
        env_name = observations['env_name']
        system_prompt = construct_system_prompt(env_name)
        print(env_name)



        observation_prompt = construct_observation_prompt(
            observations, env_name)
        

        step_prompt_constructor = construct_step_prompt
        step_instruct = step_prompt_constructor(observations)
        step_prompt = step_instruct['prompt']
        observation_prompt = observation_prompt + '\n' + step_prompt
        

        msgs = self.construct_init_messages(
            system_prompt, observation_prompt)
        
        #Chosen action, custom for training dataset
        mcts_act = "Action:\n" + move

        con = sqlite3.connect("/home/GTBench_Thesis/experiments/DPO_train.db")
        cur = con.cursor()
        data_insert = [(str(system_prompt), str(observation_prompt), str(mcts_act),str(agent_action_list), str(env_name))]
        cur.executemany("insert into DPO_ttt VALUES(?, ?, ?, ?, ?)", data_insert)        
        con.commit()
        
        
        
        
        return move, []