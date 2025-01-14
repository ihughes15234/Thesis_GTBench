import pyspiel
import numpy as np
import sqlite3 #custom add 

from open_spiel.python.algorithms import mcts
from gamingbench.agents.base_agent import BaseAgent
from gamingbench.prompts.observation_prompts import construct_observation_prompt
from gamingbench.prompts.system_prompts import construct_system_prompt
from gamingbench.prompts.step_prompts.prompt_agent import construct_step_prompt


class MCTSAgent(BaseAgent):

    def __init__(self, config, **kwargs):
        super(MCTSAgent, self).__init__(config)
        self.rollout_count = config.rollout_count
        self.uct_c = config.uct_c
        self.max_simulations = config.max_simulations
        self.solve = config.solve
        self.verbose = config.verbose
        rng = np.random.RandomState()
        evaluator = mcts.RandomRolloutEvaluator(self.rollout_count, rng)
        self.bot = mcts.MCTSBot(
            kwargs['game'],
            self.uct_c,
            self.max_simulations,
            evaluator,
            random_state=rng,
            solve=self.solve,
            verbose=self.verbose)

    def step(self, observations):        
        agent_action_list = observations['legal_moves']
        openspiel_action_list = observations['openspiel_legal_actions']
        state = observations['state']
        action = self.bot.step(state)
        
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
        mcts_act = "Action:\n" + agent_action_list[openspiel_action_list.index(action)]

        con = sqlite3.connect("") #add filepath of database, SQLLite
        cur = con.cursor()
        data_insert = [(str(system_prompt), str(observation_prompt), str(mcts_act),str(agent_action_list), str(env_name))]
        cur.executemany("insert into  VALUES(?, ?, ?, ?, ?)", data_insert)        
        con.commit()
        
        
        
        return agent_action_list[openspiel_action_list.index(action)], []

    def inform_action(self, state, player_idx, action):
        self.bot.inform_action(state, player_idx, action)

