from collections import deque
import os
import gym

from PIL import Image
from typing import List, Tuple, Dict
import openai 

from datetime import datetime
import json

from abc import ABC, abstractmethod

# all actions of overcooked - AI 
n = (0, -1)
s = (0, 1)
e = (1, 0)
w = (-1, 0)
stay = (0, 0)
interact = "interact"

STR2ACT = {
    'north': n,
    'south': s,
    'east': e,
    'west': w,
    'stay': stay,
    'interact': interact
}

ACTSEQUENCE = ['north', 'south', 'east', 'west', 'stay', 'interact']

DEBUG = False

class EmbodiedAgent(ABC):

    def __init__(self, agent_list, save_dir: str) -> None:
        self.m_agentList = agent_list
        self.m_saveDir = save_dir
        self.m_fixedPrompt = f'Please plan an action for all agents in {self.m_agentList}.'
        self.m_totalRequest = 0
        self.m_totalToken = 0
        
    def setFixedPrompt(self, query_sentence: str):
        self.m_fixedPrompt = query_sentence
    
    # Query LLM to get response
    def queryOnce(self, prompt:str, max_query: int = 3):
        response = None
        if DEBUG:
            print('------------------------------query_once begin-----------------------------------------')
        for n in range(max_query):
            print('querying : {} {} th time'.format(self.m_totalRequest, n))
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": prompt+self.m_fixedPrompt},
                    ],
                    max_tokens=512,
                    temperature=0,
                )
                usage = response['usage']["total_tokens"]
                response = response['choices'][0]['message']['content']
                if DEBUG:
                    print("\ncurrent_system_prompt: \n", prompt)
                    print("***************************************************************************")
                    print("\ncurrent_response: ", response)
                    print("***************************************************************************")
                self.m_totalRequest += 1
                self.m_totalToken += usage
                break
            except Exception as e:
                print("API error, try again, ", e)
            continue
        if DEBUG:
            print('------------------------------query_once end-----------------------------------------')

        return response, usage

    # Check if the joint action is valid
    def validJointAction(self, joint_action: str):
        feedback = ''
        agent_dicts = {agent:'' for agent in self.m_agentList}
        actions = [act.strip() for act in joint_action.strip().split('EXECUTE\n')[-1].split('\n')]
        if len(actions) != len(self.m_agentList):
            return f'The number of actions is not equal to the number of agents! Please folloing the [Action Output Instruction]!', None
        response_agents = [action.split(' ')[1].strip() for action in actions] 
        not_in_response_list = [agent for agent in self.m_agentList if agent not in response_agents]
        # 保证每个agent都有action
        if len(not_in_response_list) > 0:
            return f'Not all agents in {not_in_response_list} is given an action! Please folloing the [Action Output Instruction]!', None
        action_list = []
        for i, action in enumerate(actions):
            # flag = False
            for agent_name in agent_dicts.keys():
                if agent_name in action:
                    # flag = True
                    raw_action = action.split(' ')[-1].strip()
                    if raw_action in STR2ACT.keys():
                        true_action = STR2ACT[raw_action]
                        agent_dicts[agent_name] = raw_action
                        action_list.append(true_action)
                        break
                    else:
                        feedback += f'The action of {agent_name} is {raw_action}, which is not a valid action! Please folloing the [Action Output Instruction]!'
                        return feedback, None
                    
        if any(v == '' for v in agent_dicts.values()):    
            return f'Not all agents in {self.m_agentList} is given an action! Please folloing the [Action Output Instruction]!', None
        
        all_act_list = list(STR2ACT.values())
        return_index = [all_act_list.index(act) for act in action_list ]
        
        return None, return_index
    
    @abstractmethod
    def getAction(self, task_prompt, env_prompt, env_state_prompt, agent_state_prompt: str = None):
        raise NotImplementedError
    
    @abstractmethod
    def getPrompt(self, task_prompt, env_prompt, env_state_prompt, agent_state_prompt: str = None):
        raise NotImplementedError
    
    
class MindAgent(EmbodiedAgent):
    
    def __init__(self, agent_list, save_dir , mem_size: int = 100):
        super(MindAgent, self).__init__(agent_list=agent_list, save_dir=save_dir)
        self.m_memSize = mem_size
        # use deque as memory to avoid memory too long
        self.m_Memory = deque(maxlen=mem_size)
        self.m_name = "MindAgent"
        self.m_agentType = "mind"
        self.m_offset = 0
    
    # update memory
    def updateMemory(self, env_pos: str, agent_state: str, response: str, feedback: str):
        
        # size limitation
        if len(self.m_Memory) == self.m_memSize:
            self.m_Memory.popleft()
            self.m_offset += 1
        self.m_Memory.append((env_pos, agent_state, response, feedback))
        
    def getMemory(self):
        ret = ''
        if len(self.m_Memory) != 0:
            ret = "[History]\n"
        for i, (env_state, agent_state, response, feedback) in enumerate(self.m_Memory):
            ret += f"== Round#{i+1+self.m_offset} ==\n"
            ret += f"[Env state]: \n{env_state if env_state else 'None'}\n"
            # The environment status already contains the status information of the Agent. In order to reduce 
            # information redundancy, the field [Agent State] is omitted in the implementation here.
            ret += f"{agent_state if len(agent_state) else 'None'}\n"
            ret += f"[Response]: \n{response if len(response) else 'None'}\n"
            ret += f"[Feedback]: \n{feedback if len(feedback) else 'None'}\n"
        return ret
    
    # compose prompt 
    def getPrompt(self, task_prompt, env_prompt, env_state_prompt, agent_state_prompt: str = ''):
        ret = task_prompt.strip() + '\n'+env_prompt + '\n' + self.getMemory()+ 'At the current round:\n[Env State]\n' +env_state_prompt+'\n'
        if len(agent_state_prompt):
            ret += agent_state_prompt+'\n'
        
        self.temp_info = {
            "env_state": env_state_prompt, 
            "agent_state": agent_state_prompt
        }
        return ret
        
    def updateHistory(self, agent_state: str, response: str, feedback: str = '', env_state: str = None):
        self.updateMemory(env_state, agent_state, response, feedback)
    
    # use current state prompt to compose query prompt and then query llm
    def getAction(self, task_prompt, env_prompt, env_state_prompt, agent_state_prompt: str = ''):
        prompt = self.getPrompt(task_prompt, env_prompt, env_state_prompt, agent_state_prompt)
        response, _ = self.queryOnce(prompt)
        feedback, joint_action = self.validJointAction(response)
        return  response, feedback, joint_action
      
    
if __name__ == '__main__':
    agent = MindAgent(agent_list=['Agent1', 'Agent0'], save_dir='save_dir')
    feedback, joint_action = agent.validJointAction('xxxxxxx \n EXECUTE\nNAME Agent0 ACTION stay\nNAME Agent1 ACTION north')
    print(feedback)
    print(joint_action)