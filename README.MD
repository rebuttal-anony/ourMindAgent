#  Introduction

![mindAgent](.\\source\\mindagent.png)



Our implementation of MindAgent is mainly divided into two parts：
1. Realized the mindAgent class to manage the most important memory components, and realized the valid function and the assembly of prompt, the main code is in mindagent. py file.
2. Overcooked-AI environment is responsible for generating textualized environmental states, the main code is in the overcooked_env.py file.



# Details of our implementation

Below we will describe the relevant functions of the code block in terms of functions

1. **mindAgent.py**：

   - **EmbodiedAgent**.queryOnce(prompt:str, max_query: int *=* 3)：Call the API interface with the given prompt and get the planning results.
   - **EmbodiedAgent**.validJointAction(self, joint_action: str): Verify the planning result. If the planning result is invalid, return an feedback. Otherwise returns an executable action.
   - **MindAgent**.updateMemory(env_pos: str, agent_state: str, response: str, feedback: str): Store the current round of environmental information, planning result, feedback information  in memory.
   - **MindAgent**.getMemory()：Returns all history information in memory.
   - **MindAgent**.getPrompt(task_prompt, env_prompt, env_state_prompt, agent_state_prompt: str *=* ''): Construct prompt with given environment state information, task description information, environment description information, and historical interaction information.
   - **MindAgent**.getAction(task_prompt, env_prompt, env_state_prompt, agent_state_prompt: str *=* ''): Call the getPrompt, queryOnce, and validJointAction functions successively to get the planning result.

2. **overcooked_env.py**：

   - **OvercookedEnv**.get_action_prompt(agent_type:str): Returns a list of environment optional actions.

   - **OvercookedEnv**.get_action_example(agent_type:str): Returns an example of planning consistent with the agent type feature.

   - **OvercookedEnv**.get_env_prompt(): Return the task description and environment description.

   - **OvercookedEnv**.get_env_state_prompt(): Returns the current environment status.

   - **OvercookedEnv**.get_agent_state_prompt(): Returns the current agent status.

   - **OvercookedEnv**.get_cooked_one(): Check that the task has been completed within the specified time.

     

     
