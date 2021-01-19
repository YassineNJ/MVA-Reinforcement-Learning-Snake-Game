import torch
# import gym
import sys, os
sys.path.append('../')
import numpy as np
from Envs.environment_snake import *
import matplotlib.pyplot as plt
from Agents.dqn_agent import *
from Agents.ac_agent import *
import os
os.environ['SDL_VIDEODRIVER']='dummy'


pygame.init()

def dqn_experiment(agent_n='DQN',img=False,train_steps=10000,eval_every=1000,double=False,duelling = True,
               weighted_sampling =1,batch_size= 256,use_conv=False,learning_rate=1e-4):
    
    env = SnakeGame(w=80, h=80 , max_reward = 1,food_nb = 1 , early_stopping_factor = 10 ,gray_scale = False,
                    isdisplayed = False, use_images = img   , image_reduction = 2 )
    
    if agent_n== 'DQN':
        
        agent = DQNAgent(env, use_conv=use_conv,dueling=duelling, double = double,batch_size = batch_size,learning_rate=learning_rate)
        agent.train(train_steps,eval_every,weighted_sampling)
        agent.plot()
    
if __name__ == '__main__':
    dqn_experiment()
    