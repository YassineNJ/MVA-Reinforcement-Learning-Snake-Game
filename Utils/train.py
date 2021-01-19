import numpy as np 
from copy import deepcopy
import torch
import pickle
import matplotlib.pyplot as plt
import time
def eval(agent, n_sim=5):
    
    """
    Monte Carlo evaluation of DQN agent
    """
    rewards = np.zeros(n_sim)
    scores = np.zeros(n_sim)
    #copy_env = agent.env.copy() # Important!
    copy_env = agent.env
    # Loop over number of simulations
    for sim in range(n_sim):
        state = copy_env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, score = copy_env.step(action)
            # update sum of rewards
            rewards[sim] += reward
            scores[sim] = score
            state = next_state
    return rewards , scores


def train_agent(agent,N_EPISODES,EVAL_EVERY,BATCH_SIZE,WEIGHTED_SAMPLING):
    t0 = time.time()
    #env = agent.env.copy()
    env = agent.env
    state = env.reset()
    episodes_rewards =[]
    ep= 0
    episode_reward = 0
    total_time = 0 
    
    train_step = 0
    while  ep< N_EPISODES:
        
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        # plt.imshow(next_state.transpose(1,2,0))
        # plt.show()
        total_time +=1 
        
        
        episode_reward += reward
        transition = (state, action, reward, next_state, done)
        agent.replay_buffer.push(*transition)
        
        if len(agent.replay_buffer) > WEIGHTED_SAMPLING*BATCH_SIZE:
            train_step +=1

            if train_step%100  ==0:
                print('Training step : ',train_step)
            agent.update(BATCH_SIZE,WEIGHTED_SAMPLING)

            
        if done  :
            
            mean_rewards = -1
            
            if ep % EVAL_EVERY == 0:

                # evaluate current policy
                rewards ,scores = eval(agent,n_sim=10)
                mean_rewards = np.mean(rewards)

                mean_scores = np.mean(scores)

                t1 = time.time()

                
                print("episode =", ep, ", greedy reward = ", np.round(np.mean(rewards),2),",greedy score = ",mean_scores ,  ",exploration p reward = ", episode_reward , "time = ", t1-t0)
                # if np.mean(rewards) >= REWARD_THRESHOLD:
                #     break
                episodes_rewards.append([mean_rewards ,mean_scores,t1 ])

                torch.save(agent.model.state_dict(), f'C:/Users/yassine/Desktop/ATARI-RL-main/Experiments/snake_model_{type(agent.model).__name__}_2.pth')




            state = env.reset()
            ep += 1
            print("EPISODE",ep)
            episode_reward = 0
        
        else : 

            state  = next_state
        
       
    return episodes_rewards



def experiment(agent,N_TRIALS,N_EPISODES,EVAL_EVERY,BATCH_SIZE ,WEIGHTED_SAMPLING):
    
    save_path = "C:/Users/yassine/Desktop/ATARI-RL-main/Experiments/snake"+agent.name
    all_rewards = []
    
    for i in range(N_TRIALS):
        all_rewards.append(train_agent(agent,N_EPISODES,EVAL_EVERY,BATCH_SIZE,WEIGHTED_SAMPLING))
        np.savetxt(f"episode_rewards_double_{False}_dueling_{False}_{i+1}.txt", np.array(train_agent(agent,N_EPISODES,EVAL_EVERY,BATCH_SIZE,WEIGHTED_SAMPLING)), fmt="%s")

    with open(save_path,'wb') as handle: pickle.dump(all_rewards,handle)
        
    