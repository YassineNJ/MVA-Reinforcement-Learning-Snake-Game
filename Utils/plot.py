import matplotlib.pyplot as plt
import pickle
import numpy as np

def plot_reward(reward_file,N_TRIALS,N_EPISODES,EVAL_EVERY,
                plot_train=True,plot_test=True,separate=True):

    with open(reward_file,'rb') as handle:
        all_rewards = pickle.load(handle)

    mean = np.mean(all_rewards,axis=0)
    std = np.std(all_rewards,axis=0)

    if plot_train:

        plt.figure()
        plt.title('Performance over learning')
        plt.plot(mean[:,0], mean[:,1],label=reward_file)
        plt.fill_between(mean[:,0], mean[:,1]-std[:,1],mean[:,1]+std[:,1],alpha=0.5)
        plt.xlabel('time steps')
        plt.ylabel('total reward')
        plt.legend()
        
    if plot_test:
        
        if separate : plt.figure()
        
        plt.title('Performance on Test Env')
        xv = np.arange(EVAL_EVERY, N_EPISODES, EVAL_EVERY)
        plt.plot(mean[xv,0],mean[xv,2],':o',label=reward_file)
        #plt.plot(mean[:,0],mean[:,2],':o',label=reward_file)
        plt.fill_between(mean[xv, 0], mean[xv, 2]-std[xv, 2],mean[xv, 2]+std[xv, 2], ':o',alpha=0.3)
        plt.xlabel('time steps')
        plt.ylabel('expected total reward (greedy policy)')
        plt.legend()