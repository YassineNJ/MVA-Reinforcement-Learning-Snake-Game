if __name__ == "__main__":
    import torch
    # import gym
    import sys, os
    sys.path.append('../')
    import numpy as np
    from Envs.environment_snake import *
    import matplotlib.pyplot as plt
    from Agents.dqn_agent import *
    from Agents.ac_agent import *







    agent_name = 'ACN'

    Train_agent = False

    plot_results = False

    display = True

    BATCH_SIZE = 256
    TRAIN_STEPS = 100000
    EVAL_EVERY = 1000
    WEIGHTED_SAMPLING = 1
    Image_representation = True
    double_train = True
    double = True
    dueling = False
    use_conv = True
    # # define environement

    env = SnakeGame(w=80, h=80 , max_reward = 1,food_nb = 1 , early_stopping_factor = 10 ,gray_scale = False, isdisplayed = display, use_images = Image_representation   , image_reduction = 2 )
    if agent_name == 'DQN':
        agent = DQNAgent(env, use_conv=use_conv,dueling=dueling, double = double)
    if agent_name == 'ACN':
        agent = ACAgent(env , use_conv=use_conv ,double_train=double_train)
    if Train_agent:
        if agent_name == 'DQN':
            agent.train(TRAIN_STEPS,EVAL_EVERY,BATCH_SIZE,WEIGHTED_SAMPLING)
        if agent_name == 'ACN':
            agent.train(n_updates=TRAIN_STEPS,n_sim =100 , eval_every =EVAL_EVERY ,n_steps = 20 , bootstrap = True , double_train = double_train)

    if plot_results:
        agent.plot()

    if display:
        agent.load_model('Experiments/model_ACN_SnakeGame_True_97_1000000iter.pth')
        agent.eval(double_train=True,display = True,n_sim=10)


    # exp = np.loadtxt(f"Experiments/rewards_DQN_use_conv_False_.txt")
    # expdb =  np.loadtxt(f"Experiments/rewards_DQN_DOUBLE_use_conv_False_.txt")
    # expdl =  np.loadtxt(f"Experiments/rewards_DQN_DUELING_use_conv_False_.txt")
    # expdbdl =  np.loadtxt(f"Experiments/rewards_DQN_DUELING_DOUBLE_use_conv_False_.txt")
    # expac = np.loadtxt(f"Experiments/ACN_rewards_Use_conv_False_descreet.txt")
    # plt.figure()
    # plt.title('Mean score over learning')
    # plt.plot(exp[:,0], exp[:,4],label = 'Standard DQN')
    # plt.plot(expdb[:,0], expdb[:,4],label = 'Double DQN')
    # plt.plot(expdl[:,0], expdl[:,4],label = 'Dueling DQN')
    # plt.plot(expdbdl[:,0], expdbdl[:,4],label = 'Dueling Double DQN')
    # plt.plot(expac[:100][:,0], expac[:100][:,4],label = 'Actor critic')



    # plt.fill_between(exp[:,0], exp[:,4]+exp[:,5] ,np.clip(exp[:,4]-exp[:,5],0,None), alpha = 0.2)
    # plt.fill_between(expdb[:,0], expdb[:,4]+expdb[:,5] ,np.clip(expdb[:,4]-expdb[:,5],0,None), alpha = 0.2)
    # plt.fill_between(expdl[:,0], expdl[:,4]+expdl[:,5] ,np.clip(expdl[:,4]-expdl[:,5],0,None), alpha = 0.2)
    # plt.fill_between(expdbdl[:,0], expdbdl[:,4]+expdbdl[:,5] ,np.clip(expdbdl[:,4]-expdbdl[:,5],0,None), alpha = 0.2)
    # plt.fill_between(expac[:100][:,0], expac[:100][:,4]+expac[:100][:,5] ,np.clip(expac[:100][:,4]-expac[:100][:,5],0,None), alpha = 0.2)

    # plt.legend()
    # plt.title('Descreet representation')
    # plt.xlabel('training steps')

    # plt.show()
    







    #agent = ACAgent(env , use_conv=False ,double_train=double_train)

    # with open("Experiments/ACN_rewards_Use_conv_True.txt", 'r') as f:
    #     mainlist = [line.split(' ') for line in f]

    # cleaned = []

    # for i in range(len(mainlist)):
    #     cleaned.append(mainlist[i][0:6])
    # cleaned = np.array(cleaned)
    # np.savetxt(f"Experiments/ACN_rewards_Use_conv_{True}.txt", np.array(cleaned), fmt="%s")

    #agent.train(N_UPDATES, n_sim =100 , eval_every = 1000 , n_steps=10,double_train=double_train)
#=====================================================
# exp = np.loadtxt(f"Experiments/ACN_rewards_Use_conv_{False}.txt")
# exp2 = np.loadtxt(f"Experiments/ACN_rewards.txt")


# plt.figure()
# plt.title('Mean score over learning')
# plt.plot(exp[:,0], exp[:,4] , label = "Descreet representation")
# plt.fill_between(exp[:,0], exp[:,4]+exp[:,5] , np.clip(exp[:,4]-exp[:,5],0,None) , alpha = 0.2)
# plt.plot(exp2[:,0][0:99], exp2[:,4][0:99], label = "Images representation")
# plt.fill_between(exp2[:,0][0:99], exp2[:,4][0:99]+exp[:,5][0:99] , np.clip(exp2[:,4][0:99]-exp2[:,5][0:99],0,None) , alpha = 0.2)
# plt.xlabel('training steps')
# plt.legend()
# plt.show()





#=======================================================

    # agent.model.load_state_dict(torch.load(f'C:/Users/yassine/Desktop/ATARI-RL-main/Experiments/snake_model_{type(agent.model).__name__}.pth'))


    # experiment(agent,N_TRIALS,N_EPISODES,EVAL_EVERY,BATCH_SIZE,WEIGHTED_SAMPLING)


    #env = gym.make('CartPole-v0')

    # model = DQN(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=10000, log_interval=4)
    # model.save("dqn_pendulum")

    # del model # remove to demonstrate saving and loading

    # model = DQN.load("dqn_pendulum")

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()

    #     # save_path = "C:/Users/yassine/Desktop/ATARI-RL-main/Experiments/"+agent.name
    #     # plot_reward(save_path,N_TRIALS,N_EPISODES,EVAL_EVERY)


