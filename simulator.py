import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from environment import CarbonTaxEnvironment

from PPO import PPO

# use tensorboard to plot the carbon emitted and bank
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, env_name, revenue, expenses, money_in_bank, max_car, ev, petrol_car, power_consumption, power_generation, solar_panel, max_electricity, electricity_price, carbon_tax):
        self.env_name = env_name
        self.revenue = revenue
        self.expenses = expenses
        self.money_in_bank = money_in_bank
        self.max_car = max_car
        self.ev = ev
        self.petrol_car = petrol_car
        self.power_consumption = power_consumption
        self.power_generation = power_generation
        self.solar_panel = solar_panel
        self.electricity_price = electricity_price
        self.max_electricity = max_electricity
        self.carbon_tax = carbon_tax

        self.summary_writer = SummaryWriter(f'tensorboard/{env_name}')

    ################################### Training ###################################
    def train(self):
        print("============================================================================================")

        ####### initialize environment hyperparameters ######
        env_name = self.env_name

        has_continuous_action_space = False  # continuous action space; else discrete

        max_ep_len = 5000                   # max timesteps in one episode
        max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

        print_freq = max_ep_len * 1        # print avg reward in the interval (in num timesteps)
        log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
        save_model_freq = int(1e5)          # save model frequency (in num timesteps)

        action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
        action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
        action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
        #####################################################

        ## Note : print/log frequencies should be > than max_ep_len

        ################ PPO hyperparameters ################
        update_timestep = max_ep_len    # update policy every n timesteps
        K_epochs = 80                   # update policy for K epochs in one PPO update

        eps_clip = 0.2          # clip parameter for PPO
        gamma = 0.99            # discount factor

        lr_actor = 0.0003       # learning rate for actor network
        lr_critic = 0.001       # learning rate for critic network

        random_seed = 0         # set random seed if required (0 = no random seed)
        #####################################################

        print("training environment name : " + env_name)
        
        revenue = self.revenue
        expenses = self.expenses
        money_in_bank = self.money_in_bank

        max_car = self.max_car
        ev = self.ev
        petrol_car = self.petrol_car

        power_consumption = self.power_consumption
        power_generation = self.power_generation
        solar_panel = self.solar_panel
        max_electricity = self.max_electricity

        electricity_price = self.electricity_price
        carbon_tax = self.carbon_tax

        env = CarbonTaxEnvironment(max_ep_len, revenue, expenses, money_in_bank, max_car, ev, petrol_car, power_consumption, power_generation, solar_panel, electricity_price, max_electricity, carbon_tax)

        # state space dimension
        observation_space_array = env.observation_space  # Call the property
        state_dim = observation_space_array.shape[0] 

        action_dim = env.action_space_dim

        ###################### logging ######################

        # create a csv file for logging, with headers (sim_episode, sim_ev, sim_petrol_car, sim_solar_panel, sim_power_generation, sim_power_consumption, sim_electricity_price, sim_carbon_tax, sim_revenue, sim_expenses, sim_money_in_bank, sim_current_day, sim_max_car, sim_max_electricity)
        with open(f'simulation_log_{env_name}.csv', 'w') as f:
            f.write('sim_episode,sim_ev,sim_petrol_car,sim_solar_panel,sim_power_generation,sim_power_consumption,sim_electricity_price,sim_carbon_tax,sim_revenue,sim_expenses,sim_money_in_bank,sim_current_day,sim_max_car,sim_max_electricity,sim_surplus_power\n')

        #### log files for multiple runs are NOT overwritten
        log_dir = "PPO_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + env_name + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        #### get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run
        log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

        print("current logging run number for " + env_name + " : ", run_num)
        print("logging at : " + log_f_name)
        #####################################################

        ################### checkpointing ###################
        run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

        directory = "PPO_preTrained"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/' + env_name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)


        checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
        print("save checkpoint path : " + checkpoint_path)
        #####################################################


        ############# print all hyperparameters #############
        print("--------------------------------------------------------------------------------------------")
        print("max training timesteps : ", max_training_timesteps)
        print("max timesteps per episode : ", max_ep_len)
        print("model saving frequency : " + str(save_model_freq) + " timesteps")
        print("log frequency : " + str(log_freq) + " timesteps")
        print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
        print("--------------------------------------------------------------------------------------------")
        print("state space dimension : ", state_dim)
        print("action space dimension : ", action_dim)
        print("--------------------------------------------------------------------------------------------")
        if has_continuous_action_space:
            print("Initializing a continuous action space policy")
            print("--------------------------------------------------------------------------------------------")
            print("starting std of action distribution : ", action_std)
            print("decay rate of std of action distribution : ", action_std_decay_rate)
            print("minimum std of action distribution : ", min_action_std)
            print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
        else:
            print("Initializing a discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("PPO update frequency : " + str(update_timestep) + " timesteps")
        print("PPO K epochs : ", K_epochs)
        print("PPO epsilon clip : ", eps_clip)
        print("discount factor (gamma) : ", gamma)
        print("--------------------------------------------------------------------------------------------")
        print("optimizer learning rate actor : ", lr_actor)
        print("optimizer learning rate critic : ", lr_critic)
        if random_seed:
            print("--------------------------------------------------------------------------------------------")
            print("setting random seed to ", random_seed)
            torch.manual_seed(random_seed)
            env.seed(random_seed)
            np.random.seed(random_seed)
        #####################################################

        print("============================================================================================")

        ################# training procedure ################

        # initialize a PPO agent
        ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print("============================================================================================")

        # logging file
        log_f = open(log_f_name,"w+")
        log_f.write('episode,timestep,reward\n')

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        # training loop
        while time_step <= max_training_timesteps:

            state = env.reset()
            current_ep_reward = 0

            for t in range(1, max_ep_len+1):

                # select action with policy
                action = ppo_agent.select_action(state)
                state, reward, done, data, visualization = env.step(action)

                # unpack data
                carbon_emitted, surplus_power, carbon_tax_cost, petrol_cost, carbon_surplus, bankrupt, bank = data
                sim_ev, sim_petrol_car, sim_solar_panel, sim_power_generation, sim_power_consumption, sim_electricity_price, sim_carbon_tax, sim_revenue, sim_expenses, sim_money_in_bank, sim_current_day, sim_max_car, sim_max_electricity = visualization
                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step +=1
                current_ep_reward += reward

                if done:
                    if bankrupt:
                        print(f'###################################Simulation failed due to bankrupt###################################')
                    print(f'Training Environment: {env_name}, Simulation Ended at day: {t}, Total amount in bank: {bank} carbon emitted: {carbon_emitted}, power surplus: {surplus_power}, carbon surplus: {carbon_surplus}, carbon tax cost: {carbon_tax_cost}')

                # update PPO agent
                if time_step % update_timestep == 0 or done:
                    ppo_agent.update()
                    

                # if continuous action space; then decay action std of ouput action distribution
                if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

                # log in logging file
                if time_step % log_freq == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # save model weights
                if time_step % save_model_freq == 0:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path)
                    ppo_agent.save(checkpoint_path)
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")

                # break; if the episode is over
                if done:
                    break

            # log data into csv file
            with open(f'simulation_log_{env_name}.csv', 'a') as f:
                f.write(f'{i_episode},{sim_ev},{sim_petrol_car},{sim_solar_panel},{sim_power_generation},{sim_power_consumption},{sim_electricity_price},{sim_carbon_tax},{sim_revenue},{sim_expenses},{sim_money_in_bank},{sim_current_day},{sim_max_car},{sim_max_electricity},{surplus_power}\n')
            
            # carbon_emitted, surplus_power, carbon_tax_cost, petrol_cost, carbon_surplus, bankrupt, bank
            self.summary_writer.add_scalar('Carbon_Emitted', carbon_emitted, i_episode)
            self.summary_writer.add_scalar('Bank', bank, i_episode)
            self.summary_writer.add_scalar('Carbon_Tax_Cost', carbon_tax_cost, i_episode)
            self.summary_writer.add_scalar('Power_Surplus', surplus_power, i_episode)
            self.summary_writer.add_scalar('Carbon_Surplus', carbon_surplus, i_episode)
            self.summary_writer.add_scalar('Petrol_Cost', petrol_cost, i_episode)

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        log_f.close()
        env.close()

        # print total training time
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")
    
    
    
    
    
    
    
