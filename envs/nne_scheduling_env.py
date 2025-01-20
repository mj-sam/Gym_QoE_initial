
import csv
import heapq
import time
import logging
import random
import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
from statistics import mean
from datetime import datetime, timedelta
from envs.utils import DeploymentRequest, get_c2e_deployment_list, save_to_csv, sort_dict_by_value, \
    calculate_gini_coefficient, normalize, calculate_qoe, simulate_model, model_estimation, normalize_columns_individually

from envs.constants import *



class NNESchedulingEnv(gym.Env):
    """ NNE Scheduling env in Kubernetes - an OpenAI gym environment"""
    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, num_nodes=DEFAULT_NUM_NODES,
                 arrival_rate_r=DEFAULT_ARRIVAL_RATE,
                 call_duration_r=DEFAULT_CALL_DURATION,
                 episode_length=DEFAULT_NUM_EPISODE_STEPS,
                 reward_function=DEFAULT_REWARD_FUNTION,
                 latency_weight=LATENCY_WEIGHT,
                 gini_weight=GINI_WEIGHT,
                 cost_weight=COST_WEIGHT,
                 bandwidth_weight=BANDWIDTH_WEIGHT,
                 qoe_weight=QOE_WEIGHT,
                 seed=SEED,
                 factor=FACTOR,
                 path_csv_files=PATH_CSV_FILES,
                 file_results_name= DEFAULT_FILE_NAME_RESULTS,
                 objective_feature_in_observation = DEFAULT_OBJECTIVE_FEATURE_IN_OBSERVATION,
                 qoe_in_observation = DEFAULT_QOE_IN_OBSERVATION,
                 qoe_simulation_mode= QOE_SIMULATION_MODE,
                 qoe_simulated_accuracy= QOE_SIMULATED_ACCURACY):

        # Define action and observation space
        super(NNESchedulingEnv, self).__init__()

        #initializing self with inputs
        self.name = "qoe_gym"
        self.__version__ = "0.0.1"
        self.reward_function = reward_function
        self.num_nodes = num_nodes
        self.total_number = num_nodes #* NUM_SERVER_TYPE
        self.current_step = 0
        self.default_node_types = DEFAULT_NODE_TYPES
        self.arrival_rate_r = arrival_rate_r
        self.call_duration_r = call_duration_r
        self.episode_length = episode_length
        self.running_requests: list[DeploymentRequest] = []
        self.seed = seed
        self.np_random, seed = seeding.np_random(self.seed)
        self.factor = factor


        # different observation space configuration
        self.objective_feature_in_observation = objective_feature_in_observation
        self.qoe_in_observation = qoe_in_observation
        self.qoe_simulation_mode = qoe_simulation_mode
        self.qoe_simulated_accuracy = qoe_simulated_accuracy
        # -------------------------------------------------------
        # ------------- observation configuration ----------------
        self.feature_count = NUM_METRICS_NODES  # Base features

        if self.qoe_in_observation:
            self.feature_count  += 3  # Adding latency, jerkiness, sync

        if self.objective_feature_in_observation:
            self.feature_count  += 4  # Adding throughput, packet size, interarrival times

        #-------------------------------------------------------
        #------------- simulation configuration ----------------
        self.file_df = pd.read_csv("./mydata/simulation.csv")
        if self.qoe_simulation_mode == "Simulation":
            self.file_df = simulate_model(self.file_df, self.qoe_simulated_accuracy, columns = [DF_COLUMN_LATENCY, DF_COLUMN_JERKINESS, DF_COLUMN_SYNC])
        elif self.qoe_simulation_mode == "Real":
            self.file_df = model_estimation(self.file_df, columns = [DF_COLUMN_LATENCY, DF_COLUMN_JERKINESS, DF_COLUMN_SYNC])
        else:
            logging.info("Invalid QoE Simulation Mode")
            raise Exception("Invalid QoE Simulation Mode")

        self.file_df = normalize_columns_individually(self.file_df,
                                                               [DF_COLUMN_LATENCY, DF_COLUMN_JERKINESS, DF_COLUMN_SYNC])

        #Initialize variables
        self.initialize_rewards(latency_weight, gini_weight, cost_weight, qoe_weight)
        self.initialize_avg_metrics()
        self.initialize_metrics_arrays()

        self.node_id = np.zeros(self.total_number)
        self.server_type_id = np.zeros(self.total_number)

        #Defined as a matrix having as rows the nodes and columns their associated metrics
        self.observation_space = spaces.Box(low=MIN_OBS,
                                            high=MAX_OBS,
                                            shape=(self.total_number + 1, self.feature_count  + NUM_METRICS_REQUEST),
                                            dtype=np.float32)

        logging.info(
            "[Init] Env: {} | Version {} | Num_Nodes: {} | Total Number: {}".format(self.name, self.__version__,
                                                                                    num_nodes, self.total_number))

        # Action Space
        # deploy the service on node 1 - ID 1, node 1 ID 2,..., n + reject it
        self.num_actions = self.total_number + 1
        # Discrete action space
        self.action_space = spaces.Discrete(self.num_actions)
        # Action and Observation Space
        logging.info("[Init] Action Space: {}".format(self.action_space))
        logging.info("[Init] Observation Space: {}".format(self.observation_space))
        # logging.info("[Init] Observation Space Shape: {}".format(self.observation_space.shape))

        # Setting the experiment based on Cloud2Edge (C2E) deployments
        self.deploymentList = get_c2e_deployment_list()
        self.deployment_request = None

        #for test remove this later
        # order = [0, 1, 2, 3]
        # np.random.shuffle(order)

        logging.info("[Init] Resource Capacities... ")
        self.initialize_resources()



        # CSV files for each node
        self.path_csv_files = path_csv_files
        self.node_csv_data = []
        self.df_node = []
        self.action_valid = []

        self.df_node_selected_rows = []
        self.selected_ts = None

        self.intialize_node()

        # Choose a random timestamp to start Episode
        self.get_start_index()

        # Update network
        self.update_network_values()

        # Variables for logging
        self.current_time = 0
        self.penalty = False
        self.accepted_requests = 0
        self.offered_requests = 0
        self.ep_accepted_requests = 0
        self.next_request()

        # Info & episode over
        self.total_reward = 0
        self.episode_over = False
        self.info = {}
        self.block_prob = 0
        self.ep_block_prob = 0
        self.time_start = 0
        self.execution_time = 0
        self.episode_count = 0
        self.file_results_name = file_results_name
        self.file_results = file_results_name + ".csv"
        self.obs_csv = self.name + "_obs.csv"

    def initialize_resources(self):
        #np.random.shuffle(order)
        # New: Resource capacities based on node type
        self.cpu_capacity = np.zeros(self.total_number)
        self.memory_capacity = np.zeros(self.total_number)
        self.node_type = [0] * self.total_number
        # Keeps track of allocated resources
        self.allocated_cpu = np.zeros(self.total_number)
        self.allocated_memory = np.zeros(self.total_number)

        random_cpu = self.np_random.uniform(low=0.0, high=0.2, size=self.num_nodes)
        random_memory = self.np_random.uniform(low=0.0, high=0.2, size=self.num_nodes)

        # Keeps track of Free resources for deployment requests
        self.free_cpu = np.zeros(self.total_number)
        self.free_memory = np.zeros(self.total_number)

        j=0
        for n in range(self.num_nodes):
            #node_type = order[j] #int(self.np_random.integers(low=0, high=NUM_NODE_TYPES))
            node_type = int(self.np_random.integers(low=0, high=NUM_NODE_TYPES))
            #for i_s in range(NUM_SERVER_TYPE):
            self.node_id[j] = n
            self.node_type[j] = node_type
            self.cpu_capacity[j] = DEFAULT_NODE_TYPES[node_type]['cpu']
            self.memory_capacity[j] = DEFAULT_NODE_TYPES[node_type]['mem']

            logging.info("[Init] node: {} | Type: {} "
                        "| cpu: {} | mem: {}".format(n + 1,
                                                         DEFAULT_NODE_TYPES[node_type]['type'],
                                                         self.cpu_capacity[j],
                                                         self.memory_capacity[j]))
            j += 1



        j = 0
        for n in range(self.num_nodes):
            # for i_s in range(NUM_SERVER_TYPE):
            self.allocated_cpu[j] = random_cpu[n]
            self.allocated_memory[j] = random_memory[n]
            j += 1



    def initialize_rewards(self, latency_weight, gini_weight, cost_weight, qoe_weight):
        self.latency_weight = latency_weight
        self.gini_weight = gini_weight
        self.cost_weight = cost_weight
        #self.bandwidth_weight = bandwidth_weight
        self.qoe_weight = qoe_weight

    def initialize_metrics_arrays(self):
        # MJ: these are the environment variable
        # These are the observational values in the environment: at each step these are getting filled from the simulation
        self.throuput_in = np.zeros(self.total_number)
        self.throuput_out = np.zeros(self.total_number)
        self.packetsize_in = np.zeros(self.total_number)
        self.packetsize_out = np.zeros(self.total_number)
        self.interarrival_in = np.zeros(self.total_number)
        self.interarrival_out = np.zeros(self.total_number)

        # simulation for if have the actual labels : QOE MODEL
        self.latency_binary = np.zeros(self.total_number)
        self.jerkiness_binary = np.zeros(self.total_number)
        self.sync_binary = np.zeros(self.total_number)

        # variable for the actual reported value
        self.latency_q = np.zeros(self.total_number)
        self.jerkiness_q = np.zeros(self.total_number)
        self.sync_q = np.zeros(self.total_number)


        # self.rtt = np.zeros(self.total_number)
        # self.ul = np.zeros(self.total_number)
        # self.dl = np.zeros(self.total_number)
        # self.jitter = np.zeros(self.total_number)


    def initialize_avg_metrics(self):
        # ==========================
        # -------------------------
        self.avg_throuput_in = []
        self.avg_packetsize_in = []
        self.avg_interarrival_in = []
        # -------------------------
        self.avg_throuput_out = []
        self.avg_packetsize_out = []
        self.avg_interarrival_out = []
        # -------------------------
        self.avg_latency_binary = []
        self.avg_latency_q = []
        # -------------------------
        self.avg_jerkiness_binary = []
        self.avg_jerkiness_q = []
        # -------------------------
        self.avg_sync_binary = []
        self.avg_sync_q = []
        # ----------------------
        # self.avg_rtt = []
        # self.avg_ul = []
        # self.avg_dl = []
        # self.avg_jitter = []
        # ==========================

        self.avg_total_latency = []
        self.avg_processing_latency = []
        self.avg_access_latency = []
        self.total_latency = []
        self.avg_deployment_cost = []
        self.avg_load_served_per_provider = np.zeros(NUM_SERVER_TYPE)
        self.processing_latency = np.zeros(self.total_number)

    def intialize_node(self):
        j = 0
        #print(distributed)
        #np.random.shuffle(order)

        for n in range(self.num_nodes):
            config_random = np.random.randint(0, 3, dtype=int)
            #print(config_random)
            S = self.file_df[self.file_df['Config'] == SERVER_TYPES[config_random]]
            self.server_type_id[n] = config_random
            #S = file_df[file_df['Config'] == SERVER_TYPES[j % 4]]
            #print("CONFIG : ",SERVER_TYPES[j%4])
            #print("S : ",S)

            sampled_df = S.sample(n=1000, replace=True)

            #for i_s in range(NUM_SERVER_TYPE):
            #-------------------------------------------------------------------
            self.free_cpu[j] = self.cpu_capacity[j] - self.allocated_cpu[j]
            self.free_memory[j] = self.memory_capacity[j] - self.allocated_memory[j]

            # Update files for each node
            # self.node_csv_data.append(self.path_csv_files + file)
            self.node_csv_data.append(config_random)

            #self.df_node.append(pd.read_csv(self.node_csv_data[j]))
            self.df_node.append(sampled_df)

            server_type = SERVER_TYPES[config_random]

            # Select rows based on server type
            selected_rows = self.df_node[j][(self.df_node[j][SERVER_TYPE] == server_type)]

            # print(selected_rows)
            self.df_node_selected_rows.append(selected_rows)

            # logging.info("[Init] Node: {} | Provider: {} | Interface: {} | size of rows: {}".format(n + 1,
            #   PROVIDERS[p],INTERFACES[i], len(selected_rows)))

            # If len(rows) = 0 then provider or interface do not exist
            if len(selected_rows) == 0:
                # logging.info("[Init] Node: {} | Provider: {} | Interface: {}
                # do not exist in CSV file".format(n + 1, PROVIDERS[p], INTERFACES[i]))
                self.action_valid.append(False)
            else:
                logging.info(
                    "[Init] Node: {} "
                    "exists in CSV file".format(n + 1))
                self.action_valid.append(True)

            j += 1

    # Reset Function
    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.current_step = 0
        self.episode_over = False
        self.total_reward = 0
        self.ep_accepted_requests = 0
        self.penalty = False

        self.block_prob = 0
        self.ep_block_prob = 0

        # files for each node
        self.node_csv_data = []
        self.df_node = []
        self.action_valid = []

        # Rows for each node
        self.df_node_selected_rows = []
        self.selected_ts = None
        #==================================================
        # for test remove this later
        #order = [0, 1, 2, 3]
        #np.random.shuffle(order)
        #=================================================

        self.initialize_avg_metrics()
        # Reset Deployment Data
        self.deploymentList = get_c2e_deployment_list()
        self.initialize_metrics_arrays()
        logging.info("[Reset] Resource Capacities... ")
        self.initialize_resources()

        self.intialize_node()

        # Choose a random index to start Episode
        self.get_start_index()

        # Update network
        self.update_network_values()

        # return obs
        return np.array(self.get_state())

    # Step function
    def step(self, action):
        if self.current_step == 1:
            self.time_start = time.time()

        # Execute one time step within the environment
        self.offered_requests += 1
        self.take_action(action)

        # Calculate Reward
        reward = self.get_reward()
        self.total_reward += reward

        # Find correct action move for logging purposes
        move = ""
        if action < self.total_number:
            move = ACTIONS[0] + "-" + str(int(self.node_id[action] + 1)) \
                   + "-Server Type-" + str(int(self.server_type_id[action] + 1)) \

        elif action == self.total_number:
            move = ACTIONS[1]

        # Logging Step and Total Reward
        logging.info('[Step {}] | Action: {} | Reward: {} | Total Reward: {}'.format(self.current_step, move, reward,
                                                                                     self.total_reward))

        # Get next request
        self.next_request()

        # Update RTT values
        self.update_network_values()

        # Update observation
        ob = self.get_state()

        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # self.save_obs_to_csv(self.obs_csv, np.array(ob), date)

        # episode results to save
        self.block_prob = 1 - (self.accepted_requests / self.offered_requests)
        self.ep_block_prob = 1 - (self.ep_accepted_requests / self.current_step)

        if (len(self.avg_access_latency) == 0 \
                and len(self.avg_deployment_cost) == 0 \
                and len(self.avg_throuput_in) == 0 \
                and len(self.avg_packetsize_in) == 0 \
                and len(self.avg_interarrival_in) == 0 \
                and len(self.avg_throuput_out) == 0 \
                and len(self.avg_packetsize_out) == 0 \
                and len(self.avg_interarrival_out) == 0 \

                and len(self.avg_latency_binary) == 0 \
                and len(self.avg_jerkiness_binary) == 0 \
                and len(self.avg_sync_binary) == 0 \
                and len(self.total_latency) == 0 \
                and len(self.avg_processing_latency) == 0):
            avg_c = 1
            #-------------------
            avg_throuput_in = 1
            avg_packetsize_in = 1
            avg_interarrival_in = 1
            avg_throuput_out = 1
            avg_packetsize_out = 1
            avg_interarrival_out = 1
            avg_latency_binary = 1
            avg_jerkiness_binary = 1
            avg_sync_binary = 1
            avg_sync_q = 1
            avg_latency_q = 1
            avg_jerkiness_q = 1
            #----------------
            avg_l = 1
            total_latency = 1
            avg_proc = 1
        else:
            avg_c = mean(self.avg_deployment_cost)
            #-------------------------------------
            avg_throuput_in = mean(self.avg_throuput_in)
            avg_packetsize_in = mean(self.avg_packetsize_in)
            avg_interarrival_in = mean(self.avg_interarrival_in)
            avg_throuput_out = mean(self.avg_throuput_out)
            avg_packetsize_out = mean(self.avg_packetsize_out)
            avg_interarrival_out = mean(self.avg_interarrival_out)
            avg_latency_binary = mean(self.avg_latency_binary)
            avg_jerkiness_binary = mean(self.avg_jerkiness_binary)
            avg_sync_binary = mean(self.avg_sync_binary)
            avg_latency_q = mean(self.avg_latency_q)
            avg_jerkiness_q = mean(self.avg_jerkiness_q)
            avg_sync_q = mean(self.avg_sync_q)
            #------------------------------------------
            avg_l = mean(self.avg_access_latency)
            total_latency = mean(self.avg_total_latency)
            avg_proc = mean(self.avg_processing_latency)

        qoe = calculate_qoe(avg_sync_q, avg_latency_q, avg_jerkiness_q)
        self.info = {
            "reward_step": float("{:.2f}".format(reward)),
            "action": float("{:.2f}".format(action)),
            "reward": float("{:.2f}".format(self.total_reward)),
            "ep_block_prob": float("{:.2f}".format(self.ep_block_prob)),
            "ep_accepted_requests": float("{:.2f}".format(self.ep_accepted_requests)),
            'avg_deployment_cost': float("{:.2f}".format(avg_c)),
            'avg_total_latency': float("{:.2f}".format(total_latency)),
            'avg_access_latency': float("{:.2f}".format(avg_l)),
            'avg_processing_latency': float("{:.2f}".format(avg_proc)),
            #----------------
            # 'avg_throuput_in': float("{:.2f}".format(avg_throuput_in)),
            # 'avg_packetsize_in': float("{:.2f}".format(avg_packetsize_in)),
            # 'avg_interarrival_in': float("{:.2f}".format(avg_interarrival_in)),
            # 'avg_throuput_out': float("{:.2f}".format(avg_throuput_out)),
            # 'avg_packetsize_out': float("{:.2f}".format(avg_packetsize_out)),
            # 'avg_interarrival_out': float("{:.2f}".format(avg_interarrival_out)),
            # 'avg_latency_binary': float("{:.2f}".format(avg_latency_binary)),
            # 'avg_jerkiness_binary': float("{:.2f}".format(avg_jerkiness_binary)),
            # 'avg_sync_binary': float("{:.2f}".format(avg_sync_binary)),
            #--------------------------------
            'qoe': float("{:.2f}".format(qoe)),
            'gini': float("{:.2f}".format(calculate_gini_coefficient(self.avg_load_served_per_provider))),
            'executionTime': float("{:.2f}".format(self.execution_time))
        }

        if self.current_step == self.episode_length:
            self.episode_count += 1
            self.episode_over = True
            self.execution_time = time.time() - self.time_start

            gini = calculate_gini_coefficient(self.avg_load_served_per_provider)

            logging.info("[Step] Episode finished, saving results to csv...")
            qoe = calculate_qoe(avg_sync_q, avg_latency_q, avg_jerkiness_q)
            #print(qoe)

            # Prepare data dictionary
            data = {
                'episode': self.episode_count,
                'reward': round(self.total_reward, 2),
                'ep_block_prob': round(self.ep_block_prob, 2),
                'ep_accepted_requests': round(self.ep_accepted_requests, 2),
                'avg_deployment_cost': round(mean(self.avg_deployment_cost), 2),
                'avg_total_latency': round(mean(self.avg_total_latency), 2),
                'avg_access_latency': round(mean(self.avg_access_latency), 2),
                'avg_proc_latency': round(mean(self.avg_processing_latency), 2),
                # --------------------------
                'avg_throuput_in': round(mean(self.avg_throuput_in), 2),
                'avg_packetsize_in': round(mean(self.avg_packetsize_in), 2),
                'avg_interarrival_in': round(mean(self.avg_interarrival_in), 2),
                'avg_throuput_out': round(mean(self.avg_throuput_out), 2),
                'avg_packetsize_out': round(mean(self.avg_packetsize_out), 2),
                'avg_interarrival_out': round(mean(self.avg_interarrival_out), 2),
                'avg_latency_binary': round(mean(self.avg_latency_binary), 3),
                'avg_jerkiness_binary': round(mean(self.avg_jerkiness_binary), 3),
                'avg_sync_binary': round(mean(self.avg_sync_binary), 3),
                'avg_qoe': round(qoe, 3),
                'gini': round(gini, 3),
                'execution_time': round(self.execution_time, 3)
            }

            save_to_csv(self.file_results, data)

        # return ob, reward, self.episode_over, self.info
        return np.array(ob), reward, self.episode_over, self.info

    # Reward Function
    def get_reward(self):
        """ Calculate Rewards """
        if self.reward_function == NAIVE:
            if self.penalty:
                if not self.check_if_node_is_really_full():
                    logging.info("[Get Reward] Penalty = True, and resources "
                                 "were available, penalize the agent...")
                    return -1
                else:  # agent should not be penalized
                    logging.info("[Get Reward] Penalty = True, but resources "
                                 "were not available, do not penalize the agent...")
                    return 1
            else:
                return 1

        # Multi-objective
        elif self.reward_function == MULTI:
            if self.penalty:
                if not self.check_if_node_is_really_full():
                    logging.info("[Get Reward] Penalty = True, and resources "
                                 "were available, penalize the agent...")
                    return -1
                else:  # agent should not be penalized
                    logging.info("[Get Reward] Penalty = True, but resources "
                                 "were not available, do not penalize the agent...")
                    return 1
            else:  # Multi-objective reward function: latency + cost + gini + bandwidth

                #latency = self.deployment_request.expected_rtt + self.deployment_request.expected_access_latency + self.deployment_request.expected_processing_latency
                latency = self.deployment_request.expected_access_latency + self.deployment_request.expected_processing_latency
                logging.info('[Multi Reward] Latency components:  Lat: {} | Processing: {}'.format(
                    self.deployment_request.expected_access_latency,
                    self.deployment_request.expected_processing_latency))
                # Gini
                gini = calculate_gini_coefficient(self.avg_load_served_per_provider)

                # Cost
                cost = self.deployment_request.expected_cost

                # Bandwidth
                qoe = calculate_qoe(self.deployment_request.sync_q, self.deployment_request.jerkiness_q, self.deployment_request.latency_q)

                #bandwidth = self.deployment_request.expected_dl_bandwidth + self.deployment_request.expected_ul_bandwidth
                #latency_q = self.late
                #==============================================
                #TODO : Add QoE here
                #==========================================
                logging.info(
                    '[Multi Reward] latency: {} | gini: {} | cost: {} | qoe: {}| '.format(latency, gini, cost,qoe))

                latency = normalize(latency,  MIN_LATENCY + MIN_PROC,  MAX_LATENCY + MAX_PROC)
                cost = normalize(cost, MIN_COST, MAX_COST)
                #bandwidth = normalize(bandwidth, MIN_DL + MIN_UL, MAX_DL + MAX_UL)
                #qoe = normalize(qoe, 0, 15)

                reward = self.latency_weight * (1 - latency) + self.gini_weight * (1 - gini) + self.cost_weight * (1 - cost) + self.qoe_weight *  (1 - qoe)

                logging.info(
                    '[Multi Reward] Normalized: latency: {} | gini: {} | cost: {} | qoe: {}|'.format(latency, gini, cost, qoe))
                logging.info(
                    '[Multi Reward] Applying weights: latency {} | gini part: {} | cost: {} | qoe: {}'.format(
                        self.latency_weight * (1 - latency),
                        self.gini_weight * (1 - gini),
                        self.cost_weight * (1 - cost),
                        self.qoe_weight * (1 - qoe)
                    ))

                logging.info('[Multi Reward] Final reward: {}'.format(reward))

                return reward
        else:
            logging.info('[Get Reward] Unrecognized reward: {}'.format(self.reward_function))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return

    # Apply the action selected by the RL agent
    def take_action(self, action):
        self.current_step += 1

        # Stop if MAX_STEPS
        if self.current_step == self.episode_length:
            # logging.info('[Take Action] MAX STEPS achieved, ending ...')
            self.episode_over = True

        # Possible Actions: Place all replicas together or split them.
        # Known as NP-hard problem (Bin pack with fragmentation)
        # Any ideas for heuristic? We can later compare with an ILP/MILP model...
        # Check first if "Place all" Action can be performed
        if action < self.total_number:
            if self.check_if_node_is_full_after_full_deployment(action) or not self.action_valid[action]:
                self.penalty = True
                logging.info('[Take Action] Block the selected action since action is invalid or node will be full!')
                # Do not raise error since algorithm might not support action mask
                # raise ValueError("Action mask is not working properly. Full nodes should be always masked.")
            else:
                # accept request
                logging.info("[Take Action] Accept request...")
                self.accepted_requests += 1
                self.ep_accepted_requests += 1
                self.processing_latency[action] += PROCESSING_DELAY

                type_id = self.node_type[action]
                self.avg_load_served_per_provider[int(self.server_type_id[action])] += 1
                self.deployment_request.deployed_node = self.node_id[action]
                self.deployment_request.action_id = action
                self.deployment_request.deployed_provider = self.server_type_id[action]

                self.avg_deployment_cost.append(DEFAULT_NODE_TYPES[type_id]['cost'])

                # WE DONT NEED TO INCLUD RTT HERE AGAIN
                # self.avg_total_latency.append( DEFAULT_NODE_TYPES[type_id]['latency'] + self.processing_latency[action] + self.rtt[action])

                self.avg_total_latency.append(
                    DEFAULT_NODE_TYPES[type_id]['latency'] + self.processing_latency[action])# + self.rtt[action])

                self.avg_access_latency.append(DEFAULT_NODE_TYPES[type_id]['latency'])
                self.avg_processing_latency.append( self.processing_latency[action])

                #self.avg_rtt.append(self.rtt[action])
                #self.avg_ul.append(self.ul[action])
                #self.avg_dl.append(self.dl[action])
                #self.avg_jitter.append(self.jitter[action])

                self.avg_throuput_in.append(self.throuput_in[action])
                self.avg_packetsize_in.append(self.packetsize_in[action])
                self.avg_interarrival_in.append(self.interarrival_in[action])
                # -------------------------
                self.avg_throuput_out.append(self.throuput_out[action])
                self.avg_packetsize_out.append(self.packetsize_out[action])
                self.avg_interarrival_out.append(self.interarrival_out[action])
                # -------------------------
                self.avg_latency_binary.append(self.latency_binary[action])
                self.avg_jerkiness_binary.append(self.jerkiness_binary[action])
                self.avg_sync_binary.append(self.sync_binary[action])

                #---------------------------
                self.avg_latency_q.append(self.latency_q[action])
                self.avg_jerkiness_q.append(self.jerkiness_q[action])
                self.avg_sync_q.append(self.sync_q[action])
                #---------------------------

                #self.deployment_request.expected_dl_bandwidth = self.dl[action]
                #self.deployment_request.expected_ul_bandwidth = self.ul[action]
                #self.deployment_request.expected_rtt = self.rtt[action]


                # WE ONLY KEEP THE ACCESS LATENCY
                self.deployment_request.expected_access_latency = DEFAULT_NODE_TYPES[type_id]['latency']
                self.deployment_request.expected_cost = DEFAULT_NODE_TYPES[type_id]['cost']
                self.deployment_request.expected_processing_latency = self.processing_latency[action]

                # for QOE metrics
                self.deployment_request.latency_q = self.latency_q[action]
                self.deployment_request.jerkiness_q = self.jerkiness_q[action]
                self.deployment_request.sync_q =  self.sync_q[action]

                self.penalty = False

                # Update allocated amounts
                for n in range(len(self.node_id)):
                    if self.node_id[n] == self.node_id[action]:
                        self.allocated_cpu[n] += self.deployment_request.cpu_request
                        self.allocated_memory[n] += self.deployment_request.memory_request

                        # Update free resources
                        self.free_cpu[n] = self.cpu_capacity[n] - self.allocated_cpu[n]
                        self.free_memory[n] = self.memory_capacity[n] - self.allocated_memory[n]

                        # Update processing latency
                        self.processing_latency[n] += PROCESSING_DELAY

                # Update the request
                self.enqueue_request(self.deployment_request)

        # Reject the request: give the agent a penalty, especially if the request could have been accepted
        elif action == self.total_number:
            self.penalty = True
        else:
            logging.info('[Take Action] Unrecognized Action: {}'.format(action))

    def get_state(self):
        # Get Observation state

        #node = np.full(shape=(1, NUM_METRICS_NODES), fill_value=-1)
        node = np.full(shape=(1, self.feature_count), fill_value=-1)

        base_observation = np.stack([
            self.allocated_cpu,
            self.cpu_capacity,
            self.allocated_memory,
            self.memory_capacity
        ], axis=1)

        if self.qoe_in_observation:
            qoe_observation = np.stack([
                self.latency_binary,
                self.jerkiness_binary,
                self.sync_binary
            ], axis=1)
            base_observation = np.concatenate([base_observation, qoe_observation], axis=1)

        if self.objective_feature_in_observation:
            objective_observation = np.stack([
                self.throuput_in,
                self.throuput_out,
                self.packetsize_in,
                self.packetsize_out,
                #self.interarrival_in,
                #self.interarrival_out
            ], axis=1)
            base_observation = np.concatenate([base_observation, objective_observation], axis=1)

        # Condition the elements in the set with the current node request
        request = np.tile(
            np.array(
                [self.deployment_request.cpu_request,
                 self.deployment_request.memory_request,
                 self.deployment_request.latency_threshold,
                 self.dt]
            ),
            (self.total_number + 1, 1),
        )

        observation = np.concatenate([base_observation, node], axis=0)

        observation = np.concatenate([observation, request], axis=1)

        return observation

    # Save observation to csv file
    def save_obs_to_csv(self, obs_file, obs, date):
        file = open(obs_file, 'a+', newline='')  # append
        # file = open(file_name, 'w', newline='') # new
        fields = []
        node_obs = {}
        with file:
            fields.append('date')
            for n in range(self.num_nodes):
                fields.append("node_" + str(n + 1) + '_allocated_cpu')
                fields.append("node_" + str(n + 1) + '_cpu_capacity')
                fields.append("node_" + str(n + 1) + '_allocated_memory')
                fields.append("node_" + str(n + 1) + '_memory_capacity')
                fields.append("node_" + str(n + 1) + '_num_replicas')
                fields.append("node_" + str(n + 1) + '_cpu_request')
                fields.append("node_" + str(n + 1) + '_memory_request')
                fields.append("node_" + str(n + 1) + '_dt')

            # logging.info("[Save Obs] fields: {}".format(fields))

            writer = csv.DictWriter(file, fieldnames=fields)
            # writer.writeheader() # write header

            node_obs.update({fields[0]: date})

            for n in range(self.num_nodes):
                i = self.get_iteration_number(n)
                node_obs.update({fields[i + 1]: obs[n][0]})
                node_obs.update({fields[i + 2]: obs[n][1]})
                node_obs.update({fields[i + 3]: obs[n][2]})
                node_obs.update({fields[i + 4]: obs[n][3]})
                node_obs.update({fields[i + 5]: obs[n][4]})
                node_obs.update({fields[i + 6]: obs[n][5]})
                node_obs.update({fields[i + 7]: obs[n][6]})
                node_obs.update({fields[i + 8]: obs[n][7]})
            writer.writerow(node_obs)
        return

    def get_iteration_number(self, n):
        num_fields_per_node = 8
        return num_fields_per_node * n

    def enqueue_request(self, request: DeploymentRequest) -> None:
        heapq.heappush(self.running_requests, (request.departure_time, request))

    # Action masks
    def action_masks(self):
        valid_actions = np.ones(self.total_number + 1, dtype=bool)
        j = 0
        for n in range(self.num_nodes):
            #for i_s in range(NUM_SERVER_TYPE):
            if self.check_if_node_is_full_after_full_deployment(n) or not self.action_valid[j]:
                valid_actions[j] = False
            else:
                valid_actions[j] = True
            j += 1

        # 1 additional action: Reject
        valid_actions[self.total_number] = True
        logging.info('[Action Mask]: Valid actions {} |'.format(valid_actions))
        return valid_actions

    # Double-check if the selected cluster is full
    def check_if_node_is_full_after_full_deployment(self, action):
        total_cpu = self.deployment_request.cpu_request
        total_memory = self.deployment_request.memory_request

        if (self.allocated_cpu[action] + total_cpu > 0.95 * self.cpu_capacity[action]
                or self.allocated_memory[action] + total_memory > 0.95 * self.memory_capacity[action]):
            logging.info('[Check]: Node is full... Action id: {}'.format(action + 1))
            return True

        return False

    # Remove deployment request
    def dequeue_request(self):
        _, deployment_request = heapq.heappop(self.running_requests)
        logging.info("[Dequeue] Request will be terminated...")

        action = deployment_request.action_id
        total_cpu = self.deployment_request.cpu_request
        total_memory = self.deployment_request.memory_request

        '''
        logging.info("[Dequeue] Before")
        logging.info("[Dequeue] Action ID: {}".format(action))
        logging.info("[Dequeue] CPU allocated: {}".format(self.allocated_cpu))
        logging.info("[Dequeue] CPU free: {}".format(self.free_cpu))
        logging.info("[Dequeue] MEM allocated: {}".format(self.allocated_memory))
        logging.info("[Dequeue] MEM free: {}".format(self.free_memory))
        '''
        # Update allocated amounts
        for n in range(len(self.node_id)):
            if self.node_id[n] == self.node_id[action]:
                self.allocated_cpu[n] -= total_cpu
                self.allocated_memory[n] -= total_memory

                # Update free resources
                self.free_cpu[n] = self.cpu_capacity[n] - self.allocated_cpu[n]
                self.free_memory[n] = self.memory_capacity[n] - self.allocated_memory[n]

                # Update processing latency
                self.processing_latency[n] -= PROCESSING_DELAY

        '''
        logging.info("[Dequeue] After")
        logging.info("[Dequeue] Action ID: {}".format(action))
        logging.info("[Dequeue] CPU allocated: {}".format(self.allocated_cpu))
        logging.info("[Dequeue] CPU free: {}".format(self.free_cpu))
        logging.info("[Dequeue] MEM allocated: {}".format(self.allocated_memory))
        logging.info("[Dequeue] MEM free: {}".format(self.free_memory))
        logging.info("[Dequeue] Processing Delay: {}".format(self.processing_latency))
        '''

    # Check if all clusters are full
    def check_if_node_is_really_full(self) -> bool:
        is_full = [self.check_if_node_is_full_after_full_deployment(i) for i in range(self.num_nodes)]
        return np.all(is_full)

    # Create a deployment request
    def deployment_generator(self):
        deployment_list = get_c2e_deployment_list()
        n = self.np_random.integers(low=0, high=len(deployment_list))
        d = deployment_list[n - 1]
        return d

    # Select (random) the next deployment request
    def next_request(self) -> None:
        arrival_time = self.current_time + self.np_random.exponential(scale=1 / self.arrival_rate_r)
        departure_time = arrival_time + self.np_random.exponential(scale=self.call_duration_r)
        self.dt = departure_time - arrival_time
        self.current_time = arrival_time

        while True:
            if self.running_requests:
                next_departure_time, _ = self.running_requests[0]
                if next_departure_time < arrival_time:
                    self.dequeue_request()
                    continue
            break

        self.deployment_request = self.deployment_generator()
        self.deployment_request.cpu_request = self.factor * self.deployment_request.cpu_request
        self.deployment_request.memory_request = self.factor * self.deployment_request.memory_request

        logging.info('[Next Request]: Name: {} | CPU: {} | MEM: {}'.format(self.deployment_request.name,
                                                                           self.deployment_request.cpu_request,
                                                                           self.deployment_request.memory_request))

    # Choose random index (ts) from dataframe to start simulation with at least 300 samples left for each node
    def get_start_index(self):
        # check min size and id of the node
        min_size = MIN_SIZE
        id = 0
        for n in range(self.num_nodes):
            if len(self.df_node[n]) < min_size:
                min_size = len(self.df_node[n])
                id = n

        # Choose a random index, but making sure 300 consecutive samples exist
        # TODO : select appropriate random start index
        #start_index = np.random.randint(0, 2)

        # Get the timestamp at the random index
        #self.selected_ts = self.df_node[id].loc[0, 'timestamp']
        logging.info("Selected TS: {}".format(self.selected_ts))

        j = 0
        for n in range(self.num_nodes):
            # for i_s in range(NUM_SERVER_TYPE):
            # If provider and interface exist
            if self.action_valid[j]:
                # Select rows starting from ts
                self.df_node_selected_rows[j] = self.df_node[j]#[(self.df_node[j]['timestamp'] >= self.selected_ts)]

                # Reset index of the final DataFrame
                self.df_node_selected_rows[j].reset_index(drop=True, inplace=True)

            j += 1
        return

    def update_network_values(self):
        if self.current_step == 0:
            step = 1
        else:
            step = self.current_step + 1

        j = 0
        for n in range(self.num_nodes):
            #for i_s in range(NUM_SERVER_TYPE):
            if self.action_valid[j]:
                self.throuput_in[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_THROUPUT_DL]
                self.throuput_out[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_THROUPUT_UL]
                self.packetsize_in[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_PACKETSIZE_DL]
                self.packetsize_out[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_PACKETSIZE_UL]
                self.interarrival_in[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_INTERARRIVALTIME_DL]
                self.interarrival_out[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_INTERARRIVALTIME_UL]

                # simulation for if have the actual labels : QOE MODEL
                self.latency_binary[j]= self.df_node_selected_rows[j].at[step, DF_COLUMN_LATENCY_BINARY]
                self.jerkiness_binary[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_JERKINESS_BINARY]
                self.sync_binary[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_SYNC_BINARY]

                # variable for the actual reported value
                self.latency_q[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_LATENCY]
                self.jerkiness_q[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_JERKINESS]
                self.sync_q[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_SYNC]

                # Update values
                # self.rtt[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_RTT_Q90]
                # self.ul[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_UL]
                # self.dl[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_DL]
                # self.latency[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_LATENCY]
                # self.jitter[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_JITTER]

                logging.info("[update_network_values] Node: {} |"
                            "Throuput IN: {} | Throuput OUT: {} | Packetsize IN: {} |"
                            "Packetsize OUT: {} | Interarrival IN: {} | Interarrival OUT: {} | "
                            "Latency Binary: {} | Jerkiness Binary: {} | Sync Binary: {} | "
                            "Latency: {} | Jerkiness: {} | Sync: {} |".format(n + 1,
                                                                                #SERVER_TYPES[i_s],
                                                                                self.throuput_in[j],
                                                                                self.throuput_out[j],
                                                                                self.packetsize_in[j],
                                                                                self.packetsize_out[j],
                                                                                self.interarrival_in[j],
                                                                                self.interarrival_out[j],
                                                                                self.latency_binary[j],
                                                                                self.jerkiness_binary[j],
                                                                                self.sync_binary[j],
                                                                                self.latency_q[j],
                                                                                self.jerkiness_q[j],
                                                                                self.sync_q[j])
                                )
            else:
                self.throuput_in = -1
                self.throuput_out = -1
                self.packetsize_in = -1
                self.packetsize_out = -1
                self.interarrival_in = -1
                self.interarrival_out = -1

                # simulation for if have the actual labels : QOE MODEL
                self.latency_binary = -1
                self.jerkiness_binary = -1
                self.sync_binary = -1

                # variable for the actual reported value
                self.latency_q = -1
                self.jerkiness_q = -1
                self.sync_q = -1

            j += 1
        return