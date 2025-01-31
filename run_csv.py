import logging
import argparse

import re
import matplotlib
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO, MaskablePPO, TRPO, TQC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from envs.nne_scheduling_env import NNESchedulingEnv
from envs.ppo_deepset import PPO_DeepSets
from envs.dqn_deepset import DQN_DeepSets
from sb3_contrib.common.maskable.utils import get_action_masks
import pandas as pd

matplotlib.use('TkAgg')

# Logging
logging.basicConfig(filename='run.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description='Run RL Agent!')
parser.add_argument('--alg', default='mask_ppo',
                    help='The algorithm: ["ppo_deepsets", "recurrent_ppo", "ppo", "mask_ppo", "ppo_deepsets", "dqn_deepsets", "trpo", "tqc"]')
parser.add_argument('--env_name', default='nne', help='Env: ["nne"]')
parser.add_argument('--num_nodes', default=4, help='num_nodes: 4, 6, 8, etc')
parser.add_argument('--reward', default='multi', help='reward: ["naive", "risk", "cost", "latency"]')
parser.add_argument('--training', default=False, action="store_true", help='Training mode')
parser.add_argument('--testing', default=True, action="store_true", help='Testing mode')

parser.add_argument('--loading', default=False, action="store_true", help='Loading mode')
parser.add_argument('--load_path',
                    default='./results/nne/multi/mask_ppo_env_nne_num_nodes_4_reward_multi_totalSteps_200000_run_1/mask_ppo_env_nne_num_nodes_4_reward_multi_totalSteps_200000.zip',
                    help='Loading path, ex: logs/model/test.zip')
parser.add_argument('--test_path',
                    #default='./results/nne/multi/mask_ppo_env_nne_num_nodes_4_reward_multi_totalSteps_200000_run_1/mask_ppo_env_nne_num_nodes_4_reward_multi_totalSteps_200000.zip',
                    help='Testing path, ex: logs/model/test.zip')
parser.add_argument('--steps', default=200000, help='Save model after X steps')
parser.add_argument('--total_steps', default=200000, help='The total number of steps.')

parser.add_argument('--qoe', default=False, help='If qoe estimation is present in the observation space.')
parser.add_argument('--objective', default=False, help='If objective features are present in the observation space.')
parser.add_argument('--simulation_mode', default="Simulation", help='Simulation mode : Real or Simulation. Default: Simulation.')
parser.add_argument('--qoe_accuracy', default= 1.0 , help='qoe model accuracy simulation')
parser.add_argument('--test_trained_model', default=False, action="store_true", help='Testing mode')



# TODO: add other arguments if needed
# parser.add_argument('--k8s', default=False, action="store_true", help='K8s mode')
# parser.add_argument('--goal', default='cost', help='Reward Goal: ["cost", "latency"]')

args = parser.parse_args()

TESTING_FACTORS = False


def get_model(alg, env, tensorboard_log):
    model = 0
    if alg == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
    elif alg == 'a2c':
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)  # , n_steps=steps
    elif alg == 'mask_ppo':
        model = MaskablePPO("MlpPolicy", env, gamma=0.95, verbose=1, tensorboard_log=tensorboard_log)  # , n_steps=steps
    elif alg == 'ppo_deepsets':
        model = PPO_DeepSets(env, num_steps=100, n_minibatches=8, ent_coef=0.001, tensorboard_log=None, seed=2)
    elif alg == 'trpo':
        model = TRPO("MlpPolicy", env, verbose=1, tensorboard_log=None)
    elif alg == 'tqc':
        policy_kwargs = dict(n_critics=2, n_quantiles=25)
        model = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=1, policy_kwargs=policy_kwargs)
    elif alg == 'dqn_deepsets':
        model = DQN_DeepSets(env, num_steps=100, n_minibatches=8, tensorboard_log=None)
    else:
        logging.info('Invalid algorithm!')

    return model


def get_load_model(env, alg, tensorboard_log, load_path):
    print("tesnor_path: ", load_path)
    print("tesnsor Log : ",tensorboard_log)
    if alg == 'ppo':
        return PPO.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        return RecurrentPPO.load(load_path, reset_num_timesteps=False, verbose=1,
                                 tensorboard_log=tensorboard_log)  # n_steps=steps
    elif alg == 'a2c':
        return A2C.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log)
    elif alg == 'mask_ppo':
        return MaskablePPO.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log)
    elif alg == 'ppo_deepsets':
        agent = PPO_DeepSets(env, tensorboard_log=None)
        return agent.load(f"" + load_path)
    elif alg == 'dqn_deepsets':
        agent = DQN_DeepSets(env, tensorboard_log=None)
        return agent.load(f"" + load_path)
    elif alg == 'trpo':
        agent = TRPO.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log)
        return agent.load(f"" + load_path)
    elif alg == 'tqc':
        agent = TQC.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log)
        return agent.load(f"" + load_path)
    else:
        logging.info('Invalid algorithm!')


def get_env(env_name, num_nodes, reward_function, qoe, objective, simulation_mode, qoe_accuracy, latency_weight, cost_weight, gini_weight, qoe_weight,file_results_name):
    envs = 0
    # latency_weight = 0.0
    # cost_weight = 0.0
    # gini_weight = 0.0
    # bandwidth_weight = 1.0
    factor = 1
    path = "mydata/"


    if env_name == "nne":
        env = NNESchedulingEnv(num_nodes=num_nodes, arrival_rate_r=100, call_duration_r=1,
                               episode_length=100,
                               reward_function=reward_function,
                               latency_weight=latency_weight,
                               cost_weight=cost_weight,
                               gini_weight=gini_weight,
                               qoe_weight=qoe_weight,
                               factor=factor,
                               path_csv_files=path,
                               qoe_in_observation=qoe,
                               objective_feature_in_observation=objective,
                               qoe_simulation_mode=simulation_mode,
                               qoe_simulated_accuracy=qoe_accuracy,
                               file_results_name=file_results_name,
                               )
        # For faster training!
        # otherwise just comment the following lines

        env.reset()
        _, _, _, info = env.step(0)
        info_keywords = tuple(info.keys())

        env = SubprocVecEnv([lambda: NNESchedulingEnv(num_nodes=num_nodes, arrival_rate_r=100,
                                                      call_duration_r=1, episode_length=100,
                                                      reward_function=reward_function,
                                                      latency_weight=latency_weight,
                                                      cost_weight=cost_weight,
                                                      gini_weight=gini_weight,
                                                      qoe_weight=qoe_weight,
                                                      factor=factor,
                                                      path_csv_files=path,
                                                      qoe_in_observation=qoe,
                                                      objective_feature_in_observation=objective,
                                                      qoe_simulation_mode=simulation_mode,
                                                      qoe_simulated_accuracy=qoe_accuracy,
                                                      file_results_name=file_results_name
                                                      )
                             for i in range(1)])
        envs = VecMonitor(env, filename="vec_nne_gym_results", info_keywords=info_keywords)


    else:
        logging.info('Invalid environment!')

    return envs


def test_model(model, env, n_episodes, n_steps, smoothing_window, fig_name):
    episode_rewards = []
    reward_sum = 0
    obs = env.reset()

    print("------------Testing -----------------")
    for e in range(n_episodes):
        for step in range(n_steps):
            action_masks = get_action_masks(env)
            # print(f"Step: {step}, Action Masks: {action_masks}")

            action, _ = model.predict(obs, action_masks=action_masks)
            obs, reward, done, info = env.step(action)
            reward_sum += float(reward)

            # print(f"Action: {action}, Reward: {reward}, Done: {done}")

            if done:
                episode_rewards.append(reward_sum)
                print(f"Episode {e} | Total reward: {reward_sum}")
                reward_sum = 0
                obs = env.reset()
                break

    env.close()

    # Free memory
    del model, env

    # Plot the episode reward over time
    '''
    plt.figure()
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(fig_name, dpi=250, bbox_inches='tight')
    '''


def main():
    # Parameter ranges
    # Load configuration from CSV
    config_path = './execution_config.csv'
    config_data = pd.read_csv(config_path)
    config_data.drop_duplicates(inplace=True)
    # Base parameters
    alg = args.alg
    env_name = args.env_name
    reward = args.reward
    num_nodes = int(args.num_nodes)
    loading = args.loading
    load_path = args.load_path
    training = args.training
    testing = args.testing
    test_path = args.test_path
    steps = int(args.steps)
    total_steps = int(args.total_steps)
    test_trained_model = args.test_trained_model

    if training:
        # Loop through configurations from CSV
        for _, config in config_data.iterrows():
            obj_included = bool(config['obj_included'])
            qoe_included = bool(config['qoe_included'])
            cost_weight = config['cost_weight']
            qoe_weight = config['qoe_weight']
            qoe_simulation_mode = config['qoe_simulation_mode']
            qoe_accuracy = config['qoe_accuracy']
            latency_weight = config['latency_weight']
            gini_weight = config['gini_weight']

            # Create a dynamic name for each configuration
            name = f"{alg}_env_{env_name}_qoe_{qoe_included}_obj_{obj_included}_cw_{cost_weight}_qw_{qoe_weight}_lw_{latency_weight}_gw_{gini_weight}_sim_{qoe_simulation_mode}_acc_{qoe_accuracy}"

            print("alg : ", alg, "env_name : ", env_name, "cost_weight : ", cost_weight, "qoe_weight : ", qoe_weight, "qoe_simulation_mode : ", qoe_simulation_mode, "qoe_accuracy : ", qoe_accuracy, "latency_weight : ", latency_weight, "gini_weight : ", gini_weight, "qoe_included : ", qoe_included, "obj_included : ", obj_included)

            # Create environment with current parameters
            env = get_env(
                env_name=env_name,
                num_nodes=num_nodes,
                reward_function=reward,
                qoe=qoe_included,
                objective=obj_included,
                simulation_mode=qoe_simulation_mode,
                qoe_accuracy=qoe_accuracy,
                latency_weight=latency_weight,
                cost_weight=cost_weight,
                gini_weight=gini_weight,
                qoe_weight=qoe_weight,
                file_results_name='./run_metrics/' + name + '.csv'
            )

            tensorboard_log = f"./results/{env_name}/{reward}/" \
                              f"qoe_{qoe_included}_obj_{obj_included}_cw_{cost_weight}_qw_{qoe_weight}_lw_{latency_weight}_gw_{gini_weight}_sim_{qoe_simulation_mode}_acc_{qoe_accuracy}/"

            checkpoint_callback = CheckpointCallback(
                save_freq=steps,
                save_path=f"logs/{name}",
                name_prefix=name
            )

            # Training logic
            if training:
                model = get_model(alg, env, tensorboard_log)
                model.learn(
                    total_timesteps=total_steps,
                    tb_log_name=f"{name}_run",
                    callback=checkpoint_callback
                )
                model.save("./models/" + name)

            # Testing logic
            if testing:
                # Create environment with current parameters
                env = get_env(
                    env_name=env_name,
                    num_nodes=num_nodes,
                    reward_function=reward,
                    qoe=qoe_included,
                    objective=obj_included,
                    simulation_mode=qoe_simulation_mode,
                    qoe_accuracy=qoe_accuracy,
                    latency_weight=latency_weight,
                    cost_weight=cost_weight,
                    gini_weight=gini_weight,
                    qoe_weight=qoe_weight,
                    file_results_name='./run_metrics_test/' + name + '.csv'
                )


                model = get_load_model(env, alg, tensorboard_log, "./models/" + name)
                test_model(
                    model, env,
                    n_episodes=100,
                    n_steps=100,
                    smoothing_window=5,
                    fig_name=f"{name}_test_reward.png"
                )
    if test_trained_model:
        # Define the pattern to extract values
        #pattern = r"(?P<alg>.+)_env_(?P<env_name>.+)_cw_(?P<cost_weight>.+)_qw_(?P<qoe_weight>.+)_sim_(?P<qoe_simulation_mode>.+)_acc_(?P<qoe_accuracy>.+)"
        #pattern = r"(?P<alg>.+)_env_(?P<env_name>.+)_qoe_(?P<qoe_included>.+)_obj_(?P<obj_included>.+)_cw_(?P<cost_weight>.+)_qw_(?P<qoe_weight>.+)_sim_(?P<qoe_simulation_mode>.+)_acc_(?P<qoe_accuracy>.+)"
        pattern = r"(?P<alg>.+)_env_(?P<env_name>.+)_qoe_(?P<qoe_included>.+)_obj_(?P<obj_included>.+)_cw_(?P<cost_weight>.+)_qw_(?P<qoe_weight>.+)_lw_(?P<latency_weight>.+)_gw_(?P<gini_weight>.+)_sim_(?P<qoe_simulation_mode>.+)_acc_(?P<qoe_accuracy>.+)"

        name = test_path.split('/')[-1]
        # Match the pattern
        match = re.match(pattern, name)

        if match:
            # Extract the values
            values = match.groupdict()
            alg = values["alg"]
            env_name = values["env_name"]
            cost_weight = float(values["cost_weight"])
            qoe_weight = float(values["qoe_weight"])
            qoe_simulation_mode = values["qoe_simulation_mode"]
            qoe_accuracy = float(values["qoe_accuracy"])
            latency_weight = float(values["latency_weight"])
            gini_weight = float(values["gini_weight"])
            qoe_included = bool(values["qoe_included"])
            obj_included = bool(values["obj_included"])
            print("alg : ", alg, "env_name : ", env_name, "cost_weight : ", cost_weight, "qoe_weight : ", qoe_weight, "qoe_simulation_mode : ", qoe_simulation_mode, "qoe_accuracy : ", qoe_accuracy, "latency_weight : ", latency_weight, "gini_weight : ", gini_weight, "qoe_included : ", qoe_included, "obj_included : ", obj_included)
            # Create environment with current parameters
            env = get_env(
                env_name=env_name,
                num_nodes=num_nodes,
                reward_function=reward,
                qoe=qoe_included,
                objective=obj_included,
                simulation_mode=qoe_simulation_mode,
                qoe_accuracy=qoe_accuracy,
                latency_weight=latency_weight,
                cost_weight=cost_weight,
                gini_weight=gini_weight,
                qoe_weight=qoe_weight,
                file_results_name='./run_metrics_test/' + name + '.csv'
            )

            tensorboard_log = f"./results/{env_name}/{reward}/" \
                              f"qoe_{qoe_included}_obj_{obj_included}_cw_{cost_weight}_qw_{qoe_weight}_lw_{latency_weight}_gw_{gini_weight}_sim_{qoe_simulation_mode}_acc_{qoe_accuracy}/"

            checkpoint_callback = CheckpointCallback(
                save_freq=steps,
                save_path=f"logs/{name}",
                name_prefix=name
            )

            model = get_load_model(env, alg, tensorboard_log, "./models/" + name)
            test_model(
                model, env,
                n_episodes=100,
                n_steps=100,
                smoothing_window=5,
                fig_name=f"{name}_test_reward.png"
            )
        else:
            raise ValueError("Invalid name pattern.")


if __name__ == "__main__":
    main()
