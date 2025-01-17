import csv
import os
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pandas as pd
import logging


# DeploymentRequest Info
@dataclass
class DeploymentRequest:
    name: str
    cpu_request: float
    cpu_limit: float  # limits can be left out
    memory_request: float
    memory_limit: float
    arrival_time: float
    latency_threshold: int  # Latency threshold that should be respected
    # ul_traffic: int # Expected Uplink traffic
    # dl_traffic: int # Expected downlink traffic
    departure_time: float
    action_id: int = None  # action id
    deployed_provider: int = None

    deployed_node: int = None  # All replicas deployed in one cluster
    expected_cost: int = None  # expected cost after deployment
    expected_dl_bandwidth: int = None  # expected downlink bandwidth after deployment
    expected_ul_bandwidth: int = None  # expected uplink bandwidth after deployment

    expected_rtt: int = None  # expected RTT
    expected_access_latency: int = None  # expected latency based on node type
    expected_processing_latency: int = None  # expected processing latency


# Reverses a dict
def sort_dict_by_value(d, reverse=False):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))


# Normalize function
def normalize(value, min_value, max_value):
    if max_value == min_value:
        return 0.0  # Avoid division by zero if min_value equals max_value
    return (value - min_value) / (max_value - min_value)


def get_c2e_deployment_list():
    deployment_list = [
        # 1 adapter-amqp
        DeploymentRequest(name="adapter-amqp",
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.3, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),
        # 2 adapter-http
        DeploymentRequest(name="adapter-http",
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.3, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),
        # 3 adapter-mqtt
        DeploymentRequest(name="adapter-mqtt",
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.3, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),
        # 4 adapter-mqtt
        DeploymentRequest(name="artemis",
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.6, memory_limit=0.6,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),

        # 5 dispatch-router
        DeploymentRequest(name="dispatch-router",
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.06, memory_limit=0.25,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),

        # 6 ditto-connectivity
        DeploymentRequest(name="ditto-connectivity",
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.7, memory_limit=1.0,
                          arrival_time=0, departure_time=0,
                          latency_threshold=100),

        # 7 ditto-gateway
        DeploymentRequest(name="ditto-gateway",
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=100),

        # 8 ditto-nginx
        DeploymentRequest(name="ditto-nginx",
                          cpu_request=0.05, cpu_limit=0.15,
                          memory_request=0.016, memory_limit=0.032,
                          arrival_time=0, departure_time=0,
                          latency_threshold=100),

        # 9 ditto-policies
        DeploymentRequest(name="ditto-policies",
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=100),

        # 10 ditto-swagger-ui
        DeploymentRequest(name="ditto-swagger-ui",
                          cpu_request=0.05, cpu_limit=0.1,
                          memory_request=0.016, memory_limit=0.032,
                          arrival_time=0, departure_time=0,
                          latency_threshold=400),

        # 11 ditto-things
        DeploymentRequest(name="ditto-things",
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),

        # 12 ditto-things-search
        DeploymentRequest(name="ditto-things-search",
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),

        # 13 ditto-mongo-db
        DeploymentRequest(name="ditto-mongo-db",
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),

        # 14 service-auth
        DeploymentRequest(name="service-auth",
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.2, memory_limit=0.25,
                          arrival_time=0, departure_time=0,
                          latency_threshold=300),

        # 15 service-command-router
        DeploymentRequest(name="service-command-router",
                          cpu_request=0.15, cpu_limit=1.0,
                          memory_request=0.2, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=300),

        # 16 service-device-registry
        DeploymentRequest(name="service-device-registry",
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.4, memory_limit=0.4,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),
    ]
    return deployment_list


# TODO: modify function
'''
def save_obs_to_csv(file_name, timestamp, num_pods, desired_replicas, cpu_usage, mem_usage,
                    traffic_in, traffic_out, latency, lstm_1_step, lstm_5_step):
    file = open(file_name, 'a+', newline='')  # append
    # file = open(file_name, 'w', newline='') # new
    with file:
        fields = ['date', 'num_pods', 'cpu', 'mem', 'desired_replicas',
                  'traffic_in', 'traffic_out', 'latency', 'lstm_1_step', 'lstm_5_step']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()  # write header
        writer.writerow(
            {'date': timestamp,
             'num_pods': int("{}".format(num_pods)),
             'cpu': int("{}".format(cpu_usage)),
             'mem': int("{}".format(mem_usage)),
             'desired_replicas': int("{}".format(desired_replicas)),
             'traffic_in': int("{}".format(traffic_in)),
             'traffic_out': int("{}".format(traffic_out)),
             'latency': float("{:.3f}".format(latency)),
             'lstm_1_step': int("{}".format(lstm_1_step)),
             'lstm_5_step': int("{}".format(lstm_5_step))}
        )
'''

# def save_to_csv(file_name, episode,
#                 reward, ep_block_prob,
#                 ep_accepted_requests,
#                 avg_deployment_cost, avg_total_latency,
#                 avg_access_latency, avg_proc_latency,
#                 avg_throuput_in, avg_packetsize_in, avg_interarrival_in,
#                 avg_throuput_out, avg_packetsize_out, avg_interarrival_out,
#                 avg_latency_binary, avg_jerkiness_binary, avg_sync_binary,
#                 avg_qoe,
#                 gini,
#                 execution_time):
#     file = open(file_name, 'a+', newline='')  # append
#     # file = open(file_name, 'w', newline='')
#     with file:
#         fields = ['episode', 'reward', 'ep_block_prob', 'ep_accepted_requests', 'avg_deployment_cost',
#                   'avg_total_latency', 'avg_access_latency', 'avg_proc_latency',
#                   'avg_throuput_in', 'avg_packetsize_in', 'avg_interarrival_in', 'avg_throuput_out','avg_packetsize_out','avg_interarrival_out','avg_qoe','gini',
#                   'execution_time']
#         writer = csv.DictWriter(file, fieldnames=fields)
#         # writer.writeheader()
#         writer.writerow(
#             {
#                 'episode': episode,
#                 'reward': float("{:.2f}".format(reward)),
#                 'ep_block_prob': float("{:.2f}".format(ep_block_prob)),
#                 'ep_accepted_requests': float("{:.2f}".format(ep_accepted_requests)),
#                 'avg_deployment_cost': float("{:.2f}".format(avg_deployment_cost)),
#                 'avg_total_latency': float("{:.2f}".format(avg_total_latency)),
#                 'avg_access_latency': float("{:.2f}".format(avg_access_latency)),
#                 'avg_proc_latency': float("{:.2f}".format(avg_proc_latency)),
#                 'avg_throuput_in': float("{:.2f}".format(avg_throuput_in)),
#                 'avg_packetsize_in': float("{:.2f}".format(avg_packetsize_in)),
#                 'avg_interarrival_in': float("{:.2f}".format(avg_interarrival_in)),
#                 'avg_throuput_out': float("{:.2f}".format(avg_throuput_out)),
#                 'avg_packetsize_out': float("{:.2f}".format(avg_packetsize_out)),
#                 'avg_interarrival_out': float("{:.2f}".format(avg_interarrival_out)),
#                 'avg_qoe':float("{:.2f}".format(avg_qoe)),
#                 'gini': float("{:.2f}".format(gini)),
#                 'execution_time': float("{:.2f}".format(execution_time))
#              }
#         )


def save_to_csv(file_name, data):
    """
    Save data to a CSV file.

    Args:
        file_name (str): Name of the CSV file.
        data (dict): Data to be written to the CSV file as key-value pairs.
    """
    fields = [
        'episode', 'reward', 'ep_block_prob', 'ep_accepted_requests',
        'avg_deployment_cost', 'avg_total_latency', 'avg_access_latency',
        'avg_proc_latency', 'avg_throuput_in', 'avg_packetsize_in',
        'avg_interarrival_in', 'avg_throuput_out', 'avg_packetsize_out',
        'avg_interarrival_out', 'avg_latency_binary', 'avg_jerkiness_binary',
        'avg_sync_binary', 'avg_qoe', 'gini', 'execution_time'
    ]

    # Check if file exists to determine if headers need to be written
    file_exists = os.path.exists(file_name)

    try:
        with open(file_name, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            # Ensure values are formatted to 2 decimal places
            formatted_data = {key: round(value, 2) if isinstance(value, (float, int)) else value for key, value in
                              data.items()}
            writer.writerow(formatted_data)
    except IOError as e:
        print(f"Error writing to file {file_name}: {e}")

# Calculation of Gini Coefficient
# 0 is better - 1 is worse!
def calculate_gini_coefficient(loads):
    n = len(loads)
    total_load = sum(loads)
    mean_load = total_load / n if n != 0 else 0

    if mean_load == 0:
        return 0  # Handle the case where all loads are zero to avoid division by zero

    gini_numerator = sum(abs(loads[i] - loads[j]) for i in range(n) for j in range(n))
    gini_coefficient = gini_numerator / (2 * n ** 2 * mean_load)

    return gini_coefficient

def calculate_qoe(sync, jerkiness, latnecy, vrsq = 0):
    """
        Calculate QoE based on the given metrics.

        Args:
            Sync (int): Reported Sync value
            jerkiness (int): Reported Jerkiness value
            latency (int): Reported Latency value

        Returns:
            qoe (float): Calculated QoE value.
        """
    # qoe = (sync / 5.0) + (jerkiness / 5.) + (latnecy / 5.) + (vrsq / 5.)
    # return qoe / 3.
    qoe = (sync / 5.0) + (jerkiness / 5.) + (latnecy / 5.)

    return qoe / 3.0


def simulate_model(dataframe, accuracy_rate, columns):
    """
    Simulates model predictions with a given accuracy rate.

    Args:
    - dataframe (pd.DataFrame): Input dataframe containing ground truth columns.
    - accuracy_rate (float): Desired accuracy rate (0.0 to 1.0).
    - columns (list of str): List of column names to simulate.

    Returns:
    - pd.DataFrame: DataFrame with simulated predictions added as new columns.
    """
    simulated_df = dataframe.copy()

    for column in columns:
        if column not in simulated_df.columns:
            raise ValueError(f"Column {column} not found in the dataframe. Skipping simulation for this column.")

        ground_truth = simulated_df[column].values
        predictions = np.copy(ground_truth)

        correct_mask = np.zeros(len(ground_truth))
        for i_m in range(len(ground_truth)):
            if_simulated_model_predicted_correctly = np.random.choice([0, 1], 1, p=[1 - accuracy_rate, accuracy_rate])
            correct_mask[i_m] = if_simulated_model_predicted_correctly[0]

        # incorrect_mask = np.random.rand(len(ground_truth)) > accuracy_rate
        unique_labels = np.unique(ground_truth)

        for i in range(len(ground_truth)):
            # Assign a random incorrect label
            if correct_mask[i] == 0:
                incorrect_labels = unique_labels[unique_labels != ground_truth[i]]
                predictions[i] = np.random.choice(incorrect_labels)
            elif correct_mask[i] == 1:
                predictions[i] = ground_truth[i]  # No alternative labels, fallback

        # Add simulated predictions as a new column
        simulated_df[f"{column}_for_agent"] = predictions

    return simulated_df

def model_estimation(dataframe, columns):
    """
    Copies predicted values stored in columns with '_prediction' suffix into '_simulated' columns.

    Args:
    - dataframe (pd.DataFrame): Input dataframe containing prediction columns.
    - columns (list of str): List of column names without '_prediction' suffix to copy from.

    Returns:
    - pd.DataFrame: DataFrame with simulated predictions added as new columns.
    """
    estimated_df = dataframe.copy()

    for column in columns:
        prediction_column = f"{column}_prediction"
        simulated_column = f"{column}_simulated"

        if prediction_column not in estimated_df.columns:
            raise ValueError(f"Prediction column {prediction_column} not found in the dataframe.")

        # Copy prediction values to simulated column
        estimated_df[f"{column}_for_agent"]= estimated_df[prediction_column]
    print(estimated_df.shape)
    return estimated_df