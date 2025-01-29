import numpy.typing as npt
import gym
import numpy as np


# elif policy == LATENCY_GREEDY:
#   action = latency_greedy_policy(num_actions, action_mask, env.default_node_types[env.node_type[node]]['latency'],
#            env.deployment_request.latency_threshold)
def latency_greedy_policy(env: gym.Env, action_mask: npt.NDArray, lat_threshold: float) -> int:
    """Returns the index of a feasible node that minimizes the access latency."""
    lat_val = np.array(
        [env.default_node_types[env.node_type[node]]['latency'] + env.processing_latency[node] for node in
         range(len(action_mask) - 1)])

    # print("[Latency-Greedy] Latency Threshold: {}".format(lat_threshold))
    # print("[Latency-Greedy] Total Latency: {}".format(lat_val))

    feasible_nodes = np.where((action_mask[:-1] == True) & (lat_val <= lat_threshold))[0]
    # print("[Latency-Greedy] Feasible nodes: {}".format(feasible_nodes))

    if len(feasible_nodes) == 0:
        return len(action_mask) - 1

    # Choose randomly from the feasible nodes
    return np.random.choice(feasible_nodes)


def access_latency_greedy_policy(env: gym.Env, action_mask: npt.NDArray) -> int:
    """Returns the index of a feasible node that minimizes the access latency."""
    feasible_nodes = np.argwhere(action_mask[:-1] == True).flatten()
    # print("[Access-Latency-Greedy] Feasible nodes: {}".format(feasible_nodes))

    if len(feasible_nodes) == 0:
        return len(action_mask) - 1

    access_latencies = [env.default_node_types[env.node_type[node]]['latency'] for node in feasible_nodes]
    # print("[Access-Latency-Greedy] Access Latency of feasible nodes: {}".format(access_latencies))

    return feasible_nodes[np.argmin(access_latencies)]


def cost_greedy_policy(env: gym.Env, action_mask: npt.NDArray, ) -> int:
    """Returns the index of a feasible node that minimizes the deployment cost."""
    feasible_nodes = np.argwhere(action_mask[:-1] == True).flatten()
    # print("[Cost-Greedy] Feasible nodes: {}".format(feasible_nodes))

    # Get the endpoint with the lowest deployment cost
    if len(feasible_nodes) == 0:
        return len(action_mask) - 1

    # mean deployment cost based on node type
    deployment_costs = [env.default_node_types[env.node_type[node]]['cost'] for node in feasible_nodes]
    # print("[Cost-Greedy] Deployment Cost of feasible nodes: {}".format(deployment_costs))
    # deployment_costs = [cost if cost is not None else float('inf') for cost in deployment_costs]  # replace None with infinity

    return feasible_nodes[np.argmin(deployment_costs)]


def cpu_greedy_policy(env: gym.Env, action_mask: npt.NDArray, ) -> int:
    """Returns the index of the feasible node with the highest free CPU"""
    feasible_nodes = np.argwhere(action_mask[:-1] == True).flatten()
    # print("[CPU-Greedy] Feasible nodes: {}".format(feasible_nodes))

    if len(feasible_nodes) == 0:
        return len(action_mask) - 1

    free_cpu = [env.free_cpu[node] for node in feasible_nodes]
    # print("[Cost-Greedy] Deployment Cost of feasible nodes: {}".format(free_cpu))
    # print("[CPU-Greedy] Return: {}".format(np.argmax(env.free_cpu[feasible_nodes])))
    return feasible_nodes[np.argmax(env.free_cpu[node] for node in feasible_nodes)]


def throughput_greedy_policy(env: gym.Env, action_mask: npt.NDArray) -> int:
    """Returns the index of a feasible node that maximizes the throughput."""
    feasible_nodes = np.argwhere(action_mask[:-1] == True).flatten()
    print("[Throughput-Greedy] Feasible nodes: {}".format(feasible_nodes))

    if len(feasible_nodes) == 0:
        return len(action_mask) - 1

    print("[Throughput-Greedy] Return: {}".format(np.argmax(env.throuput_out[feasible_nodes])))
    return feasible_nodes[np.argmin(env.throuput_out[node] for node in feasible_nodes)]
