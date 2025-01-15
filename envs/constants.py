# Actions - for printing purposes
ACTIONS = ["Deploy-Node", "Reject"]

# Reward objectives
# NAIVE Strategy: +1 if agent accepts request, or -1 if rejects it (if resources were available)
NAIVE = 'naive'

# Cost defaults
MAX_COST = 16  # Defined based on the max cost in DEFAULT_CLUSTER_TYPES
MIN_COST = 1  # Defined based on the min cost in DEFAULT_CLUSTER_TYPES

# Multi-objective reward function
MULTI = 'multi'

# Node Types
# Cluster Types
NUM_NODE_TYPES = 4
DEFAULT_NODE_TYPES = [{"type": "edge_tier_1", "cpu": 2.0, "mem": 2.0, "cost": 1, "latency": 1},
                      {"type": "edge_tier_2", "cpu": 2.0, "mem": 4.0, "cost": 2, "latency": 2.5},
                      {"type": "fog_tier_1", "cpu": 2.0, "mem": 8.0, "cost": 4, "latency": 5.0},
                      {"type": "fog_tier_2", "cpu": 4.0, "mem": 16.0, "cost": 8, "latency": 7.5},]
                      #{"type": "cloud", "cpu": 8.0, "mem": 32.0, "cost": 16, "latency": 10.0}]

# DEFAULT_NODE_TYPES = [{"type": "edge", "cpu": 4.0, "mem": 4.0, "cost": 1},  # pc-engine and celerway
#                     {"type": "fog", "cpu": 8.0, "mem": 16.0, "cost": 4},
#                     {"type": "cloud", "cpu": 12.0, "mem": 32.0, "cost": 8}]

# DEFAULTS for Env configuration
DEFAULT_NUM_EPISODE_STEPS = 50
DEFAULT_NUM_NODES = 4
DEFAULT_ARRIVAL_RATE = 100
DEFAULT_CALL_DURATION = 1
DEFAULT_REWARD_FUNTION = NAIVE
DEFAULT_FILE_NAME_RESULTS = "nne_gym_results"

# Computing metrics: 4 metrics = CPU capacity, memory capacity, CPU allocated, memory allocated
# Dataset metrics: 6 metrics = provider id, interface id, ul_mbps, dl_mbps, jitter, rtt
# Processing latency added = processing_latency

# Other to consider: Latency removed
NUM_METRICS_NODES = 7

# Computing metrics: 3 metrics = cpu_request, memory_request, latency_threshold
# Bandwidth requirements? ul_traffic + dl_traffic?
# sim variables: dt
NUM_METRICS_REQUEST = 4

NUM_SERVER_TYPE = 4 # A, B, C, D
A_CSV = 1
B_CSV = 2
C_CSV = 3
D_CSV = 4

SERVER_TYPES = ["A", "B", "C", "D"]
SERVER_TYPE = "Config"
# Defaults
# Adjusted based on min/max values of dataset
MIN_RTT = 0.0  # corresponds to 0.0 ms v1: 0 - 30/v2: 0 - 40
MAX_RTT = 40.0  # corresponds to 30.0 ms

MIN_LATENCY = 1.0  # corresponds to min access latency of node - 1.0
MAX_LATENCY = 10.0  # corresponds to max access latency of node - 10.0

MIN_JITTER = 0.0  # v1: 0 - 513/v2: 0 - 229
MAX_JITTER = 229.0

MIN_UL = 0.0  # v1: 0 - 90/v2: 0 - 93
MAX_UL = 1000.0

MIN_DL = 1.0  # v1: 1 - 620/v2: 1 - 516
MAX_DL = 1000.0

MIN_PKT_LOSS_RATE = 0.0
MAX_PKT_LOSS_RATE = 1.0

MIN_OBS = 0.0
MAX_OBS = 1000.0

PROCESSING_DELAY = 2.0  # 2.0 ms
MIN_PROC = 0.0
MAX_PROC = 200.0  # 2.0 * 100 steps = 200.0 ms

# Dataframe column names For OBJECTIVE assessment
DF_COLUMN_LATENCY = "Latency"
DF_COLUMN_LATENCY_BINARY = "Latency_for_agent"

DF_COLUMN_JERKINESS = "Jerkiness"
DF_COLUMN_JERKINESS_BINARY = "Jerkiness_for_agent"

DF_COLUMN_SYNC = "Sync"
DF_COLUMN_SYNC_BINARY = "Sync_for_agent"

DF_COLUMN_THROUPUT_DL = "throuput_mean_in"
DF_COLUMN_THROUPUT_UL = "throuput_mean_out"
DF_COLUMN_PACKETSIZE_DL = "avg_packet_size_in"
DF_COLUMN_PACKETSIZE_UL = "avg_packet_size_out"
DF_COLUMN_INTERARRIVALTIME_DL = "inter_arrival_times_avg_in"
DF_COLUMN_INTERARRIVALTIME_UL = "inter_arrival_times_avg_out"

# Dataframe column names For PHYSIOLOGICAL assessment

# DF_COLUMN_PKT_LOSS_RATE = "pkt_loss_rate"
# DF_COLUMN_RTT_AVG = "rtt_avg"
# # DF_COLUMN_RTT_MEDIAN = "rtt_median"
# # DF_COLUMN_RTT_STD = "rtt_std"
# DF_COLUMN_RTT_Q90 = "throuput_dl"
# DF_COLUMN_RSSI = "throuput_ul"
# DF_COLUMN_RSRQ = "packetsize_dl"
# DF_COLUMN_RSRP = "packetsize_ul"
# DF_COLUMN_UL = "inter_arrival_dl"
# DF_COLUMN_DL = "inter_arrival_ul"
# #DF_COLUMN_LATENCY = "latency"
# DF_COLUMN_JITTER = "inter_arrival_ul"

# Defaults for Weights
LATENCY_WEIGHT = 0.0
GINI_WEIGHT = 0.0
COST_WEIGHT = 0.0
BANDWIDTH_WEIGHT = 0.0
QOE_WEIGHT = 1.0

FACTOR = 1.0
SEED = 42
PATH_CSV_FILES = "./mydata/"

# Adjusted parameter for GYM_QoE
MIN_SIZE = 72
