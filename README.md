# gym-qoe
Code repository for the paper entitled "Reinforcement Learning-Driven Service Placement in 6G Networks across the Compute Continuum", submitted to CNSM 2025.

# Gym_QoE
📘 How to Run the RL Agents
This guide explains how to run training and testing experiments using the run_csv.py script in this repository. The script supports various reinforcement learning algorithms and configurations for the Gym-QoE environment.

📁 Prerequisites
Make sure the following packages are installed:
`pip install -r requirements.txt`


🔧 Training Example Usage :

Run the following to train a model using the configurations in execution_config.csv:

`python run_csv.py \
  --alg mask_ppo \
  --env_name nne \
  --num_nodes 4 \
  --reward multi \
  --training \
  --steps 50000 \
  --total_steps 200000`

Read multiple configurations from execution_config.csv

Train and optionally test each configuration

Save trained models to ./models/

Log results to ./logs/ and ./results/

Log the model performance to : run_metrics and run_metrics_test

📂 Output Files
Logs: ./logs/[name]/
Models: ./models/[name].zip
Metrics: ./run_metrics/*.csv, ./run_metrics_test/*.csv
TensorBoard logs: ./results/[env]/[reward]/
To view training performance:
`tensorboard --logdir ./results`

Analysing The Algorithm Performance :
To view and analyze the algorithm performance you can see the `Analysis.ipynb`. 