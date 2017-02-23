# Creating RELAAX client for OpenAI Gym CartPole

This tutorial is step by step introduction on how to simple RELAAX client. The purpose of the client is to connect existing Reinforcement Learning environment to RELAAX server. In this tutuorial we are going to learn policy gradient to play OpenAI Gym CartPole enviroment.

Before we start our client we need to configure RELAAX server to train agent. Lets start from Procfile. Create new empty training directory, open new Terminal window and navigate there. Then create Procfile to start RELAAX server locally:
```bash
ps: PYTHONUNBUFFERED=true relaax-parameter-server --config config.yaml
rlx: PYTHONUNBUFFERED=true relaax-rlx-server --config config.yaml
tb: PYTHONUNBUFFERED=true tensorboard --logdir metrics
```
The file lists three server processes: parameter server to accumulate agent experience, RLX server to run learning agent (agents), TensorBoard to show learning progress.

Next file is to configure RELAAX server. config.yaml:
```yaml
---
relaax-parameter-server:
  --bind: localhost:7000
  --checkpoint-dir: checkpoints
  --checkpoint-global-step-interval: 1000
  --checkpoints-to-keep: 2
  --metrics-dir: metrics

relaax-rlx-server:
  --bind: localhost:7001
  --parameter-server: localhost:7000

algorithm:
  path: ../../algorithms/policy_grad  # use policy gradient algorithm from RELAAX repo

  action_size: 2                      # action size for the given environment (CartPole:2->1)
  state_size: 4                       # size of the input observation (flattened)
  hidden_layer_size: 10               # size of the hidden layer for simple FC-NN
  batch_size: 5                       # how many steps perform before a param update
  max_global_step: 1e8                # maximum global step to stop the training when it is reached

  learning_rate: 1e-4                 # learning rate which we use through whole training
  entropy_beta: 0.01                  # entropy regularization constant
  rewards_gamma: 0.99                 # discount factor for rewards

  RMSProp:
    decay: 0.99                       # decay parameter for RMSProp
    epsilon: 1e-5                     # epsilon parameter for RMSProp (in a denominator sum to avoid NaN)
```

