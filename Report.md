### Learning Algorithm
To resolve the challenge, has been used the algorithm **Multi-Agent Deep Deterministic Policy Gradient**. This method implement DDPG to be used for multiple agents at the same time, recording all their experiences in the replay buffer:

DDPG | MADDPG
------------ | -------------
![Image of DDPG](https://miro.medium.com/max/1084/1*BVST6rlxL2csw3vxpeBS8Q.png) | ![Image of MADDPG](https://programmersought.com/images/862/5709e3323ebc72a6499d52623798369e.png)

The **deep neural networks** (actor and critic) use two hidden layers (128 units and 128 units) accompanied with batch normalization and dropout. 


The **parameters** below have had the most successful results for this algorithm:

```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-3         # learning rate of the actor 
LR_CRITIC = 2e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

ACTOR_UNITS_l1 = 128    # DNN layers units
ACTOR_UNITS_l2 = 128
CRITIC_UNITS_l1 = 128
CRITIC_UNITS_l2 = 128

GAMES = 300
MAX_T = 1000
```


### Results

The agent has solved the problem after 2060 games:

<img src="https://github.com/Chulvi/DRL_Nanodegree_Collaboration_And_Competition/blob/main/images/rewards.png" width="800"></img>

```
Game 100  --->  Avg Reward: 0.02  ---> Reward: 0.0
Game 200  --->  Avg Reward: 0.0  ---> Reward: 0.0
Game 300  --->  Avg Reward: 0.0  ---> Reward: 0.0
Game 400  --->  Avg Reward: 0.01  ---> Reward: 0.0
Game 500  --->  Avg Reward: 0.01  ---> Reward: 0.0
Game 600  --->  Avg Reward: 0.02  ---> Reward: 0.0
Game 700  --->  Avg Reward: 0.0  ---> Reward: 0.0
Game 800  --->  Avg Reward: 0.0  ---> Reward: 0.0
Game 900  --->  Avg Reward: 0.0  ---> Reward: 0.0
Game 1000  --->  Avg Reward: 0.01  ---> Reward: 0.0
Game 1100  --->  Avg Reward: 0.04  ---> Reward: 0.0
Game 1200  --->  Avg Reward: 0.05  ---> Reward: 0.0
Game 1300  --->  Avg Reward: 0.1  ---> Reward: 0.1
Game 1400  --->  Avg Reward: 0.12  ---> Reward: 0.05
Game 1500  --->  Avg Reward: 0.13  ---> Reward: 0.05
Game 1600  --->  Avg Reward: 0.16  ---> Reward: 0.2
Game 1700  --->  Avg Reward: 0.17  ---> Reward: 0.1
Game 1800  --->  Avg Reward: 0.24  ---> Reward: 0.05
Game 1900  --->  Avg Reward: 0.19  ---> Reward: 0.4
---
Game 2000  --->  Avg Reward: 0.4  ---> Reward: 1.7
Game 2010  --->  Avg Reward: 0.37  ---> Reward: 0.3
Game 2020  --->  Avg Reward: 0.43  ---> Reward: 0.35
Game 2030  --->  Avg Reward: 0.48  ---> Reward: 0.0
Game 2040  --->  Avg Reward: 0.52  ---> Reward: 0.5
Game 2050  --->  Avg Reward: 0.5  ---> Reward: 0.3
Game 2060  --->  Avg Reward: 0.5  ---> Reward: 0.7
```

### Ideas for Future Work

- Try to implement different algorithms to compare results.
- Experiment with 'All time high' checkpoints to resume avoiding exaggerated dropping.
- Increase number of tennis courts.
