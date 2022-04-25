Requirements: numpy, PyTorch, wandb (only if training the Environment Model)

Directory:

* main.py  
This serves as the main entry to the program. Contains the training loop for the Environment Model (train_vae()), the training loop for the Double Deep Q Network (train_ddqn_agent()), and other support functions. Parameters can be altered in the load_parameters() function.

* networks.py  
Contains all networks for the Environment Model, including the Encoder, Decoder, NextState which predicts the next state from the current state and an action, PrevState which predicts the previous state from the current state and last action to be applied, and the VAE_Model which combines these networks, along with the loss function and BinaryConcrete class which performs the discretization.

* rl_networks.py  
Contains the DQN and DoubleDQN networks. The DoubleDQN class creates two DQN networks, one which acts as the online network and the other which acts as the target network. DoubleDQN contains function choose_action() for selecting the action, experience_replay() which samples transitions from memory and calculates the target values, training_update() which then updates the online network according to these target values, and other supporting functions.

* data.py  
Contains the SlidingTiles class which handles creating the sliding tile puzzle environment and handling action steps. All functions for interacting with the environment are within this class, for example reset(), step(), and render(), as well as a function for generating the dataset the Environment Model is trained on (generate_vae_dataset()).

Note: vae-state-dict.pt is necessary for running the reinforcement learning agent. It is obtained from running train_vae() in main, but running this training loop is unnecessary as the presaved weights are already there. Instead one may simply run the reinforcement learning agent which will load in these weights to encode the states.
