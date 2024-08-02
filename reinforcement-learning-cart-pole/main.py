#%% [markdown]
# ## Problem Statement
#
# Write a deep reinforcement learning program to learn a policy network $\pi_\theta(s, a)$ to regulate the angular position of an inverted pendulum on a cart to be vertical. 
# Use two networks, an actor (policy $\pi_\theta$) and a critic (value $V_\phi(s)$). 
# Use a temporal difference TD(0) method to update the networks. Use the Gymnasium [cart-pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment. 
# Note that this has a discrete action space, able to push the cart with a force a unit leftward (0) or rightward (1). 
# Train for several episodes and display the results.
#
# ## Load Packages and Set Script Parameters
#
# Begin by importing the necessary packages as follows:

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gymnasium as gym
import tensorflow as tf
import time
import animation  # Import the animation module (animation.py)

#%% [markdown]
# Set script parameters as follows:

#%%
retrain = True  # Retrain the policy and value networks
save = True  # Save the rewards and policy and value networks
visualize = False  # TODO fix True Visualize the cart/pole environment
file_rewards = 'rewards.npy'  # File to save the rewards
file_policy = 'policy_network.keras'  # File to save the policy network
file_value = 'value_network.keras'  # File to save the value network
n_episodes = 300  # Number of episodes to train from

#%% [markdown]
# ## Set Up the Cart-Pole Environment
#
# Set up the cart-pole environment as follows:

#%%
if visualize:
    env = gym.make('CartPole-v1', render_mode='human')  # Create environment
else:
    env = gym.make('CartPole-v1', render_mode=None)  # Create environment
env.reset()  # Reset the environment to the initial state

#%% [markdown]
# Print out the action space and observation space:

#%%
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

#%% [markdown]
# Test the environment by taking random actions for an episode and visualizing the results:

#%%
for i in range(20):
    if visualize:
        time.sleep(0.03)  # Pause loop for rendering
    s_new, r_new, term, trunc, info = env.step(
        env.action_space.sample()
    )  # Take a random action
    print("Step", i, "State", s_new, "Reward", r_new)
    if term:
        break

#%% [markdown]
# ## Define the Policy and Value Networks and their Optimizers
#
# Define a function that defines the architecture and compiles the policy network as follows:

#%%
def policy_network_com(optimizer):
    """Define and compile the policy network.

    Input layer: 32 units, ReLU activation, input shape of observation space
    Output layer: 2 units, softmax activation, output shape of action space

    Returns:
    Compiled policy network
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(env.observation_space.shape))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(env.action_space.n, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    return model

#%% [markdown]
# Define a function that defines the architecture and compiles the value network as follows:

#%%
def value_network_com(optimizer):
    """Define and compile the value network.
    
    Input layer: 32 units, ReLU activation, input shape of observation space
    Output layer: 1 unit, linear activation, output shape of 1 (value)
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(env.observation_space.shape))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

#%% [markdown]
# Define the optimizers for the policy and value networks:

#%%
optimizer_policy = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer_value = tf.keras.optimizers.Adam(learning_rate=0.001)

#%% [markdown]
# Define the policy and value networks:

#%%
policy_network = policy_network_com(optimizer_policy)
value_network = value_network_com(optimizer_value)

#%% [markdown]
# ## Train the Policy and Value Networks
#
# Define a function to update the policy and value networks using a temporal difference TD(0) method as follows:

#%%
def updater(policy_network, value_network, state, action, state_next, reward, gamma):
    """Update the policy and value networks using a temporal difference TD(0) method.

    Args:
    policy_network (keras.models.Sequential): Policy network
    value_network (keras.models.Sequential): Value network
    state (numpy.ndarray): State
    action (int): Action
    state_next (numpy.ndarray): Next state
    reward (float): Reward at old state
    gamma (float): Discount factor

    Returns:
    policy_network: Updated policy network
    value_network: Updated value network
    """
    with tf.GradientTape(persistent=True) as tape:
        tf.keras.utils.disable_interactive_logging()  # Suppress bars
        state = state.reshape(1, -1)  # 2D array
        state_next = state_next.reshape(1, -1)  # 2D array

        # TD(0) error
        value = value_network(state)  # Value of the current state
        value_next = value_network(state_next)  # Value of the next state
        td_target = reward + gamma * value_next  # TD(0) target
        td_error = td_target - value  # TD(0) error

        # Losses
        action_prob = policy_network(state)[0, action]  # Prob of action taken
        log_prob = tf.math.log(action_prob)  # Log prob of action taken (negative)
        policy_loss = -log_prob * td_error  # Policy loss
        value_loss = tf.square(td_error)  # Value loss (MSE)

        # Gradients
        policy_grads = tape.gradient(
            policy_loss, policy_network.trainable_variables
        )  # Gradient of the loss function wrt policy network parameters
        value_grads = tape.gradient(
            value_loss, value_network.trainable_variables
        )  # Gradient of the loss function wrt value network parameters

        # Clip gradients
        policy_grads = tf.clip_by_global_norm(policy_grads, 10.0)[0]
        value_grads = tf.clip_by_global_norm(value_grads, 10.0)[0]

        # Apply gradients
        optimizer_policy.apply_gradients(
            zip(policy_grads, policy_network.trainable_variables)
        )  # Update the policy network parameters
        optimizer_value.apply_gradients(
            zip(value_grads, value_network.trainable_variables)
        )  # Update the value network parameters

    return policy_network, value_network

#%% [markdown]
# In `animation.py`, we define two animation classes to monitor the training.
# The first, `Angle_policy_animation`, plots a sample of the policy network for the pole (pendulum) angle with all other states set to 0.
# Our intuition is that the policy network should output a higher probability of pushing the cart in the direction that will bring the pole closer to the vertical position (i.e., leftward if the pole is leaning left and rightward if the pole is leaning right).
# The second, `Angle_value_animation`, plots a sample of the value network for the pole (pendulum) angle with all other states set to 0.
# Here our intuition is that the value network should output a higher value for the pole angle closer to the vertical position.
#
# We are now ready to define a function to train the policy and value networks using a temporal difference TD(0) method (i.e., the `updater()` function we defined) as follows:

#%%
def train_policy_value_networks(
        env, policy_network, value_network, updater, 
        n_episodes=1000, gamma=0.99, max_episode_steps=100, 
        plot=False, print_=False
):
    """Train policy and value networks using a temporal difference TD(0) method

    Update the policy and value networks using the updater() function.

    Args:
    policy_network (keras.models.Sequential): Policy network
    value_network (keras.models.Sequential): Value network
    n_episodes (int): Number of episodes to train from
    gamma (float): Discount factor
    max_episode_steps (int): Maximum number of steps per episode

    Returns:
    policy_network: Trained policy network
    value_network: Trained value network
    rewards: Numpy array of episode rewards
    """
    rewards = []
    if plot:
        animation_policy = animation.Angle_policy_animation(policy_network).animate()
        animation_value = animation.Angle_value_animation(value_network).animate()
    for episode in range(n_episodes):
        if plot:
            animation_policy.policy_network = policy_network
            plt.figure(animation_policy._fig.number)  # Activate figure
            plt.pause(0.01)  # Pause for rendering
            animation_value.value_network = value_network
            plt.figure(animation_value._fig.number)  # Activate figure
            plt.pause(0.01)  # Pause for rendering
        state, _ = env.reset()  # Reset env (random initial state)
        episode_rewards = []  
        for _ in range(max_episode_steps):
            prob = policy_network(state.reshape((1, -1))).numpy()[0]
                # Get action probabilities from policy network
            action = np.random.choice(env.action_space.n, p=prob)
                # Sample action from policy network
            state_new, reward, term, trunc, _ = env.step(action)  # Enact
            episode_rewards.append(reward)
            policy_network, value_network = updater(
                policy_network, value_network, 
                state, action, state_new, 
                reward, gamma
            )  # Update the policy and value networks
            if term or trunc:
                break
            state = state_new
        if print_:
            print(f"Episode {episode + 1}/{n_episodes}. "
                  f"Reward sum: {sum(episode_rewards)}"
                  f"\t{int(sum(episode_rewards)/500*100)*'='}"
            )
        rewards.append(sum(episode_rewards))
    return policy_network, value_network, rewards

#%% [markdown]
# Train the policy and value networks

#%%
if retrain:
    policy_network, value_network, rewards = train_policy_value_networks(
        env, policy_network, value_network, updater, 
        n_episodes=n_episodes, gamma=0.99, max_episode_steps=500, 
        plot=False, print_=True
    )
    if save:
        policy_network.save(file_policy)  # Save the policy network
        value_network.save(file_value)  # Save the value network
        np.save(file_rewards, rewards)  # Save the rewards
else:
    policy_network = tf.keras.models.load_model(file_policy)
    value_network = tf.keras.models.load_model(file_value)
    rewards = np.load(file_rewards)

#%% [markdown]
# ## Plot the Results
#
# Plot the rewards as follows:

#%%
episodes = np.arange(len(rewards))
plt.figure()
plt.plot(episodes, rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.draw()

#%% [markdown]
# This plot shows the reward obtained in each episode. 
# The reward is the sum of the rewards obtained in each step of the episode (i.e., how long the pole was kept upright).
# Because our method is on-policy, the reward is expected to increase over time as the policy network improves.
# We observe that on average the reward does in fact increase over time.
#
# Plot a sample of the policy network for the pole (pendulum) angle as follows:

#%%
fig, ax = plt.subplots()
angles = np.linspace(-0.418, 0.418, 101)  # Pole angles
probs_left = np.zeros((101))  # Probability of pushing left
probs_right = np.zeros((101))  # Probability of pushing right
for i in range(len(angles)):
    state = np.array([0, 0, angles[i], 0])  # State
    prob = policy_network(state.reshape((1, -1))).numpy()[0]  # Probabilities
    probs_left[i] = prob[0]
    probs_right[i] = prob[1]
ax.bar(angles, probs_left, width=0.005, color='b')
ax.bar(angles, probs_right, width=0.005, bottom=probs_left, color='r')
ax.set_xlabel('Pole Angle')
ax.set_ylabel('Probability')
ax.legend(['Push Left', 'Push Right'])
plt.draw()

#%% [markdown]
# This plot shows the probability of pushing the cart left or right for different pole angles.
# We observe that the policy network outputs a higher probability of pushing the cart in the direction that will bring the pole closer to the vertical position (i.e., leftward if the pole is leaning left and rightward if the pole is leaning right).
# This is consistent with our intuition of an optimal policy.
#
# Plot a sample of the value network for the pole (pendulum) angle as follows:

#%%
fig, ax = plt.subplots()
angles = np.linspace(-0.418, 0.418, 101)  # Pole angles
values = np.zeros((101))  # Value
for i in range(len(angles)):
    state = np.array([0, 0, angles[i], 0])  # State
    values[i] = value_network(state.reshape(1, -1)).numpy()[0][0]
ax.plot(angles, values)
ax.set_xlabel('Pole Angle')
ax.set_ylabel('Value')
plt.show()

#%% [markdown]
# This plot shows the value of the pole angle.
# We observe that the value network outputs a higher value for the pole angle closer to the vertical position.
# This is consistent with our intuition of an optimal value function.