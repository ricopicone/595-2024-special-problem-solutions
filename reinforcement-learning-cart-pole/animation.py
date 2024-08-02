import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf

class Angle_policy_animation:
    """Animation of the policy network's output for different pole angles

    Args:
        policy_network (tf.keras.Model): Policy network
        n_points (int): Number of points to plot
        refresh_interval (int): Refresh interval for the animation
    """
    def __init__(self, policy_network, n_points=51, refresh_interval=1000):
        self.policy_network = policy_network
        self.n_points = n_points
        self.refresh_interval = refresh_interval
        self.fig, self.ax = plt.subplots()
        self.angles = np.linspace(-0.418, 0.418, n_points)  # Pole angles
        self.barcollection_left = None
        self.barcollection_right = None
        self.initialize_plot()

    def initialize_plot(self):
        self.fig, self.ax = plt.subplots()
        left_probs = np.zeros((self.n_points))
        right_probs = np.zeros((self.n_points))
        for i in range(self.n_points):
            state = np.array([0, 0, self.angles[i], 0])
            prob = self.policy_network(state.reshape((1, -1))).numpy()[0]
            left_probs[i] = prob[0]
            right_probs[i] = prob[1]
        self.barcollection_left = self.ax.bar(self.angles, left_probs, width=0.01, color='b')
        self.barcollection_right = self.ax.bar(self.angles, right_probs, width=0.01, bottom=left_probs, color='r')
        self.ax.set_xlabel('Pole Angle')
        self.ax.set_ylabel('Probability')
        self.ax.legend(['Push Left', 'Push Right'])
    
    def update(self, frame):
        left_probs = np.zeros((self.n_points))
        right_probs = np.zeros((self.n_points))
        for i in range(self.n_points):
            state = np.array([0, 0, self.angles[i], 0])
            prob = self.policy_network(state.reshape((1, -1))).numpy()[0]
            left_probs[i] = prob[0]
            right_probs[i] = prob[1]
        self.barcollection_left = self.ax.bar(self.angles, left_probs, width=0.01, color='b')
        self.barcollection_right = self.ax.bar(self.angles, right_probs, width=0.01, bottom=left_probs, color='r')
        return self.barcollection_left, self.barcollection_right
    
    def animate(self):
        anim = FuncAnimation(self.fig, self.update, frames=None, interval=self.refresh_interval,
                             blit=False, cache_frame_data=False)
        return anim


class Angle_value_animation:
    """Animation of the value network's output for different pole angles

    Args:
        value_network (tf.keras.Model): Value network
        n_points (int): Number of points to plot
        refresh_interval (int): Refresh interval for the animation
    """
    def __init__(self, value_network, n_points=51, refresh_interval=1000):
        self.value_network = value_network
        self.n_points = n_points
        self.refresh_interval = refresh_interval
        self.fig, self.ax = plt.subplots()
        self.angles = np.linspace(-0.418, 0.418, n_points)  # Pole angles
        self.values = np.zeros((n_points))
        self.initialize_plot()

    def initialize_plot(self):
        self.fig, self.ax = plt.subplots()
        for i in range(self.n_points):
            state = np.array([0, 0, self.angles[i], 0])
            self.values[i] = self.value_network(state.reshape(1, -1)).numpy()[0][0]
        self.ax.plot(self.angles, self.values)
        self.ax.set_xlabel('Pole Angle')
        self.ax.set_ylabel('Value')
    
    def update(self, frame):
        self.ax.clear()
        for i in range(self.n_points):
            state = np.array([0, 0, self.angles[i], 0])
            self.values[i] = self.value_network(state.reshape(1, -1)).numpy()[0][0]
        self.ax.plot(self.angles, self.values)
        return self.ax
    
    def animate(self):
        anim = FuncAnimation(self.fig, self.update, frames=None, interval=self.refresh_interval,
                             blit=False, cache_frame_data=False)
        return anim