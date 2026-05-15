"""
Deep Q-Network (DQN) Agent for RL-based risk management.

Implements DQN with experience replay and epsilon-greedy exploration.
Network architecture: Dense(24) -> Dense(24) -> Linear(actions)
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DQNAgent:
    """
    Deep Q-Network agent for risk level selection.
    
    Network architecture:
    - Input layer: state_size
    - Hidden layer 1: Dense(24) with ReLU activation
    - Hidden layer 2: Dense(24) with ReLU activation
    - Output layer: Dense(action_size) with linear activation
    
    Hyperparameters:
    - gamma: 0.95 (discount factor)
    - epsilon: 1.0 -> min 0.01 (exploration rate)
    - epsilon_decay: 0.995
    - learning_rate: 0.001
    - memory_size: 2000
    - batch_size: 32
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.001,
        memory_size: int = 2000,
        batch_size: int = 32
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of discrete actions (risk levels)
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            learning_rate: Learning rate for Adam optimizer
            memory_size: Maximum size of experience replay buffer
            batch_size: Batch size for training
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Build main and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self) -> keras.Model:
        """
        Build neural network model.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(24, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """Update target model with weights from main model."""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Reshape state for prediction
        state_reshaped = state.reshape(1, self.state_size)
        q_values = self.model.predict(state_reshaped, verbose=0)
        
        return np.argmax(q_values[0])
    
    def replay(self) -> float:
        """
        Train the model on a batch of experiences.
        
        Returns:
            Loss value
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample minibatch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        # Predict Q-values for current and next states
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update targets using Bellman equation
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])
        
        # Train the model
        history = self.model.fit(
            states, targets,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0
        )
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
    
    def load(self, name: str):
        """
        Load model weights from file.
        
        Args:
            name: File path to load weights from
        """
        self.model.load_weights(name)
        self.target_model.load_weights(name)
    
    def save(self, name: str):
        """
        Save model weights to file.
        
        Args:
            name: File path to save weights to
        """
        self.model.save_weights(name)
    
    def get_risk_multiplier(self, action: int) -> float:
        """
        Get risk multiplier for given action.
        
        Args:
            action: Action index
            
        Returns:
            Risk multiplier value
        """
        risk_map = {0: 0.005, 1: 0.01, 2: 0.02}
        return risk_map.get(action, 0.01)
