import numpy as np
import pickle

class QLearningAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.99, epsilon=0.2, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Q-Learning Agent for SIR Intervention.
        
        Params:
            action_space (list): List of possible intervention strengths (u_t).
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate.
        """
        self.action_space = action_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor (gamma_RL)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-Table: Dictionary mapping state_tuple -> np.array(q_values)
        self.q_table = {}
        
    def get_state_key(self, obs, total_steps):
        """
        Discretizes the continuous state into a tuple key for the Q-table.
        State: (S_fraction, I_fraction, time_fraction)
        """
        (S, I, R), _ = obs
        N = S + I + R
        
        # Bins for discretization (Adjust resolution as needed)
        # S and I are critical, so we give them reasonable resolution (e.g., 10 bins)
        # Time is also important (e.g., 10 bins)
        s_bin = int((S / N) * 10)
        i_bin = int((I / N) * 10)
        # Assuming max episode length T approx 150-200, we normalize time
        # Let's assume total_steps is passed or we normalize by a fixed T_max
        # For simplicity, let's bucket time by chunks of 20 steps if T is unknown, 
        # or fraction if T is known. Let's use raw time / 10 for now.
        t_bin = int(total_steps / 15) 
        
        return (s_bin, i_bin, t_bin)

    def get_action(self, state_key):
        """Epsilon-greedy action selection."""
        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(len(self.action_space))
        
        # Exploitation
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.action_space))
            
        return np.argmax(self.q_table[state_key])

    def update(self, state_key, action_idx, reward, next_state_key):
        """Q-Learning update rule."""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.action_space))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.action_space))
            
        # Bellman Equation
        # Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s', a')) - Q(s,a)]
        best_next_q = np.max(self.q_table[next_state_key])
        current_q = self.q_table[state_key][action_idx]
        
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q

    def decay_epsilon(self):
        """Reduces exploration over time."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)