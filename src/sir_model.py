import numpy as np

class AgentBasedSIR:
    def __init__(self, N=5000, beta=0.2, gamma=1/7, C=8, dt=1.0):
        """
        Initialize the Agent-Based SIR Model.
        
        Params:
            N (int): Population size
            beta (float): Transmission rate
            gamma (float): Recovery rate
            C (int): Average contacts per step
            dt (float): Time step in days
        """
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.C = C
        self.dt = dt
        
        # Calculate probabilities based on discrete time mapping
        self.p_trans = 1 - np.exp(-self.beta * self.dt)
        self.p_rec = 1 - np.exp(-self.gamma * self.dt)
        
        # States mapping
        self.S_CODE = 0
        self.I_CODE = 1
        self.R_CODE = 2
        
        # Initialize state
        self.reset()

    def reset(self, initial_infected=1):
        """
        Resets simulation to initial state.
        
        Returns:
            tuple: ((S, I, R), new_infections_count)
            Note: new_infections_count is 0 at reset.
        """
        self.t = 0
        self.state = np.full(self.N, self.S_CODE)
        
        # Infect random agents to start
        initial_indices = np.random.choice(self.N, initial_infected, replace=False)
        self.state[initial_indices] = self.I_CODE
        
        # Return state and 0 new infections (initial condition)
        return self._get_counts(), 0

    def step(self, intervention_strength=0.0):
        """
        Advances the simulation by one time step.
        
        Args:
            intervention_strength (float): u_t in [0, 1]. 
            Reduces contacts by factor (1 - u_t).
        
        Returns:
            tuple: ((S, I, R), new_infections_count)
        """
        # 1. Identify indices of agents in different states
        infected_indices = np.where(self.state == self.I_CODE)[0]
        
        num_infected = len(infected_indices)
        count_new_infections = 0  # Initialize counter for this step
        
        # Create a copy for synchronous update
        next_state = self.state.copy()
        
        # --- TRANSMISSION PROCESS ---
        if num_infected > 0:
            # Apply intervention: Reduce contacts
            # effective_C = C * (1 - u_t)
            effective_C = int(self.C * (1.0 - intervention_strength))
            
            if effective_C > 0:
                # Total number of contact attempts by all infected agents
                total_contacts = num_infected * effective_C
                
                # Sample targets uniformly from population (Well-mixed assumption)
                contact_targets = np.random.choice(self.N, total_contacts)
                
                # Filter: Only contacts with Susceptible agents result in potential transmission
                valid_contacts_mask = (self.state[contact_targets] == self.S_CODE)
                susceptible_targets = contact_targets[valid_contacts_mask]
                
                # Bernoulli trial for transmission
                if len(susceptible_targets) > 0:
                    transmission_rolls = np.random.random(len(susceptible_targets))
                    new_infections = susceptible_targets[transmission_rolls < self.p_trans]
                    
                    # Deduplicate infections (in case multiple agents infected the same target in one step)
                    new_infections = np.unique(new_infections)
                    
                    # Update state to Infectious
                    next_state[new_infections] = self.I_CODE
                    
                    # Record number of new infections for RL Reward
                    count_new_infections = len(new_infections)

        # --- RECOVERY PROCESS ---
        # Bernoulli trial for recovery
        if num_infected > 0:
            recovery_rolls = np.random.random(num_infected)
            recovering_agents = infected_indices[recovery_rolls < self.p_rec]
            next_state[recovering_agents] = self.R_CODE
            
        # Update simulation state
        self.state = next_state
        self.t += 1
        
        # Return updated counts AND the number of new infections
        return self._get_counts(), count_new_infections

    def _get_counts(self):
        """Returns current (S, I, R) counts."""
        unique, counts = np.unique(self.state, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        return (
            counts_dict.get(self.S_CODE, 0),
            counts_dict.get(self.I_CODE, 0),
            counts_dict.get(self.R_CODE, 0)
        )