import numpy as np

def estimate_r0_empirical(beta, gamma, C, dt=1.0, n_runs=100):
    """
    Estimates R0 empirically by simulating a single index case.
    
    Protocol:
    1. Initialize 1 infectious agent.
    2. Simulate until recovery.
    3. Count direct secondary infections caused by this agent.
    4. Average over n_runs.
    """
    p_trans = 1 - np.exp(-beta * dt)
    p_rec = 1 - np.exp(-gamma * dt)
    
    secondary_infections = []
    
    for _ in range(n_runs):
        infections_count = 0
        is_infected = True
        
        while is_infected:
            # Try to infect C contacts
            # We assume large N, so all C contacts are Susceptible
            # Bernoulli trial for each contact
            # generated random numbers compared to p_trans
            new_infections = np.sum(np.random.random(C) < p_trans)
            infections_count += new_infections
            
            # Check for recovery
            if np.random.random() < p_rec:
                is_infected = False
                
        secondary_infections.append(infections_count)
        
    return np.mean(secondary_infections)