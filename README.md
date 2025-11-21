# Agent-Based SIR Simulation with RL Intervention

This repository contains an Agent-Based Model (ABS) of the SIR epidemic dynamics and a Reinforcement Learning (RL) agent optimized to control the outbreak via social distancing interventions.

## Structure

- `src/`: Contains the core Python implementation.
  - `sir_model.py`: The agent-based environment logic.
  - `rl_agent.py`: The Q-Learning agent implementation.
  - `utils.py`: Helper functions for R0 estimation.
- `notebooks/`: Jupyter notebooks to reproduce experiments and plots.
  - `01_sir_dynamics.ipynb`: Visualizes basic SIR curves (Part A).
  - `02_r0_estimation.ipynb`: Empirically estimates R0 vs Beta (Part D).
  - `03_rl_training.ipynb`: Trains the RL agent and evaluates performance (Part C & D).
- `plots/`: Generated figures used in the report.

## How to Run

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Navigate to the `notebooks/` directory and run the notebooks in order.
3.  Check the `plots/` directory for the output figures.

### Authors and Contact

This project was created by a student from Jönköping University's School of Engineering (JTH) for the fourth seminar of the State of the Art in AI course.

For questions, feedback, or collaborations, please feel free to reach out to the author or open an issue on the project's repository.

| Name                    | Email Address            |
| ----------------------- | ------------------------ |
| **Hossein Ghadirzadeh** | `ghmo23az@student.ju.se` |

**Jönköping University, School of Engineering (JTH)**<br>
_November 2025_
