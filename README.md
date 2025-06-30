# Quantum Optimization
Solution for Quantum circuit optimization problems
This project, "Statevector or Tensor based  Entanglement Optimization for Quantum circuits," utilizes a Deep Double Q-Network (DDQN) agent to optimize quantum circuits. The primary objective is to maximize entanglement while simultaneously reducing circuit depth and the total number of gates. The optimization process is framed as a reinforcement learning problem, where the DDQN agent interacts with a custom `QuantumCircuitEnv` environment.

### Features

  * **Reinforcement Learning (DDQN):** Implements a DDQN agent with an experience replay memory and a target network for stable training.
  * **Custom Quantum Environment (`QuantumCircuitEnv`):**
      * **State Representation:** Encodes quantum circuits into a numerical state representation for the RL agent, including gate information, qubit connectivity, entanglement, depth, and gate count.
      * **Action Space:** Defines a set of actions that the agent can take to modify the quantum circuit, such as adding various gates (H, CX, RX, RZ, CZ, SWAP), removing gates, swapping gates, and replacing gates. It also includes specific actions for entanglement injection.
      * **Reward Function:** A multi-objective reward function that incentivizes:
          * Increased Quantum Fisher Information (QFI).
          * Increased entanglement.
          * Reduced circuit depth.
          * Reduced gate count.
          * Includes bonuses for achieving high entanglement thresholds and penalties for low entanglement.
      * **Noise Model Integration:** Simulates depolarizing and thermal relaxation errors to make the optimization more realistic.
      * **Circuit Optimization:** Incorporates Qiskit's `PassManager` and tket's optimization passes (`pytket`) for circuit simplification and depth optimization within the environment's step function.
      * **Adaptive Entanglement Threshold:** Dynamically adjusts the entanglement threshold to encourage continuous improvement.
      * **Entanglement Boosting:** Periodically injects entangling gates if the current entanglement falls below a certain threshold.
      * **Layer-wise Entanglement Analysis:** Analyzes entanglement at each layer and injects entanglement where needed.
  * **Attention Mechanism:** The DDQN's neural network architecture includes a custom attention layer to focus on relevant features, particularly entanglement.
  * **Adaptive Learning Rate Scheduler:** Uses a custom Keras callback to adjust the learning rate during training, promoting better convergence.
  * **Comprehensive Visualization:** Provides various plots to visualize training progress, including:
      * Learning curve (Total Reward per Episode).
      * Metrics per Episode (Depth, Entanglement, QFI, Gate Count, Depth Reduction, Gate Count Reduction).
      * Pareto Front analysis (Depth vs. Entanglement vs. QFI).
      * Optimization Progress Heatmap.
      * Error Rates over episodes.
      * Initial vs. Final Entanglement and QFI.
      * Episode execution time and memory consumption.
      * Comparison of Tket vs. Qiskit optimization counts.
  * **Circuit Loading:** Supports loading initial quantum circuits from a QASM file (`quantum_circuits_25.qasm`).

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd statevector_entanglement_optimization
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    (You will need to create a `requirements.txt` file containing the following: `numpy`, `psutil`, `qiskit`, `qiskit-aer`, `pytket`, `gym`, `tensorflow`, `matplotlib`, `seaborn`)

### Usage

1.  **Prepare your initial quantum circuits:**
    Ensure you have a QASM file named `quantum_circuits_25.qasm` in the project directory. This file should contain one or more quantum circuits in QASM 3.0 format, separated by `// Circuit` comments. An example is provided in the code.

2.  **Run the training script:**

    ```bash
    python statevector_entanglement_optimization.py
    ```

    The `train_ddqn_agent` function will start the training process. You can adjust the `num_episodes`, `num_qubits`, and `max_gates` parameters within the `if __name__ == "__main__":` block.

3.  **Monitor training progress:**
    During training, various plots will be generated and saved in a directory named `25S/` (created automatically). These plots visualize the learning curve, entanglement, QFI, depth, gate count, and other metrics over episodes.

4.  **Test the trained agent (optional):**
    After training, you can uncomment the `test_trained_agent` function call in the `if __name__ == "__main__":` block to evaluate the performance of the trained agent on a new or predefined circuit. This will save images of the initial and optimized circuits.

### Code Structure

  * `QuantumCircuitEnv(gym.Env)`: The custom Gym environment defining the quantum circuit optimization problem.
      * `_create_noise_model()`: Initializes a noise model for simulations.
      * `reset()`: Resets the environment for a new episode, either with a random circuit or a provided one.
      * `_compute_qubit_pair_entanglement()`: Calculates entanglement between a specific pair of qubits.
      * `_calculate_entanglement_gradient()`: Computes gradients of entanglement.
      * `inject_entanglement()`: Adds entangling gates to boost entanglement.
      * `_generate_random_circuit()`: Creates a random initial quantum circuit.
      * `step()`: Executes an action chosen by the agent and returns the next state, reward, and done flag.
      * `terminate()`: Defines the termination condition for an episode based on entanglement convergence.
      * `optimize_circuit()`: Applies circuit simplification and depth optimization using `CircuitOptimizer`.
      * `optimize_depth()`: Uses `pytket` for advanced depth optimization.
      * `_has_entangling_gates()`: Checks for the presence of entangling gates.
      * `_index_to_qubit_pair()`: Converts an action index to a qubit pair.
      * `_compute_layer_entanglements()`: Calculates entanglement for each layer of the circuit.
      * `_inject_layer_entanglement()`: Injects entanglement into layers with low entanglement.
      * `_get_state()`: Transforms the current circuit into the numerical state representation for the agent.
      * `_calculate_reward()`: Computes the reward based on changes in QFI, depth, entanglement, and gate count.
      * `_compute_qfi()`: Calculates the Quantum Fisher Information.
      * `_actual_qfi()`: Helper for QFI calculation, considering parameterized circuits.
      * `_compute_entanglement()`: Calculates the entanglement entropy of the circuit.
      * `simplify_circuit()`: Applies Qiskit's `PassManager` for circuit simplification.
  * `CriticNetwork(tf.keras.Model)`: A simple neural network for the critic in an Actor-Critic setup (though the primary agent is DDQN, this structure is included).
  * `AttentionLayer(layers.Layer)`: Custom Keras layer for incorporating attention in the neural network, specifically focusing on entanglement features.
  * `AdaptiveLearningRateScheduler(tf.keras.callbacks.Callback)`: Custom Keras callback for adaptive learning rate adjustment.
  * `ExpandDimsLayer(layers.Layer)`: A custom layer to expand dimensions of tensors.
  * `DDQNAgent`: The DDQN agent implementation.
      * `_build_model()`: Constructs the DDQN's neural network, including the attention layer.
      * `update_target_model()`: Copies weights from the main model to the target model.
      * `remember()`: Stores experiences in the replay memory.
      * `act()`: Selects an action based on the current state (epsilon-greedy).
      * `replay()`: Samples from replay memory to train the DDQN.
      * `load()`, `save()`: Methods for loading and saving model weights.
      * `update_epsilon()`: Adjusts the epsilon value for exploration-exploitation balance.
  * `pareto_front()`: Function to identify Pareto optimal solutions (for visualization).
  * `visualize_training_results()`: Generates and saves various plots to show training performance.
  * `load_circuits_from_file()`: Utility function to load quantum circuits from a QASM file.
  * `train_ddqn_agent()`: Main function to run the training loop.
  * `test_trained_agent()`: Function to evaluate a trained agent.
  * `create_circuit()`: Example function to create a predefined quantum circuit for testing.

### Dependencies

The core dependencies for this project include:

  * **Qiskit:** For building, simulating, and transpiling quantum circuits.
  * **Pytket:** For advanced circuit optimization passes.
  * **TensorFlow/Keras:** For building and training the deep reinforcement learning model.
  * **Gym:** For creating the reinforcement learning environment.
  * **NumPy:** For numerical operations.
  * **Matplotlib, Seaborn:** For data visualization.
  * **Psutil:** For monitoring memory usage during training.

### Contributing

Contributions are welcome\! Please feel free to open issues or submit pull requests.
