'''
 @Author: LAXMISH
 @Created on: 7/12/2024
'''
import gc
import logging
import time
from collections import deque

import numpy as np
import psutil
from pytket._tket.passes import SequencePass, FullPeepholeOptimise, RebaseCustom
from pytket._tket.predicates import GateSetPredicate
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import HGate, CXGate, RXGate, RZGate, CZGate, SwapGate
from qiskit.qasm3 import loads
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Optimize1qGates,
    CommutativeCancellation,
    CXCancellation,
    ConsolidateBlocks,
    BarrierBeforeFinalMeasurements, Optimize1qGatesDecomposition, OptimizeSwapBeforeMeasure,
    RemoveDiagonalGatesBeforeMeasure
)
from qiskit.visualization import circuit_drawer
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Statevector, partial_trace, entropy, DensityMatrix
import matplotlib.pyplot as plt
import gym
from gym import spaces
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns

from CircuitOptimizer import CircuitOptimizer
import warnings
from pytket.passes import (
            SequencePass,
            CommuteThroughMultis,
            RemoveRedundancies,
            CliffordSimp,
            PeepholeOptimise2Q,
            FullPeepholeOptimise,
            DecomposeBoxes,
            SynthesiseTket,
            RebaseTket,
        )
from pytket.predicates import CompilationUnit
from pytket.circuit import OpType
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("qiskit_aer").setLevel(logging.ERROR)
import matplotlib
matplotlib.use('Agg')




class QuantumCircuitEnv(gym.Env):
    def __init__(self, num_qubits, max_gates, p_meas=0.02, p_gate1=0.01, p_gate2=0.03, T1=50, T2=70):
        super(QuantumCircuitEnv, self).__init__()
        self.parameter_count = None
        self.num_qubits = num_qubits
        self.max_gates = max_gates
        self.action_space = spaces.Discrete(9 + num_qubits * (num_qubits - 1) // 2)  # Add H, Add CNOT, Add RX, Add RZ, Add CZ, Add SWAP, Add CRX, Remove gate, Swap gates, Replace gate
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.max_gates, 5 * num_qubits + 4), dtype=np.float32)
        self.qc = None
        self.initial_qfi = 0
        self.initial_depth = 0
        self.initial_entanglement = 0
        self.initial_gate_count = 0
        self.gate_fidelities = {
            'h': 0.9995,  # Single-qubit gate, high fidelity
            'rx': 0.9995,  # Single-qubit gate, high fidelity
            'rz': 0.9995,  # Single-qubit gate, high fidelity
            'cx': 0.995,  # Two-qubit entangling gate, slightly lower but still high
            'cz': 0.995,  # Two-qubit entangling gate, slightly lower but still high
            'swap': 0.990  # Two-qubit gate, slightly lower fidelity
        }
        self.entanglement_threshold = 0.7
        self.depth_penalty = 0.1
        self.noise_model = self._create_noise_model(p_meas, p_gate1, p_gate2, T1, T2)
        self.error_rates = {
            'measurement': p_meas,
            'single_qubit': p_gate1,
            'two_qubit': p_gate2,
            'T1': 1 - np.exp(-1 / T1),
            'T2': 1 - np.exp(-1 / T2)
        }
        self.entanglement_history = []
        self.gate_count_history = []
        self.backend = AerSimulator(method='matrix_product_state',noise_model=self.noise_model)
        print(self.backend.available_methods())

    def _create_noise_model(self, p_meas, p_gate1, p_gate2, T1, T2):
        noise_model = NoiseModel()

        # Add measurement error
        error_meas = depolarizing_error(p_meas, 1)
        noise_model.add_all_qubit_quantum_error(error_meas, "measure")

        # Add single qubit gate error
        error_1 = depolarizing_error(p_gate1, 1)
        noise_model.add_all_qubit_quantum_error(error_1, ["h", "rx", "rz"])

        # Add two qubit gate error
        error_2 = depolarizing_error(p_gate2, 2)
        noise_model.add_all_qubit_quantum_error(error_2, ["cx"])

        # Add T1/T2 relaxation error
        for qubit in range(self.num_qubits):
            relaxation_error = thermal_relaxation_error(T1, T2, p_gate1)
            noise_model.add_quantum_error(relaxation_error,["h", "rx", "rz"], [qubit])

        return noise_model

    def reset(self, circ):

        if circ is None:
            print("Generating Random Circuit")
            self.qc = QuantumCircuit(self.num_qubits)
            self._generate_random_circuit()
        else:
            self.qc = circ
            self.num_gates = len(self.qc.data)
            self.parameters = [Parameter(f'θ{i}') for i in range(self.num_gates)]


        self.initial_qfi = self._compute_qfi()
        self.initial_depth = self.qc.depth()
        self.initial_entanglement = self._compute_entanglement(None)
        self.initial_gate_count = len(self.qc.data)
        self.parameter_count = 0

        return self._get_state()

    def _compute_qubit_pair_entanglement(self, qubit1, qubit2):
        circ = QuantumCircuit(self.num_qubits)
        for operation, qubits, clbits in self.qc.data:
            circ.append(operation, qubits, clbits)
        param_binds = {param: random.uniform(0, 2 * np.pi) for param in circ.parameters}
        # Create a copy of the circuit and bind the parameters
        circ = circ.assign_parameters(param_binds)
        circ.save_matrix_product_state()
        simulator = self.backend
        job = simulator.run(circ)
        results = job.result()
        mps_data = results.data()['matrix_product_state']
        _, schmidt_coeffs = mps_data
        bond_entropies = []

        for lambdas in schmidt_coeffs:
            lambdas_sq = lambdas ** 2
            S = -np.sum(lambdas_sq * np.log2(lambdas_sq + 1e-12))  # Avoid log(0)
            bond_entropies.append(S)

        bond_entropy = np.mean(bond_entropies)
        entanglement_entropy = max(0, min(1, bond_entropy))
        return entanglement_entropy


    def _calculate_entanglement_gradient(self):
        gradients = {}
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                current_ent = self._compute_qubit_pair_entanglement(i, j)
                self.qc.cx(i, j)
                new_ent = self._compute_qubit_pair_entanglement(i, j)
                self.qc.cx(i, j)  # undo the CNOT
                gradients[(i, j)] = new_ent - current_ent
        return gradients

    def inject_entanglement(self, qubit1, qubit2):
        current_ent = self._compute_qubit_pair_entanglement(qubit1, qubit2)
        self.qc.h(qubit1)
        self.qc.cx(qubit1, qubit2)
        new_ent = self._compute_qubit_pair_entanglement(qubit1, qubit2)
        return new_ent - current_ent

    def _generate_random_circuit(self):
        num_gates = random.randint(self.num_qubits, self.max_gates)
        self.num_gates = num_gates
        self.parameters = [Parameter(f'θ{i}') for i in range(num_gates)]
        param_index = 0

        for _ in range(num_gates):
            # Weighted choice: single-qubit gates more likely than entangling gates
            gate = random.choices(
                ['h', 'rx', 'rz', 'cx', 'cz', 'swap'],
                weights=[3, 3, 3, 1, 1, 1],  # entangling gates are less likely
                k=1
            )[0]

            if gate == 'h':
                qubit = random.randint(0, self.num_qubits - 1)
                self.qc.h(qubit)

            elif gate == 'rx':
                qubit = random.randint(0, self.num_qubits - 1)
                small_angle = random.uniform(0, 0.01)

                # Alternatively use parameterized version for tunable small QFI:
                self.qc.rx(self.parameters[param_index], qubit)
                param_index += 1

            elif gate == 'rz':
                qubit = random.randint(0, self.num_qubits - 1)
                small_angle = random.uniform(0, 0.01)

                self.qc.rz(self.parameters[param_index], qubit)
                param_index += 1

            elif gate == 'cx':
                control, target = random.sample(range(self.num_qubits), 2)
                self.qc.cx(control, target)

            elif gate == 'cz':
                control, target = random.sample(range(self.num_qubits), 2)
                self.qc.cz(control, target)

            elif gate == 'swap':
                qubit1, qubit2 = random.sample(range(self.num_qubits), 2)
                self.qc.swap(qubit1, qubit2)

    def step(self, action):
        original_depth = self.qc.depth()
        original_entanglement = self._compute_entanglement(None)
        '''
        # Increase the probability of adding entangling gates
        if random.random() < 0.3:  # 30% chance to add an entangling gate
            control, target = random.sample(range(self.num_qubits), 2)
            gate_type = random.choice(['cx', 'cz', 'swap'])
            if gate_type == 'cx':
                self.qc.cx(control, target)
            elif gate_type == 'cz':
                self.qc.cz(control, target)
            else:
                self.qc.swap(control, target)
        else:
            '''
        gate_name = None
        # Proceed with the original action
        if action == 0:  # Add H gate
            qubit = random.randint(0, self.num_qubits - 1)
            self.qc.h(qubit)
            gate_name = 'h'
        elif action == 1:  # Add CNOT gate
            control, target = random.sample(range(self.num_qubits), 2)
            self.qc.cx(control, target)
            gate_name = 'cx'
        elif action == 2:  # Add RX gate
            qubit = random.randint(0, self.num_qubits - 1)
            #self.qc.rx(np.random.random() * np.pi, qubit)
            param = Parameter(f'θ{len(self.parameters)}')
            self.parameters.append(param)
            self.qc.rx(param, qubit)
            gate_name = 'rx'
        elif action == 3:  # Add RZ gate
            qubit = random.randint(0, self.num_qubits - 1)
            #self.qc.rz(np.random.random() * np.pi, qubit)
            param = Parameter(f'θ{len(self.parameters)}')
            self.parameters.append(param)
            self.qc.rz(param, qubit)
            gate_name = 'rz'
        elif action == 4:  # Add CZ gate
            control, target = random.sample(range(self.num_qubits), 2)
            self.qc.cz(control, target)
            gate_name = 'cz'
        elif action == 5:  # Add SWAP gate
            qubit1, qubit2 = random.sample(range(self.num_qubits), 2)
            self.qc.swap(qubit1, qubit2)
            gate_name = 'swap'
        elif action == 6:  # Remove last gate
            if len(self.qc.data) > 0:
                if self._has_entangling_gates(excluding_last=True):
                    self.qc.data.pop()
        elif action == 7:  # Swap two gates
            if len(self.qc.data) > 1:
                i, j = random.sample(range(len(self.qc.data)), 2)
                self.qc.data[i], self.qc.data[j] = self.qc.data[j], self.qc.data[i]
        elif action == 8:  # Replace a gate
            if len(self.qc.data) > 0:
                i = random.randint(0, len(self.qc.data) - 1)
                old_gate, old_qubits, _ = self.qc.data[i]
                gate = random.choice(['h', 'cx', 'rx', 'rz', 'cz', 'swap'])
                if gate == 'h' and len(old_qubits) >= 1:
                    new_gate = HGate()
                    self.qc.data[i] = (new_gate, [old_qubits[0]], [])
                elif gate == 'cx' and len(old_qubits) >= 2:
                    new_gate = CXGate()
                    self.qc.data[i] = (new_gate, old_qubits[:2], [])
                elif gate == 'rx' and len(old_qubits) >= 1:
                    angle = random.uniform(0, 2 * np.pi)
                    param = Parameter(f'θ{len(self.parameters)}')
                    self.parameters.append(param)
                    new_gate = RXGate(param)
                    self.qc.data[i] = (new_gate, [old_qubits[0]], [])
                elif gate == 'rz' and len(old_qubits) >= 1:
                    angle = random.uniform(0, 2 * np.pi)
                    param = Parameter(f'θ{len(self.parameters)}')
                    self.parameters.append(param)
                    new_gate = RZGate(param)
                    self.qc.data[i] = (new_gate, [old_qubits[0]], [])
                elif gate == 'cz' and len(old_qubits) >= 2:
                    new_gate = CZGate()
                    self.qc.data[i] = (new_gate, old_qubits[:2], [])
                elif gate == 'swap' and len(old_qubits) >= 2:
                    new_gate = SwapGate()
                    self.qc.data[i] = (new_gate, old_qubits[:2], [])
                else:
                    # If the chosen gate is not compatible with the number of qubits in the old gate,
                    # we can either skip the replacement or choose a different gate
                    pass
        else:  # Entangling actions
            qubit1, qubit2 = random.sample(range(self.num_qubits), 2)
            self.inject_entanglement(qubit1, qubit2)

        # Adaptive entanglement threshold
        self.entanglement_threshold = max(0.7, original_entanglement)

        # Check if the new depth exceeds the original depth
        if self.qc.depth() >= original_depth :
            # If it does, revert the last action
            if action in [0, 1, 2, 3, 4, 5, 6, 8]:  # If a gate was added
                self.qc.data.pop()
            #elif action == 7:  # If gates were swapped
                #self.qc.data[i], self.qc.data[j] = self.qc.data[j], self.qc.data[i]


        # Ensure the circuit has at least one entangling gate
        if not self._has_entangling_gates():
            control, target = random.sample(range(self.num_qubits), 2)
            self.qc.cx(control, target)

        # Periodic entanglement boosting
        if self._compute_entanglement(None) < self.entanglement_threshold:
            control, target = random.sample(range(self.num_qubits), 2)
            self.qc.h(control)
            self.qc.cx(control, target)

        # Layer-wise entanglement analysis
        '''
        layer_entanglements = self._compute_layer_entanglements()
        if min(layer_entanglements) < 0.7:
            self._inject_layer_entanglement(layer_entanglements)
        '''

        self.optimize_circuit()
        print("num_gates ",self.num_gates)
        while len(self.qc.data) > self.num_gates:
            self.qc.data.pop()
        print("current_gates ", len(self.qc.data))


        reward = self._calculate_reward()
        done = False

        if gate_name is not None:
            fidelity = self.gate_fidelities[gate_name]
            reward *= fidelity

        # Update error rates based on the action taken
        if action in [0, 2, 3]:  # Single qubit gates
            self.error_rates['single_qubit'] *= (1 + 0.01)  # Increase error rate by 1%
        elif action in [1, 4, 5]:  # Two qubit gates
            self.error_rates['two_qubit'] *= (1 + 0.02)  # Increase error rate by 2%

        return self._get_state(), reward, done, {'error_rates': self.error_rates}

    def terminate(self, reward):
        done = False
        patience = 5  # Number of steps to wait for stabilization
        entanglement_tolerance = 0.099  # Minimum change in entanglement to stop (since it's between 0 and 1)
        gate_count_tolerance = 1  # Minimum change in gate count

        # Inside your optimization loop
        entropy = self.current_entropy
        #print("entropy",entropy)
        if entropy >= 0.994:
            done = True
        self.entanglement_history.append(entropy)
        self.gate_count_history.append(len(self.qc.data))
        # Check for convergence after 'patience' steps
        if len(self.entanglement_history) > patience:
            recent_entanglement_change = max(
                abs(self.entanglement_history[-i] - self.entanglement_history[-i - 1]) for i in range(1, patience))
            recent_gate_count_change = max(
                abs(self.gate_count_history[-i] - self.gate_count_history[-i - 1]) for i in range(1, patience))
            print("recent_entanglement_change:",recent_entanglement_change)
            if recent_entanglement_change < entanglement_tolerance and recent_gate_count_change < gate_count_tolerance:
                done = True

        return done

    def optimize_circuit(self):
        # Simplify gates
        self.simplify_circuit()
        optimizer = CircuitOptimizer(self.qc.copy(),self.parameter_count)
        self.qc,parameter_count = optimizer.optimize()
        self.parameter_count = parameter_count

        #self.optimize_depth()

    def optimize_depth(self):
        # Convert Qiskit circuit to t|ket⟩ circuit
        tk_circuit = qiskit_to_tk(self.qc)
        # Define our custom gate set
        custom_gate_set = {OpType.H, OpType.CX, OpType.Rx, OpType.Rz, OpType.SWAP, OpType.CZ}
        # Create a custom optimization sequence
        custom_pass = SequencePass([
            DecomposeBoxes(),
            CommuteThroughMultis(),
            RemoveRedundancies(),
            PeepholeOptimise2Q(),
            CliffordSimp(),
            CommuteThroughMultis(),
            RemoveRedundancies(),
        ])
        # Apply the custom pass
        custom_pass.apply(tk_circuit)
        # Get the optimized circuit
        optimized_tk_circuit = tk_circuit
        # Convert back to Qiskit circuit
        optimized_qc = tk_to_qiskit(optimized_tk_circuit)
        # Compare metrics
        print(f"\nOriginal depth: {self.qc.depth()}, gates: {self.qc.count_ops()},gatecount: {len(self.qc.data)}")
        print(f"Optimized depth: {optimized_qc.depth()}, gates: {optimized_qc.count_ops()}")
        if (optimized_qc.depth() <= self.qc.depth() and len(optimized_qc.data) <= len(self.qc.data)):
            return optimized_qc
        else :
            return self.qc

    def _has_entangling_gates(self, excluding_last=False):
        gates_to_check = self.qc.data[:-1] if excluding_last else self.qc.data
        return any(gate[0].name == 'cx' for gate in gates_to_check)

    def _index_to_qubit_pair(self, index):
        # Convert an index to a unique qubit pair
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                if index == 0:
                    return i, j
                index -= 1

    def _compute_layer_entanglements(self):
        # Compute entanglement for each layer of the circuit
        layer_entanglements = []
        for i in range(self.qc.depth()):
            layer_circuit = QuantumCircuit(self.num_qubits)
            for operation, qubits, clbits in self.qc.data:
                layer_circuit.append(operation, qubits, clbits)
            layer_circuit.data = self.qc.data[:i + 1]
            layer_entanglements.append(self._compute_entanglement(layer_circuit))
        return layer_entanglements

    def _inject_layer_entanglement(self, layer_entanglements):
        worst_layer = np.argmin(layer_entanglements)
        gradients = self._calculate_entanglement_gradient()
        best_pair = max(gradients, key=gradients.get)
        self.qc.h(best_pair[0])
        self.qc.cx(best_pair[0], best_pair[1])
        return gradients[best_pair] - self.depth_penalty

    def _get_state(self):
        state = np.zeros((self.max_gates, 5 * self.num_qubits + 4))
        for i, (gate, qubits, params) in enumerate(self.qc.data):
            if i >= self.max_gates:
                break
            gate_vector = np.zeros(5 * self.num_qubits + 4)
            if gate.name == 'h':
                gate_vector[0] = 1
            elif gate.name == 'cx':
                gate_vector[1] = 1
            elif gate.name == 'rx':
                gate_vector[2] = 1
                gate_vector[3] = params[0].evalf() if params else 0
            elif gate.name == 'rz':
                gate_vector[4] = 1
                gate_vector[5] = params[0].evalf() if params else 0
            elif gate.name == 'cz':
                gate_vector[6] = 1
            elif gate.name == 'swap':
                gate_vector[7] = 1
            elif gate.name == 'crx':
                gate_vector[8] = 1
                gate_vector[9] = gate.params[0]
            state[i] = gate_vector

        current_entanglement = self._compute_entanglement(None)
        layer_entanglements = self._compute_layer_entanglements()
        state[:, -5] = np.mean(layer_entanglements)
        state[:, -4] = current_entanglement
        state[:, -3] = self.qc.depth() / self.max_gates
        state[:, -2] = len(self.qc.data) / self.max_gates
        state[:, -1] = len(self.parameters) / self.max_gates

        return state

    def _calculate_reward(self):
        self.current_qfi = self._compute_qfi()
        current_depth = self.qc.depth()
        self.current_entropy = self._compute_entanglement(None)
        current_gate_count = len(self.qc.data)

        qfi_improvement = (self.current_qfi - self.initial_qfi) / max(self.initial_qfi, 1e-10)
        depth_reduction = (self.initial_depth - current_depth) / max(self.initial_depth, 1)
        entanglement_improvement = (self.current_entropy - self.initial_entanglement) / max(self.initial_entanglement,
                                                                                            1e-10)
        gate_count_reduction = (self.initial_gate_count - current_gate_count) / max(self.initial_gate_count, 1)

        reward = (50 * qfi_improvement +
                  5 * depth_reduction +
                  15 * entanglement_improvement +  # Increased from 10 to 15
                  5 * gate_count_reduction)

        # Multiple entanglement thresholds with increasing bonuses
        if self.current_entropy >= 0.95:
            reward += 50
        elif self.current_entropy >= 0.9:
            reward += 30
        elif self.current_entropy >= 0.85:
            reward += 20
        elif self.current_entropy >= 0.7:
            reward += 10
        else:
            reward -= 5

        return reward

    def _compute_qfi(self):
        current_params = self.qc.parameters
        #print("PARA ", current_params)
        #circ = self.qc.copy()
        # Create a new circuit instead of copying
        circ = QuantumCircuit(self.num_qubits)
        for operation, qubits, clbits in self.qc.data:
            circ.append(operation, qubits, clbits)
        param_binds = {param: random.uniform(0, 2 * np.pi) for param in circ.parameters}
        # Create a copy of the circuit and bind the parameters
        circ = circ.assign_parameters(param_binds)
        return self._actual_qfi()

    def _actual_qfi(self):
        circuit = QuantumCircuit(self.num_qubits)
        for operation, qubits, clbits in self.qc.data:
            circuit.append(operation, qubits, clbits)
        """
        Calculate the Quantum Fisher Information (QFI) for a given quantum circuit.

        :param circuit: Qiskit QuantumCircuit object
        :return: QFI value
        """
        # Get the number of qubits in the circuit
        num_qubits = circuit.num_qubits

        # Get the parameters of the circuit
        params = circuit.parameters
        #print("PARAM ",params)
        param_binds = {param: random.uniform(0, 2 * np.pi) for param in params}
        # Create a copy of the circuit and bind the parameters
        #circuit = circuit.assign_parameters(param_binds)
        if not params:
            circuit.save_statevector()
            backend = self.backend
            job = backend.run(circuit)
            results = job.result()
            statevector = results.get_statevector(circuit)
            # If there are no parameters, calculate QFI based on the final state
            qfi = 4 * (1 - np.sum(np.abs(statevector.data) ** 4))
            qfi = max(0, min(1, qfi))
            return qfi
            #print("QFFFI111", qfi)

        qfi_total = 0
        total_shots = 2000
        delta = np.pi / 2
        for param in params:
            plus_circuit = circuit.copy()
            minus_circuit = circuit.copy()

            param_binds = {p: random.uniform(0,np.pi/2) for p in params if p != param}
            param_binds[param] = delta
            plus_circuit = plus_circuit.assign_parameters(param_binds)

            param_binds[param] = -delta
            minus_circuit = minus_circuit.assign_parameters(param_binds)

            plus_circuit.measure_all()
            minus_circuit.measure_all()

            plus_result = self.backend.run(plus_circuit, shots=total_shots).result()
            minus_result = self.backend.run(minus_circuit, shots=total_shots).result()

            plus_counts = plus_result.get_counts()
            minus_counts = minus_result.get_counts()

            # Compute probabilities

            plus_probs = {state: count / total_shots for state, count in plus_counts.items()}
            minus_probs = {state: count / total_shots for state, count in minus_counts.items()}

            # Compute QFI from counts
            qfi_param = 0
            for state in set(plus_probs.keys()).union(minus_probs.keys()):
                p_plus = plus_probs.get(state, 0)
                p_minus = minus_probs.get(state, 0)
                p_avg = (p_plus + p_minus) / 2
                dp_dtheta = (p_plus - p_minus) / (2 * delta)

                if p_avg > 0:  # Avoid division by zero
                    qfi_param += 4 * (dp_dtheta ** 2) / p_avg

            qfi_total += qfi_param

            # Normalize QFI
        qfi = qfi_total / len(params) if params else qfi_total
        qfi = max(0, min(1, qfi))  # Clamp within [0, 1]
        #print("QFI (from counts):", qfi)
        return qfi
    def _compute_entanglement(self, circuit):
        if circuit is None:
            circuit = QuantumCircuit(self.num_qubits)
            for operation, qubits, clbits in self.qc.data:
                circuit.append(operation, qubits, clbits)
        param_binds = {param: random.uniform(0, 2 * np.pi) for param in circuit.parameters}
        # Create a copy of the circuit and bind the parameters
        circuit = circuit.assign_parameters(param_binds)

        circuit.save_matrix_product_state()
        backend = self.backend
        job = backend.run(circuit)
        results = job.result()

        mps_data = results.data()['matrix_product_state']
        _, schmidt_coeffs = mps_data
        bond_entropies = []

        for lambdas in schmidt_coeffs:
            lambdas_sq = lambdas ** 2
            S = -np.sum(lambdas_sq * np.log2(lambdas_sq + 1e-12))  # Avoid log(0)
            bond_entropies.append(S)

        bond_entropy = np.mean(bond_entropies)
        entanglement_entropy = max(0,min(1,bond_entropy))
        return entanglement_entropy
    def simplify_circuit(self):
        pm = PassManager()
        pm.append(Optimize1qGates())
        pm.append(CommutativeCancellation())
        pm.append(CXCancellation())
        pm.append(ConsolidateBlocks())
        pm.append(Optimize1qGatesDecomposition())
        pm.append(BarrierBeforeFinalMeasurements())
        pm.append(OptimizeSwapBeforeMeasure())
        pm.append(RemoveDiagonalGatesBeforeMeasure())
        simplified_qc = pm.run(self.qc)
        self.qc = simplified_qc

class CriticNetwork(tf.keras.Model):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.q_value = layers.Dense(1, activation='linear')

    def call(self, state_action):
        x = self.dense1(state_action)
        q_value = self.q_value(x)
        return q_value

class AttentionLayer(layers.Layer):
    def __init__(self, units, entanglement_index=-3):
        super(AttentionLayer, self).__init__()
        self.units = units
        self.entanglement_index = entanglement_index
        self.W = layers.Dense(units)
        self.U = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, inputs):
        # Use all features for the main computation
        main_features = self.W(inputs)

        # Extract entanglement features (now using the specified index)
        entanglement_features = inputs[:, :, self.entanglement_index:self.entanglement_index + 1]
        entanglement_features = self.U(entanglement_features)

        # Compute attention weights
        score = self.V(tf.nn.tanh(main_features + entanglement_features))
        attention_weights = tf.nn.softmax(score, axis=1)

        # Apply attention weights to input
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # Extract the entanglement feature from the last state
        last_entanglement_feature = inputs[:, -3, self.entanglement_index]

        return context_vector, last_entanglement_feature
class AdaptiveLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr=0.001, factor=0.5, patience=10, min_lr=1e-6):
        super(AdaptiveLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.wait = 0
        self.best = float('inf')
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_lr,
            decay_steps=10000,
            decay_rate=0.9
        )

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('loss')
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = self.lr_schedule(self.model.optimizer.iterations)
                if old_lr > self.min_lr:
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=new_lr,
                        decay_steps=10000,
                        decay_rate=0.9
                    )
                    self.model.optimizer.learning_rate = self.lr_schedule
                    print(f'\nEpoch {epoch + 1}: reducing learning rate to {new_lr}.')
                self.wait = 0
class ExpandDimsLayer(layers.Layer):
    def __init__(self, axis, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)
class DDQNAgent:
    def __init__(self, state_size, action_size, critic=True):
        self.state_size = state_size
        self.action_size = action_size
        self.best_avg_reward = float('-inf')
        self.memory = deque(maxlen=2000)
        self.entanglement_memory = deque(maxlen=1000)
        self.best_episodes = deque(maxlen=10)  # Store 10 best episodes
        self.min_best_reward = float('-inf')
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.critic = critic
        self.lr_scheduler = AdaptiveLearningRateScheduler()
        if critic:
            self.critic_model = CriticNetwork()
            self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_model(self):
        inputs = keras.Input(shape=self.state_size)
        x = layers.Conv1D(64, 3, activation='relu')(inputs)
        x = layers.Conv1D(64, 3, activation='relu')(x)
        context_vector, last_entanglement = AttentionLayer(64,entanglement_index=-4)(x)
        last_entanglement_expanded = ExpandDimsLayer(axis=-1)(last_entanglement)  # Using the custom layer
        x = layers.Concatenate()([context_vector, last_entanglement_expanded])
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.action_size, activation='linear')(x)
        model = keras.Model(inputs, outputs)
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=10000,
            decay_rate=0.9
        )
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if action >= 9:  # Entangling action
            self.entanglement_memory.append((state, action, reward, next_state, done))

    def act(self, state):
        #print(state.shape)
        state = np.expand_dims(state, axis=0)
        if np.random.rand() <= self.epsilon:
            if np.random.rand() < 0.5:  # 60% chance of choosing an entangling action
                return random.choice([1, 4, 5])  # Indices of entangling gates
            else:
                return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        entanglement_batch = random.sample(self.entanglement_memory, self.batch_size // 2) if len(
            self.entanglement_memory) >= self.batch_size // 2 else []
        minibatch.extend(entanglement_batch)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state[0])
            targets_f.append(target_f[0])

        self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0,callbacks =[self.lr_scheduler])

        if self.critic:
                with tf.GradientTape() as tape:
                    state = state.reshape(state.shape[0], -1)
                    state_action = np.concatenate([state, np.array([[action]])], axis=1)
                    q_value = self.critic_model(state_action)
                    loss = tf.reduce_mean(tf.square(q_value - reward))
                grads = tape.gradient(loss, self.critic_model.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def update_epsilon(self):
        if len(self.memory) > self.batch_size:
            recent_rewards = [r for _, _, r, _, _ in list(self.memory)[-100:]]
            current_avg_reward = np.mean(recent_rewards)
            if current_avg_reward > self.best_avg_reward:
                self.best_avg_reward = current_avg_reward
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.99)
            else:
                self.epsilon = min(1.0, self.epsilon / 0.99)

def pareto_front(rewards, depths, entanglements):
    pareto_optimal = np.ones(len(rewards), dtype=bool)
    for i, (r, d, e) in enumerate(zip(rewards, depths, entanglements)):
        for j, (r2, d2, e2) in enumerate(zip(rewards, depths, entanglements)):
            if i != j:
                if r2 >= r and d2 <= d and e2 >= e and (r2 > r or d2 < d or e2 > e):
                    pareto_optimal[i] = False
                    break
    return pareto_optimal


def visualize_training_results(rewards, depths, qfis, entanglements, initial_depths,
                               gate_counts, initial_gate_counts,error_rates_list,
                               initial_entanglements,initial_qfis,episode_times,episode_memories):
    episodes = range(len(rewards))

    # Calculate overall percentage of depth reduction
    depth_reductions = [(initial - final) / initial * 100 for initial, final in zip(initial_depths, depths)]
    gate_count_reductions = [(initial - final) / initial * 100 for initial, final in zip(initial_gate_counts, gate_counts)]
    # Create separate figures for each metric
    metrics = {
        "Circuit Depth": depths,
        "Entanglement": entanglements,
        "Depth Reduction (%)": depth_reductions,
        "Gate Count Reduction (%)": gate_count_reductions,
        "QFI": qfis,
        "Total Reward": rewards
    }

    for metric_name, values in metrics.items():
        min_val = max(0,np.min(values))
        max_val = np.max(values)
        avg_val = np.mean(values)

        plt.figure(figsize=(12, 6))
        plt.plot(episodes, values, label=f'{metric_name}',color='b')
        plt.title(f'{metric_name} per Episode')
        plt.xlabel('Episode')
        plt.ylabel(metric_name)
        plt.axhline(y=min_val, color='r', linestyle='--', label=f'Min: {min_val:.2f}')
        plt.axhline(y=max_val, color='g', linestyle='--', label=f'Max: {max_val:.2f}')
        plt.axhline(y=avg_val, color='b', linestyle='--', label=f'Avg: {avg_val:.2f}')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'100T/{metric_name}_per_Episode.png', dpi=300)

        '''
        pareto_optimal = pareto_front(rewards, depths, entanglements)
        plt.figure(figsize=(12, 6))
        plt.scatter(np.array(depths)[pareto_optimal], np.array(entanglements)[pareto_optimal], c='r',
                    label='Pareto Optimal')
        plt.scatter(np.array(depths)[~pareto_optimal], np.array(entanglements)[~pareto_optimal], c='b',
                    label='Non-Optimal')
        plt.xlabel('Depth')
        plt.ylabel('Entanglement')
        plt.title('Pareto Front: Depth vs Entanglement')
        plt.legend()
        plt.savefig('100T/Pareto_Front_Depth_Entanglement.png', dpi=300)
        '''
        plt.show()

        # 1. Learning Curve Plot
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, rewards)
        plt.title('Learning Curve')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('100T/learning_curve.png', dpi=300)
        plt.close()

        # 2. Pareto Front Visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(depths, qfis, entanglements, c=rewards, cmap='viridis')
        ax.set_xlabel('Depth')
        ax.set_ylabel('QFI')
        ax.set_zlabel('Entanglement')
        plt.colorbar(scatter, label='Reward')
        plt.title('Pareto Front: Depth vs QFI vs Entanglement')
        plt.savefig('100T/pareto_front_3d.png', dpi=300)
        plt.close()

        # 4. Optimization Progress Heatmap
        progress_data = np.array([depths, qfis, entanglements]).T
        plt.figure(figsize=(12, 8))
        sns.heatmap(progress_data, cmap='YlOrRd', xticklabels=['Depth', 'QFI', 'Entanglement'], yticklabels=False)
        plt.title('Optimization Progress Heatmap')
        plt.ylabel('Episode')
        plt.savefig('100T/optimization_progress_heatmap.png', dpi=300)
        plt.close()

        # Add error rate visualization
        plt.figure(figsize=(12, 6))
        plt.plot(list(range(len(error_rates_list))), [error_rates['single_qubit'] for error_rates in error_rates_list],
                 label='Single Qubit Error')
        plt.plot(list(range(len(error_rates_list))), [error_rates['two_qubit'] for error_rates in error_rates_list],
                 label='Two Qubit Error')
        plt.plot(list(range(len(error_rates_list))), [error_rates['measurement'] for error_rates in error_rates_list],
                 label='Measurement Error')
        plt.title('Error Rates')
        plt.xlabel('Episode')
        plt.ylabel('Error Rate')
        plt.legend()
        plt.savefig('100T/error_rates.png', dpi=300)
        plt.close()

        # Plot initial vs final entanglements
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, initial_entanglements, label='Initial Entanglement', color='orange')
        plt.plot(episodes, entanglements, label='Final Entanglement', color='blue')
        plt.title('Initial vs Final Entanglement per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Entanglement')
        plt.axhline(y=np.min(initial_entanglements), color='orange', linestyle='--',
                    label=f'Initial Min: {np.min(initial_entanglements):.2f}')
        plt.axhline(y=np.max(initial_entanglements), color='orange', linestyle='--',
                    label=f'Initial Max: {np.max(initial_entanglements):.2f}')
        plt.axhline(y=np.mean(initial_entanglements), color='orange', linestyle='--',
                    label=f'Initial Avg: {np.mean(initial_entanglements):.2f}')

        plt.axhline(y=np.min(entanglements), color='blue', linestyle='--',
                    label=f'Final Min: {np.min(entanglements):.2f}')
        plt.axhline(y=np.max(entanglements), color='blue', linestyle='--',
                    label=f'Final Max: {np.max(entanglements):.2f}')
        plt.axhline(y=np.mean(entanglements), color='blue', linestyle='--',
                    label=f'Final Avg: {np.mean(entanglements):.2f}')

        plt.legend()
        plt.grid(True)
        plt.savefig('100T/Initial_vs_Final_Entanglement.png', dpi=300)
        plt.show()
        plt.close()

        # Plot initial vs final QFIs
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, initial_qfis, label='Initial QFI', color='orange')
        plt.plot(episodes, qfis, label='Final QFI', color='blue')
        plt.title('Initial vs Final QFI per Episode')
        plt.xlabel('Episode')
        plt.ylabel('QFI')
        plt.axhline(y=np.min(initial_qfis), color='orange', linestyle='--',
                    label=f'Initial Min: {np.min(initial_qfis):.2f}')
        plt.axhline(y=np.max(initial_qfis), color='orange', linestyle='--',
                    label=f'Initial Max: {np.max(initial_qfis):.2f}')
        plt.axhline(y=np.mean(initial_qfis), color='orange', linestyle='--',
                    label=f'Initial Avg: {np.mean(initial_qfis):.2f}')

        plt.axhline(y=np.min(qfis), color='blue', linestyle='--', label=f'Final Min: {np.min(qfis):.2f}')
        plt.axhline(y=np.max(qfis), color='blue', linestyle='--', label=f'Final Max: {np.max(qfis):.2f}')
        plt.axhline(y=np.mean(qfis), color='blue', linestyle='--', label=f'Final Avg: {np.mean(qfis):.2f}')

        plt.legend()
        plt.grid(True)
        plt.savefig('100T/Initial_vs_Final_QFI.png', dpi=300)
        plt.show()
        plt.close()

        # Episode-level performance metrics
        plt.figure(figsize=(12, 6))
        plt.plot(episode_times, label='Episode Time')
        plt.title('Episode Execution Time')
        plt.xlabel('Episode')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        plt.legend()
        plt.savefig('100T/episode_times.png', dpi=300)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(episode_memories, label='Episode Memory')
        plt.title('Episode Memory Consumption')
        plt.xlabel('Episode')
        plt.ylabel('Memory (MB)')
        plt.grid(True)
        plt.legend()
        plt.savefig('100T/episode_memories.png', dpi=300)
        plt.close()




def load_circuits_from_file(filename):
    circuits = []
    with open(filename, 'r',encoding='utf-8') as file:
        qasm_data = file.read().strip().split('// Circuit')
        for qasm_str in qasm_data:
            if qasm_str.strip():
                # Each QASM block starts with a comment, so skip it
                qasm_code = qasm_str.split('\n', 1)[-1]
                circuit = loads(qasm_code)
                circuits.append(circuit)
    return circuits


def train_ddqn_agent(num_episodes, num_qubits, max_gates, p_meas=0.02, p_gate1=0.01, p_gate2=0.03, T1=50, T2=70):
    process = psutil.Process()
    env = QuantumCircuitEnv(num_qubits, max_gates, p_meas, p_gate1, p_gate2, T1, T2)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DDQNAgent(state_size, action_size)
    rewards, depths, qfis, entanglements, initial_depths, gate_counts, initial_gate_counts = [], [], [], [], [], [], []
    error_rates_list,initial_entanglements,initial_qfis = [], [] ,[]
    episode_times = []
    episode_memories = []
    #loaded_circuits = load_circuits_from_file('quantum_circuits_100.qasm')
    #print("Size of load:",len(loaded_circuits))
    #size = len(loaded_circuits)
    tket_count = 0
    qiskit_count = 0
    total_memory = 0
    for e in range(num_episodes):
        gc.collect()
        episode_start_time = time.time()
        episode_start_memory = process.memory_info().rss / 1024 / 1024
        #circ = loaded_circuits[e]
        state = env.reset(None)
        initial_depths.append(env.initial_depth)
        initial_gate_counts.append(env.initial_gate_count)
        initial_entanglements.append(env.initial_entanglement)
        initial_qfis.append(env.initial_qfi)
        max_gates = env.max_gates
        total_reward = 0

        print(f"\nEpisode {e + 1}/{num_episodes}")
        print("Initial Circuit:")
        #print(env.qc)
        i_circuit_image = circuit_drawer(env.qc, output='mpl', fold=50)
        i_circuit_image.savefig("100T/initial.png")
        print(f"Initial depth: {env.initial_depth}, QFI: {env.initial_qfi:.4f}, Entanglement: {env.initial_entanglement:.4f}, Gate Count: {env.initial_gate_count}")

        best_reward = 0
        best_circuit = env.qc.copy()
        best_depth = env.initial_depth
        best_entanglement = env.initial_entanglement
        best_qfi = env.initial_qfi
        best_gates = len(env.qc.data)
        for _ in range (50):  # Allow more steps for optimization
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            current_entanglement = env.current_entropy
            current_qfi = env.current_qfi
            c_depth = env.qc.depth()
            c_gates = len(env.qc.data)
            if (current_entanglement >= 0.999 and c_gates <= env.initial_gate_count):
                done = True
            print(f"--reward:{reward},entropy:{current_entanglement},qfi:{current_qfi},depth:{env.qc.depth()},gates:{len(env.qc.data)}")
            if(best_entanglement <= current_entanglement or current_qfi >= best_qfi) :
                best_circuit = env.qc.copy()
                best_depth = env.qc.depth()
                best_entanglement = current_entanglement
                best_qfi = current_qfi
                best_gates = len(env.qc.data)
            if done:
                break

        agent.replay()
        agent.update_epsilon()
        if e % 3 == 0:
            agent.update_target_model()
        '''    
        if e % 50 == 0 and e > 0:  # Reset every 50 episodes
            state = env.reset()
            print("Resetting circuit...")
        '''
        env.qc = best_circuit

        final_qfi = best_qfi
        final_entanglement = best_entanglement

        optimized_qc = env.optimize_depth()
        env.qc = optimized_qc
        final_qfi1 = env._compute_qfi()
        final_entanglement1 = env._compute_entanglement(None)
        print(f"QFI1: {final_qfi1:.4f}, Entanglement1: {final_entanglement1:.4f},")
        if(final_qfi1 >= final_qfi and final_entanglement1 >= final_entanglement) :
            final_qfi = final_qfi1
            final_entanglement = final_entanglement1
            tket_count += 1
        else :

            env.qc = best_circuit
            qiskit_count += 1

        final_gate_count = len(env.qc.data)
        final_depth = env.qc.depth()
        print(
            f"Final depth: {final_depth}, QFI: {final_qfi:.4f}, Entanglement: {final_entanglement:.4f},"
            f" Gate Count: {final_gate_count}")
        print("\nOptimized Circuit:")
        #print(env.qc)
        final_circuit_image = circuit_drawer(env.qc, output='mpl', fold=50)
        final_circuit_image.savefig("100T/final.png")


        rewards.append(total_reward)
        depths.append(final_depth)
        if final_qfi is not None:
            qfis.append(final_qfi)
        else:
            print("finalQFI NULL",final_qfi)

        entanglements.append(final_entanglement)
        gate_counts.append(final_gate_count)
        error_rates_list.append(env.error_rates)
        print(f"Depth change: {env.initial_depth - final_depth}")
        print(f"QFI change: {final_qfi - env.initial_qfi:.4f}")
        print(f"Entanglement change: {final_entanglement - env.initial_entanglement:.4f}")
        print(f"Gate count change: {env.initial_gate_count - final_gate_count}")
        print(f"Total reward: {total_reward:.2f}")
        print("-------------------------")
        episode_end_time = time.time()
        episode_end_memory = process.memory_info().rss / 1024 / 1024
        episode_times.append(episode_end_time - episode_start_time)
        memory_change = episode_end_memory - episode_start_memory
        episode_memories.append(memory_change)
        total_memory += memory_change
        np.save("100T/episode_memories.npy", episode_memories)

    agent.save(f"100T/ddqn_model_episode_{e}_100T.weights.h5")
    print("Total memory consumption",total_memory)
    import pickle

    # Dictionary to store all the lists
    data_dict = {
        'depths': depths,
        'qfis': qfis,
        'entanglements': entanglements,
        'initial_depths': initial_depths,
        'gate_counts': gate_counts,
        'initial_gate_counts': initial_gate_counts,
        'error_rates_list': error_rates_list,
        'initial_entanglements': initial_entanglements,
        'initial_qfis': initial_qfis
    }

    # Save to a pickle file
    with open('data.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

    visualize_training_results(rewards, depths, qfis, entanglements, initial_depths,
                               gate_counts, initial_gate_counts,error_rates_list,initial_entanglements,initial_qfis,episode_times,episode_memories)
    # Plot the results
    plt.close()
    plt.figure(figsize=(12, 8))
    labels = ['Tket', 'Qiskit']
    counts = [tket_count, qiskit_count]
    print(counts)
    plt.bar(labels, counts, color=['blue', 'green'])
    plt.ylabel('Number of Optimizations')
    plt.title('Optimization Comparison: Tket vs Qiskit')
    plt.savefig('100T/tket_vs_qiskit.png', dpi=300)
    plt.close()
    return agent


def test_trained_agent(agent, num_qubits, num_gates):
    env = QuantumCircuitEnv(num_qubits, num_gates)
    loaded_circuits = load_circuits_from_file('quantum_circuits_25.qasm')
    initial_state = env.reset(loaded_circuits[0])
    #env.qc = create_circuit()
    print("Initial Circuit:")
    print(env.qc)
    initial_circuit_image = circuit_drawer(env.qc, output='mpl',fold=50)
    initial_circuit_image.savefig("100T/initial_QCirc55.png")
    initial_depth = env.qc.depth()
    initial_qfi = env._compute_qfi()
    initial_entanglement = env._compute_entanglement(None)
    initial_gate_count = len(env.qc.data)

    state = initial_state
    best_circuit = env.qc.copy()
    best_qfi = initial_qfi
    best_depth = initial_depth
    best_entanglement = initial_entanglement
    best_reward = 0
    rewards = []
    for _ in range(500):  # Allow more steps for optimization
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        current_qfi = env._compute_qfi()
        current_depth = env.qc.depth()
        current_entanglement = env._compute_entanglement(None)
        state = next_state
        # Update best circuit if current QFI is better
        if current_entanglement >= best_entanglement and current_depth <= best_depth :
            best_circuit = env.qc.copy()
            best_qfi = current_qfi
            best_depth = current_depth
            best_entanglement = current_entanglement
            best_reward =  reward

        if done:
            break

    # 1. Learning Curve Plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(rewards)), rewards)
    plt.title('Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('100T/learning_curve_test.png', dpi=300)
    plt.close()
    print("Rewards:",rewards)

    print("\nBest Optimized Circuit:")
    print(best_circuit)
    final_circuit_image = circuit_drawer(best_circuit,
                                         output='mpl',fold=50)
    final_circuit_image.savefig("100T/final_QCirc55.png")
    final_depth = best_depth
    final_qfi = best_qfi
    final_entanglement = best_entanglement
    final_gate_count = len(best_circuit.data)

    print(f"\nInitial depth: {initial_depth}, Final depth: {final_depth}")
    print(f"Initial QFI: {initial_qfi:.4f}, Final QFI: {final_qfi:.4f}")
    print(f"Initial entanglement: {initial_entanglement:.4f}, Final entanglement: {final_entanglement:.4f}")
    print(f"Depth reduction: {initial_depth - final_depth}")
    print(f"QFI improvement: {final_qfi - initial_qfi:.4f}")
    print(f"Entanglement improvement: {final_entanglement - initial_entanglement:.4f}")
    print(f"\nInitial gates: {initial_gate_count}, Final gates: {final_gate_count}")
    print(f"Gate count change: {initial_gate_count - final_gate_count}")

    optimized_qc = env.optimize_depth()
    env.qc = optimized_qc
    final_qfi1 = env._compute_qfi()
    final_entanglement1 = env._compute_entanglement(None)
    final_depth1 = env.qc.depth()
    final_gate1 = len(env.qc.data)
    print(f"QFI1: {final_qfi1:.4f}, Entanglement1: {final_entanglement1:.4f},depth1: {final_depth1:.4f},gate counts1: {final_gate1:.4f}")
    if(final_qfi1 >= final_qfi and final_entanglement1 >= final_entanglement) :
        print(optimized_qc)
        # Adjust the parameters for better representation
        final_circuit_image = circuit_drawer(
            optimized_qc,
            output='mpl',
            plot_barriers = False,
            fold=30,  # Increase the fold to reduce the number of lines
            initial_state=True,  # Optionally display the initial state
            cregbundle=False,  # Prevent classical registers from bundling (if they exist)
            idle_wires=False,  # Hide idle wires to remove distractions
            style={'fontsize': 10, 'figwidth': 50, 'figheight': 20}  # Set fontsize and figure size
        )
        final_circuit_image.savefig("100T/reduced_circ.png")

    return best_circuit, final_qfi, final_depth, final_entanglement


def create_circuit():
    qc = QuantumCircuit(8)

    # Layer 1
    qc.h(0)
    qc.rx(0.01, 1)
    qc.rz(0.02, 2)
    qc.rx(0.015, 3)
    qc.rz(0.025, 4)
    qc.h(5)
    qc.rx(0.03, 6)
    qc.rz(0.035, 7)


    # Layers 2-17
    for _ in range(8):
        qc.cx(0, 1)
        qc.cx(2, 3)
        qc.cx(4, 5)
        qc.cx(6, 7)
        qc.rx(0.01, 0)
        qc.rz(0.01, 1)
        qc.rx(0.01, 2)
        qc.rz(0.01, 3)

    # Layer 18
    qc.h(0)
    qc.rx(0.01, 1)
    qc.rz(0.02, 2)
    qc.rx(0.015, 3)
    qc.rz(0.025, 4)
    qc.h(5)
    qc.rx(0.03, 6)
    qc.rz(0.035, 7)

    return qc

if __name__ == "__main__":
    # Train the agent

    start_time = time.time()
    trained_agent = train_ddqn_agent(num_episodes=1, num_qubits=60, max_gates=100)
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    '''''
    env = QuantumCircuitEnv(25, 150)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    trained_agent = DDQNAgent(state_size, action_size)
    trained_agent.load("100T/ddqn_model_episode_4_100T.weights.h5")
    # Test the trained agent
    test_trained_agent(trained_agent, num_qubits=25, num_gates=150)
    '''''


