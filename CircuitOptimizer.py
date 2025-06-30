'''
 @Author: LAXMISH
 @Created on: 7/15/2024
'''
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import partial_trace, Statevector, entropy
from qiskit_aer import Aer


class CircuitOptimizer:
    def __init__(self, circuit,parameter_count):
        self.original_circuit = circuit
        self.reset_optimization()
        self.parameter_counter = parameter_count

    def reset_optimization(self):
        self.circuit = self.original_circuit.copy()
        self.gates = self._extract_gates()

    def _extract_gates(self):
        gates = []
        for instruction in self.circuit.data:
            name = instruction[0].name
            qubits = tuple(self.circuit.find_bit(q).index for q in instruction[1])
            params = instruction[0].params if hasattr(instruction[0], 'params') else []
            gates.append((name, qubits, params))
        return gates

    def optimize(self):
        iteration = 0
        max_iteration = 10

        while iteration <= max_iteration:
            previous_gate_count = len(self.gates)
            self._optimize_once()
            current_gate_count = len(self.gates)

            if current_gate_count <= previous_gate_count:
                break

            iteration += 1
            print(f"Iteration {iteration}: Reduced from {previous_gate_count} to {current_gate_count} gates")
            #opt_circuit = self._create_optimized_circuit()
            #print(opt_circuit)

        return self._create_optimized_circuit(),self.parameter_counter

    def _optimize_once(self):
        simplified = True
        while simplified:
            simplified = False
            i = 0
            while i < len(self.gates) - 1:
                if self._simplify_sequence(i):
                    simplified = True
                else:
                    i += 1
        self._remove_identity_gates()

    def _simplify_sequence(self, i):
        # Check for H - CX - H pattern
        if i + 2 < len(self.gates):
            gate1, gate2, gate3 = self.gates[i], self.gates[i + 1], self.gates[i + 2]
            if (gate1[0] == 'h' and gate2[0] == 'cx' and gate3[0] == 'h' and
                    gate1[1][0] == gate3[1][0] == gate2[1][0]):
                # Replace H - CX - H with reversed CX
                self.gates[i] = ('cx', (gate2[1][1], gate2[1][0]), [])
                self.gates[i + 1] = ('id', gate2[1][0:1], [])
                self.gates[i + 2] = ('id', gate2[1][1:2], [])
                return True

        # Other simplification rules
        if i + 1 < len(self.gates):
            return self._simplify_pair(i)

        return False

    def _simplify_pair(self, i):
        gate1, gate2 = self.gates[i], self.gates[i + 1]

        # Rule 1: Simplify adjacent identical single-qubit gates
        if gate1[0] == gate2[0] and gate1[1] == gate2[1] and gate1[0] in ['h', 'rx', 'rz']:
            if gate1[0] in ['rx', 'rz']:
                # Handle parameterized gates
                if isinstance(gate1[2][0], Parameter) or isinstance(gate2[2][0], Parameter):
                    new_param = Parameter(f"{gate1[0]}_{self.parameter_counter}")
                    self.parameter_counter += 1
                    self.gates[i] = (gate1[0], gate1[1], [new_param])
                else:
                    new_angle = (gate1[2][0] + gate2[2][0]) % (2 * np.pi)
                    self.gates[i] = (gate1[0], gate1[1], [new_angle])
            self.gates[i + 1] = ('id', gate1[1], [])
            return True

        # Rule 2: Simplify adjacent CNOT gates
        if gate1[0] == 'cx' and gate2[0] == 'cx':
            if gate1[1] == gate2[1]:  # Same control and target
                self.gates[i + 1] = ('id', gate1[1][0:1], [])
                return True
            if gate1[1] == gate2[1][::-1]:  # Swapped control and target
                self.gates[i:i + 2] = [
                    ('h', gate1[1][0:1], []), ('h', gate1[1][1:2], []),
                    ('cx', gate1[1], []),
                    ('h', gate1[1][0:1], []), ('h', gate1[1][1:2], [])
                ]
                return True

        # Rule 3: Simplify CZ to CX
        if gate1[0] == 'h' and gate2[0] == 'cz' and i + 2 < len(self.gates):
            gate3 = self.gates[i + 2]
            if gate3[0] == 'h' and gate1[1] == gate3[1] == gate2[1][1:2]:
                self.gates[i:i + 3] = [('cx', gate2[1], [])]
                return True

        # Rule 4: Simplify SWAP gate
        if gate1[0] == 'swap':
            self.gates[i:i + 1] = [
                ('cx', gate1[1], []),
                ('cx', gate1[1][::-1], []),
                ('cx', gate1[1], [])
            ]
            return True

        return False

    def _remove_identity_gates(self):
        self.gates = [gate for gate in self.gates if gate[0] != 'id']



    def _create_optimized_circuit(self):
        optimized_circuit = QuantumCircuit(self.circuit.num_qubits)
        for gate in self.gates:
            if gate[0] == 'h':
                optimized_circuit.h(gate[1][0])
            elif gate[0] == 'cx':
                optimized_circuit.cx(gate[1][0], gate[1][1])
            elif gate[0] == 'rx':
                optimized_circuit.rx(gate[2][0], gate[1][0])
            elif gate[0] == 'rz':
                optimized_circuit.rz(gate[2][0], gate[1][0])
            elif gate[0] == 'cz':
                optimized_circuit.cz(gate[1][0], gate[1][1])
            elif gate[0] == 'swap':
                optimized_circuit.swap(gate[1][0], gate[1][1])
        return optimized_circuit


