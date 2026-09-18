import numpy as np
import cirq

def TrotterStep(Number_of_Fock_States, Number_of_Bosonic_Modes, Time):
    qubits = cirq.LineQubit.range(Number_of_Bosonic_Modes * (Number_of_Fock_States+1))
    Trotter_circuit = cirq.Circuit()

    for j in range(Number_of_Bosonic_Modes):
        for i in range(0, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

        for i in range(1, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

    for j in range(0, Number_of_Bosonic_Modes-1, 2):
        Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range(1, Number_of_Bosonic_Modes-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range (Number_of_Bosonic_Modes):
         Trotter_circuit.append(cirq.CX(qubits[(j+1)*Number_of_Fock_States-1],qubits[Number_of_Fock_States*Number_of_Bosonic_Modes+j]))

    return Trotter_circuit, qubits

def TrotterStepCZ(Number_of_Fock_States, Number_of_Bosonic_Modes, Time):
    qubits = cirq.LineQubit.range(Number_of_Bosonic_Modes * (Number_of_Fock_States+1))
    Trotter_circuit = cirq.Circuit()

    for j in range(Number_of_Bosonic_Modes):
        for i in range(0, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

        for i in range(1, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

    for j in range(0, Number_of_Bosonic_Modes-1, 2):
        Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range(1, Number_of_Bosonic_Modes-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range (Number_of_Bosonic_Modes):
         Trotter_circuit.append(cirq.CZ(qubits[(j+1)*Number_of_Fock_States-1],qubits[Number_of_Fock_States*Number_of_Bosonic_Modes+j]))

    return Trotter_circuit, qubits