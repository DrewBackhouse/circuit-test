import numpy as np
import cirq
from qutip import about, basis, tensor, destroy, mcsolve, mesolve, expect, qeye, sigmax, sigmay, sigmaz, fock, wigner, coherent
from scipy.optimize import minimize_scalar

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

def TrotterStepCRZ(Number_of_Fock_States, Number_of_Bosonic_Modes, Time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent):
    qubits = cirq.LineQubit.range(Number_of_Bosonic_Modes * (Number_of_Fock_States+1))
    Trotter_circuit = cirq.Circuit()

    for j in range(Number_of_Bosonic_Modes):
        for i in range(0, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Displacement_Coefficent*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

        for i in range(1, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Displacement_Coefficent*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

    for j in range(0, Number_of_Bosonic_Modes-1, 2):
        Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Spin_Interaction_Coefficent*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range(1, Number_of_Bosonic_Modes-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Spin_Interaction_Coefficent*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range (Number_of_Bosonic_Modes):
         Trotter_circuit.append(cirq.rz(2*Time*Spin_Boson_Interaction_Coefficent).controlled()(qubits[(j+1)*Number_of_Fock_States-1],qubits[Number_of_Fock_States*Number_of_Bosonic_Modes+j]))

    return Trotter_circuit, qubits

def QutipHamiltonian(NumberOfBosonicModes, NumberOfFockStates, displacement_coefficient, spin_interaction_coefficient, SpinBosonInteractionCoefficent):
    
    sx_list, sy_list, sz_list = [], [], []                       # Define pauli X and Y operator lists 
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2),qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]       # Define list of 2D and ND identity operators for each lattice site
        op_list[i] = tensor(sigmax(), qeye(NumberOfFockStates))                      # Replace ith entry with pauli X and I_N
        sx_list.append(tensor(op_list))                             # Add to Pauli X operator list
        op_list[i] = tensor(sigmay(), qeye(NumberOfFockStates))                      # Repeat for Pauli Y
        sy_list.append(tensor(op_list))
        op_list[i] = tensor(sigmaz(), qeye(NumberOfFockStates))                      # Repeat for Pauli Z
        sz_list.append(tensor(op_list))

    a_list = []                                     # Repeat for ladder and number operators
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2),qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]
        op_list[i] = tensor(qeye(2), destroy(NumberOfFockStates))
        a_list.append(tensor(op_list))

    n_list = []
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2),qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]
        op_list[i] = tensor(qeye(2), destroy(NumberOfFockStates).dag() * destroy(NumberOfFockStates))
        n_list.append(tensor(op_list))

    n_max_list = []                                 # Treat |n*><n*| as an operator and repeat
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2),qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]
        op_list[i] = tensor(qeye(2), fock(NumberOfFockStates, NumberOfFockStates-1) * fock(NumberOfFockStates, NumberOfFockStates-1).dag())
        n_max_list.append(tensor(op_list))

    H = sum(displacement_coefficient * (a_list[i] + a_list[i].dag()) + SpinBosonInteractionCoefficent * n_max_list[i] * sx_list[i] for i in range(NumberOfBosonicModes))
    H += sum(spin_interaction_coefficient * (sx_list[i] * sx_list[i+1] + sy_list[i] * sy_list[i+1])/2 for i in range(NumberOfBosonicModes-1))
    
    return H, n_list, sz_list