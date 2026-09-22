import numpy as np
import cirq
import matplotlib.pyplot as plt
from qutip import about, basis, tensor, destroy, mcsolve, mesolve, expect, qeye, sigmax, sigmay, sigmaz, fock, wigner, coherent
from scipy.optimize import minimize_scalar

from Functions import TrotterStepCRZ, QutipHamiltonian

Number_of_Fock_States = 15
Number_of_Bosonic_Modes = 1
Displacement_Coefficent = 1
Spin_Interaction_Coefficent = 1
Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (Number_of_Fock_States -0.5)**0.5

Time= np.pi/(2*Displacement_Coefficent*(Number_of_Fock_States-0.5)**0.5)
Timesteps=40
Number_of_Shots = 2000

#---Qutip Sim---#

Qutip_Time = Time * Timesteps

state_list = [tensor(basis(2,0),fock(Number_of_Fock_States,0)) for _ in range(Number_of_Bosonic_Modes)]       # Tensor product of spin and boson vectors in ground state at each lattice point in a list 
psi0 = tensor(state_list)                                           # Tensor product of all entries in the list
Qutip_Time_Data = np.linspace(0, Qutip_Time, int(Qutip_Time*10))

H, n_list, sz_list = QutipHamiltonian(Number_of_Bosonic_Modes, Number_of_Fock_States, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)

result = mesolve(H, psi0, Qutip_Time_Data, args={'Displacement_Coefficent': Displacement_Coefficent, 'Spin_Interaction_Coefficent': Spin_Interaction_Coefficent, 'Spin_Boson_Interaction_Coefficent': Spin_Boson_Interaction_Coefficent})
states = result.states

# Calculate expectation values
exp_n = np.array([expect(n_list[i], states) for i in range(Number_of_Bosonic_Modes)])
exp_sz = np.array([expect(sz_list[i], states) for i in range(Number_of_Bosonic_Modes)])


#---Cirq Sim---#

Trotter_circuit, qubits = TrotterStepCRZ(Number_of_Fock_States, Number_of_Bosonic_Modes, Time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)
print(Trotter_circuit)

All_Results = []
for i in range(1, Timesteps+1):
    circuit = cirq.Circuit()
    for j in range(Number_of_Bosonic_Modes):
        circuit.append(cirq.X(qubits[j*Number_of_Fock_States]))
    for j in range(Number_of_Bosonic_Modes):
            circuit.append(cirq.H(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j]))
    circuit.append([Trotter_circuit]*i)
    for j in range(Number_of_Bosonic_Modes):
                circuit.append(cirq.H(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j]))
    circuit.append(cirq.measure(*qubits, key='m'))
    if i == 3:
          print(circuit)
    Full_Results = cirq.Simulator().run(circuit, repetitions=Number_of_Shots).measurements['m']
    # print(Full_Results) # shape = (repetitions, Number of Qubits).
    Averaged_Results = Full_Results.mean(axis=0)
    # print(Averaged_Results) # Each bosonic mode still always adds to 1.
    Boson_Results = Averaged_Results[:Number_of_Bosonic_Modes*Number_of_Fock_States].reshape(Number_of_Bosonic_Modes, Number_of_Fock_States)
    # print(Boson_Results)
    Boson_Occupation_Number = np.sum(Boson_Results * np.arange(0, Number_of_Fock_States, 1), axis=1)
    # print(Boson_Occupation_Number)
    Timestep_Results = np.concatenate([Boson_Occupation_Number, Averaged_Results[Number_of_Bosonic_Modes*Number_of_Fock_States:]])
    # print(Timestep_Results)
    All_Results.append(Timestep_Results)
    print(f'timestep {i} complete')

All_Results = np.array(All_Results)
print(All_Results)

Time_Data = np.linspace(Time, Time*Timesteps, Timesteps)
# print(Time_Data)

plt.figure()
for i in range(Number_of_Bosonic_Modes):
    plt.plot(Qutip_Time_Data, exp_n[i])
    plt.scatter(Time_Data, All_Results[:,i], label = f'Mode {i}')
plt.legend()
plt.title(f'N={Number_of_Fock_States}, L={Number_of_Bosonic_Modes}, Trotter time ={Time:.2f}, Shots={Number_of_Shots}')
plt.ylabel('Average Bosonic occupation number')
plt.xlabel('Time')

plt.figure()
for i in range(Number_of_Bosonic_Modes):
    plt.plot(Qutip_Time_Data, -(exp_sz[i]-1)/2)
    plt.scatter(Time_Data, All_Results[:,i+Number_of_Bosonic_Modes], label = f'Spin {i}')
plt.legend()
plt.title(f'N={Number_of_Fock_States}, L={Number_of_Bosonic_Modes}, Trotter time ={Time:.2f}, Shots={Number_of_Shots}')
plt.ylabel('Average Spin state')
plt.xlabel('Time')

plt.show()