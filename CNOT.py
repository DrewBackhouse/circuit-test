import numpy as np
import cirq
import matplotlib.pyplot as plt

from Functions import TrotterStep

Number_of_Fock_States = 6
Number_of_Bosonic_Modes = 1
Time= 0.7
Timesteps=40
Number_of_Shots = 5000

Trotter_circuit, qubits = TrotterStep(Number_of_Fock_States, Number_of_Bosonic_Modes, Time)
print(Trotter_circuit)

All_Results = []
for i in range(1, Timesteps+1):
    circuit = cirq.Circuit()
    for j in range(Number_of_Bosonic_Modes):
        circuit.append(cirq.X(qubits[j*Number_of_Fock_States]))
    circuit.append([Trotter_circuit]*i)
    circuit.append(cirq.measure(*qubits, key='m'))
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
# print(All_Results)

Time_Data = np.linspace(Time, Time*Timesteps, Timesteps)
# print(Time_Data)

plt.figure()
for i in range(Number_of_Bosonic_Modes):
    plt.plot(Time_Data, All_Results[:,i], label = f'Mode {i}')
plt.legend()
plt.title(f'N={Number_of_Fock_States}, L={Number_of_Bosonic_Modes}, Trotter time ={Time}, Shots={Number_of_Shots}')
plt.ylabel('Average Bosonic occupation number')
plt.xlabel('Time')

plt.figure()
for i in range(Number_of_Bosonic_Modes):
    plt.plot(Time_Data, All_Results[:,i+Number_of_Bosonic_Modes], label = f'Spin {i}')
plt.legend()
plt.title(f'N={Number_of_Fock_States}, L={Number_of_Bosonic_Modes}, Trotter time ={Time}, Shots={Number_of_Shots}')
plt.ylabel('Average Spin state')
plt.xlabel('Time')

plt.show()