import numpy as np
import cirq
import cirq_google
import qsimcirq
import matplotlib.pyplot as plt
from qutip import about, basis, tensor, destroy, mcsolve, mesolve, expect, qeye, sigmax, sigmay, sigmaz, fock, wigner, coherent

from Functions import TrotterStepCRZ, QutipHamiltonian, MapQubitsToDevice, PlotQubitEmbedding, UnaryPostSelection, TrotterStepCZ_New

#---Parameters---#

Number_of_Fock_States = 6
Number_of_Bosonic_Modes = 3
Displacement_Coefficent = 1 # Independnet of the physics (as far as I am aware, at least when it is global)
Spin_Interaction_Coefficent = 0.1
Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (Number_of_Fock_States -0.5)**0.5 # Value for a full spin flip

Time= np.pi/(2*Displacement_Coefficent*(Number_of_Fock_States-0.5)**0.5) # Trotter step time s.t. the controlled RZ gate becomes a CZ
Timesteps=20
Number_of_Shots = 500

#---Qutip Sim---#

Qutip_Time = Time * Timesteps
state_list = [tensor(basis(2,0),fock(Number_of_Fock_States,0)) for _ in range(Number_of_Bosonic_Modes)]       # Tensor product of spin and boson vectors in ground state at each lattice point in a list 
psi0 = tensor(state_list)                                           # Tensor product of all entries in the list
Qutip_Time_Data = np.linspace(0, Qutip_Time, int(Qutip_Time*10))
H, n_list, sz_list = QutipHamiltonian(Number_of_Bosonic_Modes, Number_of_Fock_States, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)
result = mesolve(H, psi0, Qutip_Time_Data, args={'Displacement_Coefficent': Displacement_Coefficent, 'Spin_Interaction_Coefficent': Spin_Interaction_Coefficent, 'Spin_Boson_Interaction_Coefficent': Spin_Boson_Interaction_Coefficent})
states = result.states
exp_n = np.array([expect(n_list[i], states) for i in range(Number_of_Bosonic_Modes)])
exp_sz = np.array([expect(sz_list[i], states) for i in range(Number_of_Bosonic_Modes)])

#---QVM Setup---#

processor_id = "willow_pink"
noise_props = cirq_google.engine.load_device_noise_properties(processor_id)
noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(noise_props)
qsim_options = qsimcirq.QSimOptions(cpu_threads=4, max_fused_gate_size=3)       # Fastest measured on this machine (default is 1 thread, fusion 2)
sim = qsimcirq.QSimSimulator(qsim_options, noise=noise_model)
device = cirq_google.engine.create_device_from_processor_id(processor_id)
cal = cirq_google.engine.load_median_device_calibration(processor_id)
sim_processor = cirq_google.engine.SimulatedLocalProcessor(
    processor_id=processor_id, sampler=sim, device=device, calibrations={cal.timestamp // 1000: cal}
)
sim_engine = cirq_google.engine.SimulatedLocalEngine([sim_processor])

#---Trotter step circuit preperation---#

Trotter_circuit, qubits = TrotterStepCZ_New(Number_of_Fock_States, Number_of_Bosonic_Modes, Time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)
print('Trotter step circuit')
print(Trotter_circuit)
qubits_willow = MapQubitsToDevice(qubits, Trotter_circuit, device, cal)
Trotter_circuit = Trotter_circuit.transform_qubits(dict(zip(qubits, qubits_willow)))
qubits = qubits_willow
PlotQubitEmbedding(cal, qubits_willow, Trotter_circuit, Number_of_Fock_States, Number_of_Bosonic_Modes)
Trotter_circuit = cirq.optimize_for_target_gateset(Trotter_circuit, gateset=cirq.CZTargetGateset())
print('Trotter step circuit mapped to native gates')
print(Trotter_circuit.to_text_diagram(qubit_order=qubits_willow))

input('Press enter to continue to simulation...')

#---Cirq Sim---#

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
    Full_Results = sim_engine.get_sampler(processor_id).run(circuit, repetitions=Number_of_Shots).measurements['m']
    Full_Results_Postselected = UnaryPostSelection(Full_Results, Number_of_Fock_States, Number_of_Bosonic_Modes)
    Averaged_Results = Full_Results_Postselected.mean(axis=0)
    Boson_Results = Averaged_Results[:Number_of_Bosonic_Modes*Number_of_Fock_States].reshape(Number_of_Bosonic_Modes, Number_of_Fock_States)
    Boson_Occupation_Number = np.sum(Boson_Results * np.arange(0, Number_of_Fock_States, 1), axis=1)
    Timestep_Results = np.concatenate([Boson_Occupation_Number, Averaged_Results[Number_of_Bosonic_Modes*Number_of_Fock_States:]])
    All_Results.append(Timestep_Results)
    print(f'timestep {i} complete, {len(Full_Results_Postselected)}/{Number_of_Shots} shots kept')

All_Results = np.array(All_Results)
Time_Data = np.linspace(Time, Time*Timesteps, Timesteps)

#---Plots---#

plt.figure()
for i in range(Number_of_Bosonic_Modes):
    plt.plot(Qutip_Time_Data, exp_n[i])
    plt.scatter(Time_Data, All_Results[:,i], label = f'Mode {i}')
plt.legend()
plt.title(f'N={Number_of_Fock_States}, L={Number_of_Bosonic_Modes}, Trotter time ={Time:.2f}, Shots={Number_of_Shots}, with noise and postselection')
plt.ylabel('Average Bosonic occupation number')
plt.xlabel('Time')

plt.figure()
for i in range(Number_of_Bosonic_Modes):
    plt.plot(Qutip_Time_Data, -(exp_sz[i]-1)/2)
    plt.scatter(Time_Data, All_Results[:,i+Number_of_Bosonic_Modes], label = f'Spin {i}')
plt.legend()
plt.title(f'N={Number_of_Fock_States}, L={Number_of_Bosonic_Modes}, Trotter time ={Time:.2f}, Shots={Number_of_Shots}, with noise and postselection')
plt.ylabel('Average Spin state')
plt.xlabel('Time')

plt.show()