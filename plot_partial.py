import numpy as np
import matplotlib.pyplot as plt

d = np.load('CRZ_partial_results.npz')

Time_Data = d['Time_Data']
All_Results = d['All_Results']
Qutip_Time_Data = d['Qutip_Time_Data']
exp_n = d['exp_n']
exp_sz = d['exp_sz']

Number_of_Fock_States = int(d['Number_of_Fock_States'])
Number_of_Bosonic_Modes = int(d['Number_of_Bosonic_Modes'])
Time = float(d['Time'])
Number_of_Shots = int(d['Number_of_Shots'])
title = f'N={Number_of_Fock_States}, L={Number_of_Bosonic_Modes}, Trotter time ={Time}, Shots={Number_of_Shots}'
title += f' ({int(d["Timesteps_completed"])}/{int(d["Timesteps_planned"])} timesteps)'

plt.figure()
for i in range(Number_of_Bosonic_Modes):
    plt.plot(Qutip_Time_Data, exp_n[i])
    plt.scatter(Time_Data, All_Results[:, i], label=f'Mode {i}')
plt.legend()
plt.title(title)
plt.ylabel('Average Bosonic occupation number')
plt.xlabel('Time')

plt.figure()
for i in range(Number_of_Bosonic_Modes):
    plt.plot(Qutip_Time_Data, -(exp_sz[i]-1)/2)
    plt.scatter(Time_Data, All_Results[:, i+Number_of_Bosonic_Modes], label=f'Spin {i}')
plt.legend()
plt.title(title)
plt.ylabel('Average Spin state')
plt.xlabel('Time')

plt.show()
