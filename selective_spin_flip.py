# %% [markdown]
# # selective spin flip, full Hamiltonian
# Here we consider the full system Hamiltonian (cavity + auxiliary qubit + coupling terms)
# drift Hamiltoinan is given by,
# $$
# H_d = \text{wr} * n_{\text{cavity}} + \text{Ez} * \sigma_z + \text{chi} * \sigma_z * (n_{\text{cavity}} + 1/2 )
# $$
# where,
# - wr: resonator frequency
# - Ez: qubit anharmonicity
# - chi: qubit-cavity coupling strength
# 
# Real control channels in the Hamiltonain are given by:
# $$
# H_c = a + a^{\dagger} + -1j*(a - a^{\dagger}) + \sigma_x
# $$
# state preparation optimization target:
# \begin{align*}
#     \psi_0 &= \frac{1}{\sqrt{2}} (\ket{0}_{\text{cavity}} \otimes \ket{0}_{\text{qubit}}
#                 + \ket{1}_{\text{cavity}} \otimes \ket{0}_{\text{qubit}})\\
#     \psi_{\text{targ}} &= \frac{1}{\sqrt{2}} (\ket{0}_{\text{cavity}} \otimes \ket{0}_{\text{qubit}}
#                 + \ket{1}_{\text{cavity}} \otimes \ket{1}_{\text{qubit}})\\
# \end{align*}
# where qubit spin flip is selective on cavity state

# %%
!pip install qutip

# %%
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import datetime
from qutip import *
import random
import qutip.logging_utils as logging
logger = logging.get_logger()
#Set this to None or logging.WARN for 'quiet' execution
log_level = logging.INFO
#QuTiP control modules
import qutip.control.pulseoptim as cpo

# %% [markdown]
# ## Physical system parameters

# %%
# time = 1 -> GHz and ns
# time = 1000 -> MHz and us
time = 1
# spin flip, from chiyuan, pulse length is ns scale
# vac2cat, from chiyuan, pulse length is ms scale
g = 0.05*2*np.pi *time # cavity qubit coupling strength
N = 16 # Fock space truncation
wr = 1.0*2*np.pi *time# resonator frequency
Ez = 0.22*2*np.pi *time# qubit anharmonicity
chi = 0.007 *time # qubit-cavity coupling strength

#### set up the operators ####
a = tensor(destroy(N),  qeye(2))
n_cavity = tensor(num(N), qeye(2))
Sigmaz = tensor(qeye(N),sigmaz())
Sigmax = tensor(qeye(N),sigmax())
Sigmam = tensor(qeye(N),  destroy(2))
Id = tensor(qeye(N),  qeye(2))
Sigmap = Sigmam.dag()

# H_d = wr * n_cavity + Ez * Sigmaz + chi * Sigmaz * (n_cavity + Id/2)
# corrected:
H_d = wr * n_cavity + Ez/2 * Sigmaz - chi * Sigmaz * (n_cavity + Id/2 )

H_c = [a + a.dag(),
       -1j*(a - a.dag()),
       Sigmax]


#### set Hamiltonian ####
n_ctrls = len(H_c)

H_labels = [r'$ReD$',
            r'$ImD2$',
            r'$R$']

#### initial and target states ####
alpha = np.sqrt(1) # coherent state complex amplitude

# vacuum to cat
# psi_0 = tensor(fock(N, 0), basis(2, 0))
# psi_targ = tensor((coherent(N, alpha)+coherent(N, -alpha)).unit(),  basis(2, 0))

# unselective spin rotation
# psi_0 = tensor(fock(N, 0), basis(2, 0))
# psi_targ = tensor(fock(N, 0), basis(2, 1))

# selective spin flip
psi_0 = (tensor(fock(N, 1), basis(2, 0)) + tensor(fock(N, 0), basis(2, 0))).unit()
psi_targ = (tensor(fock(N, 1), basis(2, 0)) + tensor(fock(N, 0), basis(2, 1))).unit() # selective wrt cavity state
U = psi_targ * psi_0.dag()

# %% [markdown]
# # Optimization parameters

# %%
# Number of time slots
n_ts = 100_000
# Time allowed for the evolution
evo_time = 500

# Fidelity error target
fid_err_targ = 1e-8
# Maximum iterations for the optisation algorithm
max_iter = 10000
# Maximum (elapsed) time allowed in seconds
max_wall_time = 7200

# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
# for GRAPE, this is initial pulse type
p_type = 'SINE'

# Set to None to suppress output files
# f_ext = "{}_n_ts{}_ptype{}.txt".format('v2cat', n_ts, p_type)
f_ext = None

# %% [markdown]
# ## Run the optimisation

# %%
# GRAPE
result = cpo.optimize_pulse_unitary(H_d, H_c, Id, U,
                n_ts, evo_time,
                fid_err_targ=fid_err_targ,
                max_iter=max_iter, max_wall_time=max_wall_time,
                method_params={'xtol':1e-3},
                init_pulse_type=p_type,
                out_file_ext=f_ext,
                log_level=log_level, gen_stats=True)

# result  = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ,
#                                      num_tslots=None, evo_time=None, tau=None, amp_lbound=None, amp_ubound=None,
#                                      fid_err_targ=1e-10, min_grad=1e-10, max_iter=500, max_wall_time=180,
#                                      alg='GRAPE', alg_params=None, optim_params=None, optim_method='DEF', method_params=None,
#                                      optim_alg=None, max_metric_corr=None, accuracy_factor=None, phase_option='PSU',
#                                      dyn_params=None, prop_params=None, fid_params=None,
#                                      tslot_type='DEF', tslot_params=None, amp_update_mode=None,
#                                      init_pulse_type='DEF', init_pulse_params=None, pulse_scaling=1.0, pulse_offset=0.0,
#                                      ramping_pulse_type=None, ramping_pulse_params=None, log_level=0, out_file_ext=None, gen_stats=False)

# %%
result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Final fidelity error {}".format(result.fid_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print("Completed in {} HH:MM:SS.US".format(
        datetime.timedelta(seconds=result.wall_time)))

# %% [markdown]
# ## Plot control pulses

# %%
def plot_amp(result,i):
    # i is the index of the control amplitude
    plt.figure(figsize=(15, 2))
    # title and labels
    plt.title(f"Control {i+1} Amplitudes")
    plt.xlabel("Time $ns$")
    plt.ylabel("Control Amplitude $GHz$")

    # Plot the initial control amplitudes in blue
    plt.step(result.time,
             np.hstack((result.initial_amps[:, i], result.initial_amps[-1, i])),
             where='post',alpha = 0.7, label='Initial Amplitude')

    # Plot the final control amplitudes in red
    plt.step(result.time,
             np.hstack((result.final_amps[:, i], result.final_amps[-1, i])),
             where='post', color='red',alpha = 0.7, label='Final Amplitude')

    # Add a legend to the plot
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_amp(result,0)
plot_amp(result,1)
plot_amp(result,2)

# %% [markdown]
# # Simulation

# %%
def cavityFidelity(psi1, psi2):
    '''
    psi1, psi2: Qobj
    '''
    return fidelity(ptrace(psi1, 0), ptrace(psi2, 0))

def qubitFidelity(psi1, psi2):
    '''
    psi1, psi2: Qobj
    '''
    return fidelity(ptrace(psi1, 1), ptrace(psi2, 1))

# %%
# H_c amplitudes (from pulse optimization)
amplitudes_0 = result.final_amps[:, 0]
amplitudes_1 = result.final_amps[:, 1]
amplitudes_2 = result.final_amps[:, 2]


tlist = np.linspace(0, evo_time, n_ts)
H_array_form =QobjEvo([H_d, [H_c[0], amplitudes_0], [H_c[1], amplitudes_1],[H_c[2], amplitudes_2]], tlist=tlist)

# res = qt.mesolve(H_array_form, psi_0, tlist)
res = sesolve(H_array_form, psi_0, tlist,options=Options(nsteps=1000000))


fidelities1 = []
for i in range(0, n_ts):
    fidelities1.append(qutip.metrics.fidelity(res.states[i].unit(), psi_targ.unit()))

print("sesolve final fidelity: ", fidelities1[-1])

print('cavity ptrace fidelity: ', cavityFidelity(res.states[-1], psi_targ))
print('qubit ptrace fidelity: ', qubitFidelity(res.states[-1], psi_targ))

### self-implemented forward propagation ###
from scipy.linalg import expm

psi_init = psi_0.full()
psi_end = psi_targ.full()
fidelities = []
for n in range(len(tlist) - 1):
    H = H_d + sum([H_c[i] * result.final_amps[n, i] for i in range(n_ctrls)])
    H = H.full()
    dt = tlist[n + 1] - tlist[n]
    o = expm(-1j * H * dt)
    psi_init = np.matmul(o,psi_init)
    fid = np.abs(np.matmul(np.transpose(np.conjugate(psi_init)),psi_end))
    #
    fidelities.append(fid[0])
# print(shape(psi_init))

print("self-implemented final fidelity: ", fidelities[-1][0])

# %%
# Create a new figure
plt.figure(figsize=(20, 2))

# Plot fidelities
plt.plot(fidelities, label='self-implemented')
plt.plot(fidelities1, label='sesolve')

# Set the title and labels for the plot
plt.title('Fidelity over time')
plt.xlabel('Time step')
plt.ylabel('Fidelity')

# Add a legend
plt.legend()

plt.grid(True)
# Show the plot
plt.show()


