# %%
import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qualang_tools.bakery.bakery import baking
from two_qubit_rb_config import *
from qm import SimulationConfig

%matplotlib qt

qubit0_qe = "q1"
qubit1_qe = "q2"
qubit0_aux_qe = "q1_aux"
qubit1_aux_qe = "q2_aux"
qubit0_x_pulse = "x180"
qubit1_x_pulse = "x180"
cr_c0t1 = "cr01"
cr_c0t1_pulse = "cw"
minus_cr_c0t1_pulse = "minus_cw"

a=0
z=0
x=1

local_config = add_aux_elements(config, "q1", "q2")

with baking(local_config) as b:
    b.frame_rotation_2pi(-0.25, qubit0_aux_qe)
    b.play(qubit0_x_pulse, qubit0_aux_qe, amp=1)
    b.frame_rotation_2pi(0.25, qubit0_aux_qe)
    b.frame_rotation_2pi(-1.0, qubit1_aux_qe)
    b.play(qubit1_x_pulse, qubit1_aux_qe, amp=0.5)
    b.frame_rotation_2pi(1.0, qubit1_aux_qe)
    b.align(qubit1_aux_qe, qubit0_aux_qe, cr_c0t1)
    b.play(cr_c0t1_pulse, cr_c0t1)
    b.align(cr_c0t1, qubit0_aux_qe)
    b.play(qubit0_x_pulse, qubit0_aux_qe)
    b.align(qubit0_aux_qe, cr_c0t1)
    b.play(minus_cr_c0t1_pulse, cr_c0t1)

with program() as prog:
    update_frequency(qubit0_qe, 0)
    update_frequency(qubit1_qe, 0)
    update_frequency(cr_c0t1, 0)
    align()
    b.run()

qmm = QuantumMachinesManager(host='172.16.33.100')
job = qmm.simulate(local_config, prog, SimulationConfig(1000))
job.get_simulated_samples().con2.plot()
# %%