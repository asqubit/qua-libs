# %%
import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qualang_tools.bakery.bakery import Baking
from TwoQubitRB import TwoQubitRb

from two_qubit_rb_config import *

qubit0_qe = "q1"
qubit1_qe = "q2"
qubit0_aux_qe = "q1_aux"
qubit1_aux_qe = "q2_aux"
qubit0_x_pulse = "x180"
qubit1_x_pulse = "x180"
cr_c0t1 = "cr01"
cr_c0t1_pulse = "cw"
minus_cr_c0t1_pulse = "minus_cw"

# %%


def bake_phased_xz(baker: Baking, q, x, z, a):
    qe = qubit0_qe if q == 0 else qubit1_qe
    pulse = qubit0_x_pulse if q == 0 else qubit1_x_pulse

    baker.frame_rotation_2pi(-a, qe)
    baker.play(pulse, qe, amp=x)
    baker.frame_rotation_2pi(a + z, qe)


def bake_sqrt_iswap(baker: Baking, q1, q2):
    baker.play(iswap_pulse, qubit0_flux_qe)


def bake_cnot(baker: Baking, q1, q2):
    """
    The way it is written is uses cr01 for both cirq.CNOT(q1,q2) and cirq.CNOT(q2,q1)
    # TODO: add components to accept cr10
    """
    if q1 == 0 and q2 == 1:
        baker.frame_rotation_2pi(-0.25, qubit0_aux_qe)
        baker.play(qubit0_x_pulse, qubit0_aux_qe, amp=1)
        baker.frame_rotation_2pi(0.25, qubit0_aux_qe)
        baker.frame_rotation_2pi(-1.0, qubit1_aux_qe)
        baker.play(qubit1_x_pulse, qubit1_aux_qe, amp=0.5)
        baker.frame_rotation_2pi(1.0, qubit1_aux_qe)

        baker.wait(x180_len // 4, cr_c0t1)
        baker.play(cr_c0t1_pulse, cr_c0t1)

        baker.wait(const_len // 4, qubit0_aux_qe)
        baker.play(qubit0_x_pulse, qubit0_aux_qe)

        baker.wait((x180_len + const_len) // 4, cr_c0t1)
        baker.play(minus_cr_c0t1_pulse, cr_c0t1)

    elif q1 == 1 and q2 == 0:
        # note that cirq.CNOT(q2,q1) rewritten still in cirq.CNOT(q1,q2) terms
        # with H to q1, q2 before
        baker.frame_rotation_2pi(-0.5, qubit0_aux_qe)
        baker.play(qubit0_x_pulse, qubit0_aux_qe, amp=0.5)
        baker.frame_rotation_2pi(0.5 + 0.5, qubit0_aux_qe)
        baker.frame_rotation_2pi(-0.0, qubit1_aux_qe)
        baker.play(qubit1_x_pulse, qubit1_aux_qe, amp=0.5)
        baker.frame_rotation_2pi(0.0 + 0.5, qubit1_aux_qe)

        baker.wait(x180_len // 4, cr_c0t1)
        baker.play(cr_c0t1_pulse, cr_c0t1)

        baker.wait(const_len // 4, qubit0_aux_qe)
        baker.play(qubit0_x_pulse, qubit0_aux_qe)

        baker.wait((x180_len + const_len) // 4, cr_c0t1)
        baker.play(minus_cr_c0t1_pulse, cr_c0t1)

        # with H to q1, q2 before
        baker.wait(const_len // 4, qubit0_aux_qe)
        baker.wait(const_len // 4, qubit1_aux_qe)

        baker.frame_rotation_2pi(-0.5, qubit0_aux_qe)
        baker.play(qubit0_x_pulse, qubit0_aux_qe, amp=0.5)
        baker.frame_rotation_2pi(0.5 + 1.0, qubit0_aux_qe)
        baker.frame_rotation_2pi(-0.5, qubit1_aux_qe)
        baker.play(qubit1_x_pulse, qubit1_aux_qe, amp=0.5)
        baker.frame_rotation_2pi(0.5 + 1.0, qubit1_aux_qe)


# def bake_cz(baker: Baking, q1, q2):
#     baker.play(cz_pulse, qubit0_flux_qe)


def prep():
    wait(4)


def meas():
    q0 = declare(bool)
    q1 = declare(bool)
    measure_qubit_0(q0)
    measure_qubit_1(q1)
    return q0, q1


# %%

local_config = add_aux_elements(config, "q1", "q2")

qmm = QuantumMachinesManager(host="172.16.33.100", port=80)

rb = TwoQubitRb(local_config, bake_phased_xz, {"CNOT": bake_cnot}, prep, meas, verify_generation=True)

res = rb.run(qmm, sequence_depths=[10, 15, 20, 25, 30], num_repeats=4, num_averages=10)

# %%

res.plot_hist()
plt.show()

res.plot_fidelity()
plt.show()