import inspect
from pathlib import Path
from typing import Union
import warnings

from qm.qua import *
from quam_components import QuAM


def apply_all_flux_to_min(quam: "QuAM"):
    align()
    for q in quam.active_qubits:
        q.z.to_min()
    align()


def apply_all_flux_to_idle(quam: "QuAM"):
    align()
    for q in quam.active_qubits:
        q.z.to_joint_idle()
    align()


def qua_declaration(num_qubits):
    """
    Macro to declare the necessary QUA variables

    :param nb_of_qubits: Number of qubits used in this experiment
    :return:
    """
    n = declare(int)
    n_st = declare_stream()
    I = [declare(fixed) for _ in range(num_qubits)]
    Q = [declare(fixed) for _ in range(num_qubits)]
    I_st = [declare_stream() for _ in range(num_qubits)]
    Q_st = [declare_stream() for _ in range(num_qubits)]
    # Workaround to manually assign the results variables to the readout elements
    # for i in range(num_qubits):
    #     assign_variables_to_element(f"rr{i}", I[i], Q[i])
    return I, I_st, Q, Q_st, n, n_st


def multiplexed_readout(qubits, I, I_st, Q, Q_st, sequential=False, amplitude=1.0, weights=""):
    """Perform multiplexed readout on two resonators"""

    for ind, q in enumerate(qubits):
        # TODO: demod.accumulated?
        q.resonator.measure("readout", qua_vars=(I[ind], Q[ind]), amplitude_scale=amplitude)  # TODO: implement amplitude sweep

        if I_st is not None:
            save(I[ind], I_st[ind])
        if Q_st is not None:
            save(Q[ind], Q_st[ind])

        if sequential and ind < len(qubits) - 1:
            align(q.resonator.name, qubits[ind + 1].resonator.name)


def node_save(quam: QuAM, name: str, data: dict, additional_files: Union[dict, bool] = None):
    # Save results
    if isinstance(additional_files, dict):
        quam.data_handler.additional_files = additional_files
    elif additional_files is True:
        additional_files = {}
        files = ["../configuration/calibration_db.json", "optimal_weights.npz"]
        try:
            files.append(inspect.currentframe().f_back.f_locals['__file__'])
        except Exception:
            warnings.warn("Unable to save script file")

        for file in files:
            file = Path(file)
            if not file.exists():
                warnings.warn(f"Unable to save additional file {file}")
            additional_files[str(file)] = file.name
    #     Path(__file__).parent.parent / 'configuration' / v: v for v in 
    #     [Path(__file__), "calibration_db.json", "optimal_weights.npz"]
    # }

    quam.data_handler.save_data(data=data, name=name)

    # Save QuAM to the data folder
    quam.save(path=quam.data_handler.path / "quam_state", content_mapping={"wiring.json": {"wiring", "network"}})

    # Save QuAM to configuration directory / `state.json`
    config_dir = Path(__file__).parent.parent / 'configuration'
    quam.save(path=config_dir / "quam_state", content_mapping={"wiring.json": {"wiring", "network"}})
