from dataclasses import field
from typing import List, Union, Dict
from quam import QuamComponent
from quam.components.channels import IQChannel, SingleChannel, InOutIQChannel
from quam.components.octave import Octave
from quam.core import QuamRoot, quam_dataclass

from qm.qua import set_dc_offset, align

# import macros

__all__ = ["Transmon", "FluxLine", "ReadoutResonator", "QuAM"]


@quam_dataclass
class FluxLine(SingleChannel):
    """Example QuAM component for a transmon qubit."""

    independent_offset: float = 0.0
    joint_offset: float = 0.0
    min_offset: float = 0.0

    def to_independent_idle(self):  # TODO: put the functions here
        set_dc_offset(self.name, "single", self.independent_offset)

    def to_joint_idle(self):
        set_dc_offset(self.name, "single", self.joint_offset)

    def to_min(self):
        set_dc_offset(self.name, "single", self.min_offset)


@quam_dataclass
class ReadoutResonator(InOutIQChannel):
    """QuAM component for a readout resonator

    :params depletion_time: the resonator depletion time in ns.
    :params frequency_bare: the bare resonator frequency in Hz.
    """

    depletion_time: int = 1000
    frequency_bare: float = 0.0

    @property
    def f_01(self):
        """The optimal frequency for discriminating the qubit between |0> and |1> in Hz"""
        return self.frequency_converter_up.LO_frequency + self.intermediate_frequency


@quam_dataclass
class Transmon(QuamComponent):
    """
    Example QuAM component for a transmon qubit.

    Args:
        thermalization_time (int): An integer.
        T1 (str): A string.
    """

    id: Union[int, str]

    xy: IQChannel = None
    z: FluxLine = None

    resonator: ReadoutResonator = None

    T1: int = 10_000
    T2ramsey: int = 10_000
    T2echo: int = 10_000
    thermalization_time_factor: int = 5
    anharmonicity: int = 150e6

    @property
    def thermalization_time(self):
        return self.thermalization_time_factor * self.T1

    @property
    def f_01(self):
        """The 0-1 (g-e) transition frequency in Hz"""
        return self.xy.frequency_converter_up.LO_frequency + self.xy.intermediate_frequency

    @property
    def f_12(self):
        """The 0-2 (e-f) transition frequency in Hz"""
        return self.xy.frequency_converter_up.LO_frequency + self.xy.intermediate_frequency - self.anharmonicity

    @property
    def name(self):
        return self.id if isinstance(self.id, str) else f"q{self.id}"


@quam_dataclass
class QuAM(QuamRoot):
    """Example QuAM root component."""

    @classmethod
    def load(self, *args, **kwargs) -> "QuAM":
        return super().load(*args, **kwargs)

    octave: Octave = None

    qubits: Dict[str, Transmon] = field(default_factory=dict)
    wiring: dict = field(default_factory=dict)

    active_qubit_names: List[str] = field(default_factory=list)

    @property
    def active_qubits(self) -> List[Transmon]:
        """Return the the list of active qubits"""
        return [self.qubits[q] for q in self.active_qubit_names]

    @property
    def get_depletion_time(self) -> int:
        """Return the longest depletion time amongst the active qubits"""
        return max([q.resonator.depletion_time for q in self.active_qubits])

    @property
    def get_thermalization_time(self) -> int:
        """Return the longest thermalization time amongst the active qubits"""
        return max([q.thermalization_time for q in self.active_qubits])

    def apply_all_flux_to_min(self) -> None:
        """Apply the offsets that bring all the active qubits to the minimum frequency point."""
        align()
        for q in self.active_qubits:
            q.z.to_min()
        align()