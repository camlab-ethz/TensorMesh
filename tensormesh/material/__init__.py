import csv
from dataclasses import dataclass
from importlib.resources import files
from typing import Optional, Tuple

_FRICTIONAL_PRESET_FILE = "frictional_materials.csv"


@dataclass
class IsotropicMaterial:
    name: str
    E: float  # Young's Modulus (Pa)
    nu: float # Poisson's Ratio
    rho: float # Density (kg/m^3)
    sigma_y: float = None # Yield Stress (Pa)
    H: float = 0.0 # Hardening Modulus (Pa)

    @property
    def lame_params(self):
        mu = self.E / (2 * (1 + self.nu))
        lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        return mu, lam

Steel = IsotropicMaterial("Steel", E=210e9, nu=0.3, rho=7850, sigma_y=250e6)
Aluminum = IsotropicMaterial("Aluminum", E=70e9, nu=0.33, rho=2700, sigma_y=100e6, H=700e6) # Example H
Rubber = IsotropicMaterial("Rubber", E=10e6, nu=0.48, rho=1100)
Glass = IsotropicMaterial("Glass", E=70e9, nu=0.2, rho=2500)


@dataclass(frozen=True)
class FrictionalMaterial:
    """Pressure-dependent (Drucker-Prager / Mohr-Coulomb) soil or rock material.

    Holds the parameters consumed by the Drucker-Prager constitutive primitive
    and the :class:`~tensormesh.assemble.DruckerPragerPlasticity` assembler.
    Angles are in degrees.

    The required fields are ``E``, ``nu``, ``cohesion`` and ``friction_angle``;
    ``dilatancy_angle``, ``H``, ``rho`` and ``name`` are optional, so a one-off
    soil can be built directly::

        soil = FrictionalMaterial(E=50e6, nu=0.30, cohesion=5e3, friction_angle=35.0)

    A ``dilatancy_angle`` of ``None`` selects associated plastic flow (the
    friction angle is used).  Named presets are loaded from a small CSV table::

        soil = FrictionalMaterial.from_preset("DenseSand")

    Parameters
    ----------
    E : float
        Young's modulus in Pa.
    nu : float
        Poisson's ratio.
    cohesion : float
        Cohesion in Pa.
    friction_angle : float
        Mohr-Coulomb friction angle in degrees.
    dilatancy_angle : float or None, optional
        Dilatancy angle in degrees; ``None`` (default) means associated flow.
    H : float, optional
        Linear isotropic hardening modulus in Pa. Default ``0.0``.
    rho : float or None, optional
        Density in kg/m^3 (metadata; not used by the constitutive update).
    name : str, optional
        Label for the material. Default ``"Custom"``.
    description : str, optional
        Free-text note (set for presets).
    """

    E: float
    nu: float
    cohesion: float
    friction_angle: float
    dilatancy_angle: Optional[float] = None
    H: float = 0.0
    rho: Optional[float] = None
    name: str = "Custom"
    description: str = ""

    @property
    def lame_params(self):
        mu = self.E / (2 * (1 + self.nu))
        lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        return mu, lam

    @staticmethod
    def _load_preset_table() -> dict:
        text = files(__package__).joinpath(_FRICTIONAL_PRESET_FILE).read_text(encoding="utf-8")
        table = {}
        for row in csv.DictReader(text.splitlines()):
            table[row["name"]] = row
        return table

    @classmethod
    def preset_names(cls) -> Tuple[str, ...]:
        """Return the names of the available frictional-material presets."""
        return tuple(cls._load_preset_table().keys())

    @classmethod
    def from_preset(cls, name: str) -> "FrictionalMaterial":
        """Build a :class:`FrictionalMaterial` from a named preset in the CSV table.

        The preset values are *illustrative example presets*, not design-grade
        soil/rock parameters.

        Raises
        ------
        KeyError
            If ``name`` is not one of :meth:`preset_names`.
        """
        table = cls._load_preset_table()
        if name not in table:
            available = ", ".join(sorted(table))
            raise KeyError(
                f"Unknown frictional-material preset {name!r}. Available presets: {available}."
            )
        row = table[name]

        def _opt(value: str) -> Optional[float]:
            value = value.strip()
            return None if value == "" else float(value)

        return cls(
            E=float(row["E"]),
            nu=float(row["nu"]),
            cohesion=float(row["cohesion"]),
            friction_angle=float(row["friction_angle"]),
            dilatancy_angle=_opt(row.get("dilatancy_angle", "")),
            H=float(row["H"]),
            rho=_opt(row.get("rho", "")),
            name=row["name"],
            description=row.get("description", ""),
        )

