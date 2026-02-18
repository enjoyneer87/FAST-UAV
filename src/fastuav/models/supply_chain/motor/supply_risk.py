"""
Motor Supply Risk Model

Computes per-material and aggregate supply risk scores for a motor.

Methodology
-----------
Adapted from the European Commission's Critical Raw Materials (CRM)
methodology and Graedel et al. (2012):

  Material Risk Score (MRS_i):
      HHI_norm   = HHI_i / 10000           (0→1; 1 = complete monopoly)
      country_risk = 1 - WGI_score_i       (0→1; 1 = worst governance)
      MRS_i = HHI_norm × country_risk × (1 - substitutability_i)
                                          × (1 - recyclability_i)

  Motor Aggregate Supply Risk Index (ASRI):
      mass_frac_i = mass_i / Σ mass_j
      ASRI = Σ (MRS_i × mass_frac_i)

  Critical Material Mass Fraction (CMMF):
      CMMF = Σ mass_i  for EU-critical materials / total mass

References
----------
  Graedel et al. (2012) "Methodology of Metal Criticality Determination",
      Environ. Sci. Technol. 46, 1063–1070.
  EC (2023) "Study on the Critical Raw Materials for the EU 2023".
"""
import pathlib
import warnings

import numpy as np
import pandas as pd
import fastoad.api as oad
import openmdao.api as om

_DATA_DIR = pathlib.Path(__file__).parents[4] / "data" / "supply_chain"
_DEFAULT_RISK_CSV = _DATA_DIR / "supply_risk_indicators.csv"

# Mapping: CSV 'material' key → OpenMDAO variable suffix (same as in material_cost.py)
_MAT_VARNAME = {
    "copper":           "copper",
    "ndfeb_magnet":     "magnet",
    "electrical_steel": "steel",
    "aluminum":         "aluminum",
    "other":            "other",
}

# Built-in fallback risk indicators
_FALLBACK_RISK = {
    "copper":           dict(hhi=1289, wgi_score=0.65, eu_critical=False, substitutability=0.85, recyclability=0.43),
    "ndfeb_magnet":     dict(hhi=8547, wgi_score=0.40, eu_critical=True,  substitutability=0.15, recyclability=0.13),
    "electrical_steel": dict(hhi=642,  wgi_score=0.40, eu_critical=False, substitutability=0.75, recyclability=0.55),
    "aluminum":         dict(hhi=789,  wgi_score=0.40, eu_critical=False, substitutability=0.85, recyclability=0.42),
    "other":            dict(hhi=500,  wgi_score=0.55, eu_critical=False, substitutability=0.50, recyclability=0.20),
}


class SupplyRiskLoader:
    """Loads supply risk indicators from the package CSV."""

    @staticmethod
    def load(csv_path=None) -> dict:
        """Return {material_key: {indicator: value, …}} dict."""
        path = pathlib.Path(csv_path) if csv_path else _DEFAULT_RISK_CSV
        if not path.exists():
            warnings.warn(
                f"Supply risk CSV not found at {path}. Using built-in fallbacks.",
                UserWarning
            )
            return dict(_FALLBACK_RISK)

        df = pd.read_csv(path, comment="#")
        result = {}
        bool_map = {"True": True, "False": False, True: True, False: False}
        for _, row in df.iterrows():
            mat = row["material"]
            result[mat] = dict(
                hhi=float(row["hhi"]),
                wgi_score=float(row["wgi_score"]),
                eu_critical=bool_map.get(row["eu_critical_2023"], False),
                substitutability=float(row["substitutability"]),
                recyclability=float(row["recyclability"]),
            )
        for mat, vals in _FALLBACK_RISK.items():
            result.setdefault(mat, vals)
        return result


def _compute_mrs(hhi, wgi_score, substitutability, recyclability) -> float:
    """Material Risk Score on [0, 1] scale."""
    hhi_norm     = hhi / 10000.0
    country_risk = 1.0 - np.clip(wgi_score, 0.0, 1.0)
    mrs = hhi_norm * country_risk * (1.0 - substitutability) * (1.0 - recyclability)
    return float(np.clip(mrs, 0.0, 1.0))


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.motor.supply_chain.supply_risk")
class MotorSupplyRisk(om.ExplicitComponent):
    """
    Computes supply risk scores for a motor based on material masses and
    supply risk indicators (HHI, WGI, substitutability, recyclability).

    Inputs
    ------
    data:propulsion:motor:supply_chain:mass:<material>  [kg]

    Outputs
    -------
    data:propulsion:motor:supply_chain:risk:score:<material>  [-]  per-material MRS (0–1)
    data:propulsion:motor:supply_chain:risk:index             [-]  mass-weighted ASRI (0–1)
    data:propulsion:motor:supply_chain:risk:critical_mass_fraction  [-]  EU-critical material fraction
    """

    def initialize(self):
        self.options.declare(
            "risk_csv", default=None, allow_none=True,
            desc="Path to supply_risk_indicators.csv. None → package default."
        )

    def setup(self):
        indicators = SupplyRiskLoader.load(self.options["risk_csv"])
        # Store for use in compute (no OpenMDAO inputs needed – these are data, not design vars)
        self._indicators = indicators
        self._mat_order  = list(_MAT_VARNAME.items())  # (csv_key, var_suffix)

        for _, var_suffix in self._mat_order:
            self.add_input(
                f"data:propulsion:motor:supply_chain:mass:{var_suffix}",
                val=np.nan, units="kg"
            )
            self.add_output(
                f"data:propulsion:motor:supply_chain:risk:score:{var_suffix}",
                val=0.0,
                desc=f"Material Risk Score for {var_suffix} (0=safe, 1=critical)"
            )

        self.add_output(
            "data:propulsion:motor:supply_chain:risk:index",
            val=0.0,
            desc="Motor Aggregate Supply Risk Index – mass-weighted (0=safe, 1=critical)"
        )
        self.add_output(
            "data:propulsion:motor:supply_chain:risk:critical_mass_fraction",
            val=0.0,
            desc="Fraction of motor mass attributed to EU Critical Raw Materials"
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        masses = {}
        scores = {}

        for mat_key, var_suffix in self._mat_order:
            m = float(inputs[f"data:propulsion:motor:supply_chain:mass:{var_suffix}"])
            ind = self._indicators.get(mat_key, _FALLBACK_RISK[mat_key])
            mrs = _compute_mrs(
                hhi=ind["hhi"],
                wgi_score=ind["wgi_score"],
                substitutability=ind["substitutability"],
                recyclability=ind["recyclability"],
            )
            masses[mat_key] = m
            scores[mat_key] = mrs
            outputs[f"data:propulsion:motor:supply_chain:risk:score:{var_suffix}"] = mrs

        total_mass = sum(masses.values())
        if total_mass <= 0.0:
            outputs["data:propulsion:motor:supply_chain:risk:index"] = 0.0
            outputs["data:propulsion:motor:supply_chain:risk:critical_mass_fraction"] = 0.0
            return

        # Mass-weighted aggregate supply risk index
        asri = sum(
            scores[mat_key] * masses[mat_key] / total_mass
            for mat_key, _ in self._mat_order
        )
        outputs["data:propulsion:motor:supply_chain:risk:index"] = np.clip(asri, 0.0, 1.0)

        # EU-critical material mass fraction
        critical_mass = sum(
            masses[mat_key]
            for mat_key, _ in self._mat_order
            if self._indicators.get(mat_key, _FALLBACK_RISK[mat_key])["eu_critical"]
        )
        outputs["data:propulsion:motor:supply_chain:risk:critical_mass_fraction"] = (
            critical_mass / total_mass
        )
