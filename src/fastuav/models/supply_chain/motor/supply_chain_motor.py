"""
Motor Supply Chain – Top-Level OpenMDAO Group

Wires together:
  1. MotorMaterialComposition  – splits motor mass into raw-material masses
  2. MotorMaterialCost         – raw material cost from masses × prices
  3. MotorSupplyRisk           – HHI/WGI-based supply risk scores

Registration
------------
  fastuav.propulsion.motor.supply_chain

Usage in YAML model file
------------------------
  - id: fastuav.propulsion.motor.supply_chain
    # optional options:
    # supply_chain_options:
    #   f_copper:    0.25
    #   f_magnet:    0.12
    #   f_steel:     0.45
    #   f_aluminum:  0.10
    #   prices_csv:  /path/to/material_prices.csv
    #   risk_csv:    /path/to/supply_risk_indicators.csv

Programmatic usage
------------------
  import openmdao.api as om
  from fastuav.models.supply_chain.motor.supply_chain_motor import MotorSupplyChain

  prob = om.Problem()
  prob.model.add_subsystem("motor_sc", MotorSupplyChain(), promotes=["*"])
"""
import fastoad.api as oad
import openmdao.api as om

from fastuav.models.supply_chain.motor.material_composition import MotorMaterialComposition
from fastuav.models.supply_chain.motor.material_cost import MotorMaterialCost
from fastuav.models.supply_chain.motor.supply_risk import MotorSupplyRisk


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.motor.supply_chain")
class MotorSupplyChain(om.Group):
    """
    Motor supply chain analysis group.

    Computes raw material decomposition, material cost, and supply risk
    for a single motor, given its total mass.

    Options
    -------
    f_copper   : float, default 0.25   Copper mass fraction
    f_magnet   : float, default 0.12   NdFeB magnet mass fraction
    f_steel    : float, default 0.45   Electrical steel mass fraction
    f_aluminum : float, default 0.10   Aluminum mass fraction
    prices_csv : str,   default None   Path to material_prices.csv (None = built-in)
    risk_csv   : str,   default None   Path to supply_risk_indicators.csv (None = built-in)
    """

    def initialize(self):
        self.options.declare("f_copper",    default=0.25, desc="Copper mass fraction")
        self.options.declare("f_magnet",    default=0.12, desc="NdFeB magnet mass fraction")
        self.options.declare("f_steel",     default=0.45, desc="Electrical steel mass fraction")
        self.options.declare("f_aluminum",  default=0.10, desc="Aluminum mass fraction")
        self.options.declare("prices_csv",  default=None, allow_none=True,
                             desc="Custom path to material_prices.csv")
        self.options.declare("risk_csv",    default=None, allow_none=True,
                             desc="Custom path to supply_risk_indicators.csv")

    def setup(self):
        self.add_subsystem(
            "material_composition",
            MotorMaterialComposition(
                f_copper=self.options["f_copper"],
                f_magnet=self.options["f_magnet"],
                f_steel=self.options["f_steel"],
                f_aluminum=self.options["f_aluminum"],
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "material_cost",
            MotorMaterialCost(prices_csv=self.options["prices_csv"]),
            promotes=["*"],
        )
        self.add_subsystem(
            "supply_risk",
            MotorSupplyRisk(risk_csv=self.options["risk_csv"]),
            promotes=["*"],
        )
