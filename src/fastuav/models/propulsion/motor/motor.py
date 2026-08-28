"""
Motor component
"""
import fastoad.api as oad
import openmdao.api as om
from fastuav.models.propulsion.motor.definition_parameters import MotorDefinitionParameters
from fastuav.models.propulsion.motor.estimation_models import MotorEstimationModels
from fastuav.models.propulsion.motor.catalogue import MotorCatalogueSelection
from fastuav.models.propulsion.motor.performance_analysis import MotorPerformanceGroup
from fastuav.models.propulsion.motor.constraints import MotorConstraints
from fastuav.models.supply_chain.motor.supply_chain_motor import MotorSupplyChain


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.motor")
class Motor(om.Group):
    """
    Group containing the Motor MDA.

    Options
    -------
    off_the_shelf : bool
        If True, snap motor sizing to the nearest catalogue entry.
    supply_chain : bool
        If True, append a MotorSupplyChain sub-group that computes raw-material
        mass breakdown, material cost, and supply risk for the sized motor.
    supply_chain_options : dict
        Keyword arguments forwarded to MotorSupplyChain (f_copper, f_magnet,
        f_steel, f_aluminum, prices_csv, risk_csv).  Ignored when
        supply_chain=False.
    """

    def initialize(self):
        self.options.declare("off_the_shelf", default=False, types=bool)
        self.options.declare("supply_chain", default=False, types=bool,
                             desc="Enable motor supply chain analysis")
        self.options.declare("supply_chain_options", default={}, types=dict,
                             desc="Options forwarded to MotorSupplyChain")

    def setup(self):
        self.add_subsystem("definition_parameters", MotorDefinitionParameters(), promotes=["*"])
        self.add_subsystem("estimation_models", MotorEstimationModels(), promotes=["*"])
        self.add_subsystem("catalogue_selection" if self.options["off_the_shelf"] else "skip_catalogue_selection",
                           MotorCatalogueSelection(off_the_shelf=self.options["off_the_shelf"]),
                           promotes=["*"],
        )
        self.add_subsystem("performance_analysis", MotorPerformanceGroup(), promotes=["*"])
        self.add_subsystem("constraints", MotorConstraints(), promotes=["*"])

        if self.options["supply_chain"]:
            sc_opts = self.options["supply_chain_options"]
            self.add_subsystem(
                "supply_chain",
                MotorSupplyChain(**sc_opts),
                promotes=["*"],
            )
