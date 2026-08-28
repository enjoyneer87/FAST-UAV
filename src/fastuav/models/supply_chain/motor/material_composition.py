"""
Motor Material Composition Model

Decomposes total motor mass into raw material fractions.

Typical BLDC motor composition (drone scale, literature-based):
  - Copper     : ~25%  (stator windings)
  - NdFeB magnet: ~12% (permanent magnets, rotor)
  - Elec. steel : ~45% (stator/rotor laminations)
  - Aluminum    : ~10% (housing / endcaps)
  - Other       :  ~8% (insulation, epoxy, fasteners)

References:
  Riba et al. (2016) "Rare-earth-free propulsion motors for EVs"
  Nordelof et al. (2019) "A scalable life cycle inventory of an automotive
      electric motor - Part I & II", Int. J. Life Cycle Assess.
"""
import numpy as np
import fastoad.api as oad
import openmdao.api as om


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.motor.supply_chain.material_composition")
class MotorMaterialComposition(om.ExplicitComponent):
    """
    Splits motor mass into constituent raw materials.

    Input
    -----
    data:propulsion:motor:mass  [kg]  Total motor mass (from estimation / catalogue)

    Outputs
    -------
    data:propulsion:motor:supply_chain:mass:<material>  [kg]  per-material mass
    """

    def initialize(self):
        # Material mass fractions (dimensionless, must sum to 1.0)
        self.options.declare(
            "f_copper", default=0.25,
            desc="Copper mass fraction (stator windings)"
        )
        self.options.declare(
            "f_magnet", default=0.12,
            desc="NdFeB permanent magnet mass fraction (rotor)"
        )
        self.options.declare(
            "f_steel", default=0.45,
            desc="Electrical (silicon) steel mass fraction (stator/rotor laminations)"
        )
        self.options.declare(
            "f_aluminum", default=0.10,
            desc="Aluminum mass fraction (housing, endcaps)"
        )
        # 'other' fraction = 1 - sum of the above (insulation, epoxy, fasteners, …)

    def setup(self):
        self.add_input("data:propulsion:motor:mass", val=np.nan, units="kg",
                       desc="Total motor mass")

        for mat in ("copper", "magnet", "steel", "aluminum", "other"):
            self.add_output(
                f"data:propulsion:motor:supply_chain:mass:{mat}",
                val=0.0, units="kg",
                desc=f"Motor mass attributed to {mat}"
            )

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        m = inputs["data:propulsion:motor:mass"]

        f_cu  = self.options["f_copper"]
        f_mag = self.options["f_magnet"]
        f_st  = self.options["f_steel"]
        f_al  = self.options["f_aluminum"]
        f_oth = max(0.0, 1.0 - f_cu - f_mag - f_st - f_al)

        outputs["data:propulsion:motor:supply_chain:mass:copper"]   = f_cu  * m
        outputs["data:propulsion:motor:supply_chain:mass:magnet"]   = f_mag * m
        outputs["data:propulsion:motor:supply_chain:mass:steel"]    = f_st  * m
        outputs["data:propulsion:motor:supply_chain:mass:aluminum"] = f_al  * m
        outputs["data:propulsion:motor:supply_chain:mass:other"]    = f_oth * m
