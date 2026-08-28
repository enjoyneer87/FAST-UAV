"""
Estimation models for the motor
"""
import openmdao.api as om
import numpy as np
from fastuav.utils.uncertainty import add_subsystem_with_deviation


class MotorEstimationModels(om.Group):
    """
    Group containing the estimation models for the motor.
    Estimation models take a reduced set of definition parameters and estimate the main component characteristics from it.
    """

    def setup(self):
        add_subsystem_with_deviation(
            self,
            "nominal_torque",
            NominalTorque(),
            uncertain_outputs={"data:propulsion:motor:torque:nominal:estimated": "N*m"},
        )

        add_subsystem_with_deviation(
            self,
            "friction_torque",
            FrictionTorque(),
            uncertain_outputs={"data:propulsion:motor:torque:friction:estimated": "N*m"},
        )

        add_subsystem_with_deviation(
            self,
            "resistance",
            Resistance(),
            uncertain_outputs={"data:propulsion:motor:resistance:estimated": "V/A"},
        )

        add_subsystem_with_deviation(
            self,
            "weight",
            Weight(),
            uncertain_outputs={"data:weight:propulsion:motor:mass:estimated": "kg"},
        )

        self.add_subsystem("geometry", Geometry(), promotes=["*"])

        self.add_subsystem("iron_loss", IronLoss(), promotes=["*"])


class NominalTorque(om.ExplicitComponent):
    """
    Compute nominal torque
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:max:estimated", val=np.nan, units="N*m")
        self.add_input("models:propulsion:motor:torque:nominal:reference", val=np.nan, units="N*m")
        self.add_input("models:propulsion:motor:torque:max:reference", val=np.nan, units="N*m")
        self.add_output("data:propulsion:motor:torque:nominal:estimated", units="N*m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        T_mot_nom_ref = inputs["models:propulsion:motor:torque:nominal:reference"]
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]

        T_mot_nom = T_mot_nom_ref * T_mot_max / T_mot_max_ref  # [N.m] nominal torque

        outputs["data:propulsion:motor:torque:nominal:estimated"] = T_mot_nom

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        T_mot_nom_ref = inputs["models:propulsion:motor:torque:nominal:reference"]
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]

        partials["data:propulsion:motor:torque:nominal:estimated",
                 "models:propulsion:motor:torque:nominal:reference"] = T_mot_max / T_mot_max_ref

        partials["data:propulsion:motor:torque:nominal:estimated",
                 "models:propulsion:motor:torque:max:reference"] = - T_mot_nom_ref * T_mot_max / T_mot_max_ref ** 2

        partials["data:propulsion:motor:torque:nominal:estimated",
                 "data:propulsion:motor:torque:max:estimated"] = T_mot_nom_ref / T_mot_max_ref


class FrictionTorque(om.ExplicitComponent):
    """
    Computes friction torque.
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:max:estimated", val=np.nan, units="N*m")
        self.add_input("models:propulsion:motor:torque:max:reference", val=np.nan, units="N*m")
        self.add_input("models:propulsion:motor:torque:friction:reference", val=np.nan, units="N*m")
        self.add_output("data:propulsion:motor:torque:friction:estimated", units="N*m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        Tf_ref = inputs["models:propulsion:motor:torque:friction:reference"]
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]

        Tf = Tf_ref * (T_mot_max / T_mot_max_ref) ** (3 / 3.5)  # [N.m] Friction torque

        outputs["data:propulsion:motor:torque:friction:estimated"] = Tf

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        Tf_ref = inputs["models:propulsion:motor:torque:friction:reference"]
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]

        partials["data:propulsion:motor:torque:friction:estimated",
                 "models:propulsion:motor:torque:friction:reference"
        ] = (T_mot_max / T_mot_max_ref) ** (3 / 3.5)

        partials["data:propulsion:motor:torque:friction:estimated",
                 "models:propulsion:motor:torque:max:reference"
        ] = - (3 / 3.5) * Tf_ref * T_mot_max ** (3 / 3.5) / T_mot_max_ref ** (6.5 / 3.5)

        partials["data:propulsion:motor:torque:friction:estimated",
                 "data:propulsion:motor:torque:max:estimated"
        ] = (3 / 3.5) * Tf_ref / T_mot_max_ref ** (3 / 3.5) * T_mot_max ** (- 0.5 / 3.5)


class Resistance(om.ExplicitComponent):
    """
    Computes motor resistance.
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:max:estimated", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:speed:constant:estimated", val=np.nan, units="rad/V/s")
        self.add_input("models:propulsion:motor:torque:max:reference", val=np.nan, units="N*m")
        self.add_input("models:propulsion:motor:resistance:reference", val=np.nan, units="V/A")
        self.add_input("models:propulsion:motor:speed:constant:reference", val=np.nan, units="rad/V/s")
        self.add_output("data:propulsion:motor:resistance:estimated", units="V/A")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        R_ref = inputs["models:propulsion:motor:resistance:reference"]
        Kv_ref = inputs["models:propulsion:motor:speed:constant:reference"]
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]
        Kv = inputs["data:propulsion:motor:speed:constant:estimated"]

        R = (
            R_ref * (T_mot_max / T_mot_max_ref) ** (-5 / 3.5) * (Kv / Kv_ref) ** (-2)
        )  # [Ohm] motor resistance

        outputs["data:propulsion:motor:resistance:estimated"] = R

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        R_ref = inputs["models:propulsion:motor:resistance:reference"]
        Kv_ref = inputs["models:propulsion:motor:speed:constant:reference"]
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]
        Kv = inputs["data:propulsion:motor:speed:constant:estimated"]

        partials["data:propulsion:motor:resistance:estimated",
                 "models:propulsion:motor:torque:max:reference"
        ] = (5 / 3.5) * R_ref * T_mot_max ** (-5 / 3.5) * T_mot_max_ref ** (1.5 / 3.5) * (Kv / Kv_ref) ** (-2)

        partials["data:propulsion:motor:resistance:estimated",
                 "models:propulsion:motor:resistance:reference"
        ] = (T_mot_max / T_mot_max_ref) ** (-5 / 3.5) * (Kv / Kv_ref) ** (-2)

        partials["data:propulsion:motor:resistance:estimated",
                 "models:propulsion:motor:speed:constant:reference"
        ] = 2 * R_ref * (T_mot_max / T_mot_max_ref) ** (-5 / 3.5) * Kv_ref / Kv ** 2

        partials["data:propulsion:motor:resistance:estimated",
                 "data:propulsion:motor:torque:max:estimated"
        ] = (-5 / 3.5) * R_ref / T_mot_max_ref ** (-5 / 3.5) * T_mot_max ** (-8.5 / 3.5) * (Kv / Kv_ref) ** (-2)

        partials["data:propulsion:motor:resistance:estimated",
                 "data:propulsion:motor:speed:constant:estimated"
        ] = -2 * R_ref * (T_mot_max / T_mot_max_ref) ** (-5 / 3.5) * Kv_ref ** 2 / Kv ** 3


class Weight(om.ExplicitComponent):
    """
    Weight calculation of an electrical Motor
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:max:estimated", val=np.nan, units="N*m")
        self.add_input("models:propulsion:motor:torque:max:reference", val=np.nan, units="N*m")
        self.add_input("models:weight:propulsion:motor:mass:reference", val=np.nan, units="kg")
        self.add_output("data:weight:propulsion:motor:mass:estimated", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        m_mot_ref = inputs["models:weight:propulsion:motor:mass:reference"]

        m_mot = m_mot_ref * (T_mot_max / T_mot_max_ref) ** (3 / 3.5)  # [kg] Motor mass (estimated)

        outputs["data:weight:propulsion:motor:mass:estimated"] = m_mot

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        T_mot_max = inputs["data:propulsion:motor:torque:max:estimated"]
        T_mot_max_ref = inputs["models:propulsion:motor:torque:max:reference"]
        m_mot_ref = inputs["models:weight:propulsion:motor:mass:reference"]

        partials["data:weight:propulsion:motor:mass:estimated",
                 "data:propulsion:motor:torque:max:estimated"
        ] = (3 / 3.5) * m_mot_ref / T_mot_max_ref ** (3 / 3.5) * T_mot_max ** (-0.5 / 3.5)

        partials["data:weight:propulsion:motor:mass:estimated",
                 "models:propulsion:motor:torque:max:reference"
        ] = - (3 / 3.5) * m_mot_ref * T_mot_max ** (3 / 3.5) / T_mot_max_ref ** (6.5 / 3.5)

        partials["data:weight:propulsion:motor:mass:estimated",
                 "models:weight:propulsion:motor:mass:reference"
        ] = (T_mot_max / T_mot_max_ref) ** (3 / 3.5)


class Geometry(om.ExplicitComponent):
    """
    Computes motor geometry (length and outer diameter)
    """

    def setup(self):
        self.add_input("models:propulsion:motor:length:reference", val=np.nan, units="m")
        self.add_input("models:propulsion:motor:diameter:reference", val=np.nan, units="m")
        self.add_input("models:weight:propulsion:motor:mass:reference", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:motor:mass:estimated", val=np.nan, units="kg")
        self.add_output("data:propulsion:motor:length:estimated", units="m")
        self.add_output("data:propulsion:motor:diameter:estimated", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        L_mot_ref = inputs["models:propulsion:motor:length:reference"]
        D_ext_ref = inputs["models:propulsion:motor:diameter:reference"]
        m_mot_ref = inputs["models:weight:propulsion:motor:mass:reference"]
        m_mot = inputs["data:weight:propulsion:motor:mass:estimated"]

        # L ~ m^(1/3), D ~ m^(1/3) ~ T^(1/3.5)
        L_mot = L_mot_ref * (m_mot / m_mot_ref) ** (1.0 / 3.0)
        D_ext = D_ext_ref * (m_mot / m_mot_ref) ** (1.0 / 3.0)

        outputs["data:propulsion:motor:length:estimated"] = L_mot
        outputs["data:propulsion:motor:diameter:estimated"] = D_ext

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        L_mot_ref = inputs["models:propulsion:motor:length:reference"]
        D_ext_ref = inputs["models:propulsion:motor:diameter:reference"]
        m_mot_ref = inputs["models:weight:propulsion:motor:mass:reference"]
        m_mot = inputs["data:weight:propulsion:motor:mass:estimated"]

        # --- length partials ---
        partials["data:propulsion:motor:length:estimated",
                 "models:propulsion:motor:length:reference"
        ] = (m_mot / m_mot_ref) ** (1.0 / 3.0)

        partials["data:propulsion:motor:length:estimated",
                 "data:weight:propulsion:motor:mass:estimated"
        ] = (1.0 / 3.0) * L_mot_ref / m_mot_ref ** (1.0 / 3.0) * m_mot ** (-2.0 / 3.0)

        partials["data:propulsion:motor:length:estimated",
                 "models:weight:propulsion:motor:mass:reference"
        ] = -(1.0 / 3.0) * L_mot_ref * m_mot ** (1.0 / 3.0) / m_mot_ref ** (4.0 / 3.0)

        # --- diameter partials ---
        partials["data:propulsion:motor:diameter:estimated",
                 "models:propulsion:motor:diameter:reference"
        ] = (m_mot / m_mot_ref) ** (1.0 / 3.0)

        partials["data:propulsion:motor:diameter:estimated",
                 "data:weight:propulsion:motor:mass:estimated"
        ] = (1.0 / 3.0) * D_ext_ref / m_mot_ref ** (1.0 / 3.0) * m_mot ** (-2.0 / 3.0)

        partials["data:propulsion:motor:diameter:estimated",
                 "models:weight:propulsion:motor:mass:reference"
        ] = -(1.0 / 3.0) * D_ext_ref * m_mot ** (1.0 / 3.0) / m_mot_ref ** (4.0 / 3.0)


class IronLoss(om.ExplicitComponent):
    """
    Motor iron loss scaled from reference using geometric scaling law.
    Aroua et al., eTransportation 2023, eq.15:  P_fer = KA*KR^2*P0_fer
    For isotropic UAV outrunner scaling: KA*KR^2 = m/m_ref
    Speed scaling (Steinmetz): P_fer(f) = Ph_ref*(f/f0) + Pc_ref*(f/f0)^2
    where f/f0 = n/n0  (pole-pair factor cancels in the ratio)
    """

    def setup(self):
        self.add_input("data:weight:propulsion:motor:mass:estimated", val=np.nan, units="kg")
        self.add_input("models:weight:propulsion:motor:mass:reference", val=np.nan, units="kg")
        # Propeller speed at takeoff (rad/s) — converted to rpm internally.
        # Uses propeller speed (= motor speed for N_red=1 multirotor) which is
        # computed upstream of this group and thus available without a solver loop.
        self.add_input("data:propulsion:propeller:speed:takeoff", val=np.nan, units="rad/s")
        self.add_input("models:propulsion:motor:iron:Ph_ref", val=np.nan, units="W")
        self.add_input("models:propulsion:motor:iron:Pc_ref", val=np.nan, units="W")
        self.add_input("models:propulsion:motor:iron:Ppm_ref", val=np.nan, units="W")
        self.add_input("models:propulsion:motor:iron:segPM", val=1.0, units=None)
        self.add_input("models:propulsion:motor:iron:n0", val=np.nan, units="rpm")
        self.add_input("models:propulsion:motor:pole_pairs", val=7.0, units=None)
        self.add_output("data:propulsion:motor:iron_loss:estimated", val=0.0, units="W")
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        m = inputs["data:weight:propulsion:motor:mass:estimated"]
        m_ref = inputs["models:weight:propulsion:motor:mass:reference"]
        omega = inputs["data:propulsion:propeller:speed:takeoff"]   # [rad/s]
        n = omega * 60.0 / (2.0 * np.pi)                           # convert to [rpm]
        Ph0 = inputs["models:propulsion:motor:iron:Ph_ref"]
        Pc0 = inputs["models:propulsion:motor:iron:Pc_ref"]
        Ppm0 = inputs["models:propulsion:motor:iron:Ppm_ref"]
        segPM = inputs["models:propulsion:motor:iron:segPM"]
        n0 = inputs["models:propulsion:motor:iron:n0"]

        KT = m / m_ref        # geometric mass scaling factor (KA*KR^2 for isotropic scaling)
        f_ratio = n / n0      # frequency ratio (f/f0 = n/n0, pole pairs cancel)

        # Total iron loss = hysteresis + stator eddy + PM eddy  (Aroua 2023 eq.15-16)
        # PM eddy current loss scales with (f/f0)^2 and PM segmentation factor
        outputs["data:propulsion:motor:iron_loss:estimated"] = KT * (
            Ph0 * f_ratio + Pc0 * f_ratio ** 2 + segPM * Ppm0 * f_ratio ** 2
        )

    def compute_partials(self, inputs, partials):
        m = inputs["data:weight:propulsion:motor:mass:estimated"]
        m_ref = inputs["models:weight:propulsion:motor:mass:reference"]
        omega = inputs["data:propulsion:propeller:speed:takeoff"]   # [rad/s]
        n = omega * 60.0 / (2.0 * np.pi)                           # [rpm]
        Ph0 = inputs["models:propulsion:motor:iron:Ph_ref"]
        Pc0 = inputs["models:propulsion:motor:iron:Pc_ref"]
        Ppm0 = inputs["models:propulsion:motor:iron:Ppm_ref"]
        segPM = inputs["models:propulsion:motor:iron:segPM"]
        n0 = inputs["models:propulsion:motor:iron:n0"]

        KT = m / m_ref
        f_ratio = n / n0
        P_scaled = Ph0 * f_ratio + Pc0 * f_ratio ** 2 + segPM * Ppm0 * f_ratio ** 2

        # d(n_rpm)/d(omega) = 30/pi  [rpm/(rad/s)]
        dn_domega = 30.0 / np.pi

        out = "data:propulsion:motor:iron_loss:estimated"
        partials[out, "data:weight:propulsion:motor:mass:estimated"] = P_scaled / m_ref
        partials[out, "models:weight:propulsion:motor:mass:reference"] = -KT * P_scaled / m_ref
        # Chain rule: dP/d(omega) = dP/d(n) * d(n)/d(omega)
        partials[out, "data:propulsion:propeller:speed:takeoff"] = KT * (
            Ph0 / n0 + 2 * Pc0 * n / n0 ** 2 + 2 * segPM * Ppm0 * n / n0  ** 2
        ) * dn_domega
        partials[out, "models:propulsion:motor:iron:Ph_ref"] = KT * f_ratio
        partials[out, "models:propulsion:motor:iron:Pc_ref"] = KT * f_ratio ** 2
        partials[out, "models:propulsion:motor:iron:Ppm_ref"] = KT * segPM * f_ratio ** 2
        partials[out, "models:propulsion:motor:iron:segPM"] = KT * Ppm0 * f_ratio ** 2
        partials[out, "models:propulsion:motor:iron:n0"] = KT * (
            -Ph0 * n / n0 ** 2 - 2 * Pc0 * n ** 2 / n0 ** 3 - 2 * segPM * Ppm0 * n ** 2 / n0 ** 3
        )
        partials[out, "models:propulsion:motor:pole_pairs"] = 0.0
