"""
Supply-chain models.

Two layers with distinct roles -- keep them separate:

1. ``motor/`` -- IN-LOOP discipline model.
   OpenMDAO components registered as ``fastuav.propulsion.motor.supply_chain*``
   (material composition, raw-material cost, supply risk). Sourced prices in
   ``fastuav/data/supply_chain/material_prices.csv`` (LME / USGS, 2024).
   This is the authoritative model for the MOTOR, and the only one that can
   participate in MDA / MDO / Sobol UQ.

2. ``model.py`` -- POST-PROCESSING procurement layer.
   Plain pandas, runs outside the MDAO loop. Scores a whole-drone BOM for
   cost / lead time / supply risk and feeds the Streamlit dashboard
   (``notebooks/supply_chain_app.py``).

The procurement layer must NOT re-derive motor raw-material numbers of its
own -- it defers to layer 1 for the motor row so a single motor cannot be
priced two different ways. See ``MATERIAL_COMPOSITION`` in ``model.py``.
"""

from .model import *  # noqa: F401,F403  (procurement layer public helpers)
