"""
Motor Material Cost Model

Computes per-material and total raw material cost of one motor unit.

Data sources (default CSV shipped with the package):
  src/fastuav/data/supply_chain/material_prices.csv

Optional API refresh (World Bank commodity price API, free, no auth required):
  PriceFetcher.refresh_from_worldbank(csv_path)
"""
import pathlib
import warnings

import numpy as np
import pandas as pd
import fastoad.api as oad
import openmdao.api as om

_DATA_DIR = pathlib.Path(__file__).parents[4] / "data" / "supply_chain"
_DEFAULT_PRICES_CSV = _DATA_DIR / "material_prices.csv"

MATERIALS = ("copper", "ndfeb_magnet", "electrical_steel", "aluminum", "other")
# Mapping from supply_chain material key → OpenMDAO variable suffix
_MAT_VARNAME = {
    "copper":           "copper",
    "ndfeb_magnet":     "magnet",
    "electrical_steel": "steel",
    "aluminum":         "aluminum",
    "other":            "other",
}
# Fallback prices [USD/kg] if CSV is missing or row is absent
_FALLBACK_PRICES = {
    "copper":           9.50,
    "ndfeb_magnet":    85.00,
    "electrical_steel": 1.80,
    "aluminum":         2.40,
    "other":            5.00,
}


class PriceFetcher:
    """
    Utility class to load and optionally refresh material prices.

    Usage
    -----
    prices = PriceFetcher.load()                 # from default CSV
    prices = PriceFetcher.load(my_csv_path)      # from custom CSV

    # One-time refresh from World Bank API (writes updated CSV):
    PriceFetcher.refresh_from_worldbank(my_csv_path)
    """

    @staticmethod
    def load(csv_path=None) -> dict:
        """Return {material: price_usd_per_kg} dict from CSV."""
        path = pathlib.Path(csv_path) if csv_path else _DEFAULT_PRICES_CSV
        if not path.exists():
            warnings.warn(
                f"Material prices CSV not found at {path}. "
                "Using built-in fallback values.", UserWarning
            )
            return dict(_FALLBACK_PRICES)

        df = pd.read_csv(path, comment="#")
        prices = {}
        for _, row in df.iterrows():
            prices[row["material"]] = float(row["price_usd_per_kg"])
        # Fill any missing materials with fallback
        for mat, val in _FALLBACK_PRICES.items():
            prices.setdefault(mat, val)
        return prices

    @staticmethod
    def refresh_from_worldbank(csv_path=None, timeout=10):
        """
        Fetch latest annual average prices from the World Bank commodity API
        for materials that have a `world_bank_indicator` entry in the CSV,
        then write updated prices back to the CSV file.

        Currently supported via World Bank API:
          - copper   → indicator PCOPP  (USD / metric ton → /1000 for USD/kg)
          - aluminum → indicator PALUM  (USD / metric ton → /1000 for USD/kg)

        NdFeB magnet and electrical steel prices are updated manually
        (no free public API available).

        Parameters
        ----------
        csv_path : str or Path, optional
            Path to the prices CSV. Defaults to the package data file.
        timeout : int
            HTTP request timeout in seconds.
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "The 'requests' package is required for API refresh. "
                "Install it with: pip install requests"
            )

        path = pathlib.Path(csv_path) if csv_path else _DEFAULT_PRICES_CSV
        df = pd.read_csv(path, comment="#")

        WB_BASE = "https://api.worldbank.org/v2/en/indicator"
        updated = []

        for idx, row in df.iterrows():
            indicator = str(row.get("world_bank_indicator", "")).strip()
            if not indicator or indicator == "nan":
                updated.append(row["price_usd_per_kg"])
                continue

            url = (
                f"{WB_BASE}/PCOMM{indicator}.MTL.USD.M"
                "?format=json&per_page=12&mrv=12"
            )
            try:
                resp = requests.get(url, timeout=timeout)
                data = resp.json()
                # World Bank returns [metadata, [records]]
                records = data[1] if isinstance(data, list) and len(data) > 1 else []
                values = [
                    float(r["value"])
                    for r in records
                    if r.get("value") is not None
                ]
                if values:
                    # price in USD/metric-ton → USD/kg
                    new_price = np.mean(values) / 1000.0
                    updated.append(round(new_price, 4))
                    print(
                        f"  {row['material']}: updated {row['price_usd_per_kg']:.4f}"
                        f" → {new_price:.4f} USD/kg"
                    )
                else:
                    updated.append(row["price_usd_per_kg"])
                    warnings.warn(f"No data returned for {row['material']} ({indicator}).")
            except Exception as exc:
                updated.append(row["price_usd_per_kg"])
                warnings.warn(f"Could not fetch price for {row['material']}: {exc}")

        df["price_usd_per_kg"] = updated
        import datetime
        df["year"] = datetime.datetime.now().year
        df.to_csv(path, index=False)
        print(f"Prices saved to {path}")


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.motor.supply_chain.material_cost")
class MotorMaterialCost(om.ExplicitComponent):
    """
    Computes raw material cost of one motor from per-material masses and prices.

    Inputs
    ------
    data:propulsion:motor:supply_chain:mass:<material>  [kg]
    data:propulsion:motor:supply_chain:price:<material> [USD/kg]  (loaded from CSV as defaults)

    Outputs
    -------
    data:propulsion:motor:supply_chain:cost:<material>      [USD]
    data:propulsion:motor:supply_chain:cost:total_material  [USD]
    """

    def initialize(self):
        self.options.declare(
            "prices_csv", default=None, allow_none=True,
            desc="Path to material_prices.csv. None → package default."
        )

    def setup(self):
        prices = PriceFetcher.load(self.options["prices_csv"])

        for mat_key, var_suffix in _MAT_VARNAME.items():
            self.add_input(
                f"data:propulsion:motor:supply_chain:mass:{var_suffix}",
                val=np.nan, units="kg"
            )
            self.add_input(
                f"data:propulsion:motor:supply_chain:price:{var_suffix}",
                val=prices.get(mat_key, _FALLBACK_PRICES[mat_key]),
                desc=f"{mat_key} price [USD/kg]"
            )
            self.add_output(
                f"data:propulsion:motor:supply_chain:cost:{var_suffix}",
                val=0.0,
                desc=f"Raw material cost for {mat_key} [USD]"
            )

        self.add_output(
            "data:propulsion:motor:supply_chain:cost:total_material",
            val=0.0,
            desc="Total raw material cost for one motor [USD]"
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        total = 0.0
        for var_suffix in _MAT_VARNAME.values():
            mass  = inputs[f"data:propulsion:motor:supply_chain:mass:{var_suffix}"]
            price = inputs[f"data:propulsion:motor:supply_chain:price:{var_suffix}"]
            cost  = float(mass) * float(price)
            outputs[f"data:propulsion:motor:supply_chain:cost:{var_suffix}"] = cost
            total += cost

        outputs["data:propulsion:motor:supply_chain:cost:total_material"] = total
