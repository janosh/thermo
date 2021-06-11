import pandas as pd
from pymatgen import MPRester
from pymatgen.ext.cod import COD

from thermo.utils import ROOT


def fetch_mp(criteria={}, properties=[], save_to=None):
    """Fetch data from the Materials Project (MP).
    Docs at https://docs.materialsproject.org.
    Pymatgen MP source at https://pymatgen.org/_modules/pymatgen/ext/matproj.

    Note: Unlike ICSD - a database of materials that actually exist - MP has
    all structures where DFT+U converges. Those can be thermodynamically
    unstable if they lie above the convex hull. Set criteria = {"e_above_hull": 0}
    to get stable materials only.

    Args:
        criteria (dict, optional): filter criteria which returned items must
            satisfy, e.g. criteria = {"material_id": {"$in": ["mp-7988", "mp-69"]}}.
            Supports all features of the Mongo query syntax.
        properties (list, optional): quantities of interest, can be selected from
            https://materialsproject.org/docs/api#resources_1 or
            MPRester().supported_properties.
        save_to (str, optional): Pass a file path to save the data returned by MP
            API as CSV. Defaults to None.

    Returns:
        df: pandas DataFrame with a column for each requested property
    """

    properties = list({*properties, "material_id"})  # use set to remove dupes

    # MPRester connects to the Material Project REST interface.
    # API keys available at https://materialsproject.org/dashboard.
    with MPRester() as mp:
        # mp.query performs the actual API call.
        data = mp.query(criteria, properties)

    if data:
        df = pd.DataFrame(data)[properties]  # ensure same column order as in properties

        df = df.set_index("material_id")

        if save_to:
            data.to_csv(ROOT + save_to, float_format="%g")

        return df
    else:
        raise ValueError("query returned no data")


def fetch_cod(formulas=None, ids=None, get_ids_for=None):
    """Fetch data from the Crystallography Open Database (COD).
    Docs at https://pymatgen.org/pymatgen.ext.cod.
    Needs the mysql binary to be in path to run queries. Installable
    via `brew install mysql`.
    """
    cod = COD()
    if formulas:
        return [cod.get_structure_by_formula(f) for f in formulas]
    if ids:
        return [cod.get_structure_by_id(i) for i in ids]
    if get_ids_for:
        return [cod.get_cod_ids(i) for i in get_ids_for]
    raise ValueError("fetch_cod() requires formulas or ids.")
