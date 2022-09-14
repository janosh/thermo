from pymatgen.ext.cod import COD


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
