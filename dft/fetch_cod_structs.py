# %%
import os
from glob import glob
from os.path import basename, dirname, isfile

import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.ext.cod import COD
from pymatgen.io.vasp.sets import MPStaticSet
from tqdm import tqdm

from thermo.utils import ROOT


# %%
gurobi_candidates = pd.read_csv(
    f"{ROOT}/notebooks/screen/mnf_magpie/gurobi_candidates.csv",
    index_col=[0, "id", "T"],
)
greedy_candidates = pd.read_csv(
    f"{ROOT}/notebooks/screen/mnf_magpie/greedy-candidates-epochs=140-batch=32-n_preds=50.csv",
    index_col=[0, "id", "T"],
)

(n_gurobi_candidates := len(gurobi_candidates))


# %%
(n_candidates := len(gurobi_candidates))

# COD() methods require mysql in path (brew install mysql)
cod = COD()

candidates = (
    pd.concat([greedy_candidates.head(n_gurobi_candidates), gurobi_candidates])
    .reset_index()
    .drop_duplicates("id")
    .set_index(["level_0", "id", "T"])
)


# %%
invalid_cifs = []
for (_, cod_id, _), (formula, database) in tqdm(
    candidates[candidates.database == "COD"][["formula", "database"]].iterrows()
):
    path = f"{ROOT}/dft/{database}-{cod_id}-{formula}/structure.json"
    if isfile(path):
        continue
    try:
        os.makedirs(dirname(path), exist_ok=True)
        cod.get_structure_by_id(cod_id).to(filename=path)
    except ValueError:
        invalid_cifs.append([cod_id, formula])
        continue

print(f"{invalid_cifs=}")
# crashing due to supposedly invalid CIF files (might be Pymatgen bug):
# UserWarning: Issues encountered while parsing CIF: No _symmetry_equiv_pos_as_xyz
# type key found. Spacegroup from _symmetry_space_group_name_H-M used.
# Some occupancies ([1, 2, 6, 6]) sum to > 1! If they are within the occupancy_tolerance,
# they will be rescaled. The current occupancy_tolerance is set to: 1.0

# invalid_cifs = [
#     [5910072, "Sb4Tl5"],
#     [5910260, "Hg2I3NaO2"],
#     [5910295, "CdGa2Te4"],
#     [5910274, "Ga2Te4Zn"],
#     [5910282, "Ga2HgSe4"],
#     [5910326, "In2Se4Zn"],
# ]


# %%
structure_paths = glob(f"{ROOT}/dft/COD-*/structure.json")

crashing_structs = []


# %%
for struct_path in structure_paths:

    path = dirname(struct_path)
    if isfile(f"{path}/INCAR"):
        continue

    struct = Structure.from_file(struct_path)
    mps = MPStaticSet(struct, reciprocal_density=1e3)
    print(f"\ngenerating VASP input files for {path=}\n")
    try:
        mps.write_input(path)
    except (OSError, KeyError):
        # OSError: You do not have the right POTCAR with functional
        # PBE and label <element> in your VASP_PSP_DIR
        crashing_structs.append(basename(path))
        continue

print(f"{crashing_structs=}")
# crashing_structs = [
#     "COD-4312037-CsTe3TmZn",
#     "COD-4030531-NaS2Tm",
#     "COD-4030787-ErRbS2",
#     "COD-4312033-CsDyTe3Zn",
#     "COD-2020013-HoKS2",
#     "COD-4030786-HoRbS2",
#     "COD-1523090-HgPo",
#     "COD-4310304-CsGeSe4Sm",
#     "COD-4312035-CsHoTe3Zn",
#     "COD-4313793-CdCsSe3Tb",
#     "COD-4313802-CsHgSe3Sm",
#     "COD-4030532-NaS2Yb",
#     "COD-4312036-CsErTe3Zn",
# ]
