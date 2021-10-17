# %%
import pandas as pd
from pymatgen.ext.cod import COD
from pymatgen.ext.matproj import MPRester
from tqdm import tqdm


# %%
gurobi_candidates = pd.read_csv("gurobi_candidates.csv", index_col=[0, "id", "T"])
greedy_candidates = pd.read_csv(
    "greedy-candidates-epochs=140-batch=32-n_preds=50.csv", index_col=[0, "id", "T"]
)

gurobi_candidates["source"] = "gurobi"
greedy_candidates["source"] = "greedy"

(n_gurobi_candidates := len(gurobi_candidates))


# %% COD() methods require mysql in path (brew install mysql)
cod = COD()
mpr = MPRester()

# pick same number of greedy as Gurobi candidates for fair comparison
# of both selection methods (greedy contains the full screening set
# just ordered descending by predicted zT)
candidates = pd.concat([greedy_candidates.head(n_gurobi_candidates), gurobi_candidates])

# change source to 'both' on duplicate rows
candidates.loc[
    candidates.reset_index([0, 2]).index.duplicated(keep=False), "source"
] = "both"

# remove duplicates
candidates = (
    candidates.reset_index().drop_duplicates("id").set_index(["level_0", "id", "T"])
)


cod_candidates = candidates[candidates.database == "COD"]
icsd_candidates = candidates[candidates.database == "ICSD"]


# %%
invalid_cod_cifs = []
missing_icsd_ids = []


# %%
for idx, formula, database, *_ in tqdm(candidates[130:].itertuples()):
    _, db_id, _ = idx
    if database == "COD":
        try:
            # advice from @shyuep how to get from COD IDs to MP IDs https://git.io/JOPI1
            struct = cod.get_structure_by_id(db_id)
            struct.merge_sites(tol=0.003, mode="delete")

            mp_id = mpr.find_structure(struct)
            candidates.loc[idx, "mp_id"] = mp_id
        except ValueError:
            invalid_cod_cifs.append([db_id, formula])
            continue
    else:
        data = mpr.query({"icsd_ids": db_id}, ["material_id"])
        if len(data) == 1:
            mp_id = data[0]["material_id"]
            candidates.loc[idx, "mp_id"] = mp_id
        elif len(data) == 0:
            missing_icsd_ids.append([db_id, formula])
        else:
            ValueError("An ICSD ID returned more than 1 MP ID, this shouldn't happen.")

# make sure, the same number of COD IDs that could not be matched to an MP ID
# ended up in invalid_cod_cifs
assert candidates.mp_id.isna().sum() == len([*invalid_cod_cifs, *missing_icsd_ids])

# 23 COD IDs yield invalid_cifs:
# [[2204243, 'Bi2Cs2Se5Zn'],
#  [1527367, 'Bi8Te9'],
#  [5910072, 'Sb4Tl5'],
#  [5910260, 'Hg2I3NaO2'],
#  [1523090, 'HgPo'],
#  [4002338, 'Bi8Cs2Se13'],
#  [1010622, 'BiI3'],
#  [1529984, 'HgI4K2'],
#  [1521988, 'TeTl'],
#  [1523087, 'PbPo'],
#  [9011287, 'AuTe2'],
#  [1528242, 'CrI6Tl4'],
#  [4000782, 'GaSe2Tl'],
#  [5910295, 'CdGa2Te4'],
#  [5910274, 'Ga2Te4Zn'],
#  [5910282, 'Ga2HgSe4'],
#  [9008069, 'AgCuSe'],
#  [5910050, 'Ag3Hg4'],
#  [4001921, 'Ga4Se7Sn'],
#  [1523046, 'Ga5Ir3'],
#  [5910326, 'In2Se4Zn'],
#  [2106208, 'In6Se7'],
#  [4304497, 'In9Se14']]

# 5 ICSD IDs yield no associated MP IDs
# [[656155, 'Cu2GeHgTe4'],
#  [634431, 'GaInSe3'],
#  [152189, 'Sb16Te3'],
#  [610367, 'AsCuTe'],
#  [69680, 'Cs2Se7Sn3']]


# %% ICSD IDs can be matched directly to MP IDs by MPRester (uses structural similarity)
# https://git.io/JOPJM, finds 270 out of 275 IDs
icsd_ids = icsd_candidates.reset_index(level=1).id
data = mpr.query(
    {"icsd_ids": {"$in": icsd_ids.to_list()}},
    ["material_id", "icsd_ids", "e_above_hull", "pretty_formula"],
)
(icsd_df := pd.DataFrame(data))


# %%
data = mpr.query(
    {"pretty_formula": {"$in": candidates.formula.to_list()}},
    ["material_id", "e_above_hull", "pretty_formula"],
)
(formula_match_df := pd.DataFrame(data))


# %% not recommended to try match IDs from different DBs based on chemical formula
candidates.reset_index(level=[1, 2]).merge(
    icsd_ids.rename(columns={"pretty_formula": "formula", "material_id": "mp_id"}),
    on="formula",
).to_csv("mnf_candidates_in_ricci_dataset/mnf-candidates-with-mp-ids.csv", index=False)


# %%
candidates.reset_index([1, 2]).to_csv("mnf-candidates-with-mp-ids.csv", index=False)
