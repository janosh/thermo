# %%
import matplotlib.pyplot as plt
import pandas as pd
import pymatviz as pmv
from matminer.utils.io import load_dataframe_from_json


# %% this file is available at
# https://github.com/janosh/matbench/blob/main/data/carrier_transport.json.gz
carrier_transport = load_dataframe_from_json(
    "/Users/janosh/Repos/matbench/data/carrier_transport_with_strucs.json.gz"
)


# %%
mnf_candidates = pd.read_csv("mnf-candidates-with-mp-id.csv")


# %% 269 out of 446 MNF candidates
mnf_candidates.mp_id.isin(carrier_transport.index).sum()


# %%
mnf_in_carrier = carrier_transport.loc[
    mnf_candidates[mnf_candidates.mp_id.isin(carrier_transport.index)].mp_id
]


# %%
ax = pmv.ptable_heatmap(mnf_in_carrier.pretty_formula.dropna(), log=True)
ax.set_title(
    "Elemental prevalence of MNF candidates in the Ricci carrier transport dataset"
)
pmv.save_fig(ax, "mnf-in-carrier-elements-log.pdf")


# %%
top_power_factors = carrier_transport.sort_values(by="PF.p [µW/cm/K²/s]").tail(
    len(mnf_in_carrier)
)
ax = pmv.ptable_heatmap(top_power_factors.pretty_formula.dropna(), log=True)
ax.set_title(
    "Elemental prevalence among highest power factor materials in "
    "the Ricci carrier transport dataset"
)
pmv.save_fig(ax, "top-pf-in-carrier-elements-log.pdf")


# %%
ax = mnf_in_carrier.hist(bins=50, log=True, figsize=[30, 16])
plt.suptitle(
    "Properties of MNF Candidates according to Ricci carrier transport Dataset", y=1.05
)
pmv.save_fig(ax, "mnf-candidates-in-carrier-transport-hists.pdf")


# %%
dependent_vars = [
    "Sᵉ.p.v [µV/K]",
    "Sᵉ.n.v [µV/K]",
    "σᵉ.p.v [1/Ω/m/s]",
    "σᵉ.n.v [1/Ω/m/s]",
    "PFᵉ.p.v [µW/cm/K²/s]",
    "PFᵉ.n.v [µW/cm/K²/s]",
    "κₑᵉ.p.v [W/K/m/s]",
    "κₑᵉ.n.v [W/K/m/s]",
]

ax = mnf_in_carrier[dependent_vars].hist(
    bins=50, log=True, figsize=[15, 15], layout=[4, 2]
)
plt.suptitle(
    "Carrier transport property distributions of MNF candidates "
    "present in Ricci carrier transport Dataset",
    y=1.05,
)
pmv.save_fig(ax, "mnf-candidates-carrier-transport-hists-dependent-vars.pdf")


# %%
mnf_in_carrier[dependent_vars].describe()


# %%
carrier_transport[dependent_vars].describe()


# %%
# probably not a reliable way of computing zT_el due to
# https://nature.com/articles/sdata201785#Sec15
# > It is also important to note that when a derived property is needed (e.g., the power
# > factor S^2 σ), it would be wrong to operate on eigenvalues (since they might not
# > refer to corresponding directions). Therefore, we strongly suggest to instead
# > perform the operations on the full tensors. Eigenvalues can be obtained by running
# > an adequate algorithm on the resulting full tensor.
carrier_transport["zT_el"] = (
    (carrier_transport.dropna()["Sᵉ.p.v [µV/K]"] * 1e-6) ** 2
    * carrier_transport.dropna()["σᵉ.p.v [1/Ω/m/s]"]
    / carrier_transport.dropna()["κₑᵉ.p.v [W/K/m/s]"]
    * 300
)


# %%
mnf_in_carrier["zT_el"] = (
    (mnf_in_carrier["Sᵉ.p.v [µV/K]"] * 1e-6) ** 2
    * mnf_in_carrier["σᵉ.p.v [1/Ω/m/s]"]
    / mnf_in_carrier["κₑᵉ.p.v [W/K/m/s]"]
    * 300
)


# %%
carrier_transport.zT_el.describe()
