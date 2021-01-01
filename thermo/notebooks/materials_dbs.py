"""
This notebook plots the number of materials databases over time and writes them
to LaTeX tables.
"""


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermo.utils import ROOT

# %%
dbs = pd.read_csv(ROOT + "/data/materials_dbs.csv", comment="#")


# %% [markdown]
# # Timeline of new materials databases
# Adapted from https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/timeline.


# %%
# Choose some nice levels
levels = np.arange(len(dbs)) + 2
levels[::2] *= -1

# Create figure and plot a stem plot with the date
plt.figure(figsize=(12, 8))

markerline, stemline, baseline = plt.stem(dbs.year, levels, linefmt="C3-", basefmt="k-")

plt.setp(markerline, mec="k", mfc="w", zorder=3)

# annotate lines
v_align = np.where(levels < 0, "top", "bottom")

text_pos = lambda level, va: dict(
    xytext=(-3, np.sign(level) * 3),
    textcoords="offset points",
    verticalalignment=va,
    horizontalalignment="right",
)

prev_year = 0

for year, name, level, va in zip(
    *dbs[["year", "abbreviation"]].values.T, levels, v_align
):
    plt.annotate(name, xy=(year, level), **text_pos(level, va))
    if year > prev_year + 3:
        prev_year = year
        plt.annotate(year, xy=(year, 0), **text_pos(level, va))

plt.axis("off")


# %% [markdown]
# # Cumulative count of materials databases over time


# %%
dbs.reset_index().plot(
    x="year", y="index", drawstyle="steps", legend=None, figsize=(10, 5)
)
plt.fill_between(dbs.year, dbs.index, step="pre", alpha=0.4)

for idx, [year, abbr, url] in dbs[["year", "abbreviation", "url"]].T.items():
    plt.annotate(
        abbr, [year, idx], [year - 7, idx + 2], arrowprops={"arrowstyle": "->"}, url=url
    )


plt.ylabel("database count")
plt.xlim(dbs.year.min() - 10, dbs.year.max())
plt.ylim(0, dbs.index.max() + 4)
plt.savefig("material-dbs-cumulative-over-time.pdf", bbox_inches="tight")


# %%
def data_to_latex(subset="experiment"):
    # convert names and urls to LaTeX hyperlinks
    # https://stackoverflow.com/a/52854800
    link_name = lambda x: f"\\href{{{x.url}}}{{{x['name']}}}"
    dbs["linked_name"] = dbs.apply(link_name, axis=1)

    # convert sizes to siunitx-formatted numbers
    dbs["num_size"] = dbs["size"].apply(lambda x: f"\\num{{{x:.0f}}}" if x > 0 else x)

    dbs[dbs.data_source == subset][
        ["linked_name", "abbreviation", "year", "num_size"]
    ].rename(columns={"linked_name": "name", "num_size": "size"}).to_latex(
        f"{subset}-dbs.tex", escape=False, index=False, na_rep="n/a"
    )


# %%
# pd.set_option("display.max_colwidth", -1)
# data_to_latex("experiment")
# data_to_latex("simulation")
