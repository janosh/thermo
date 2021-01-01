from collections import Counter

from thermo.utils.atom_count import (
    flatten_formula,
    formula_to_dict,
    total_atom_count,
)

formulas = ["(H2O)2Ge", "(H2(GaKSb4)2O2)2Ge", "GaKSb4", "Bi2Te3"]
counts = [7, 33, 6, 5]
flattened = ["GeH4O2", "Ga4GeH4K4O4Sb16", "GaKSb4", "Bi2Te3"]
dicts = [
    Counter({"H": 4, "O": 2, "Ge": 1}),
    Counter({"H": 4, "Ga": 4, "K": 4, "Sb": 16, "O": 4, "Ge": 1}),
    Counter({"Ga": 1, "K": 1, "Sb": 4}),
    Counter({"Bi": 2, "Te": 3}),
]


def test_formula_to_dict():
    for form, dic in zip(formulas, dicts):
        assert formula_to_dict(form) == dic


def test_total_atom_count():
    for form, count in zip(formulas, counts):
        assert total_atom_count(form) == count


def test_flatten_formula():
    for form, flat in zip(formulas, flattened):
        assert flatten_formula(form) == flat
