"""This module was adapted from https://codereview.stackexchange.com/questions/181191.
Pass the total_atom_count/flatten_formula functions a chemical formula to get the
total number of atoms or the number of atoms for each element, respectively.
"""

import re
from collections import Counter

RE = re.compile(
    r"(?P<atom>[A-Z][a-z]*)(?P<atom_count>\d*)|"
    r"(?P<new_group>\()|"
    r"\)(?P<group_count>\d*)|"
    r"(?P<UNEXPECTED_CHARACTER_IN_FORMULA>.+)"
)


def atom_count(stack, atom, atom_count="", **_):
    """Handle an atom with an optional count, e.g. H or Mg2."""
    stack[-1][atom] += 1 if atom_count == "" else int(atom_count)


def new_group(stack, **_):
    """Handle an opening parenthesis."""
    stack.append(Counter())


def group_count(stack, group_count="", **_):
    """Handle a closing parenthesis with an optional group count."""
    group_count = 1 if group_count == "" else int(group_count)
    group = stack.pop()
    for atom in group:
        group[atom] *= group_count
    stack[-1] += group


def formula_to_dict(formula):
    """Generate a stack of formula transformations with successively unpacked
    groups and return the last one."""
    stack = []
    new_group(stack)
    for match in RE.finditer(formula):
        globals()[match.lastgroup](stack, **match.groupdict())
    return stack[-1]


def flatten_formula(formula):
    d = formula_to_dict(formula)
    return "".join(
        atom + (str(count) if count > 1 else "") for atom, count in sorted(d.items())
    )


def total_atom_count(formula):
    return sum(formula_to_dict(formula).values())
