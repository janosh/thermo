"""This module was adapted from https://codereview.stackexchange.com/questions/181191.
Pass the total_atom_count/flatten_formula functions a chemical formula to get the
total number of atoms or the number of atoms for each element, respectively.
"""

import re


RE = re.compile(
    r"(?P<atom>[A-Z][a-z]*)(?P<atom_count>\d*)|"
    r"(?P<new_group>\()|"
    r"\)(?P<group_count>\d*)|"
    r"(?P<UNEXPECTED_CHARACTER_IN_FORMULA>.+)"
)


def atom_count(stack, atom, atom_count="", **_):
    """Handle an atom with an optional count, e.g. H or Mg2."""
    stack[-1][atom] += 1 if atom_count == "" else int(atom_count)


def group_count(stack, group_count="", **_):
    """Handle a closing parenthesis with an optional group count."""
    group_count = 1 if group_count == "" else int(group_count)
    group = stack.pop()
    for atom in group:
        group[atom] *= group_count
    stack[-1] += group
