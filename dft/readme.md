# Automate VASP Calculations

To allow Pymatgen to generate input files for VASP runs based on `Structure`s (stored in `data/structures`), it needs to know where to find VASP's proprietary pseudo-potential files. This can be done with

```sh
pmg config --add PMG_VASP_PSP_DIR /Users/janosh/Repos/thermo/data/vasp-pot
```

Note that Pymatgen requires that the folders for different functionals inside the VAS Ppseudo-potential directory be named `POT_LDA_PAW_PBE`, `POT_GGA_PAW_PBE`, etc.

## VASP Error Glossary

### WARNING: small aliasing (wrap around) errors must be expected

Probably not important.

### Warning: The distance between some ions is very small. Please check the nearest neigbor list in the OUTCAR file. I HOPE YOU KNOW, WHAT YOU ARE DOING.

One or more atoms in POSCAR file have nearly identical coordinates. There must be something wrong with the CIF/JSON/... structure source file used to generate the POSCAR. If Pymatgen's `Structure.from_str(struct_str)` was involved, it's likely that there was a mismatch between the CIF file and Pymatgen's symmetry tolerances. In that case try increasing `merge_tol`, e.g. `Structure.from_str(struct_str, merge_tol=0.01)` or similar. Also, check atom positions for that structure on Materials Project, if it exists. If all else fails and willing to invest the compute time, recreate VASP input files with `MPRelaxSet` instead of `MPStaticSet` and relax the crystal sructure yourself. Then use the `CONTCAR` file from that run for the `MPRelaxSet`.
