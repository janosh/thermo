# Example: Bi2Te3

> Example calculation from Yunwei with steps to reproduce.

1. Please read the BoltzTrap manual (under the file boltztrap-1.2.5/doc). I have already installed it under your HPC account. You can execute "x_trans" to have a check.

2. Make a working directory and name it as “Bi2Te3”. Do scf calculation via VASP with a very large k-point (a normal kpoint grid is 7 7 1, but for this case, I set it as 21 21 3). To perform scf calculation, you need four input files: POSCAR (structure information), POTCAR (psedo-potential), INCAR (vasp input, the parameter I used works for most of compounds, at present you don't need to change it) and KPOINTS (k-point). For other compounds, you need to set your own POSCAR, POTCAR and KPOINTS. Submit the job by using the "slurm_submit.peta4-skylake" script to run the scf calculation (make sure you have the "vasp_std" executable file under your working directory). OUTCAR is the output file of the scf calculation, check it before move to the next step.

3. After the scf calculation, you need to generate a python file, i.e. Bi2Te3.py for this example. Then give the space group number of the crystal structure (166 for this case) in Bi2Te3.py file. Then execute: python Bi2Te3.py
This step is to generate the input files for BoltzTrap calculation.

4. After step 3, you will get three files: hte.intrans, hte.struct and energies.boltztrap. Rename them into Bi2Te3.intrans, Bi2Te3.struct and Bi2Te3.energy, respectively. Notice: these files must have the same name as your working directory.

5. Change the first line “HTE…” into the “directory" name, i.e. “Bi2Te3” for this example, in Bi2Te3.struct and Bi2Te3.energy files. You can also specify the temperature region in Bi2Te3.intrans file. Read the BoltzTrap manual for more details.

6. Execute: x_trans

7. All kinds of parameters for electronic parts can be found in output files: Bi2Te3.trace and Bi2Te3.condtens

8. You can write a script to deal with these output files, for example to extract the data at 300 K by using the following command. `awk '((NR-1)*16+NR-1)%16==6' Bi2Te3.trace > 300k.dat`
