# PySCF Input for MoleditPy PySCF Calculator plugin
# Plugin Version: test
# Job Type: TS Optimization + Frequency
# Method: RKS
# Functional: b3lyp
# Basis: ma-def2-svp
# Charge: -1
# Multiplicity: 1
# Threads: 16
# Memory: 2000 MB
# Max Cycle: 200
# Conv Tol: 1e-10

from pyscf import gto, scf, dft
mol = gto.M(atom='''C -0.000446 0.000909 0.446129
Cl -0.001541 0.000978 -2.169025
H 1.079174 0.001023 0.320011
H -0.540230 0.935999 0.320729
H -0.540320 -0.934072 0.320434
Br 0.000428 0.000584 2.826129''', 
    basis='ma-def2-svp', 
    charge=-1, 
    spin=0, 
    max_memory=2000, 
    verbose=4)
mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.grids.level = 3
mf.max_cycle = 200
mf.conv_tol = 1e-10
mf.kernel()
