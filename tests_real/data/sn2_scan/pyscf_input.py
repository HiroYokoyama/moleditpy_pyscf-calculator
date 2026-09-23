# PySCF Input for MoleditPy PySCF Calculator plugin
# Plugin Version: test
# Job Type: Relaxed Surface Scan
# Method: RKS
# Functional: b3lyp
# Basis: ma-def2-svp
# Charge: -1
# Multiplicity: 1
# Threads: 16
# Memory: 2000 MB
# Max Cycle: 200
# Conv Tol: 1e-10
# Scan Parameters:
#   type: Dist
#   atoms: [0, 5]
#   start: 2.8
#   end: 1.96
#   steps: 5

from pyscf import gto, scf, dft
mol = gto.M(atom='''C 0.000 0.000 0.000
Cl 0.000 0.000 -1.800
H 1.028 0.000 0.357
H -0.514 0.890 0.357
H -0.514 -0.890 0.357
Br 0.000 0.000 2.800''', 
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
