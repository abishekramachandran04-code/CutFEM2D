This is a 2D incompressible fluid flow solver usinf CutFEM - with SUPG and PSPG stabilization. 
SUPG has been turned off in navierstokes.m and navierstokes_CutFEM.m for the purpose of running the DFG 2D-2 benchmark as it is a diffusion dominated flow. 
To turn it on, dynamic SUPG parameter lines have been written in these files which can be uncommented. 
mainCutFEM.m and mainNS.m both currently run the DFG 2D-2 benchmark; while NS_CutFEM_2.m compares the difference in solution results of the 2 files. 
mainCutFEM3.m has the cylinder under the same initial and boundary conditions as the DFG 2D-2 benchmark, 
but has the cylinder oscillating at 3.0Hz corresponding to the ~0.3 St number of the benchmark to view 2S vortex shedding. 
Its output files are under vtk_output which can be downloaded and viewed on Paraview.
