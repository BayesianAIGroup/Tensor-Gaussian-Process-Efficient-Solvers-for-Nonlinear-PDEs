# [Tensor Gaussian Process: Efficient Solvers for Nonlinear PDEs] (AISTATS 2026)

This repository contains the official implementation of the paper "[Tensor Gaussian Process: Efficient Solvers for Nonlinear PDEs]" accepted at AISTATS 2026.

Python libraries necessary for running codes can be checked in requirements.txt.

Each folder consists of the implementation to one Partial Differential Equation problem specified by the folder name. 

Each folder contains all necessary scripts for running. You can just download one folder and run the code.

## How to reproduce the results in paper?

We provide the training records containing all necessary parameters to reproduce the results in paper.

For example, if we want to reproduce Allen Cahen a=20 with number of collocation points equal 6400 in Table 2(c)

<p align="center">
  <img src="./Pictures/Table2.png" width="50%" title="Table 2">
</p>

The records are in "./Allen_Cahn2D/result_paper/Table2/"; 

"TGPS_PF80x80_a20.ini" represents the Allen_Cahn Problem with "a=20", using Partial Freeze Method under 6400(80x80) collocation points;
"TGPS_NT80x80_a20.ini" share same information except using Newton Method.

Inside record, there exist 2 parts: The front one showing the specific papameters while the below training and testing result based on those parameters.



