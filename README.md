# Optimization of Permutation Flowshop Scheduling Using an Island Genetic Algorithm for Makespan Minimization

The Permutation Flowshop Scheduling Problem is a well-known NP-hard combinatorial optimization problem that involves the sequencing of n jobs across m machines in the same order to minimize the total makespan value. This project proposes a Heterogeneous Island Genetic Algorithm framework (HIGA). Each island represents a group of solutions that evolve in parallel using different initialization heuristics, crossover and mutation operators, and adaptive parameters. A dynamic, stagnation-based migration strategy is proposed to maintain targeted communication between the islands.

The proposed HIGA approach was compared against the basic Standard Genetic Algorithm (SGA) and a more advanced Niche-based Genetic Algorithm (NEH-NGA) on Taillard’s benchmark dataset. Experimental results indicate HIGA effectively balances solution quality and efficiency, matching the best-known makespan value or coming close to it, particularly for larger instances, while being several fold faster than NEH-NGA and achieving significantly better results than the SGA.

**`testcase.txt`**:
    An example input file containing processing times for jobs on machines. The format is:
        ```
        
        machine1_job1 machine1_job2 ... machine1_jobN
        machine2_job1 machine2_job2 ... machine2_jobN
        ...
        machineM_job1 machineM_job2 ... machineM_jobN
        ```
