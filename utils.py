from typing import List, Tuple

def read_processing_times(filename):
    """Read input file with machines as rows and jobs as columns."""
    with open(filename, 'r') as f:
        machines = []
        for line in f:
            machines.append([int(x) for x in line.strip().split()])
    # transpose so that each row corresponds to a job
    return list(map(list, zip(*machines)))

def calculate_makespan(sequence: List[int], processing_times: List[List[int]]) -> int:
    """
    Calculate the makespan for a given job sequence (Corrected 2-array DP).
    
    Args:
        sequence: List of job indices
        processing_times: Processing times for each job on each machine
        
    Returns:
        Makespan (completion time of last job on last machine)
    """
    num_jobs = len(sequence)
    if not processing_times:
        raise ValueError("Processing times matrix is empty")
    num_machines = len(processing_times[0])
    if num_machines == 0:
         return 0

    prev = [0] * num_machines 

    for i in range(num_jobs):
        job_index = sequence[i]
        curr = [0] * num_machines
        
        for j in range(num_machines):
            # curr[j-1] is completion time of current job on previous machine
            # prev[j]   is completion time of previous job on current machine
            time_of_current_job_on_previous_machine = curr[j-1] if j > 0 else 0
            time_of_previous_job_on_current_machine = prev[j]
            curr[j] = max(time_of_current_job_on_previous_machine, time_of_previous_job_on_current_machine) + processing_times[job_index][j]
            
        # Update prev for the next job's calculation
        prev = curr

    return curr[num_machines - 1] 

def calculate_makespan_2d(sequence: List[int], processing_times: List[List[int]]) -> int:
    """
    Calculate the makespan using a 2D DP table (precursor to space optimization).
    
    Args:
        sequence: List of job indices (0-based).
        processing_times: Processing times[job][machine] (0-based).
        
    Returns:
        Makespan (completion time of last job on last machine).
    """
    num_jobs = len(sequence)

    if not processing_times:
        raise ValueError("Processing times matrix is empty")
    
    num_machines = len(processing_times[0])
    if num_machines == 0:
        return 0
    
    completion_times = [[ 0 for j in range(num_machines)] for i in range(num_jobs)]

    for i in range(num_jobs):
        for j in range(num_machines):
            time_of_current_job_on_previous_machine = completion_times[i][j-1] if j > 0 else 0
            time_of_previous_job_on_current_machine = completion_times[i-1][j] if i > 0 else 0
            completion_times[i][j] = processing_times[sequence[i]][j] + \
                            max(time_of_current_job_on_previous_machine, time_of_previous_job_on_current_machine)

    return completion_times[num_jobs-1][num_machines-1]

def neh_heuristic(processing_times: List[List[int]]) -> List[int]:
    """
    Nawaz, Enscore, and Ham heuristic for the flowshop scheduling problem.

    Args:
        processing_times: Processing times of jobs on machines

    Returns:
        A permutation of job indices
    """
    num_jobs = len(processing_times)

    # calculate total processing time for each job
    job_times = [(i, sum(processing_times[i])) for i in range(num_jobs)]
    # sort jobs by total processing time (descending)
    job_times.sort(key=lambda x: x[1], reverse=True)

    # start with the job with highest total processing time
    result = [job_times[0][0]]

    # add remaining jobs one by one at the best position
    for i in range(1, num_jobs):
        job = job_times[i][0]
        best_position = 0
        best_makespan = float('inf')

        # try inserting the job at each possible position
        for pos in range(len(result) + 1):
            # create a new permutation with the job inserted at this position
            new_perm = result.copy()
            new_perm.insert(pos, job)

            # calculate makespan
            makespan = calculate_makespan(new_perm, processing_times)

            # update best position if this gives a better makespan
            if makespan < best_makespan:
                best_makespan = makespan
                best_position = pos

        # insert the job at the best position
        result.insert(best_position, job)

    return result