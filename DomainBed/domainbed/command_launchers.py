# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import os
import subprocess
from multiprocessing import Pool
import time
import torch

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    n_gpus = torch.cuda.device_count()
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()


def slurm_launcher(commands):
    """
    Parallel job launcher for computationnal cluster using SLURM workload manager.
    An example of SBATCH options:
        #!/bin/bash
        #SBATCH --job-name=<job_name>
        #SBATCH --output=<job_name>.out
        #SBATCH --error=<job_name>_error.out
        #SBATCH --ntasks=4
        #SBATCH --cpus-per-task=8
        #SBATCH --gres=gpu:4
        #SBATCH --time=1-00:00:00
        #SBATCH --mem=81Gb
    Note: --cpus-per-task should match the N_WORKERS defined in datasets.py (default 8)
    Note: there should be equal number of --ntasks and --gres
    """

    if len(commands) == 0:
        return

    # large_mem = True
    large_mem = False

    use_qos = 'normal'
    # use_qos = 'legacy'
    # use_qos = 'deadline'

    if use_qos == 'legacy':
        assert not large_mem
        max_proc = 39
        partition = 'p100'
        qos = '--account=legacy --qos=legacy'
    elif use_qos == 'deadline':
        max_proc = 16
        if not large_mem:
            partition = 'p100,t4v2'
        else:
            partition = 't4v2'
        qos = '--account=deadline --qos=deadline'
    else:
        assert use_qos == 'normal'
        max_proc = 200
        if not large_mem:
            partition = 'p100,t4v2,rtx6000'
        else:
            partition = 't4v2,rtx6000'
        qos = '--qos=normal'

    group_num = 10
    with Pool(processes=max_proc) as pool:

        processes = []
        if group_num == 1:
            for command in commands:
                out_dir = command.split("output_dir")[1].split(" ")[1]
                out_path = os.path.join(out_dir, "out.txt")
                err_path = os.path.join(out_dir, "err.txt")
                script_path = os.path.join(out_dir, "run.sh")

                with open(script_path, 'w+') as f:
                    f.write("#!/bin/sh\n")
                    f.write(command)
                os.chmod(script_path, 0o764)

                process = pool.apply_async(
                    subprocess.run,
                    [f'sbatch -o {out_path} -e {err_path} --gres=gpu:1 --mem=48G -c 8 -p {partition} {qos} {script_path}'],
                    {"shell": True}
                    )
                processes.append(process)
                time.sleep(0.1)
        else:  # group several jobs together
            def split(arr, size):
                arrs = []
                while len(arr) > size:
                    pice = arr[:size]
                    arrs.append(pice)
                    arr = arr[size:]
                arrs.append(arr)
                return arrs

            commands_grouped = split(commands, group_num)
            print("Grouping {} jobs to {} groups, {} jobs each.".format(len(commands), len(commands_grouped), group_num))
            for cmd_grp in commands_grouped:
                out_dir_first = cmd_grp[0].split("output_dir")[1].split(" ")[1]  # use the output dir of the first job
                out_path = os.path.join(out_dir_first, "out_group.txt")
                err_path = os.path.join(out_dir_first, "err_group.txt")
                script_path = os.path.join(out_dir_first, "run_group.sh")
                with open(script_path, 'w+') as f:
                    f.write("#!/bin/sh\n")
                    for cmd in cmd_grp:
                        f.write(cmd + "\n")
                os.chmod(script_path, 0o764)

                process = pool.apply_async(
                    subprocess.run,
                    [f'sbatch -o {out_path} -e {err_path} --gres=gpu:1 --mem=48G -c 8 -p {partition} {qos} {script_path}'],
                    {"shell": True}
                )

                processes.append(process)
                time.sleep(0.1)



        for i, process in enumerate(processes):
            process.wait()
            print("//////////////////////////////")
            print("//// Completed ", i , " / ", len(processes), "////")
            print("//////////////////////////////")


REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher,
    'slurm_launcher': slurm_launcher
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
