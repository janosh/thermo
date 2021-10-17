# CSD3 Guide

## Submissions Scripts

To submit a CPU or GPU job, use `sbatch hpc/(c|g)pu_submit` after editing those files (if necessary). Rather than changing parameters directly in those files, you can pass in variables via the command line as follows:

```sh
sbatch --export var1='foo',var2='bar' hpc/(c|g)pu_submit
```

and then using those variables as e.g.

```sh
echo var1 is '$var1' and var2 is '$var1'
```

To change the job name and run time, use `sbatch -J job_name -t 1:0:0` (time format `h:m:s`). A complete example would be

```sh
sbatch -J job_name -t 1:0:0 --export CMD='python path/to/script.py --cli-arg 42' hpc/gpu_submit
```

## Array Jobs

To submit an array of, say 16 jobs, use

```sh
sbatch -J job_name -t 1:0:0 --array 0-15 --export CMD="python path/to/script.py --cli-arg 42 --random-seed \$SLURM_ARRAY_TASK_ID" hpc/gpu_submit
```

Note the backslash in front of `\$SLURM_ARRAY_TASK_ID` which ensures the variable isn't expanded at job submission time but at execution time where it will have a value.

You may also read the task ID directly in the Python script via

```py
task_id = int(sys.argv[1])
```

This can for instance be used to run a grid of experiments:

```py
task_id = int(sys.argv[1])

drop_rates, learning_rates = [0.1, 0.2, 0.3, 0.5], [1e-4, 3e-4, 1e-3, 3e-3]
drop_rate, learning_rate = tuple(itertools.product(drop_rates, learning_rates))[task_id]
```

## Environment

To setup dependencies, use `conda`

```sh
conda create -n py38 python
pip install -r requirements.txt
```

## Running Short Experiments

Short interactive sessions are a good way to ensure a long job submitted via `(c|g)pu_submit` will run without errors in the actual HPC environment.

[To request a 10-minute interactive CPU session](https://docs.hpc.cam.ac.uk/hpc/user-guide/interactive.html#sintr):

```sh
sintr -A LEE-SL3-CPU -p skylake -N2 -n2 -t 0:10:0 --qos=INTR
module load rhel7/default-peta4
script job_name.log
```

- `sintr`: SLURM interactive
- `-A LEE-SL3-CPU`: charge the session to account `LEE-SL3-CPU`
- `-p skylake`: run on the Skylake partition
- `-N1 -n1`: use single node
- `-t 0:10:0` set session duration to 10 min
- `--qos=INTR`: set quality of service to interactive

To request two nodes for an hour (the maximum interactive session duration), use

```sh
sintr -A LEE-SL3-CPU -p skylake -N2 -n2 -t 1:0:0 --qos=INTR
```

Useful for testing a job will run successfully in the actual environment it's going to run in without having to queue much.

The last line `script job_name.log` is optional but useful as it ensures everything printed to the terminal during the interactive session will be recorded in `job_name.log`. [See `script` docs](https://man7.org/linux/man-pages/man1/script.1.html).

To use service level 2, include your CRSId, i.e. `LEE-JR769-SL2-CPU` instead of `LEE-SL3-CPU`.

Similarly, for a 10-minute interactive GPU session:

```sh
sintr -A LEE-SL3-GPU -p ampere,pascal -N1 -n1 -t 0:10:0 --qos=INTR --gres=gpu:1
module load rhel7/default-gpu
script job_name.log
```

Before doing anything, that requires a GPU, remember to load

```sh
module load rhel7/default-gpu
```

To specify CUDA version:

```sh
module load cuda/11.0
```

Check current version with `nvcc --version`.

To check available hardware:

```sh
nvidia-smi
```

This should print something like

```text
Thu Oct  8 20:15:44 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  On   | 00000000:04:00.0 Off |                    0 |
| N/A   35C    P0    28W / 250W |      0MiB / 16280MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Debugging Tips

If the interactive window won't launch over SSH, see [vscode-python#12560](https://github.com/microsoft/vscode-python/issues/12560).

If VS Code fails to connect to a remote (encountered once with exitCode 24), follow the steps to [clean up VS Code Server on the remote](https://code.visualstudio.com/docs/remote/troubleshooting#_cleaning-up-the-vs-code-server-on-the-remote) followed by reconnecting which reinstalls the remote extension.

## Syncing Results

To sync results back from CSD3 to your local machine, use

```sh
rsync -av --delete login.hpc.cam.ac.uk:repo/results .
```

`-a`: archive mode, `-v`: increase verbosity, `--delete`: remove files from target not found in source.

If CSD3 was setup as an SSH alias in `~/.ssh/config`,

```text
Host csd3
  Hostname login.hpc.cam.ac.uk
```

Then it's simply:

```sh
rsync -av --delete csd3:repo/results .
```

Add `-n` to test the command in a dry-run first. Will list each action that would have been performed.
