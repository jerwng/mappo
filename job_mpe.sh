#!/bin/bash
### Sets the job's name.
#SBATCH --job-name=mpe

### Sets the job's output file and path.
#SBATCH --output=mpe.out.%j

### Sets the job's error output file and path.
#SBTACH --error=mpe.err.%j

### Requested number of nodes for this job. Can be a single number or a range.
#SBATCH -N 1

### Requested partition (group of nodes, i.e. compute, bigmem, gpu, etc.) for the resource allocation.
#SBATCH -p kimq

### Requested number of gpus
#SBATCH --gres=gpu:1

### Limit on the total run time of the job allocation.
#SBATCH --time=10:00:00

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "Activating Jerry's Python Venv"
source /home/chenyuanwang01/MAT/Multi-Agent-Transformer/venv/bin/activate

echo "Running train_smac.py"
bash /home/chenyuanwang01/mappo/onpolicy/scripts/train_mpe_scripts/train_mpe_spread_mat.sh

echo "Deactivating Jerry's Python Venv"
deactivate

echo "Done."
