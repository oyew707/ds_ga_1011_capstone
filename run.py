"""
-------------------------------------------------------

-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "11/22/24"
-------------------------------------------------------
"""


# Imports
import sys
import os
import time
from src.logger import getlogger
from src.parser import parse_args

# Constants
log = getlogger(__name__, 'info')
SLURM_SCRIPT = """#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=ds_ga_1011-2024fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --time=60:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=job-{job_count}:1
#SBATCH --mail-user=eo2233@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output={run_path}/output-{job_count}:1.out
#SBATCH --requeue

module purge

singularity exec --bind /scratch --nv --overlay /scratch/$USER/singularity_{job_count}/overlay-25GB-500K.ext3:rw /scratch/$USER/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c '
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate nlp_env
export SSL_CERT_DIR=/etc/ssl/certs
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Copy project files
# scp -r greene-dtn:/home/eo2233/dsga1011_ep_{job_count}/ds_ga_1011_capstone/. dsga1011/ds_ga_1011_capstone/
# cd dsga1011/ds_ga_1011_capstone/

# huggingface cli login
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

python main.py {args} -lm {model}'
"""

def create_slurm_script(args: str, model: str, run_path: str, job_count: int) -> str:
    """
    -------------------------------------------------------
    Creates a SLURM script for running the model training job
    -------------------------------------------------------
    Parameters:
         args - Command line arguments for the script (str)
         model - Model name to run (str)
         run_path - Path to the run directory (str)
         job_count - Job count for the script (int)
    Returns:
        script - SLURM script content (str)
    -------------------------------------------------------
    """
    script = SLURM_SCRIPT.format(args=args, model=model, run_path=run_path, job_count=job_count)
    return script


def main():
    """
    -------------------------------------------------------
    Main function to create and submit SLURM jobs
    -------------------------------------------------------
    """
    # Get command line arguments (excluding script name)
    args = ' '.join(sys.argv[1:])
    parsed_args = parse_args()

    # Check llm-model isn't set in the command line arguments
    args_list = args.split()
    for i, arg in enumerate(args_list):
        if arg.startswith('-lm') or arg.startswith('--llm-model'):
            raise ValueError("Model argument should not be specified in the command line arguments")

    # Define models to run
    models = ['gpt', 'gemma', 'llama']

    # Create output directory if it doesn't exist
    run_name = parsed_args.run_name
    base_path = parsed_args.output_path
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        log.info(f"Created output directory: {base_path}")

    # Create and submit job for each model
    for job_count, model in enumerate(models):
        # Create model specific run path
        run_path = os.path.join(base_path, f"{run_name}_{model}")
        if not os.path.exists(run_path):
            os.makedirs(run_path)
            log.info(f"Created run directory: {run_path}")

        # Create SLURM script
        script_content = create_slurm_script(args, model, run_path, job_count)

        # Write script to file
        script_path = os.path.join(run_path, 'submit.sbatch')
        with open(script_path, 'w') as f:
            f.write(script_content)
        log.debug(f"Created SLURM script: {script_path}")

        # Submit job
        os.system(f'sbatch {script_path}')
        log.info(f"Submitted job for model: {model}")

        # Wait 2 minutes before submitting next job
        if job_count < len(models) - 1:
            log.info(f"Waiting 2 minutes before submitting next job...")
            time.sleep(120)


if __name__ == '__main__':
    main()
