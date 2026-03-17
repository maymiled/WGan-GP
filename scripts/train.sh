#!/bin/bash
name="wgan_gp_train"
outdir="outputs"
n_gpu=1
export DATA="./data/"  # Update this path for your cluster

echo "Launching $name"

mkdir -p ${outdir}

sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:${n_gpu}
#SBATCH --time=04:00:00
#SBATCH --mem=256G
#SBATCH --account=<your_account>  # Replace with your Slurm account
#SBATCH --job-name=wgan_gp_train
#SBATCH --output=${outdir}/%x_%j.out
#SBATCH --error=${outdir}/%x_%j.err
source venv/bin/activate
python train.py --epochs 500 --lr 1e-4 --batch_size 128 --mode wgan --n_critic 5 --lambda_gp 10 --ema_decay 0.9999 --gpus ${n_gpu}
EOT

