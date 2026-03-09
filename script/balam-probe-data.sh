#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --output=slurm-%j-probe-cls-ce.out
#SBATCH --mail-type=FAIL
## SBATCH -p compute_full_node


# module load CCEnv gentoo/2023 gcccore/.12.3 gcc hwloc/2.9.1 ucx/1.14.1 libfabric/1.18.0 pmix/4.2.4 openmpi 
# module load flexiblas imkl StdEnv/2023 python/3.11.5 cudacompat/.12.6 cudacore/.12.6.2 cuda/12.6 arrow/18.1.0 
# module load ipykernel/2025a scipy-stack/2025a rust/1.85.0 ucc/1.2.0 nccl/2.26.2 ucc-cuda/1.2.0

source /scratch/suyuelyu/deimm/deimm/.venv/bin/activate
module load gcc   # or whatever your cluster uses
export CC=$(which gcc)
export CXX=$(which g++)

export DATA_DIR=/scratch/suyuelyu/deimm/data
export HOME_DATA_DIR=/scratch/suyuelyu/deimm/data
export CKPT_DIR=/scratch/suyuelyu/deimm/ckpt
export CODE_DIR=/scratch/suyuelyu/deimm/deimm
export PRETRAIN_DIR=/scratch/suyuelyu/deimm/ckpt/evodiff2/msa-seq-jamba-3b
export TRANSFORMERS_CACHE=/scratch/suyuelyu/.cache/huggingface/hub
export HF_HOME=/scratch/suyuelyu/.cache/huggingface/hub
export HF_HUB_CACHE=/scratch/suyuelyu/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export TRITON_CACHE_DIR=/scratch/suyuelyu/.cache/triton
export UV_CACHE_DIR=/scratch/suyuelyu/.cache/uv
# export LD_PRELOAD="/usr/lib64/libnvidia-ml.so.1 /usr/lib64/libcuda.so.1"
# export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

cd $CODE_DIR


# python /scratch/suyuelyu/deimm/deimm/scripts/data_prep/save_last_protein_hidden.py --train_chunk_idx 4
# python /scratch/suyuelyu/deimm/deimm/scripts/probe_taxon.py 

# python scripts/build_probe_cache.py --rank phylum --cache_dtype float16 --mmap_shard_rows 2000000
TAX_RANK=phylum
# /scratch/suyuelyu/deimm/results/probe_taxon/class_ce_mmap
OUTPUT_DIR=/scratch/suyuelyu/deimm/results/probe_taxon/${TAX_RANK}_ce_mmap_lyr16
python /scratch/suyuelyu/deimm/pp_dayhoff/script/analysis/probe_taxon_linear_ce_stream.py \
  --rank $TAX_RANK \
  --data_mode mmap \
  --shuffle_mode chunk \
  --shuffle_device cuda \
  --host_dtype float16 \
  --prefetch_batches 1 --disk_prefetch_batches 2 \
  --eval_every 1 \
  --batch_size_positions 500000 \
  --amp \
  --weight_decay 0 \
  --lr_scheduler reduce_on_plateau \
  --lr_scheduler_patience 200 --lr_scheduler_factor 0.5 \
  --lr 1e-3 --epochs 10 --save_every_epoch --cache_num_workers 16 \
  --output_dir $OUTPUT_DIR 

# python /scratch/suyuelyu/deimm/deimm/scripts/get_ppl_probe_steer.py \
  