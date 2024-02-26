"starting calculation at $(date)"
export OMP_SCHEDULE=dynamic

export KMP_BLOCKTIME=0

export OMP_NUM_THREADS=1

source /data/home/prabu/miniconda3/bin/activate
conda activate dr3
python search_all.py >stdout 2>stderr

echo "finishing at $(date)"
