
for dataset in tp_fr tp_de tp_us tp_dk tp_uk
do
    mkdir -p baseline_${dataset} &&
    python main.py baseline_${dataset} ${dataset} -i 16 -L 1 -l 64 -w 32 -W 64 --baseline > baseline_${dataset}/baseline_log.txt & 
done
