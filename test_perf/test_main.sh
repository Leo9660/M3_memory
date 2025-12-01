topk=5
nprobe=8

for ds in agentgym gsm8k prm800k ultrachat ultrafeedback xlam_function_calling; do
  python ratio_throughput.py --backend quake --dataset $ds --mode item_search_insert \
    --faiss-index /data/msarco_dataset/IVF4096Flat/msmarco_ivf_flat.index \
    --insert-batch 256 --search-batch 256 --ops-per-run 200 --top-k "$topk" --nprobe "$nprobe" --limit 10000 \
    --log-file test_perf/perf_main.csv
done
