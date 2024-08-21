# DSTORE_SIZE_DICT = {
#     'it': 3613350,
#     'medical': 6903320, 
#     'law': 19070000,
#     'wikitext-103': 103225485,
# }

DATASET="it"
python prune_datastore.py \
    --dataset $DATASET \
    --dstore_filename datastore/$DATASET \
    --faiss_index datastore/$DATASET/knn_index \
    --dstore_size 3613350 \
    --dstore_fp16 \
    --k 8 \
    --retrieve_k 128 \
    --use_gpu_to_search \
    --batch_size 128 \
    --seed 1 \
    --interpolation \
    --ordering "seq_order" \
    --store_dstore_filename datastore/$DATASET/seq_order \
    --load_to_mem  --use_cache --load_cache_to_mem \
    --cache_path cache/$DATASET/retrieve_results