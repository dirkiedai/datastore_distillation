# Datastore Distillation for Nearest Neighbor Machine Translation

This repository contains the official implementation of our paper, “Datastore Distillation for Nearest Neighbor Machine Translation.”

### Preparation
To begin, set up the kNN-MT environment using the Fairseq framework. Complete the following steps:
* Install Fairseq.
* Construct the overall keys and values for the datastore.

For more detailed instructions, refer to the original [kNN-MT repository](https://github.com/urvashik/knnmt).

### Prune the Datastore
Prune the datastore by executing the provided shell script.
Here’s an explanation of the parameters for the script:

	•	--dataset: Specifies the dataset to use.
	•	--dstore_filename: Path to the memory-mapped file storing keys and values.
	•	--dstore_size: Number of items in the datastore.
	•	--dimension: Size of each key, default is 1024.
	•	--dstore_fp16: Use 16-bit floating point precision.
	•	--seed: Random seed for sampling vectors, default is 1.
	•	--ncentroids: Number of centroids for FAISS to learn, default is 4096.
	•	--code_size: Size of quantized vectors, default is 64.
	•	--probe: Number of clusters to query, default is 32.
	•	--faiss_index: Path to store the FAISS index.
	•	--num_keys_to_add_at_a_time: Limits the number of keys loaded into memory at a time, default is 500,000.
	•	--starting_point: Index to start adding keys, default is 0.
	•	--load_to_mem: Load the datastore into memory.
	•	--no_load_keys_to_mem: Avoid loading keys into memory.
	•	--batch_size: Batch size for processing, default is 128.
	•	--k: Number of nearest neighbors to retrieve, default is 8.
	•	--retrieve_k: Number of clusters to query, default is 128.
	•	--use_gpu_to_search: Utilize GPU for searching.
	•	--pca: Perform PCA to reduce dimensionality, default is 0 (no PCA).
	•	--store_dstore_filename: Path to store the new datastore.
	•	--vocab_size: Size of the vocabulary, default is 42,024.
	•	--token_constrained: Constrain to specific tokens.
	•	--interpolation: Enable interpolation strategy.
	•	--use_cache: Use cache during retrieval.
	•	--load_cache_to_mem: Load cache into memory.
	•	--cache_path: Path to the memory-mapped file storing cache keys and values.

These parameters provide flexibility in configuring the datastore, retrieval process, and other key aspects of nearest neighbor machine translation.

### Evaluation
Evaluate the pruned datastore using the methods outlined in the [adaptive-knn-mt repository](https://github.com/zhengxxn/adaptive-knn-mt)
## Citation
If you find this repo helpful for your research, please cite the following paper:
```
@ARTICLE{10334021,
  author={Dai, Yuhan and Zhang, Zhirui and Du, Yichao and Liu, Shengcai and Liu, Lemao and Xu, Tong},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Datastore Distillation for Nearest Neighbor Machine Translation}, 
  year={2024},
  volume={32},
  number={},
  pages={807-817},
  keywords={Merging;Machine translation;Speech processing;Optimization;Iterative methods;Iterative decoding;Task analysis;Nearest neighbor machine translation;datastore distillation},
  doi={10.1109/TASLP.2023.3337633}}
```
