import argparse
import os
import numpy as np
import faiss
import time
import torch
from tqdm import tqdm
from logging import getLogger
import logging

logger = getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# The following code is used to parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='dataset')
    parser.add_argument('--dstore_filename', type=str, help='memmap where keys and vals are stored')
    parser.add_argument('--dstore_size', type=int, help='number of items saved in the datastore memmap')
    parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
    parser.add_argument('--dstore_fp16', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for sampling the subset of vectors to train the cache')
    parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
    parser.add_argument('--code_size', type=int, default=64, help='size of quantized vectors')
    parser.add_argument('--probe', type=int, default=32, help='number of clusters to query')
    parser.add_argument('--faiss_index', type=str, help='file to write the faiss index')
    parser.add_argument('--num_keys_to_add_at_a_time', default=500000, type=int,
                        help='can only load a certain amount of data to memory at a time.')
    parser.add_argument('--starting_point', type=int, default=0, help='index to start adding keys at')

    parser.add_argument('--load_to_mem', default=False, action='store_true')
    parser.add_argument('--no_load_keys_to_mem', default=False, action='store_true')


    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--k', type=int, default=8, help='batch size')
    parser.add_argument('--retrieve_k', type=int, default=128, help='batch size')
    parser.add_argument('--use_gpu_to_search', default=False, action='store_true')

    parser.add_argument('--pca', type=int, default=0, help='pca dimension')

    parser.add_argument('--store_dstore_filename', type=str, help='memmap where keys and vals are stored')

    parser.add_argument('--vocab_size', type=int, default=42024)

    parser.add_argument('--token_constrained', default=False, action='store_true' )
    parser.add_argument('--interpolation', default=False, action='store_true', help='interpolation strategy')

    parser.add_argument('--use_cache', default=False, action='store_true')
    parser.add_argument('--load_cache_to_mem', default=False, action='store_true')
    parser.add_argument('--cache_path', type=str, help='memmap where keys and vals are stored')

    args = parser.parse_args()
    return args


# The following code is used to load the datastore and train the faiss index
def load_datastore(args):

    if args.load_to_mem:
        logger.info("Loading datastore to memory")

        vals_from_memmap = np.memmap(args.dstore_filename + '/vals.npy', dtype=int, mode='r', shape=(args.dstore_size, 1))

        vals = np.zeros(vals_from_memmap.shape, dtype = vals_from_memmap.dtype)
        vals[:] = vals_from_memmap[:]
        del vals_from_memmap

        if args.no_load_keys_to_mem:
            keys = np.memmap(args.dstore_filename + '/keys.npy', dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
                                shape=(args.dstore_size, args.dimension))
        else:
        
            keys_from_memmap = np.memmap(args.dstore_filename + '/keys.npy', dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
                                shape=(args.dstore_size, args.dimension))
            keys = np.zeros(keys_from_memmap.shape, dtype=keys_from_memmap.dtype)
            keys[:] = keys_from_memmap[:]
            del keys_from_memmap

    else:
        keys = np.memmap(args.dstore_filename + '/keys.npy', dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
                                shape=(args.dstore_size, args.dimension))
        vals = np.memmap(args.dstore_filename + '/vals.npy', dtype=int, mode='r', shape=(args.dstore_size, 1))
    return keys, vals


# The following code is used to train the faiss index
def train_faiss_index(args, keys, vals, dstore_size, use_gpu):

    index_dim = args.pca if args.pca > 0 else args.dimension

    quantizer = faiss.IndexFlatL2(index_dim)
    index = faiss.IndexIVFPQ(quantizer, index_dim,
                            args.ncentroids, args.code_size, 8)
    index.nprobe = args.probe

    if args.pca > 0:
        pca_matrix = faiss.PCAMatrix(args.dimension, args.pca, 0, True)
        index = faiss.IndexPreTransform(pca_matrix, index)

    # TODO, we may remove useFloat16 when the GPU satisfy the condition
    if use_gpu:
        logger.info("Start to put index to gpu when `use_gpu` is set to True")
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)

    logger.info("Start to train the faiss index")
    random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(1000000, dstore_size)], replace=False)

    # Faiss does not handle adding keys in fp16 as of writing this.
    index.train(keys[random_sample].astype(np.float32))
    index = faiss.index_gpu_to_cpu(index) if use_gpu else index

    return index
        

# The following code is used to add keys to the faiss index
def add_keys_to_knn_index(keys, indice, use_gpu, trained_index = None):
    logger.info("Start to add keys to the knn index")

    if trained_index is not None:
        index = trained_index
    elif os.path.exists(args.faiss_index + ".trained"):
        index = faiss.read_index(args.faiss_index + ".trained")
    else:
        raise ValueError

    if use_gpu:
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
    
    if indice is None:
        indice = np.arange(keys.shape[0])

    size = indice.shape[0]

    start = args.starting_point

    while start < size:
        end = min(size, start + args.num_keys_to_add_at_a_time)
        to_add = keys[indice[start:end]].copy()
        index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
        start += args.num_keys_to_add_at_a_time

    return index


def save_selected_keys_values_index(keys, vals, weights, indices, output_path, dstore_size, use_gpu):
    logger.info(f'Start to save the selected keys and values to {output_path}')

    os.makedirs(output_path, exist_ok=True)

    if not os.path.exists(output_path + '/vals.npy'):
        store_vals = np.memmap(output_path + '/vals.npy', dtype=int, mode='w+', shape=(dstore_size, 1))
        store_vals[:] = vals[indices][:]

    if not os.path.exists(output_path + '/keys.npy'):
        store_keys = np.memmap(output_path + '/keys.npy', dtype=np.float16 if args.dstore_fp16 else np.float32, mode='w+', shape=(dstore_size, args.dimension))
        store_keys[:] = keys[indices][:]

    if not os.path.exists(output_path + '/weights.npy'):
        store_weights = np.memmap(output_path + '/weights.npy', dtype=np.int32, mode='w+', shape=(dstore_size, 1))
        store_weights[:] = weights[indices][:]
    
    if not os.path.exists(output_path + f'/knn_index'):
        trained_index = train_faiss_index(args, keys[indices], vals[indices], dstore_size, use_gpu)
        index = add_keys_to_knn_index(keys, indices, use_gpu, trained_index)
        faiss.write_index(faiss.index_gpu_to_cpu(index) if use_gpu else index, output_path + f'/knn_index')
        del index

def main(args):

    logger.info("Start to prune the datastore.")
    dstore_size = args.dstore_size
    batch_size = args.batch_size

    if args.seed is not None:
        np.random.seed(args.seed)

    keys, vals = load_datastore(args)
    use_gpu = args.use_gpu_to_search and torch.cuda.is_available()

    if args.use_cache:

        cache_path = args.cache_path
        
        if not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)

        retrieve_required = False

        # Load the retrieve knns if it exists
        cache_knns_path = cache_path + f'/init_retrieve_knns_k_{args.retrieve_k}.npy'
        if os.path.exists(cache_knns_path):
            logger.info(f'Found retrieve knns cache file, trying to load from {cache_knns_path}')

            # Load the data to memory if the setting is True
            if args.load_cache_to_mem:
                logger.info('Loading retrieve knns to memory with `load_cache_to_mem` setting to True')

                try:
                    # Try to load the data directly
                    retrieve_knns = np.load(cache_knns_path, allow_pickle=True)
                except:
                    # If there exists format error, we use memmap to load the data
                    retrieve_knns_mmap = np.memmap(
                        cache_knns_path,
                        dtype=np.int32,
                        mode='r',
                        shape=(args.dstore_size, args.retrieve_k + 1)
                    )
                    # Create a new array to store the data
                    retrieve_knns = np.zeros(shape=(args.dstore_size, 1 + args.retrieve_k), dtype=np.int32)
                    retrieve_knns[:] = retrieve_knns_mmap[:]
                    del retrieve_knns_mmap
            else:
                logger.info('Loading retrieve knns without loading to memory with `load_cache_to_mem` setting to False')
                retrieve_knns = np.memmap(cache_knns_path, dtype=np.int32, mode='r', shape=(args.dstore_size, args.retrieve_k + 1))
        else:
            logger.info('Retrieve knns cache file not found, need to retrieve the knns. Setting `retrieve_required` to True')
            retrieve_required = True

        # Load the retrieve dists if it exists
        cache_dists_path = cache_path + f'/init_retrieve_dists_k_{args.retrieve_k}.npy'
        if os.path.exists(cache_dists_path):
            logger.info(f'Found retrieve dists cache file, trying to load from {cache_dists_path}')

            if args.load_cache_to_mem:
                logger.info('Loading retrieve dists to memory with `load_cache_to_mem` setting to True')

                try:
                    # Try to load the data directly
                    retrieve_dists = np.load(cache_dists_path, allow_pickle=True)
                except:
                    # If there exists format error, we use memmap to load the data
                    retrieve_dists_mmap = np.memmap(cache_dists_path, dtype=np.float16, mode='r', shape=(args.dstore_size, args.retrieve_k + 1))
                    retrieve_dists = np.zeros(shape=(args.dstore_size, 1 + args.retrieve_k), dtype=np.float16)
                    retrieve_dists[:] = retrieve_dists_mmap[:]
                    del retrieve_dists_mmap
            else:
                logger.info('Loading retrieve dists without loading to memory with `load_cache_to_mem` setting to False')
                retrieve_dists = np.memmap(cache_dists_path, dtype=np.float16, mode='r', shape=(args.dstore_size, args.retrieve_k + 1))

            logger.info('Loaded retrieve dists')
        else:
            logger.info('Retrieve dists cache file not found, need to retrieve the knns. Setting `retrieve_required` to True')
            retrieve_required = True
    else:
        logger.info("`use_cache` is set to False, need to retrieve the knns. Setting `retrieve_required` to True")
        retrieve_required = True

    # Retrieve the knns if needed
    if retrieve_required:
        logger.info('Start to retrieve the knns')

        index = faiss.read_index(args.dstore_filename + '/knn_index', faiss.IO_FLAG_ONDISK_SAME_DIR)
        index.nprobe = args.probe

        if use_gpu:
            logger.info("Start to put index to gpu when `use_gpu` is set to True")
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(res, 0, index, co)

        # Create the empty arrays to store the knns and dists
        if args.use_cache and not args.load_cache_to_mem:
            retrieve_knns = np.memmap(cache_knns_path, dtype=np.int32, mode='w+', shape=(dstore_size, args.retrieve_k + 1))
            retrieve_dists = np.memmap(cache_dists_path, dtype=np.float16, mode='w+', shape=(dstore_size, args.retrieve_k + 1))
        
        else:
            retrieve_knns = np.zeros((args.dstore_size, args.retrieve_k + 1), dtype=np.int32)
            retrieve_dists = np.zeros((args.dstore_size, args.retrieve_k + 1), dtype=np.float16)

        
        # Retrieve the knns in batch
        start = args.starting_point
        for i in tqdm(range(start, args.dstore_size, args.batch_size)):

            end = min(args.dstore_size, start + batch_size)
            batch = np.array(keys[start:end], dtype=np.float32)

            dists, knns = index.search(batch, args.retrieve_k + 1)
            retrieve_knns[start: end] = knns
            retrieve_dists[start: end] = dists

            start += batch_size
        
        # Save the knns and dists if needed
        if args.use_cache:
            np.save(cache_knns_path, retrieve_knns)
            np.save(cache_dists_path, retrieve_dists)

    # We define the status of the datastore as follows:
    # 0: deleted, the point is deleted from the datastore
    # 1: available, the point is available for deletion
    # 2: retained, the point is retained in the datastore and cannot be deleted anymore

    class Status():
        DELETED = 0
        AVAILABLE = 1
        RETAINED = 2

    # Now we have the knns and dists, we can start to prune the datastore
    # First we initialize the status, weights and in_degree

    status = np.ones((dstore_size, 1), dtype=np.int32) * Status.AVAILABLE
    weights = np.ones((dstore_size, 1), dtype=np.int32)

    cache_in_degree_path = cache_path + f'/init_in_degree_k_{args.retrieve_k}.npy'

    if args.use_cache and os.path.exists(cache_in_degree_path):
        logger.info(f'Found in_degree cache file, trying to load from {cache_in_degree_path}')
        in_degree = np.load(cache_in_degree_path)
        in_degree = in_degree.astype(np.int32)

    else:
        logger.info('In_degree cache file not found, need to compute the in_degree')
        in_degree = np.zeros((dstore_size, 1), dtype=np.int32)

        for i in tqdm(range(dstore_size)):
            # We start from 1 because the first one is itself (at most circumstances)
            in_degree[retrieve_knns[i, 1:].flatten()] += 1

        if args.use_cache:
            np.save(cache_in_degree_path, in_degree)

        
    s = time.time()
    size = (status != Status.DELETED).sum()

    # We save the datastore every 10% of the original size
    part_size = dstore_size // 10

    for iter in range(1, args.k + 1):

        # We need to define the order of the points to be pruned
        if args.ordering == "dist_order":
            # We sort the points based on the distance to the k-th neighbor
            iter_dists = np.array(retrieve_dists[:, iter].flatten(), dtype=np.int32)
            order = np.argsort(iter_dists).astype(np.int32)  

        elif args.ordering == "rd_order":
            # We randomly shuffle the points
            order = np.arange(args.dstore_size).astype(np.int32)
            np.random.shuffle(order)
    
        elif args.ordering == "gm_order":
            order = np.arange(args.dstore_size).astype(np.int32)
            start = 0
            while start < args.dstore_size:
                end = min(args.dstore_size, start + args.batch_size)
                np.random.shuffle(order[start: end])
                start += args.batch_size
        elif args.ordering == "seq_order":
            order = np.arange(args.dstore_size).astype(np.int32)
        else:
            raise ValueError(f"Unknown ordering: {args.ordering}. It must be chosen frm [`dist_order`, `rd_order`, `gm_order`, `seq_order`]")

        logger.info(f'Ordering: {args.ordering}, Iteration: {iter}, Time: {time.time() - s}')
        for i, id in enumerate(order):
            # The current point 
            retrieve_point = id
            retrieve_topk = iter

            if not vals[retrieve_point]:
                if status[retrieve_point] != Status.DELETED:
                    size = size - 1
                status[retrieve_point] = Status.DELETED
                weights[retrieve_point] = 0

            if status[retrieve_point] != Status.AVAILABLE:
                continue

            neighbor_point = retrieve_knns[retrieve_point, retrieve_topk]
            
            if status[neighbor_point] == Status.DELETED or status[neighbor_point] == Status.RETAINED:
                continue
            else:
                
                if vals[retrieve_point] != vals[neighbor_point]:

                    if args.token_constrained:
                        status[retrieve_point] = Status.RETAINED
                    
                else:
                    # We merge the point with lower in_degree into the point with higher in_degree
                    deleted_point = retrieve_point if in_degree[retrieve_point] <= in_degree[neighbor_point] else neighbor_point
                    merge_point = neighbor_point if in_degree[retrieve_point] <= in_degree[neighbor_point] else retrieve_point

                    status[deleted_point] = Status.DELETED  

                    weights[merge_point] += weights[deleted_point]
                    weights[deleted_point] = 0 

                    size = size - 1

                    if args.interpolation:
                        # The interpolation strategy described in the paper
                        if in_degree[retrieve_point] == 0 and in_degree[neighbor_point] == 0:
                            keys[merge_point] = 0.5 * keys[retrieve_point] + 0.5 * keys[neighbor_point]
                        else:
                            lambda_a = 1.0 * in_degree[retrieve_point] / (in_degree[retrieve_point] + in_degree[neighbor_point])
                            keys[merge_point] = lambda_a * keys[retrieve_point] + (1 - lambda_a) * keys[neighbor_point]

                        in_degree[merge_point] += in_degree[deleted_point]
            
            
            if size % part_size == 0:
                # Save the datastore every part_size
                indice = np.where(status != Status.DELETED)[0]

                save_dstore_size = indice.shape[0]
                output_path = args.store_dstore_filename + f'/dsize-{save_dstore_size}/'
                save_selected_keys_values_index(keys, vals, weights, indice, output_path, save_dstore_size, use_gpu)
        
        indice = np.where(status != Status.DELETED)[0]
        save_dstore_size = indice.shape[0]


    # Save the datastore at the end
    indice = np.where(status != Status.DELETED)[0]
    save_dstore_size = indice.shape[0]
                    
    output_path = args.store_dstore_filename + f'/dsize-{save_dstore_size}/'
    save_selected_keys_values_index(keys, vals, weights, indice, output_path, save_dstore_size, use_gpu)

if __name__ == "__main__":
    args = parse_args()
    main(args)