# Datastore Distillation for Nearest Neighbor Machine Translation

Official Code for our paper "Datastore Distillation for Nearest Neighbor Machine Translation".

### Preparation
First, you must prepare the environment for kNN-MT using the fairseq framework. To make it concrete, you have to finish the following steps.
* You have to install the fairseq.
* You have to create the overall keys and values.

Please refer to the original [kNN-MT](https://github.com/urvashik/knnmt).

### Prune the Datastore
You can prune the datastore by simply running our provided shell script.

### Evaluation
Once you have prepared the pruned datastore, you can evaluate the results under [adaptive-knn-mt](https://github.com/zhengxxn/adaptive-knn-mt)
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
