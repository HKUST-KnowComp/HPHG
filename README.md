# HPHG

The code and data for CIKM '19 paper "[Hyper-Path-Based Representation Learning for Hyper-Networks](<https://arxiv.org/abs/1908.09152>)".

Readers are welcomed to star/fork this repository to reproduce the experiments and train your own model. If you find this code useful, please kindly cite our paper:

```
@inproceedings{huang2019hyper,
  title={Hyper-Path-Based Representation Learning for Hyper-Networks},
  author={Huang, Jie and Liu, Xin and Song, Yangqiu},
  booktitle={CIKM},
  year={2019}
}
```



### Examples

**Train embeddings (HPHG)**

```
python src/main.py --input graph/GPS/train.edgelist --output emb/GPS/HPHG.emb --save-model model/GPS/model.h5 --alpha 100 --iter 5
```

```
python src/main.py --input graph/Drugs/train.edgelist --output emb/Drugs/HPHG.emb --save-model model/Drugs/model.h5 --alpha 100 --iter 1
```

**Train embeddings (HPSG)**

```
python src/main.py --input graph/GPS/train.edgelist --output emb/GPS/HPSG.emb --alpha 100 --iter 15 --method hpsg
```

```
python src/main.py --input graph/Drugs/train.edgelist --output emb/Drugs/HPSG.emb --alpha 100 --iter 5 --method hpsg
```



**Link prediction (HPHG)**

```
python src/main.py --input graph/GPS/train.edgelist --load-model model/GPS/model.h5
```

**Link prediction (HPSG)**

```
python src/link_prediction.py --input emb/GPS/HPSG.emb
```



### Options

You can check out the other options available to use with *HPHG* or *HPSG* using:

```
python src/main.py --help
```



### Input

The supported input format is an edgelist:

	1            # 1 for heterogeneous and 0 for homogeneous
	146 70 5     # number of nodes of each node type (heterogeneous)
	0 93 203 220 # edge_name node_id node_id node_id
	1 17 153 216
	...



### Output

The output file has *n+1* lines for a graph with *n* nodes. 
The first line has the following format:

	num_of_nodes dim_of_representation

The next *n* lines are as follows:	
	node_name dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation learned by *HPHG* or *HPSG*.



### Misc

- Please store the edgelist file in `graph/<dataset>/*` and store the embedding file in `emb/<dataset>/*`.
- This implementation is only applied to 3-uniform heterogeneous hyper-networks, if you have any question or need a compatible version, you are welcome to open an issue or send me an email.





