# HPHG

This is an implementation of "Hyper-Path-Based Representation Learning for Hyper-Networks" (CIKM 2019).



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



**Attention**: 

- Please store the edgelist file in `graph/<dataset>/*` and store the embedding file in `emb/<dataset>/*`.
- This implementation only applied to 3-uniform heterogeneous hyper-networks, if you need a compatible version or have any question, feel free to contact us.



