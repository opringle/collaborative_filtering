# Deep Learning Recommender Systems

A repository to experiment building state of the art recommender systems using apache MXNet.

## Repo structure

`/data/`: directory used for storing model input data
`/results/`: directory used for storing model reports and results
`/src/`: python code to build models, preprocess data etc
/`docs/`: research notes etc

## Running the code

- download the dataset from kaggle: https://www.kaggle.com/prajitdatta/movielens-100k-dataset/data
- `$ pip install -r requirements.txt` from root
- `$cd ./src && python recsys.py`

## ToDo
    
1. Create custom data iterators with negative sampling ratio as an arguement: https://github.com/apache/incubator-mxnet/blob/master/example/recommenders/negativesample.py.
2. Think through a clean way to organize all input data and infer vocab size for each categorical variable (eg adding one more categorical variable does not require changes to model code)
3. Understand why X is described as determining the model capability
4. Find clean way to display HR@x as an mxnet custom evaluation metric
