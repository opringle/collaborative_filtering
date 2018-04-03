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
    
1. Add validation data iterator to ensure not overfitting
2. Understand what X should be in the final fully connected layer (easy)
3. Train model to comparible loss value (easy)
4. Create custom evaluation metrics to analyze model (hard)