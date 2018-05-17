# Deep Learning Recommender Systems

A repository to experiment building recommender systems using apache MXNet.

## Repo structure

- `/data` directory used for storing model input data
- `/libs` python modules used by model training script
- `./recysys.py` trains the recommender system 
- `./results` directory used for storing model reports and results
- `./docs`: research notes etc

## Running the code

1. Clone this repo and cd to root
2. Download, unzip and store the ratings data:
    1. `$ mkdir data`
    2. `$ wget 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'` 
    3. `$ unzip -a ml-1m.zip`
    4. `$ mv ./ml-1m/ratings.dat ./data/`
    5. `$ rm -rf ml-1m`
    6. `$ rm ml-1m.zip`
6. Train the model: `$ python recsys.py`

## ToDo
  
- [ ] Use normal data iterators with data in memory
- [ ] Use callback functions to reproduce HR@K plots
