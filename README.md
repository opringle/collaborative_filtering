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
  
- [x] Ensure data iterator is feeding correct data to model
- [x] Compute HR@K per epoch with callback function
- [x] Download original data to ensure it is the same
- [x] Do not specify batch size in network symbol
- [x] Ensure network symbol is correct
- [x] Print log loss during training with single output neuron
- [ ] Achieve similar training loss after similar epochs as paper
- [ ] Regularize model in the same way they did in the paper
- [ ] Get model training robustly with a single negative training sample (could be to do with output layer size) 
- [ ] Drastically increase model size and train on instance with dropout
