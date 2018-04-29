# Deep Learning Recommender Systems

A repository to experiment building recommender systems using apache MXNet.

## Repo structure

- `/data` directory used for storing model input data
- `/libs` python modules used by model training script
- `./recysys.py` trains the recommender system 
- `./results` directory used for storing model reports and results
- `./docs`: research notes etc

## Running the code

- `$ pip install -r requirements.txt`
- `$ wget data` 
- `$ python recsys.py`

## ToDo
  
- [x] Ensure data iterator is feeding correct data to model
- [x] Compute HR@K per epoch with callback function
- [ ] Get model training robustly with a single negative training sample (could be to do with output layer size)
- [ ] Print log loss during training 
- [ ] Achieve similar training loss after similar epochs as paper
- [ ] Drastically increase model size and train on instance with dropout
