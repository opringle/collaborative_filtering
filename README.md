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
  
- [ ] Is sampling all negatives in the training set cheating?
    - Find out why positive samples are not drawn if we draw lots of negatives...
- [x] Compute HR@K per epoch with callback function
- [ ] Investigate low accuracy:
    - If train/test set contains 99 negatives per positive, accuracy should be around 99% immediately
    - This is likely a problem with the iterator
