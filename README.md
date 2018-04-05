# Deep Learning Recommender Systems

A repository to experiment building state of the art recommender systems using apache MXNet.

## Repo structure

`/data/`: directory used for storing model input data
`/results/`: directory used for storing model reports and results
`/src/`: python code to build models, preprocess data etc
/`docs/`: research notes etc

## Running the code

- `$ pip install -r requirements.txt`
- `$cd ./src `
- `$python preprocess.py`
- `$python recsys.py`

## ToDo

### Implement Custom  Iterator that allows negative samples to be created as model is fed and can compute HR@K

1. iterator class should:
    + ingest a pandas dataframe
    + extract data as 3 numpy arrays (X_user, X_item, Y)
    + each feature array is shape = (n_examples, categorical feature count)
    + add x negative samples per label when next is called (to save memory)
    + return mx.NDArrayIer(), dictionary of vocab sizes
    + return mx.NDArrayIer(), dictionary of vocab sizes
    + to compute  HR@K we need the iterator to:
        + produce a batch for each purchase
        + for each batch sample n items that that user did not buy (that were not negatively sampled in the training set)
            + alternatively, you could sample same num negatives in train and test (kinda dumb)
            + therefore, to create ValIter you need to know about TrainIter
                + therefore, train iter should produce a dict of negatives in selected
                + val iter will take this dict as a required input
    + pass sampling matrix between iterators
    + have `compute_HR=False` parameter in order to produce batches for HR@k
    + OR keep sampling rate the same
    
2. Would be great to be able to compute HR@K with a custom mxnet metric:

    + once we have `label, pred`, we can order the labels by the predictions, slice the top k & sum the array
    + this could all be done in preprocessing then just pass ordered array to NdArrayIter()

3. might be worth just using the iterators in this example:
    + provided they keep negative samples as unseen train samples
    + provided you can add an option to produce batch per user
    + https://github.com/apache/incubator-mxnet/blob/master/example/recommenders/negativesample.py

## Notes

- since train and test do not have the sample number of negatives, the loss on the test set can't be related to the loss on the training set
- the model in training will learn to say numbers close to 0, since the dataset is imbalanced
- then on the test set there are usually many more 0's, so loss is lower
- this means making test=train error is not the measure for overfitting
- instead we should just optimize the validation score