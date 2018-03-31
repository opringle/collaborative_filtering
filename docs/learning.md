# What are recommender systems?

A recommender system is a machine learning model, which predicts the preference a user would assign to an item.  These systems are used to understand what facebook articles you'd like to read, which youtube videos you will watch, what amazon items you'll most likely buy and so on.

# What is the current state of art?

I did not used to think recommender systems were particularly interesting.  I remember briefly covering matrix factorization and collaborative filtering (two established recommender techniques) during the Master of Data Science program at the University of British Columbia.  My initial impression was that there must be far more interesting problems to solve that selling an extra item to a user on amazon.com or predicting what videos to show users  on youtube.com.  It all seemed kind of depressing and vacuous when compared to apllications in self driving cars and natural language understanding.

However, being fascinated by all applications of deep learning, I've recently been thinking "I wonder if deep learning is used in state of the art recommenders?".  Turns out, yes, everywhere.... The results have been quite astonishing, both in improved system performance and cash money for the companies leveraging it.

If you want to read up on the current state of the art check these papers out:

- [neural collaborative filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)
- [ Visual Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1510.01784.pdf)
- [wide and deep learning for recommender systems](https://arxiv.org/pdf/1606.07792.pdf)
- [A Neural Autoregressive Approach to Collaborative Filtering](https://arxiv.org/pdf/1605.09477.pdf)
- [A fast learning algorithm for deep belief nets](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)
- [Restricted Boltzmann Machines for Collaborative Filtering](https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf)

And here are a few great videos:

- [Alexandros Karatzoglou: Deep Learning for Recommender Systems](https://www.youtube.com/watch?v=KZ7bcfYGuxw)
- [Deep Learning for Personalized Search and Recommender Systems part 1](https://www.youtube.com/watch?v=0DYQzZp68ok&t=4999s)

Having read up on the following research papers, this post will be my delve into implementing a state of the art recomender architecture myself.

# Ok but why are they so interesting?

They ingest a massive quantity and diversity of data.  Since deep learning attempts to learn valid features, the more user and product features we input the better. This includes product images, user search queries, click history, purchase sequences and just about everything you can find on a person from their online footprint. As a result state of the art systems utilize convolutional layers, recurrent layers, embedding of categorical feature and deep/wide model architectures.  Further to this, the models train on such massive data they genuinely benefit from being enourmous.  xxx from xxx recently said the mdoels the research team train are made as large as tehy can fit on a single gpu.  With the recent advances in gpu hardware, this means we are talking a model with so many parameters it is 500Gb!  Now we have to distribute the training of models on many gpu's in the cloud, and keep prediction time in mind for our ebourmous model.  Sounds interesting right?!

The second reason recommender systems are interesting to me is that many companies make massive online revenue, but are not up to speed on the state of the art. Well...you tell me.  Do you think google's youtube recommender is doing some tricks Nike's is not? The potential for a small team of data scientists to massively increase online sales revenue, while pretty souless, is an exciting impact to have. I would expect data scientists specializing in state of the art recommender system implentation to  make some serious salaries.

In this blog I will build such a system.  I will start by showing how deep learning can be used for a system that only requires a record of each users' purchases.  After we implement the state of the art there, we will move on to ingesting user features, such as gender, height etc.  Then we will move on to ingesting more complex features such as item images and user queries (time to use my NLP skillz). I will then scale up the data to moderately serious numbers, train a moderately large model and show how to deploy it.

There are many more directions I could take it from there.  We'll see how keen I am to keep going.  Future work could include how to use multiobjective optimization to decide how to treat model recommendations or investigation into how to deploy this kind of system at massive scale etc etc.

Sounds interesting right! Lego.

# Model 1: Neural Collaborative filtering

- Paper: [neural collaborative filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)

- This model is an example of the core deep learning framework for building a ranking model, and as such only ingests user_id & item_id features.
- This is a great starting point to understand the input data,  model architecture, loss functions, evaluation metrics/tricks required.

## Understanding the input data

- Implicit feedback denotes whether a user interacted with an item or not (0 or 1) as opposed to explicit feedback such as a movie review (6/10).
- We don't want a model to predict what a user would rate a movie.  We want a model to predict if a user will watch a movie!  Because the later makes money.
- Therefore, we treat input data as implicit feedback, regardless of its form.
- Recommender input data is incredibly sparse.  In the case of the MovieLens 1M Dataset, we  have 1,000,000 interactions (1's) in over 22 million records! (0's & 1's)
- Dataset only contains 1's.  One approach is to generate all negative data. Another is to randomly generate less of it.
- The test set is the last interaction (in time) of every user.

## Understanding the architecture

##### MLP architecture

- For each training record X is a single user to single item combination. Y is the implicit feedback (0 or 1).
- The input user_id and item_id are both embedded to a some length (latent features).
- Embeddings are concatenated and passed through hidden layers.
- Finally the layer size is reduced to 1 unit, with sigmoid activation (can only be between 0 & 1).
- The binary cross entropy loss is applied & weights updated with some optimizer.  In real systems the pairwise loss could be a superior extension (interaction data more important than non interaction data).

##### GMF architecture

- For each training record X is a single user to single item combination. Y is the implicit feedback (0 or 1).
- The input user_id and item_id are both embedded to a some length (latent features).
- Embeddings are concatenated and passed through hidden layers.
- Finally the layer size is reduced to 1 unit, with sigmoid activation (can only be between 0 & 1).
- The logistic loss is applied & weights updated with some optimizer.

##### Hybrid architecture

## Understanding important recommender test metrics

- HR@10, NCDG@10 used
- For evaluation, randomly sample  100 non-interacted items for each user and rank the test item amoung them.
- Compute HR (hit ratio) and NDCK (Normalized Discounted Cumulative Gain) on this example.
- Only take 10 rankings.
- Otherwise when you compute test error, you would need to compute the probability of interaction of all user and item combinations.
- process is for each user to randomly select 100 random samples not interacted with.

## Implementing in MXNet

```python
print("helloworld")
```

# Model 2: Adding user features 

```python
pass
```

# Tricks

- You can do product to vec.  Products with similar vectors appear around the same other kind of products.  Context represents the other purchases of the user. A sentence is the product history.

- store user and item features in a distributed key value store, so you can look them up for each user at scale

- wide and deep models appear to be the state of the art

- item utility is defined by business decisions (revenue, diversity, CTR) and optimized using lagrange methods


