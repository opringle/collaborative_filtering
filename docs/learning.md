# What are recommender systems?

A recommender system is a machine learning model, which predicts the preference a user would assign to an item.  These systems are used to understand what facebook articles you'd like to read, which youtube videos you will watch, what amazon items you'll most likely buy and so on.

# What is the current state of art?

I did not used to think recommender systems were particularly interesting.  I remember briefly covering matrix factorization and collaborative filtering (two established recommender techniques) during the Master of Data Science program at the University of British Columbia.  My initial impression was that there must be far more interesting problems to solve that selling an extra item to a user on amazon.com or predicting what videos to show users  on youtube.com.  It all seemed kind of depressing and vacuous when compared to apllications in self driving cars and natural language understanding.

However, being fascinated by all applications of deep learning, I've recently been thinking "I wonder if deep learning is used in state of the art recommenders?".  Turns out, yes, everywhere.... The results have been quite astonishing, both in improved system performance and cash money for the companies leveraging it.

If you want to read up on the current state of the art check these papers out:


- [ Visual Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1510.01784.pdf)
- [wide and deep learning for recommender systems](https://arxiv.org/pdf/1606.07792.pdf)
- [neural collaborative filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)
- [A Neural Autoregressive Approach to Collaborative Filtering](https://arxiv.org/pdf/1605.09477.pdf)
- [A fast learning algorithm for deep belief nets](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)
- [Restricted Boltzmann Machines for Collaborative Filtering](https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf)

And here are a few great videos:

- [Alexandros Karatzoglou: Deep Learning for Recommender Systems](https://www.youtube.com/watch?v=KZ7bcfYGuxw)
- [Deep Learning for Personalized Search and Recommender Systems part 1](https://www.youtube.com/watch?v=0DYQzZp68ok&t=4999s)

Having read up on the following research papers, this post will be my delve into implementing a state of the art recomender architecture myself.

## Ok but why are they so interesting?

They ingest a massive quantity and diversity of data.  Since deep learning attempts to learn valid features, the more user and product features we input the better. This includes product images, user search queries, click history, purchase sequences and just about everything you can find on a person from their online footprint. As a result state of the art systems utilize convolutional layers, recurrent layers, embedding of categorical feature and deep/wide model architectures.  Further to this, the models train on such massive data they genuinely benefit from being enourmous.  xxx from xxx recently said the mdoels the research team train are made as large as tehy can fit on a single gpu.  With the recent advances in gpu hardware, this means we are talking a model with so many parameters it is 500Gb!  Now we have to distribute the training of models on many gpu's in the cloud, and keep prediction time in mind for our ebourmous model.  Sounds interesting right?!

The second reason recommender systems are interesting to me is that many companies make massive online revenue, but are not up to speed on the state of the art. Well...you tell me.  Do you think google's youtube recommender is doing some tricks Nike's is not? The potential for a small team of data scientists to massively increase online sales revenue, while pretty souless, is an exciting impact to have. I would expect data scientists specializing in state of the art recommender system implentation to  make some serious salaries.

In this blog I will build such a system.  I will start by showing how deep learning can be used for a system that only requires a record of each users' purchases.  After we implement the state of the art there, we will move on to ingesting user features, such as gender, height etc.  Then we will move on to ingesting more complex features such as item images and user queries (time to use my NLP skillz). I will then scale up the data to moderately serious numbers, train a moderately large model and show how to deploy it.

There are many more directions I could take it from there.  We'll see how keen I am to keep going.  Future work could include how to use multiobjective optimization to decide how to treat model recommendations or investigation into how to deploy this kind of system at massive scale etc etc.

Sounds interesting right! Lego.

## Model 1: Neural Collaborative filtering

```python
print("hellloworddllsslddlqd")
```

## Model 2: Adding user features 

```python
print("hellloworddllsslddlqd")
```


# Tricks

- embed each user (ie their latent factors).  Then you can train a supervised model and the embedded features represent how similar two customers are!  Exact same with products.  Users prefer items etc can be derived.

- If you pass each user in as a one hot encoded vector, then once you embedd the input you actually learn a user feauture matrix.

- You can pass then factorize the user feature matrix into a user feature vector and item feature vector.  You know the item feature vector from previous data. What you train teaches you then user feature vector.  You then predict user preference over items.

- You can do product to vec.  Products with similar vectors appear around the same other kind of products.  Context represents the other purchases of the user. A sentence is the product history.

- store user and item features in a distributed key value store, so you can look them up for each user at scale

- all the features are used in wide and deep models to predict item utility

- item utility is defined by business decisions (revenue, diversity, CTR)