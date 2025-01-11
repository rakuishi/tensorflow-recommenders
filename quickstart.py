from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# MapDataset
ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(
    lambda x: {"movie_title": x["movie_title"], "user_id": x["user_id"]}
)
movies = movies.map(lambda x: x["movie_title"])

# for item in ratings.take(10):
#     movie_title = item["movie_title"].numpy().decode("utf-8")
#     user_id = item["user_id"].numpy()
#     print(f"movie_title: {movie_title}, user_id: {user_id}")

# for item in movies.take(10):
#     print(item.numpy().decode("utf-8"))

movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

# print(unique_movie_titles[:10])
# print(unique_user_ids[:10])

# Build vocabularies to convert user ids and movie titles into integer indices for embedding layers:
user_ids_vocabulary = tf.keras.layers.StringLookup(
    vocabulary=unique_movie_titles, mask_token=None
)
movie_titles_vocabulary = tf.keras.layers.StringLookup(
    vocabulary=unique_movie_titles, mask_token=None
)

# Define user and movie models.
user_model = tf.keras.Sequential(
    [
        user_ids_vocabulary,
        tf.keras.layers.Embedding(user_ids_vocabulary.vocabulary_size(), 64),
    ]
)
movie_model = tf.keras.Sequential(
    [
        movie_titles_vocabulary,
        tf.keras.layers.Embedding(movie_titles_vocabulary.vocabulary_size(), 64),
    ]
)

# Define your objectives.
task = tfrs.tasks.Retrieval(
    metrics=tfrs.metrics.FactorizedTopK(movies.batch(128).map(movie_model))
)


class MovieLensModel(tfrs.Model):
    def __init__(
        self,
        user_model: tf.keras.Model,
        movie_model: tf.keras.Model,
        task: tfrs.tasks.Retrieval,
    ):
        super().__init__()

        self.user_model = user_model
        self.movie_model = movie_model
        self.task = task

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_title"])
        return self.task(user_embeddings, movie_embeddings)


# Create a retrieval model.
model = MovieLensModel(user_model, movie_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

# Train for 3 epochs.
model.fit(ratings.batch(4096), epochs=3)

# Use brute-force search to set up retrieval using the trained representations.
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
    movies.batch(100).map(lambda title: (title, model.movie_model(title)))
)

# Get some recommendations.
_, titles = index(np.array(["42"]))
print(f"Top 3 recommendations for user 42: {titles[0, :3]}")
