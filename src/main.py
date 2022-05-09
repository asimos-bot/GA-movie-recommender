import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import pickle
import math
import random
import time

from pathlib import Path
from enum import Enum


class Genre(Enum):
    ACTION = 0
    ADVENTURE = 1
    ANIMATION = 2
    CHILDREN = 3
    COMEDY = 4
    CRIME = 5
    DOCUMENTARY = 6
    DRAMA = 7
    FANTASY = 8
    FILM_NOIR = 9
    HORROR = 10
    MUSICAL = 11
    MYSTERY = 12
    ROMANCE = 13
    SCI_FI = 14
    THRILLER = 15
    WAR = 16
    WESTERN = 17


class Profile:

    def __init__(
            self,
            rating: float,
            age: int,
            gender: str,
            occupation: str,
            genres: list[int]):

        rating = rating / 5
        age /= 100

        self.attrs = [rating, age, gender, occupation]
        self.attrs += genres

    @classmethod
    def diff(cls, A, B, attr_idx):
        A_attr = A.attrs[attr_idx]
        B_attr = B.attrs[attr_idx]
        if type(A_attr) == str:
            return int(A_attr != B_attr)
        else:
            return abs(A_attr - B_attr)


class User:
    def __init__(self, idx):
        self.weights = np.random.rand(22)
        # make genres less important
        for i in range(4, len(self.weights)):
            self.weights[i] *= 0.05
        self.id = idx  # 0 based id
        self.row_id = idx-1
        self.fitness = 1000

    def normalize_weights(self):
        self.weights = self.weights/np.linalg.norm(self.weights)

    def copy(self):
        user = User(self.id)
        user.weights = np.copy(self.weights)
        user.fitness = self.fitness
        return user


class Dataset:
    def __init__(self, profile_filename):
        dataset_folder = Path(__file__).parent.parent.joinpath("ml-100k")
        self.users = pd.read_csv(
                dataset_folder.joinpath("u.user"),
                delimiter="|",
                header=None).to_numpy()
        self.movies = pd.read_csv(
                dataset_folder.joinpath("u.item"),
                encoding="ISO-8859-1",
                delimiter="|",
                header=None).to_numpy()
        self.ratings = pd.read_csv(
                dataset_folder.joinpath("u.data"),
                delimiter="\t",
                header=None).to_numpy()
        self.profile_filename = profile_filename

    def get_profiles(self):
        try:
            with open(self.profile_filename, "rb") as file:
                return pickle.loads(file.read())
        except Exception:
            return self._generate_profiles()

    def get_users(self):
        users = []
        for i in range(self.users.shape[0]):
            users.append(User(i))
        return users

    def _generate_profiles(self):

        profiles = []
        for i in range(self.users.shape[0]):
            profiles.append([])
            for j in range(self.movies.shape[0]):
                profiles[i].append(None)

        for idx in range(self.ratings.shape[0]):
            row = self.ratings[idx, :]
            i = row[0]-1
            j = row[1]-1
            age = self.users[i, 1]
            gender = self.users[i, 2]
            occupation = self.users[i, 3]
            genres = list(self.movies[j, 6:24])
            rating = row[2]
            profile = Profile(rating, age, gender, occupation, genres)
            profiles[i][j] = profile

        with open(self.profile_filename, "wb") as file:
            file.write(pickle.dumps(profiles))
        return profiles

    def get_mean_vote(self, A: User):
        row_idxs = np.where(self.ratings[:, 0] == A.id)[0]
        if len(row_idxs) == 0:
            return 0
        s = 0
        for idx in row_idxs:
            s += self.ratings[idx][2]
        return s/len(row_idxs)

    def get_user_rated_movies(self, A: User):
        row_idxs = np.where(self.ratings[:, 0] == A.id)[0]
        for idx in row_idxs:
            yield (self.ratings[idx][1], self.ratings[idx][2])


class RecommenderConfig:
    def __init__(
            self,
            random_sampling=50,
            neighborhood_threshold=0.2,
            top_kept=0.25,
            top_parents=0.4,
            mutation=0.01):
        self.random_sampling = random_sampling
        self.neighborhood_threshold = neighborhood_threshold
        self.top_kept = top_kept
        self.top_parents = top_parents
        self.mutation = mutation


class Recommender:
    def __init__(self, profiles_filename, config: RecommenderConfig):
        np.random.seed(int(time.time() * 1000) % (2**32 - 1))
        self.dataset = Dataset(profiles_filename)
        self.profiles = self.dataset.get_profiles()
        self.users = self.dataset.get_users()
        self.config = config
        self.first_plot = True

    def random_user_sampling(self):
        number_of_users = int(self.config.random_sampling)
        return random.sample(self.users, k=number_of_users)

    def reduce_to_threshold(self, A: User, users: list[User]):
        users = sorted(users, key=lambda user: self.euclidean(A, user))
        return users[:int(len(users) * self.config.neighborhood_threshold)]

    def euclidean(self, A: User, B: User):
        if A.id == B.id:
            return 0

        A.normalize_weights()
        s = 0
        A_movies = self.dataset.get_user_rated_movies(A)
        for movie_id, _ in A_movies:

            B_profile = self.profiles[B.row_id][movie_id-1]
            if B_profile is not None:

                A_profile = self.profiles[A.row_id][movie_id-1]
                for idx, weight in enumerate(A.weights):
                    s += weight * Profile.diff(A_profile, B_profile, idx)**2

        return math.sqrt(s)

    def predict_vote(self, A: User, movie_id: int, neighbors: list[User]):
        A_mean = self.dataset.get_mean_vote(A)
        s = 0
        euclidean_sum = 0
        for user in neighbors:
            # check if user has also seen this movie
            if self.profiles[user.row_id][movie_id-1] is not None:
                euclidean = self.euclidean(A, user)
                euclidean_sum += euclidean
                user_mean = self.dataset.get_mean_vote(user)
                s += euclidean * (self.profiles[user.row_id][movie_id-1].attrs[0] - user_mean)
        if euclidean_sum != 0:
            s /= euclidean_sum
        return s + A_mean

    def fitness_score(self, A: User, neighbors: list[User], movies=None):

        s = 0
        if movies is None:
            movies = list(self.dataset.get_user_rated_movies(A))

        if len(movies) == 0:
            return 0

        for movie_id, rating in movies:
            prediction = self.predict_vote(A, movie_id, neighbors)
            s += abs(prediction - rating)

        return s/len(movies)

    def crossover(self, A: User, B: User):
        sis = A.copy()
        bro = B.copy()
        crossover_point = random.randint(1, len(sis.weights)-1)

        tmp = sis.weights[:crossover_point].copy()
        sis.weights[:crossover_point], bro.weights[:crossover_point] = bro.weights[:crossover_point], tmp

        for i in range(len(sis.weights)):
            if random.uniform(0, 1) <= self.config.mutation:
                sis.weights[i] = random.uniform(0, 1)
            if random.uniform(0, 1) <= self.config.mutation:
                bro.weights[i] = random.uniform(0, 1)

        return sis, bro

    def sort_by_fitness(self, candidates: list[User], train_set, random_users):

        # evaluate fitness
        for candidate in candidates:
            neighbors = self.reduce_to_threshold(candidate, random_users)
            candidate.fitness = self.fitness_score(candidate, neighbors, train_set)

        # sort by fitness
        return sorted(candidates, key=lambda c: c.fitness)

    def genetic_iteration(self, candidates: list[User], train_set, random_users):

        # keep some
        number_of_kept = int(len(candidates) * self.config.top_kept)
        number_of_parents = int(len(candidates) * self.config.top_parents)

        # replace the rest of the candidates with offspring
        offsprings = [None] * (len(candidates) - number_of_kept)
        for i in range(0, len(offsprings), 2):
            mamma_idx = random.randint(0, number_of_parents)
            pappa_idx = random.randint(0, number_of_parents)
            mamma = candidates[mamma_idx]
            pappa = candidates[pappa_idx]

            offsprings[i], offsprings[i+1] = self.crossover(mamma, pappa)

        population_size = len(candidates)
        candidates = candidates[:number_of_kept]
        candidates += offsprings
        self.sort_by_fitness(candidates, train_set, random_users)

        return candidates[:population_size]

    def get_iter_info(self, candidates: list[User], train_set, random_users: list[User]):
        candidates = self.sort_by_fitness(candidates, train_set, random_users)
        best_current_fitness = candidates[0].fitness
        number_of_kept = int(len(candidates) * self.config.top_kept)
        top_kept_avg_fitness = sum([x.fitness for x in candidates[:number_of_kept]])/number_of_kept
        avg = sum([x.fitness for x in candidates])/len(candidates)
        return {
                "best current fitness": best_current_fitness,
                "elite average": top_kept_avg_fitness,
                "average": avg
                }

    def plot_iter_info(self, iterations_info):

        # limit size
        if len(iterations_info) > 20:
            iterations_info = iterations_info[len(iterations_info)-20]

        x = list(range(len(iterations_info)))
        lines = dict()
        for k in iterations_info[0]:
            lines[k] = []

        for d in iterations_info:
            for k in d:
                lines[k].append(d[k])

        if self.first_plot:
            self.first_plot = False

            plt.ion()
            self.fig, self.ax = plt.subplots()
        self.ax.clear()
        for k in lines:
            self.ax.plot(x, lines[k], label=k)
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.canvas.show()
        plt.show()

    def train(
            self,
            user_id: int,
            population_size: int = 50,
            training_percentage: float = 1/3,
            iterations: int = 100,
            live_plot=True):

        rated_movies = list(self.dataset.get_user_rated_movies(User(user_id)))
        train_set_size = int(len(rated_movies) * training_percentage)
        train_set = random.sample(rated_movies, k=train_set_size)
        test_set = [x for x in rated_movies if x not in train_set]

        random_users = self.random_user_sampling()

        # generate random population
        candidates = []
        for i in range(population_size):
            candidates.append(User(user_id))
        self.sort_by_fitness(candidates, train_set, random_users)

        iterations_info = [self.get_iter_info(candidates, test_set, random_users)]
        best = candidates[0]
        for i in range(iterations):
            candidates = self.genetic_iteration(candidates, train_set, random_users)
            if best.fitness > candidates[0].fitness:
                best = candidates[0]
            print("iteration", i, ":", candidates[0].fitness)
            iterations_info.append(self.get_iter_info(candidates, test_set, random_users))
            if live_plot:
                self.plot_iter_info(iterations_info)

        return iterations_info, best


if __name__ == "__main__":
    config = RecommenderConfig(random_sampling=10, neighborhood_threshold=0.2)
    recommender = Recommender("profiles.data", config)
    iterations = recommender.train(user_id=465, iterations=10)
    input()
