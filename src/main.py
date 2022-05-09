import numpy as np
import pandas as pd

import pickle
import math
import random
import time

from pathlib import Path
from enum import Enum


class Gender(Enum):
    MALE = 'M'
    FEMALE = 'F'


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
            gender: Gender,
            occupation: str,
            genres: list[int]):

        rating = rating.value / 5
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
            return A_attr - B_attr


class User:
    def __init__(self, idx):
        self.weights = np.random.rand(22)
        # make genres less important
        for i in range(4, len(self.weights)):
            self.weights[i] *= 1/18
        # normalize entire weight vector
        s = np.sum(self.weights)
        self.weights = self.weights/s
        self.id = idx  # 0-based id
        self.fitness = 1000

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
            print("generating profiles file...")
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
            random_sampling=0.1,
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

    def random_user_sampling(self):
        number_of_users = int(self.config.random_sampling * len(self.users))
        return random.sample(self.users, k=number_of_users)

    def reduce_to_threshold(self, A: User, users: list[User]):
        for user in users:
            if self.euclidean(A, user) <= self.config.neighborhood_threshold:
                yield user

    def euclidean(self, A: User, B: User):
        if A.id == B.id:
            return 0

        s = 0
        for i in range(len(self.profiles[A.id])):
            if None not in [self.profiles[A.id][i], self.profiles[B.id][i]]:

                A_profile = self.profiles[A.id][i]
                B_profile = self.profiles[B.id][i]
                for idx, weight in enumerate(A.weights):
                    s += weight * Profile.diff(A_profile, B_profile, idx)**2

        return math.sqrt(s)

    def predict_vote(self, A: User, i: int, neighbors: list[User]):
        A_mean = self.dataset.get_mean_vote(A)
        s = 0
        euclidean_sum = 0
        for user in neighbors:
            # check if user has also seen this movie
            if self.profiles[user.id][i] is not None:
                euclidean = self.euclidean(A, user)
                euclidean_sum += euclidean
                user_mean = self.dataset.get_mean_vote(user)
                s += euclidean * (self.profiles[user.id][i].attrs[0] - user_mean)

        if euclidean_sum != 0:
            s /= euclidean_sum
        return s + A_mean

    def fitness_score(self, A: User, neighbors: list[User]):

        s = 0
        movies = list(self.dataset.get_user_rated_movies(A))
        for movie_id, rating in movies:
            prediction = self.predict_vote(A, movie_id, neighbors)
            s += abs(prediction - rating)
        return s/len(movies)

    def crossover(self, A: User, B: User):
        A = A.copy()
        B = B.copy()
        crossover_point = random.randint(1, 20)
        B_new_slice = A.weights[crossover_point:]
        A_new_slice = B.weights[crossover_point:]

        A.weights = np.array(list(A.weights[:crossover_point]) + list(A_new_slice))
        B.weights = np.array(list(B.weights[:crossover_point]) + list(B_new_slice))

        for i in range(len(A.weights)):
            if random.uniform(0, 1) <= self.config.mutation:
                A.weights[i] *= self.config.mutation
            if random.uniform(0, 1) <= self.config.mutation:
                B.weights[i] *= self.config.mutation

        return A, B

    def sort_by_fitness(self, candidates: list[User]):

        random_users = self.random_user_sampling()

        # evaluate fitness
        for candidate in candidates:
            neighbors = self.reduce_to_threshold(candidate, random_users)
            candidate.fitness = self.fitness_score(candidate, neighbors)

        # sort by fitness
        candidates.sort(key=lambda c: c.fitness)

    def genetic_iteration(self, candidates: list[User]):

        self.sort_by_fitness(candidates)

        # keep some
        number_of_kept = int(len(candidates) * self.config.top_kept)
        number_of_parents = int(len(candidates) * self.config.top_parents)

        # replace the rest of the candidates with offspring
        offsprings = []
        for i in range(number_of_kept, len(candidates), 2):
            mamma_idx = random.randint(0, number_of_parents)
            pappa_idx = random.randint(0, number_of_parents)
            mamma = candidates[mamma_idx]
            pappa = candidates[pappa_idx]

            offsprings += self.crossover(mamma, pappa)

        population_size = len(candidates)
        candidates = candidates[:number_of_kept] + offsprings
        return candidates[:population_size]

    def train(
            self,
            user_id: int,
            population_size: int = 100,
            iterations: int = 100):

        print("generating random candidates...")
        # generate random population
        candidates = []
        for i in range(population_size):
            candidates.append(User(user_id))

        print("starting iterations...")
        for i in range(iterations):
            # list comes sorted by fitness score
            candidates = self.genetic_iteration(candidates)
            avg = sum([x.fitness for x in candidates])/len(candidates)
            print("iteration", i, ":", candidates[0].fitness, avg, candidates[-1].fitness)

        self.sort_by_fitness(candidates)
        return candidates[0]


if __name__ == "__main__":
    config = RecommenderConfig(random_sampling=0.05, neighborhood_threshold=0.2)
    print("setting up recommender system...")
    recommender = Recommender("profiles.data", config)
    print("starting iteration algorithm...")
    print(recommender.train(user_id=random.randint(1,100)))
