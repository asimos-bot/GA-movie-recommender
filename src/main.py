import numpy as np
import pandas as pd

import pickle
import math
import random

from pathlib import Path
from enum import Enum


class Rating(Enum):
    VERY_BAD = 0
    BAD = 1
    AVERAGE = 2
    GOOD = 3
    VERY_GOOD = 4


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
            rating: Rating,
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
            return abs(A_attr - B_attr)


class User:
    def __init__(self, idx):
        self.weights = np.random.rand(22)
        self.id = idx  # 0-based id


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


class RecommenderConfig:
    def __init__(
            self,
            random_sampling=0.1,
            neighborhood_threshold=0.2):
        self.random_sampling = random_sampling
        self.neighborhood_threshold = neighborhood_threshold


class Recommender:
    def __init__(self, profiles_filename, config: RecommenderConfig):
        self.dataset = Dataset(profiles_filename)
        self.profiles = self.dataset.get_profiles()
        self.users = self.dataset.get_users()
        self.config = config

    def random_user_sampling(self):
        number_of_users = int(self.config.random_sampling * len(self.users))
        return random.sample(self.users, k=number_of_users)

    def euclidean(self, A: User, B: User):
        s = 0
        for i in range(len(self.profiles[A.id])):
            if None not in [self.profiles[A.id][i], self.profiles[B.id][i]]:

                A_profile = self.profiles[A.id][i]
                B_profile = self.profiles[B.id][i]
                for idx, weight in enumerate(A.weights):
                    s += weight * Profile.diff(A_profile, B_profile, idx)**2
        return math.sqrt(s)


if __name__ == "__main__":
    recommender = Recommender("profiles.data")
    A = User(2)
    B = User(3)
    print(recommender.euclidean(A, B))
