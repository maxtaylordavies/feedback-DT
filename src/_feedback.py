import json
import os
import re
from abc import ABC, abstractmethod

import numpy as np
from numpy.random import default_rng

from argparsing import get_args
from _datasets import get_dataset, name_dataset

OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,  # goal world object does not exist in BabyAI environments
    "lava": 9,
    "agent": 10,  # agent does not seem to be included as an object in BabyAI environment grids
}

COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}

ACTION_TO_STR = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}


class Feedback(ABC):
    def __init__(self, args):
        self.feedback_mode = args["feedback_mode"]
        self.feedback_freq_type = args["feedback_freq_type"]
        self.feedback_freq_steps = args["feedback_freq_steps"]
        self.env_name = args["env_name"]
        self.num_episodes = args["num_episodes"]
        self.feedback_type = args["feedback_type"]
        self.dataset = get_dataset(args)
        self.dataset_name = name_dataset(args)

        self.goal_color, self.goal_object = self._get_goal_metadata()

        self.episode_data = self._get_episode_data()

        self.rng = default_rng(args["seed"])

        self.feedback_data = {self.feedback_type: {}}

    def _get_goal_metadata(self):
        env_name = re.split("_", self.dataset_name)[0]
        with open("env_metadata.json", "r") as env_metadata:
            metadata = json.load(env_metadata)
        mission = metadata[env_name]["mission_string"]
        return re.split(" ", mission.replace("go to the ", ""))

    def _get_episode_data(self):
        episode_data = {}
        episode_data["goal_positions"] = []
        episode_data["agent_positions"] = []
        episode_data["direction_observations"] = []
        episode_data["observations"] = []
        episode_data["actions"] = []
        total_steps = 0
        for episode in self.dataset.episodes:
            previous_total_steps = total_steps
            total_steps += len(episode)
            episode_data["goal_positions"].append(
                self.dataset.goal_positions[previous_total_steps:total_steps]
            )
            episode_data["agent_positions"].append(
                self.dataset.agent_positions[previous_total_steps:total_steps]
            )
            episode_data["direction_observations"].append(
                self.dataset.direction_observations[previous_total_steps:total_steps]
            )
            episode_data["observations"].append(episode.observations)
            episode_data["actions"].append(episode.actions)
        return episode_data

    @abstractmethod
    def _get_polarity(self):
        raise NotImplementedError

    @abstractmethod
    def generate_feedback(self):
        raise NotImplementedError

    def save_feedback(self):
        feedback_path = f"{os.path.abspath('')}/feedback_data/{self.dataset_name}.json"
        with open(feedback_path, "w+") as outfile:
            json.dump(self.feedback_data, outfile)


class DirectionFeedback(Feedback):
    def __init__(self, *args):
        super().__init__(*args)

    def _get_relative_goal_position(self, current_episode, current_step):
        north = False
        east = False
        south = False
        west = False

        goal_y = self.episode_data["goal_positions"][current_episode][current_step][1]
        goal_x = self.episode_data["goal_positions"][current_episode][current_step][0]
        agent_y = self.episode_data["agent_positions"][current_episode][current_step][1]
        agent_x = self.episode_data["agent_positions"][current_episode][current_step][1]

        if goal_x > agent_x:
            east = True
        if goal_y > agent_y:
            south = True
        if goal_x < agent_x:
            west = True
        if goal_y < agent_y:
            north = True

        # Ids of directions is based on direction encodings:
        # AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}
        return [east, south, west, north]

    def _facing_goal(self, direction_observation, relative_goal_position):
        if direction_observation in np.where(relative_goal_position)[0]:
            return True
        else:
            return False

    def _get_polarity(self, direction_observation, relative_goal_position):
        if self._facing_goal(direction_observation, relative_goal_position):
            return "positive"
        else:
            return "negative"

    def generate_feedback(self):
        with open(
            f"{os.path.abspath('')}/feedback_variants.json", encoding="utf-8"
        ) as json_file:
            feedback_variants = json.load(json_file)[self.feedback_type]

        self.feedback_data[self.feedback_type][self.feedback_mode] = {
            f"{self.feedback_freq_type}_{self.feedback_freq_steps}": []
        }
        for e, episode in enumerate(self.episode_data["direction_observations"]):
            episode_feedback = []
            if self.feedback_freq_type.lower() == "poisson":
                feedback_freq = 0
                while feedback_freq == 0:
                    feedback_freq = np.random.poisson(self.feedback_freq_steps)
            else:
                feedback_freq = self.feedback_freq_steps
            for i, direction_observation in enumerate(episode):
                if i % feedback_freq == 0:
                    relative_goal_position = self._get_relative_goal_position(e, i)
                    polarity = self._get_polarity(
                        direction_observation, relative_goal_position
                    )
                    feedback = feedback_variants[polarity][self.feedback_mode]
                    if not isinstance(feedback, str):
                        random_id = self.rng.integers(len(feedback))
                        random_feedback = feedback[random_id]
                        episode_feedback.append(random_feedback)
                    else:
                        episode_feedback.append(feedback)
                else:
                    episode_feedback.append("")
            self.feedback_data[self.feedback_type][self.feedback_mode][
                f"{self.feedback_freq_type}_{self.feedback_freq_steps}"
            ].append(episode_feedback)


# class DistanceFeedback(Feedback):
# def __init__(self, *args):
#     super().__init__(*args)


class ActionFeedback(Feedback):
    def __init__(self, *args):
        super().__init__(*args)
        self.valid_actions = np.arange(
            0, 3, dtype=int
        )  # not sure if done is actually needed?

    def _get_polarity(self, action):
        if action in self.valid_actions:
            return "positive"
        else:
            return "negative"

    def generate_feedback(self):
        with open(
            f"{os.path.abspath('')}/feedback_variants.json", encoding="utf-8"
        ) as json_file:
            feedback_variants = json.load(json_file)[self.feedback_type]

        self.feedback_data[self.feedback_type][self.feedback_mode] = {
            f"{self.feedback_freq_type}_{self.feedback_freq_steps}": []
        }
        for episode in self.episode_data["actions"]:
            episode_feedback = []
            if self.feedback_freq_type.lower() == "poisson":
                feedback_freq = 0
                while feedback_freq == 0:
                    feedback_freq = np.random.poisson(self.feedback_freq_steps)
            else:
                feedback_freq = self.feedback_freq_steps
            for i, action in enumerate(episode):
                if i % feedback_freq == 0:
                    polarity = self._get_polarity(action)
                    feedback = feedback_variants[polarity][self.feedback_mode]
                    if not isinstance(feedback, str):
                        random_id = self.rng.integers(len(feedback))
                        random_feedback = feedback[random_id]
                        episode_feedback.append(random_feedback)
                    else:
                        episode_feedback.append(feedback)
                else:
                    episode_feedback.append("")
            self.feedback_data[self.feedback_type][self.feedback_mode][
                f"{self.feedback_freq_type}_{self.feedback_freq_steps}"
            ].append(episode_feedback)


# class AdjacencyFeedback(Feedback):
# def __init__(self, *args):
#     super().__init__(*args)


if __name__ == "__main__":
    args = get_args()
    type = args["feedback_type"]
    type_to_generator = {
        "direction": DirectionFeedback(args),
        # "distance": DistanceFeedback(args),
        "action": ActionFeedback(args),
        # "adjacency": AdjacencyFeedback(args),
    }
    feedback_generator = type_to_generator[type]
    feedback_generator.generate_feedback()
    feedback_generator.save_feedback()
