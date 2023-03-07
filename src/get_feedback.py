import json
import os
import re
from abc import ABC, abstractmethod

import numpy as np
from numpy.random import default_rng

from argparsing import get_feedback_args
from _datasets import load_dataset, name_dataset

OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,  # goal object type
    "box": 7,  # distractor object type
    "goal": 8,  # goal world object does not exist in BabyAI environments - is always ball
    "lava": 9,
    "agent": 10,  # agent does not seem to be included in BabyAI environments
}

COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}


class Feedback(ABC):
    def __init__(self, dataset_name, feedback_mode, feedback_n_steps, freq_type):
        self.feedback_type = "direction"
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_name)
        self.feedback_mode = feedback_mode
        self.feedback_n_steps = feedback_n_steps
        self.freq_type = freq_type
        self.goal_color, self.goal_object = self._get_goal_metadata()

        self.episode_data = self._get_episode_data()

        self.rng = default_rng(42)

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
    def generate_feedback(self):
        raise NotImplementedError

    @abstractmethod
    def save_feedback(self):
        raise NotImplementedError


class DirectionFeedback(Feedback):
    def __init__(self, *args):
        super().__init__(*args)
        self.feedback_data = {self.feedback_type: {}}

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
        if goal_y < agent_y:
            south = True
        if goal_x < agent_x:
            west = True
        if goal_y > agent_y:
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

        self.feedback_data[self.feedback_type][self.feedback_mode] = {f"{self.feedback_n_steps}_{self.freq_type}": []}
        for e, episode in enumerate(self.episode_data["direction_observations"]):
            episode_feedback = []
            if self.feedback_n_steps < 2:
                feedback_freq = 2
                if e == 0:
                    print("Feedback can be provided at most after every other step")
            else:
                if self.freq_type.lower() == "poisson":
                    feedback_freq = 0
                    while feedback_freq < 2:
                        feedback_freq = np.random.poisson(self.feedback_n_steps)
                else:
                    feedback_freq = self.feedback_n_steps
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
            self.feedback_data[self.feedback_type][self.feedback_mode][f"{self.feedback_n_steps}_{self.freq_type}"].append(
                episode_feedback
            )

    def save_feedback(self):
        feedback_path = f"{os.path.abspath('')}/feedback_data/{self.dataset_name}.json"
        with open(feedback_path, "w+") as outfile:
            json.dump(self.feedback_data, outfile)


class DistanceFeedback(Feedback):
    def __init__(self):
        super().__init__()


class ActionFeedback(Feedback):
    def __init__(self):
        super().__init__()


class AdjacencyFeedback(Feedback):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    args = get_feedback_args()
    type = args["type"]
    feedback_mode = args["mode"]
    feedback_n_steps = args["n_steps"]
    freq_type = args["freq_type"]
    env_name = args["env_name"]
    num_episodes = args["num_episodes"]
    include_timeout = args["include_timeout"]
    dataset_name = name_dataset(env_name, num_episodes, include_timeout)
    type_to_generator = {
        "direction": DirectionFeedback(dataset_name, feedback_mode, feedback_n_steps, freq_type)
    }
    feedback_generator = type_to_generator[type]
    feedback_generator.generate_feedback()
    feedback_generator.save_feedback()
