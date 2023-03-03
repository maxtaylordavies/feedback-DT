from abc import ABC, abstractmethod
from get_datasets import load_dataset, name_dataset
import json
import re
from argparsing import get_feedback_args
import numpy as np

OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6, # goal object type
    "box": 7, # distractor object type
    "goal": 8, # goal world object does not exist in BabyAI environments - is always ball
    "lava": 9,
    "agent": 10, # agent does not seem to be included in BabyAI environments
}

COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}


class Feedback(ABC):
    def __init__(self, dataset_name, mode):
        self.mode = mode
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_name)

        self.goal_color, self.goal_object = self._get_goal_metadata()

        self.episode_data = self._get_episode_data()

        self.feedback_data = {}

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

    # def _get_goal_position(self, observation):

    @abstractmethod
    def generate_feedback(self):
        raise NotImplementedError

    @abstractmethod
    def save_feedback(self):
        raise NotImplementedError


class DirectionFeedback(Feedback):
    def __init__(self, *args):
        super().__init__(*args)

    def _get_relative_goal_position(self, step):
        self.feedback_data["relative_goal_position"] = np.zeros(len(self.dataset.episodes))
        for i in range(len(self.dataset.episodes)):
            north = False
            east = False
            south = False
            west = False
            self.feedback_data["relative_goal_position"][i] = np.zeros(len(self.episode_data["goal_positions"][i]))
            goal_y = self.episode_data["goal_positions"][i][step][1]
            goal_x = self.episode_data["goal_positions"][i][step][0]
            agent_y = self.episode_data["agent_positions"][i][step][1]
            agent_x = self.episode_data["agent_positions"][i][step][1]
            if goal_y > agent_y:
                north = True
            if goal_y < agent_y:
                south = True
            if goal_x < agent_x:
                west = True
            if goal_x > agent_x:
                east = True                
            
            # Order of directions is based on direction encodings 
            # AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}
            self.feedback_data["relative_goal_position"][i][step] = [east, south, west, north]
    
    def _facing_goal(self, step):
        for i in range(len(self.dataset.episodes)):
            direction = self.episode_data["direction_observations"][i][step]
            if direction in np.where(self.feedback_data["relative_goal_position"][i][step])[0]:
                return True
            else:
                return False

    def generate_feedback(self):
        pass

    def save_feedback(self):
        pass

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
    mode = args["mode"]
    env_name = args["env_name"]
    num_episodes = args["num_episodes"]
    include_timeout = args["include_timeout"]
    dataset_name = name_dataset(env_name, num_episodes, include_timeout)
    type_to_class = {"direction": DirectionFeedback(dataset_name, mode)}
    feedback = type_to_class[type]
