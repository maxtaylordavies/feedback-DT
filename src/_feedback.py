import json
import os
import re
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from scipy.spatial.distance import cdist

from src._datasets import get_dataset, name_dataset
from src.argparsing import get_args
from src.utils import log

FEEDBACK_DIR = f"{os.path.abspath('')}/feedback_data"

OBJECT_TO_STR = {
    0: "unseen",
    1: "empty",
    2: "wall",
    3: "floor",
    4: "door",
    5: "key",
    6: "ball",
    7: "box",
    8: "goal",  # goal world object does not exist in BabyAI environments
    9: "lava",
    10: "agent",  # agent does not seem to be included as an object in BabyAI environment grids
}

COLOR_TO_STR = {0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}

ACTION_TO_STR = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

AGENT_DIR_TO_STR = {0: "east", 1: "south", 2: "west", 3: "north"}


class Feedback(ABC):
    def __init__(self, args, dataset):
        self.feedback_mode = args["feedback_mode"]
        self.feedback_freq_type = args["feedback_freq_type"]
        self.feedback_freq_steps = args["feedback_freq_steps"]
        self.feedback_freq = f"{self.feedback_freq_type}_{self.feedback_freq_steps}"
        self.env_name = args["env_name"]
        self.num_episodes = args["num_episodes"]
        self.feedback_type = args["feedback_type"]
        self.dataset_name = name_dataset(args)
        self.dataset = dataset

        self.goal_color, self.goal_type = self._get_goal_metadata()
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
        episode_data["rgb_observations"] = []
        episode_data["symbolic_observations"] = []
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
            episode_data["rgb_observations"].append(episode.observations)
            episode_data["symbolic_observations"].append(
                self.dataset.symbolic_observations[previous_total_steps:total_steps]
            )
            episode_data["actions"].append(episode.actions)
        return episode_data

    @abstractmethod
    def _get_polarity(self):
        raise NotImplementedError

    def generate_feedback(self, attribute):
        with open(
            f"{os.path.abspath('')}/feedback_variants.json", encoding="utf-8"
        ) as json_file:
            feedback_variants = json.load(json_file)[self.feedback_type]

        self.feedback_data[self.feedback_type][self.feedback_mode] = {self.feedback_freq: []}
        for e, episode in enumerate(self.episode_data[attribute]):
            episode_feedback = []
            if self.feedback_freq_type.lower() == "poisson":
                feedback_freq = 0
                while feedback_freq == 0:
                    feedback_freq = np.random.poisson(self.feedback_freq_steps)
            else:
                feedback_freq = self.feedback_freq_steps
            for i, attribute_value in enumerate(episode):
                if i == 0 and self.feedback_type == "distance":
                    self._save_previous_agent_position(attribute_value)
                    episode_feedback.append("")
                    continue
                if (i + 1) % feedback_freq == 0:
                    polarity = self._get_polarity(attribute_value, e, i)
                    feedback = feedback_variants[polarity][self.feedback_mode]
                    if not isinstance(feedback, str):
                        random_id = self.rng.integers(len(feedback))
                        random_feedback = feedback[random_id]
                        if (
                            self.feedback_type == "adjacency"
                            and self.adjacent_object is not None
                        ):
                            final_feedback = (
                                random_feedback.replace(
                                    "{current object col}", self.adjacent_object_color
                                )
                                .replace("{goal object col}", self.goal_color)
                                .replace("{goal/current object col}", self.goal_color)
                                .replace("{current/goal object col}", self.goal_color)
                                .replace("{current object type}", self.adjacent_object_type)
                                .replace("{goal object type}", self.goal_type)
                                .replace("{goal/current object type}", self.goal_type)
                                .replace("{current/goal object type}", self.goal_type)
                            )
                        else:
                            final_feedback = random_feedback
                        episode_feedback.append(final_feedback)
                    else:
                        episode_feedback.append(feedback)
                else:
                    episode_feedback.append("")
            else:
                final_episode_feedback = episode_feedback
            self.feedback_data[self.feedback_type][self.feedback_mode][
                self.feedback_freq
            ].append(final_episode_feedback)

        return self.feedback_data

    def save_feedback(self):
        if not os.path.exists(FEEDBACK_DIR):
            os.mkdir(FEEDBACK_DIR)
        feedback_path = f"{FEEDBACK_DIR}/{self.dataset_name}.json"
        if os.path.exists(feedback_path):
            with open(feedback_path, encoding="utf-8") as json_file:
                existing_feedback_data = json.load(json_file)
            if self.feedback_type in existing_feedback_data:
                if self.feedback_mode in existing_feedback_data[self.feedback_type]:
                    if (
                        self.feedback_freq
                        in existing_feedback_data[self.feedback_type][self.feedback_mode]
                    ):
                        existing_feedback_data[self.feedback_type][self.feedback_mode][
                            self.feedback_freq
                        ] = self.feedback_data[self.feedback_type][self.feedback_mode][
                            self.feedback_freq
                        ]
                    else:
                        existing_feedback_data[self.feedback_type][self.feedback_mode].update(
                            self.feedback_data[self.feedback_type][self.feedback_mode]
                        )
                else:
                    existing_feedback_data[self.feedback_type].update(
                        self.feedback_data[self.feedback_type]
                    )
            else:
                existing_feedback_data.update(self.feedback_data)

            with open(feedback_path, "w+") as outfile:
                json.dump(existing_feedback_data, outfile)
        else:
            with open(feedback_path, "w+") as outfile:
                json.dump(self.feedback_data, outfile)


class DirectionFeedback(Feedback):
    def __init__(self, *args):
        super().__init__(*args)

    def _get_agent_coordinates(self, current_episode, current_step):
        agent_x = self.episode_data["agent_positions"][current_episode][current_step][0]
        agent_y = self.episode_data["agent_positions"][current_episode][current_step][1]
        return agent_x, agent_y

    def _get_goal_coordinates(self, current_episode, current_step):
        goal_x = self.episode_data["goal_positions"][current_episode][current_step][0]
        goal_y = self.episode_data["goal_positions"][current_episode][current_step][1]
        return int(goal_x), int(goal_y)

    def _get_relative_goal_position(self, current_episode, current_step):
        north = False
        east = False
        south = False
        west = False

        goal_x, goal_y = self._get_goal_coordinates(current_episode, current_step)
        agent_x, agent_y = self._get_agent_coordinates(current_episode, current_step)

        if goal_x > agent_x:
            east = True
        if goal_y > agent_y:
            south = True
        if goal_x < agent_x:
            west = True
        if goal_y < agent_y:
            north = True

        # Ids of directions is based on direction encodings:
        # AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"} or
        # AGENT_DIR_TO_STR = {0: "east", 1: "south", 2: "west", 3: "north"}
        return [east, south, west, north]

    def _get_offsets(self, current_episode, current_step):
        goal_x, goal_y = self._get_goal_coordinates(current_episode, current_step)
        agent_x, agent_y = self._get_agent_coordinates(current_episode, current_step)
        x_offset = abs(agent_x - goal_x)
        y_offset = abs(agent_y - goal_y)
        return x_offset, y_offset

    def _facing_goal(self, direction_observation, relative_goal_position, x_offset, y_offset):
        if direction_observation in np.where(relative_goal_position)[0]:
            # AGENT_DIR_TO_STR = {0: "east", 1: "south", 2: "west", 3: "north"}
            if len(np.where(relative_goal_position)[0]) > 1:
                if x_offset >= y_offset and direction_observation in [0, 2]:
                    return True
                if y_offset >= x_offset and direction_observation in [1, 3]:
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False

    def _get_polarity(self, direction_observation, current_episode, current_step):
        relative_goal_position = self._get_relative_goal_position(
            current_episode, current_step
        )
        x_offset, y_offset = self._get_offsets(current_episode, current_step)
        if self._facing_goal(
            direction_observation, relative_goal_position, x_offset, y_offset
        ):
            return "positive"
        else:
            return "negative"

    def generate_feedback(self):
        return super().generate_feedback("direction_observations")


class DistanceFeedback(Feedback):
    def __init__(self, *args):
        super().__init__(*args)
        self.previous_agent_position = None

    def _save_previous_agent_position(self, agent_position):
        self.previous_agent_position = agent_position

    def _get_distance(self, goal_position, agent_position):
        return cdist(
            np.reshape(goal_position, (1, 2)),
            np.reshape(agent_position, (1, 2)),
            metric="cityblock",
        )[0][0]

    def _get_polarity(self, agent_position, current_episode, current_step):
        goal_position = self.episode_data["goal_positions"][current_episode][current_step]
        current_agent_position = agent_position
        d_current = self._get_distance(goal_position, current_agent_position)
        d_previous = self._get_distance(goal_position, self.previous_agent_position)
        self.previous_agent_position = current_agent_position
        if d_current < d_previous:
            return "positive"
        else:
            return "negative"

    def generate_feedback(self):
        return super().generate_feedback("agent_positions")


class ActionFeedback(Feedback):
    def __init__(self, *args):
        super().__init__(*args)
        self.valid_actions = np.arange(
            0, 3, dtype=int
        )  # not sure if done is actually needed?

    def _get_polarity(self, action, current_episode, current_step):
        if action in self.valid_actions:
            return "positive"
        else:
            return "negative"

    def generate_feedback(self):
        return super().generate_feedback("actions")


class AdjacencyFeedback(Feedback):
    def __init__(self, *args):
        super().__init__(*args)
        self.adjacent_object = None
        self.adjacent_object_color = None
        self.adjacent_object_type = None
        # self.facing_adjacent_object = False

    def _get_agent_coordinates(self, current_episode, current_step):
        agent_x = self.episode_data["agent_positions"][current_episode][current_step][0]
        agent_y = self.episode_data["agent_positions"][current_episode][current_step][1]
        return agent_x, agent_y

    def _next_to_agent(self, x, agent_x, y, agent_y):
        if agent_y == y and abs(agent_x - x) == 1:
            return True
        if agent_x == x and abs(agent_y - y) == 1:
            return True
        else:
            return False

    # def _get_relative_position_to_agent(self, x, agent_x, y, agent_y):
    #     if agent_x > x:
    #         return "<"
    #     if agent_x < x:
    #         return ">"
    #     if agent_y > y:
    #         return "^"
    #     if agent_y < y:
    #         return "V"

    # def _agent_facing_object(self, relative_object_position, agent_direction):
    #     print(
    #         f"The agent faces this way {AGENT_DIR_TO_STR[agent_direction]} and the object is located this way {relative_object_position}"
    #     )
    #     if relative_object_position == AGENT_DIR_TO_STR[agent_direction]:
    #         return True
    #     else:
    #         return False

    def _get_adjacent_objects(self, observation, current_episode, current_step):
        agent_x, agent_y = self._get_agent_coordinates(current_episode, current_step)
        # agent_direction = self.episode_data["direction_observations"][current_episode][
        #     current_step
        # ]
        adjacent_objects = []
        for x, row in enumerate(observation):
            for y, object in enumerate(row):
                # Check if there is a key, ball or box in this position
                if object[0] in [5, 6, 7]:
                    if self._next_to_agent(x, agent_x, y, agent_y):
                        adjacent_objects.append(object)
                        # # Check if the object is either above, below, left or right of the agent
                        # relative_object_position = self._get_relative_position_to_agent(
                        #     x, agent_x, y, agent_y
                        # )
                        # if self._agent_facing_object(
                        #     relative_object_position, agent_direction
                        # ):
                        #     self.facing_adjacent_object = True
        return adjacent_objects

    def _same_color(self, object):
        object_color = COLOR_TO_STR[object[1]]
        if self.goal_color == object_color:
            return True
        else:
            return False

    def _same_type(self, object):
        object_type = OBJECT_TO_STR[object[0]]
        if self.goal_type == object_type:
            return True
        else:
            return False

    def _set_adjacent_object(self, object):
        self.adjacent_object = object
        self.adjacent_object_color = COLOR_TO_STR[object[1]]
        self.adjacent_object_type = OBJECT_TO_STR[object[0]]

    def _get_polarity(self, observation, current_episode, current_step):
        adjacent_objects = self._get_adjacent_objects(
            observation, current_episode, current_step
        )
        self.adjacent_object = None
        found_goal = False
        polarity = None
        for object in adjacent_objects:
            if found_goal:
                continue
            if self._same_type(object) and self._same_color(object):
                self._set_adjacent_object(object)
                polarity = "positive_next_to_goal"
                found_goal = True
            elif self._same_type(object) and not self._same_color(object):
                self._set_adjacent_object(object)
                polarity = "positive_same_type"
            elif self._same_color(object) and not self._same_type(object):
                self._set_adjacent_object(object)
                polarity = "positive_same_color"
            elif not self._same_color(object) and not self._same_type(object):
                self._set_adjacent_object(object)
                polarity = "negative_no_shared_attributes"
        if self.adjacent_object is None:
            polarity = "negative_no_adjacent_object"
        return polarity

    def generate_feedback(self):
        return super().generate_feedback("symbolic_observations")


def _feedback_contains_config(feedback, args):
    if args["feedback_type"] in feedback:
        if args["feedback_mode"] in feedback[args["feedback_type"]]:
            if (
                f"{args['feedback_freq_type']}_{args['feedback_freq_steps']}"
                in feedback[args["feedback_type"]][args["feedback_mode"]]
            ):
                return True
    return False


def _get_feedback_with_config(feedback, args):
    return feedback[args["feedback_type"]][args["feedback_mode"]][
        f"{args['feedback_freq_type']}_{args['feedback_freq_steps']}"
    ]


def get_feedback(args, dataset):
    generator = {
        "direction": DirectionFeedback,
        "distance": DistanceFeedback,
        "action": ActionFeedback,
        "adjacency": AdjacencyFeedback,
    }[args["feedback_type"]](args, dataset)

    fn = os.path.join(FEEDBACK_DIR, f"{name_dataset(args)}.json")
    should_generate = True

    # if we don't already have feedback for this dataset, generate it
    if os.path.exists(fn):
        log("found existing feedback file for this dataset, loading...")
        with open(fn) as f:
            feedback = json.load(f)
            should_generate = not _feedback_contains_config(feedback, args)

    if should_generate:
        log("generating feedback...")
        feedback = generator.generate_feedback()
        generator.save_feedback()

    return _get_feedback_with_config(feedback, args)


if __name__ == "__main__":
    args = get_args()
    dataset = get_dataset(args)
    feedback = get_feedback(args, dataset)
