import json
import os
import random
from itertools import combinations_with_replacement

import gymnasium as gym
import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.envs.babyai.core.verifier import LOC_NAMES, OBJ_TYPES_NOT_DOOR, OpenInstr
from tqdm import tqdm

from src.dataset.custom_feedback_verifier import TaskFeedback

ENVS_CONFIGS = {
    "GoToRedBallGrey": ["BabyAI-GoToRedBallGrey-v0"],
    "GoToRedBall": ["BabyAI-GoToRedBall-v0"],
    "GoToRedBallNoDists": ["BabyAI-GoToRedBallNoDists-v0"],
    "GoToObj": ["BabyAI-GoToObj-v0", "BabyAI-GoToObjS4-v0"],
    "GoToLocal": [
        "BabyAI-GoToLocal-v0",
        "BabyAI-GoToLocalS5N2-v0",
        "BabyAI-GoToLocalS6N2-v0",
        "BabyAI-GoToLocalS6N3-v0",
        "BabyAI-GoToLocalS6N4-v0",
        "BabyAI-GoToLocalS7N4-v0",
        "BabyAI-GoToLocalS7N5-v0",
        "BabyAI-GoToLocalS8N2-v0",
        "BabyAI-GoToLocalS8N3-v0",
        "BabyAI-GoToLocalS8N4-v0",
        "BabyAI-GoToLocalS8N5-v0",
        "BabyAI-GoToLocalS8N6-v0",
        "BabyAI-GoToLocalS8N7-v0",
    ],
    "GoTo": [
        "BabyAI-GoTo-v0",
        "BabyAI-GoToOpen-v0",
        "BabyAI-GoToObjMaze-v0",
        "BabyAI-GoToObjMazeOpen-v0",
        "BabyAI-GoToObjMazeS4R2-v0",
        "BabyAI-GoToObjMazeS4-v0",
        "BabyAI-GoToObjMazeS5-v0",
        "BabyAI-GoToObjMazeS6-v0",
        "BabyAI-GoToObjMazeS7-v0",
    ],
    "GoToImpUnlock": ["BabyAI-GoToImpUnlock-v0"],
    "GoToSeq": ["BabyAI-GoToSeq-v0", "BabyAI-GoToSeqS5R2-v0"],
    "GoToRedBlueBall": ["BabyAI-GoToRedBlueBall-v0"],
    "GoToDoor": ["BabyAI-GoToDoor-v0"],
    "GoToObjDoor": ["BabyAI-GoToObjDoor-v0"],
    "Open": ["BabyAI-Open-v0"],
    "OpenDoor": [
        "BabyAI-OpenDoor-v0",
        "BabyAI-OpenDoorColor-v0",
        "BabyAI-OpenDoorLoc-v0",
    ],
    "OpenTwoDoors": ["BabyAI-OpenTwoDoors-v0", "BabyAI-OpenRedBlueDoors-v0"],
    "OpenDoorsOrder": ["BabyAI-OpenDoorsOrderN2-v0", "BabyAI-OpenDoorsOrderN4-v0"],
    "Pickup": ["BabyAI-Pickup-v0"],
    "UnblockPickup": ["BabyAI-UnblockPickup-v0"],
    "PickupLoc": ["BabyAI-PickupLoc-v0"],
    "PickuDist": ["BabyAI-PickupDist-v0"],
    "PickupAbove": ["BabyAI-PickupAbove-v0"],
    "PutNextLocal": [
        "BabyAI-PutNextLocal-v0",
        "BabyAI-PutNextLocalS5N3-v0",
        "BabyAI-PutNextLocalS6N4-v0",
    ],
    "PutNext": [
        "BabyAI-PutNextS4N1-v0",
        "BabyAI-PutNextS5N1-v0",
        "BabyAI-PutNextS5N2-v0",
        "BabyAI-PutNextS6N3-v0",
        "BabyAI-PutNextS7N4-v0",
    ],
    "Unlock": ["BabyAI-Unlock-v0"],
    "UnlockLocal": ["BabyAI-UnlockLocal-v0", "BabyAI-UnlockLocalDist-v0"],
    "KeyInBox": ["BabyAI-KeyInBox-v0"],
    "UnlockPickup": ["BabyAI-UnlockPickup-v0", "BabyAI-UnlockPickupDist-v0"],
    "BlockedUnlockPickup": ["BabyAI-BlockedUnlockPickup-v0"],
    "UnlockToUnlock": ["BabyAI-UnlockToUnlock-v0"],
    "ActionObjDoor": ["BabyAI-ActionObjDoor-v0"],
    "FindObj": ["BabyAI-FindObjS5-v0", "BabyAI-FindObjS6-v0", "BabyAI-FindObjS7-v0"],
    "KeyCorridor": [
        "BabyAI-KeyCorridor-v0",
        "BabyAI-KeyCorridorS3R1-v0",
        "BabyAI-KeyCorridorS3R2-v0",
        "BabyAI-KeyCorridorS3R3-v0",
        "BabyAI-KeyCorridorS4R3-v0",
        "BabyAI-KeyCorridorS5R3-v0",
    ],
    "OneRoom": [
        "BabyAI-OneRoomS8-v0",
        "BabyAI-OneRoomS12-v0",
        "BabyAI-OneRoomS16-v0",
        "BabyAI-OneRoomS20-v0",
    ],
    "MoveTwoAcross": ["BabyAI-MoveTwoAcrossS5N2-v0", "BabyAI-MoveTwoAcrossS8N9-v0"],
    "Synth": [
        "BabyAI-Synth-v0",
    ],
    "SynthLoc": ["BabyAI-SynthLoc-v0"],
    "SynthSeq": ["BabyAI-SynthSeq-v0"],
    "MiniBossLevel": ["BabyAI-MiniBossLevel-v0"],
    "BossLevel": ["BabyAI-BossLevel-v0"],
    "BossLevelNoUnlock": ["BabyAI-BossLevelNoUnlock-v0"],
}


class SeedFinder:
    """
    Class to find in-domain and ood seeds for a given environment and config.

    In some cases, this involves all seeds for a given environment or config.
    In other cases, this involves a subset of seeds for a given environment or config.
    """

    def __init__(self):
        """
        Initialise the SeedFinder class.
        """
        random.seed(42)
        self.random_colors = self._pick_random_colors()
        self.random_types = self._pick_random_types()
        self.random_rel_loc = self._pick_random_rel_location()
        self.in_domain_room_size_min = 6
        self.in_domain_room_size_max = 8
        self.in_domain_num_cols = 3
        self.in_domain_num_rows = 3
        self.random_room_quadrant = self._pick_random_room_quadrant()
        self.random_maze_quadrant = self._pick_random_maze_quadrant()
        self.ood_types = {
            "size": self._check_size,
            "color_type": self._check_color_type,
            "agent_loc": self._check_agent_loc,
            "rel_loc": self._check_rel_loc,
            "object_task": self._check_object_task,
            "task_task": self._check_task_task,
        }
        self.in_domain_filename = "in_domain_seeds.json"
        self.ood_filename = "ood_seeds.json"

    def _pick_random_colors(self):
        """
        Pick a random color from the list of colors.

        Parameters
        ----------
            i (int): index of the color to pick.

        Returns
        -------
            str: a random object color from a list of colors.
        """
        return random.sample(COLOR_NAMES, 2)

    def _pick_random_types(self):
        """
        Pick a random type from the list of types.

        Parameters
        ----------
            i (int): index of the type to pick.

        Returns
        -------
            str: a random object type from a list of types.
        """
        return random.sample(OBJ_TYPES_NOT_DOOR, 2)

    def _pick_random_rel_location(self):
        """
        Pick a random relative goal location from the list of possible descriptions.

        This applies to Loc environments only.

        Returns
        -------
            str: a random relative goal object location.
        """
        return random.sample(LOC_NAMES, 1)[0]

    def _get_task_list(self, env):
        """
        Get the list of mission tasks for a given environment.

        If this is a Sequence type environment, then the list of tasks is a list of up to four subtasks.

        If this is not a Sequence type environment, then the list of tasks consists of a single task.

        Parameters
        ----------
            env: instance of the environment made using a seed.

        Returns
        -------
            list: list of mission tasks.
        """
        return TaskFeedback(env).tasks

    def _is_maze(self, env):
        """
        Check if the environment is a maze.

        Parameters
        ----------
            env: instance of the environment made using a seed.

        Returns
        -------
            bool: True if the environment is a maze, False otherwise.
        """
        return env.num_rows > 1 or env.num_cols > 1

    def _get_all_positions(self, size, n):
        """
        Get all possible positions in the grid for a given axis.

        Parameters
        ----------
            env: instance of the environment made using a seed.
            size (int): size of the room.
            n (int): number of rows or columns in the grid.

        Returns
        -------
            list: list of tuples of x or y coordinates of all possible positions in the grid.
        """
        all_positions = [c for c in range(1, size * n - n) if c % (size - 1) != 0]
        return np.asarray(all_positions)

    def _split_positions_by_room(self, positions, n):
        """
        Split an array of positions into arrays of quadrants.

        Parameters
        ----------
            positions (np.array): array of x or y coordinates of all possible positions in the grid.
            n (int): number of rows or columns in the grid.

        Returns
        -------
            array: array of arrays of x or y coordinates of all possible positions in the grid, by room.
        """
        return np.array_split(
            np.asarray(positions),
            n,
        )

    def _split_rooms_into_quadrants(self, room_positions):
        """
        Split an array of room positions into arrays of quadrants.

        Parameters
        ----------
            room_positions (np.array): array of x or y coordinates of all possible positions in the grid, by room.

        Returns
        -------
            array: array of arrays of x or y coordinates of all possible positions in the grid, by room quadrant.
        """
        return [np.array_split(x, 2) for x in room_positions]

    def _get_quadrant_lookup(self, quadrant_positions):
        """
        Get a lookup of quadrants to positions.

        Parameters
        ----------
            quadrant_positions (list): list of arrays of x or y coordinates of all possible positions in the grid, by room quadrant.

        Returns
        -------
            dict: dict of quadrants to positions.
        """
        quadrant_lookup = {}
        for _, quadrants in enumerate(quadrant_positions):
            for quadrant, positions in enumerate(quadrants):
                if quadrant + 1 not in quadrant_lookup:
                    quadrant_lookup[quadrant + 1] = list(positions)
                else:
                    quadrant_lookup[quadrant + 1].extend(list(positions))
        return quadrant_lookup

    def _agent_pos_to_room_quadrant(self, env):
        """
        Convert the agent position into a quadrant in the room.

        Parameters
        ----------
            env: instance of the environment made using a seed.

        Returns
        -------
            tuple: tuple of x and y coordinates of the quadrant of the room.

        """
        all_positions_x = self._get_all_positions(env.room_size, env.num_cols)
        all_positions_y = self._get_all_positions(env.room_size, env.num_rows)

        room_positions_x = self._split_positions_by_room(all_positions_x, env.num_cols)
        room_positions_y = self._split_positions_by_room(all_positions_y, env.num_rows)

        quadrant_positions_x = self._split_rooms_into_quadrants(room_positions_x)
        quadrant_positions_y = self._split_rooms_into_quadrants(room_positions_y)

        for x_quadrant, x_positions in self._get_quadrant_lookup(
            quadrant_positions_x
        ).items():
            for y_quadrant, y_positions in self._get_quadrant_lookup(
                quadrant_positions_y
            ).items():
                if env.agent_pos[0] in x_positions and env.agent_pos[1] in y_positions:
                    return x_quadrant, y_quadrant

    def _agent_pos_to_maze_quadrant(self, env):
        """
        Convert the agent position into a quadrant in the maze.

        Parameters
        ----------
            env: instance of the environment made using a seed.

        Returns
        -------
            tuple: tuple of x and y coordinates of the quadrant of the maze.
        """

        all_positions_x = self._get_all_positions(env.room_size, env.num_cols)
        all_positions_y = self._get_all_positions(env.room_size, env.num_rows)

        room_positions_x = self._split_positions_by_room(all_positions_x, env.num_cols)
        room_positions_y = self._split_positions_by_room(all_positions_y, env.num_rows)

        for x_room, x_positions in enumerate(room_positions_x):
            for y_room, y_positions in enumerate(room_positions_y):
                if env.agent_pos[0] in x_positions and env.agent_pos[1] in y_positions:
                    return x_room, y_room

    def _get_agent_quadrants(self, env):
        """
        Get the room and maze quadrants of the agent in the grid.

        Parameters
        ----------
            env: instance of the environment made using a seed.

        Returns
        -------
            tuple: tuple of tuple of x and y quadrant of the agent in the room and maze.
        """
        if self._is_maze(env):
            return self._agent_pos_to_room_quadrant(
                env
            ), self._agent_pos_to_maze_quadrant(env)
        return self._agent_pos_to_room_quadrant(env), None

    def _get_possible_room_quadrants(self):
        """
        Get a list of possible room grid quadrants.

        Parameters
        ----------
            env: instance of the environment made using a seed.

        Returns
        -------
            list: list of tuples of x and y coordinates for all possible quadrants of the room.
        """
        return [q for q in combinations_with_replacement([1, 2], 2)]

    def _get_possible_maze_quadrants(self):
        """
        Get a list of possible maze grid quadrants.

        Parameters
        ----------
            env: instance of the environment made using a seed.

        Returns
        -------
            list: list of tuples of x and y coordinates for all possible quadrants of the maze.
        """
        return [
            q
            for q in combinations_with_replacement(
                [self.in_domain_num_cols, self.in_domain_num_rows], 2
            )
        ]

    def _pick_random_room_quadrant(self):
        """
        Pick a random room quadrant from a list of possible room quadrants.

        Returns
        -------
            tuple: tuple of x and y coordinates for a randomly picked quadrant of the room.
        """
        return random.sample(
            self._get_possible_room_quadrants(),
            counts=[100] * len(self._get_possible_room_quadrants()),
            k=1,
        )[0]

    def _pick_random_maze_quadrant(self):
        """
        Pick a random maze quadrant from a list of possible maze quadrants.

        Returns
        -------
            tuple: tuple of x and y coordinates for a randomly picked quadrant of the maze.
        """
        return random.sample(
            self._get_possible_maze_quadrants(),
            counts=[100] * len(self._get_possible_maze_quadrants()),
            k=1,
        )[0]

    def _goal_doors_locked(self, env, task):
        """
        Check if the goal doors are locked.

        Parameters
        ----------
            env: instance of the environment made using a seed.
            task: instance of the subtask for the env that is to be checked.

        Returns
        -------
            bool: True if any of the goal doors are locked, False otherwise.
        """
        for pos in task.desc.obj_poss:
            cell = env.grid.get(*pos)
            if cell and cell.type == "door" and cell.is_locked:
                return True
        return False

    def _has_two_doors_unlock(self, env):
        """
        Check if the task explicitly requires unlocking two doors.

        Parameters
        ----------
            env: instance of the environment made using a seed.

        Returns
        -------
            bool: True if the environment has two goal doors to unlock, False otherwise.
        """
        if len(self._get_task_list(env)) > 1:
            door_count = 0
            for task in self._get_task_list(env):
                try:
                    if (
                        isinstance(task, OpenInstr)
                        and task.desc.type == "door"
                        and self._goal_doors_locked(env, task)
                    ):
                        door_count += 1
                except AttributeError:
                    continue
            if door_count > 1:
                return True
        return False

    def _check_size(self, env):
        """
        Check if the environment contains the unseen size of the room or maze.

        Parameters
        ----------
            env: instance of the environment made using a seed.

        Returns
        -------
            bool: True if the room size is 8 and maze size is 3x3, False otherwise.
        """
        if self._is_maze(env):
            return (
                env.room_size < self.in_domain_room_size_min
                or env.room_size > self.in_domain_room_size_max
                or env.num_cols != 3
                or env.num_rows != 3
            )
        return (
            env.room_size < self.in_domain_room_size_min
            or env.room_size > self.in_domain_room_size_max
        )

    def _check_color_type(self, env):
        """
        Check if the environment contains the unseen color-type combinations for goal objects.

        Parameters
        ----------
            env: instance of the environment made using a seed.

        Returns
        -------
            bool: True if (any of) the goal object/s is/are of the unseen color and type, False otherwise.
        """
        for task in self._get_task_list(env):
            try:
                if (
                    task.desc.color == self.random_colors[0]
                    and task.desc.type == self.random_types[0]
                ):
                    return True
            except AttributeError:
                if (
                    task.desc_move.color == self.random_colors[0]
                    and task.desc_move.type == self.random_types[0]
                ) or (
                    task.desc_fixed.color == self.random_colors[0]
                    and task.desc_fixed.type == self.random_types[0]
                ):
                    return True
        return False

    def _check_agent_loc(self, env):
        """
        Check if the environment contains an agent starting position in an unseen room (and for mazes, maze) quadrant(s).

        Parameters
        ----------
            env: instance of the environment made using a seed.

        Returns
        -------
            bool: True if agent starting location is in the unseed quadrant in the room (single room) and maze (for maze only), False otherwise.
        """
        room_quadrant, maze_quadrant = self._get_agent_quadrants(env)
        if room_quadrant == self.random_room_quadrant or (
            maze_quadrant and maze_quadrant == self.random_maze_quadrant
        ):
            return True
        return False

    def _check_object_task(self, env):
        """
        Check if the environment contains the unseen object-task combination.

        The object color-type combination is different from the one used in self._check_color_type.

        Parameters
        ----------
            env: instance of the environment made using a seed.

        Returns
        -------
            bool: True if the obj.desc_fixed for a PutNext task is of the second unseen color and type, False otherwise.
        """
        for task in self._get_task_list(env):
            try:
                if (
                    task.desc_fixed.color == self.random_colors[1]
                    and task.desc_fixed.type == self.random_types[1]
                ):
                    return True
            except AttributeError:
                continue
        return False

    def _check_rel_loc(self, env):
        """
        Check if the environment contains the unseen goal location description.

        This applies to Loc environments only and refers to the loc attribute of goal objects, which for most envs is None.

        Parameters
        ----------
            env (str): instance of the environment made using a seed.

        Returns
        -------
            bool: True if (any of) the goal object/s is/are of the unseen loc, False otherwise.
        """
        for task in self._get_task_list(env):
            try:
                if task.desc.loc and task.desc.loc == self.random_rel_loc:
                    return True
            except AttributeError:
                if (
                    task.desc_move.loc and task.desc_move.loc == self.random_rel_loc
                ) or (task.desc_fixed and task.desc_fixed.loc == self.random_rel_loc):
                    return True
        return False

    def _check_task_task(self, env):
        """
        Check if the environment contains the unseen task-task combination.

        Parameters
        ----------
            env: instance of the environment made using a seed.

        Returns
        -------
            bool: True if SeqInstr which involves opening two doors, False otherwise.
        """
        return self._has_two_doors_unlock(env)

    def _find_in_domain_seeds(self, config):
        """
        Find in-domain seeds for the given environment and config.

        Seeds are in-domain with respect all of the below;
        - size: seeds that involve an seen room or maze size.
        - color-type: seeds that involve an seen combination of color and type of object.
        - agent-loc: seeds that involve an seen agent start position in the room or maze.
        - object-task: seeds that involve an seen combination of object and task.
        - rel-loc: seeds that involve an seen goal location relative to the agent.
        - task-task: seeds (and envs/configs) that involve seen task combinations in sequence tasks.

        Parameters
        ----------
            config (str): environment configuration name.

        Returns
        -------
            list: list of seeds for a given environment configuration.
        """

        n_seeds = 100
        seed_list = []
        for seed in tqdm(range(0, 1000)):
            env = gym.make(config)
            try:
                env.reset(seed=seed)
                _ = env.instrs.surface(env)
            except:
                continue
            if (
                not self._check_size(env)
                and not self._check_color_type(env)
                and not self._check_object_task(env)
                and not self._check_agent_loc(env)
                and not self._check_rel_loc(env)
                and not self._check_task_task(env)
            ):
                seed_list.append(seed)
                if len(seed_list) == n_seeds:
                    break

        return seed_list

    def _find_ood_seeds(self, config):
        """
        Find in-domain and ood seeds for the given environment and config.

        Seeds are out-of-domain with respect any or all of the below:
        - size: seeds that involve an unseen room or maze size.
        - color-type: seeds that involve an unseen combination of color and type of object.
        - object-task: seeds that involve an unseen combination of object and task.
        - agent-loc: seeds that involve an unseen agent start position in the room or maze.
        - rel-loc: seeds that involve an unseen goal location relative to the agent.
        - task-task: seeds (and envs/configs) that involve unseen task combinations in sequence tasks.

        Parameters
        ----------
            config (str): environment configuration name.
            num_configs (int): number of configs that exist for a given environment.

        Returns
        -------
            list: list of seeds for a given environment configuration.
        """
        seed_lists = {ood_type: [] for ood_type in self.ood_types.keys()}
        n_seeds = 10
        for ood_type, seed_list in seed_lists.items():
            if ood_type in ["object_task", "task_task"]:
                max_seeds_to_check = 1000
            else:
                max_seeds_to_check = 100
            for seed in tqdm(range(0, 10000)):
                if seed >= max_seeds_to_check and len(seed_list) == 0:
                    break
                env = gym.make(config)
                try:
                    env.reset(seed=seed)
                    _ = env.instrs.surface(env)
                except:
                    continue
                if self.ood_types[ood_type](env) and len(seed_list) < n_seeds:
                    if ood_type == "agent_loc" and self.ood_types["size"](env):
                        continue
                    seed_list.append(seed)
        return seed_lists

    def save_in_domain_seeds(self):
        """
        Find and save in domain seeds for all (missing) environments and configs.

        """

        if os.path.exists(self.in_domain_filename):
            seed_dict = json.load(open(self.in_domain_filename, "r+"))
        else:
            seed_dict = {}
        for env, configs in ENVS_CONFIGS.items():
            if env not in seed_dict.keys():
                seed_dict[env] = {}
            for config in configs:
                if config not in seed_dict[env].keys():
                    seed_dict[env][config] = self._find_in_domain_seeds(config)
                    json.dump(seed_dict, open(self.in_domain_filename, "w"))

    def save_ood_seeds(self):
        """
        Find and save ood seeds for all (missing) environments and configs.

        """
        if os.path.exists(self.ood_filename):
            seed_dict = json.load(open(self.ood_filename, "r+"))
        else:
            seed_dict = {}
        for env, configs in ENVS_CONFIGS.items():
            if env not in seed_dict.keys():
                seed_dict[env] = {}
            for config in configs:
                if config not in seed_dict[env].keys():
                    seed_dict[env][config] = self._find_ood_seeds(config)
                    json.dump(seed_dict, open(self.ood_filename, "w"))

    def load_in_domain_seeds(self):
        """
        Get the in-domain and ood seeds for all environments and configs.

        Returns
        -------
            tuple: tuple of dictionary of lists of in-domain and ood seeds.
        """
        if not os.path.exists(self.in_domain_filename):
            self.save_in_domain_seeds()
        return json.load(open(self.in_domain_filename, "r+"))

    def load_ood_seeds(self):
        """
        Get the in-domain and ood seeds for all environments and configs.

        Returns
        -------
            tuple: tuple of dictionary of lists of in-domain and ood seeds.
        """
        if not os.path.exists(self.ood_filename):
            self.save_ood_seeds()
        return json.load(open(self.ood_filename, "r+"))