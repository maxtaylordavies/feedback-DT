class Seeds:
    """
    Class to find in-domain and ood seeds for a given environment and config.

    In some cases, this involves all seeds for a given environment or config.
    In other cases, this involves a subset of seeds for a given environment or config.
    """

    def __init__(self, envs: dict[str, str], ood_type="all"):
        """
        Initialize the class.

        """

        self.envs = envs
        self.ood_type = ood_type

    def _check_color_type(self):
        """
        Check if the environment contains the unseen color-type combinations for goal objects.

        Returns:
            bool: True if 'blue ball', False otherwise.
        """

    def _check_object_task(self):
        """
        Check if the environment contains the unseen object-task combination.

        Returns:
            bool: True if 'yellow box' is the obj.desc_fixed for PutNext tasks, False otherwise.
        """

    def _check_agent_loc(self):
        """
        Check if the environment contains the unseen agent starting location.

        Returns:
            bool: True if agent location is in bottom left corner (for single rooms) or room (for maze), False otherwise.
        """

    def _check_goal_loc(self):
        """
        Check if the environment contains the unseen goal location.

        Returns:
            bool: True if goal location is in top left corner (for single rooms) or room (for maze), False otherwise.
        """

    def _check_task_task(self):
        """
        Check if the environment contains the unseen task-task combination.

        Returns:
            bool: True if SeqInstr which involves opening two doors, or unblocking and unlocking, False otherwise.
        """

    def _find_ood_seeds(self):
        """
        Find in-domain and ood seeds for the given environment and config.

        Depending on the type, ood is defined differently.
        - all: seeds (and envs/configs) that are not in-domain with respect to all of the below.
        - color-type: seeds that involve an unseen combination of color and type of object.
        - object-task: seeds that involve an unseen combination of object and task.
        - agent-loc: seeds that involve an unseen agent start position in the room or maze.
        - goal-loc: seeds that involve an unseen goal location in the room or maze.
        - task-task: seeds (and envs/configs) that involve unseen task combinations.


        Returns:
            list: list of seeds.
        """

    def _find_in_domain_seeds(self):
        """
        Find in-domain seeds for the given environment and config.

        In-domain is defined as seeds that are in-domain with respect to one or all of the below.
        - color-type: seeds that involve a seen combination of color and type of object.
        - object-task: seeds that involve a seen combination of object and task.
        - agent-loc: seeds that involve a seen agent start position in the room or maze.
        - goal-loc: seeds that involve a seen goal location in the room or maze.
        - task-task: seeds (and envs/configs) that involve seen task combinations.


        Returns:
            dict: dict of environment configurations and seeds.
        """

    def find_seeds(self):
        """
        Find in-domain and ood seeds for the given environment and config.

        Returns:
            tuple: tuple of dicts of environment configurations and seeds.
        """
        return self._find_in_domain_seeds(), self._find_ood_seeds()
