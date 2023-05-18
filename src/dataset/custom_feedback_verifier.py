from minigrid.core.world_object import Door, Key, Wall


class RuleFeedback:
    """
    Class for generating feedback for actions on MiniGrid environments.
    """

    def __init__(self, env, action):
        """
        Initialize the Feedback object.

        Parameters
        ---------
        env : MiniGridEnv
            The environment after the agent has taken a given action. MiniGridEnv is a subclass of gym.Env.
        action : int
            The action taken by the agent.
        """
        # TO-DO: Implement sequence instruction feedback
        self.instr_type = type(env)
        self.env = env
        self.action = action
        # Get the position in front of the agent
        self.front_cell = env.grid.get(*env.front_pos)
        # Get the object carried by the agent (if any)
        self.carrying = self.env.carrying
        print(self.instr_type)
        # access description properties (e.g. desc) and methods (e.g. verify())
        # in sequence tasks using instrs.instr_a / instrs.instr_b
        # check if BeforeInstr / AfterInstr sequence tasks include nested AndInstr's
        # and use instrs.instr_a.instr_a ... if this is the case

        # PutNext instructions don't have an attribute desc - use desc_move and desc_fixed

    def verify_feedback(self):
        """
        Return the feedback for the action taken by the agent.

        Returns
        -------
        str
            The feedback for the action taken by the agent. This is either the rule violation feedback or the task feedback, or an empty string if no feedback is triggered by the agents action.
        """
        return self._get_rule_feedback()

    def _is_empty_cell(self):
        """
        Check if the agent is positioned in front of an empty cell.

        Returns
        -------
        bool
            True if the agent is positioned in front of an empty cell, False otherwise.
        """
        if self.front_cell is None:
            return True
        else:
            return False

    def _is_door(self):
        """
        Check if the agent is positioned in front of a door.

        Returns
        -------
        bool
            True if the agent is positioned in front of a door, False otherwise.
        """
        if isinstance(self.front_cell, Door):
            return True
        else:
            return False

    def _is_open_door(self):
        """
        Check if the agent is positioned in front of an open door.

        Returns
        -------
        bool
            True if the agent is positioned in front of an open door, False otherwise.
        """
        if self._is_door() and self.front_cell.is_open:
            return True
        else:
            return False

    def _is_closed_door(self):
        """
        Check if the agent is positioned in front of a closed door.

        Returns
        -------
        bool
            True if the agent is positioned in front of an cloed door, False otherwise.
        """
        if self._is_door():
            if not self.front_cell.is_open:
                return True
        return False

    def _is_locked_door(self):
        """
        Check if the agent is positioned in front of a locked door.

        Returns
        -------
        bool
            True if the agent is positioned in front of a locked door, False otherwise.
        """
        if self._is_door():
            if self.front_cell.is_locked:
                return True
        return False

    def _is_wall(self):
        """
        Check if the agent is positioned in front of a wall.

        Returns
        -------
        bool
            True if the agent is positioned in front of a wall, False otherwise.
        """
        return isinstance(self.front_cell, Wall)

    def _is_obstacle(self):
        """
        Check if the .

        Returns
        -------
        bool
            True if the agent can move forward, False otherwise.
        """
        if not self.front_cell.can_overlap() and not (
            self._is_closed_door() or self._is_locked_door() or self._is_wall()
        ):
            return True
        return False

    def _is_carrying(self):
        """
        Check if the agent is carrying an object.

        Returns
        -------
        bool
            True if the agent is carrying an object, False otherwise.
        """
        return self.carrying is not None

    def _is_carrying_correct_key(self):
        """
        Check if the agent is carrying a correct key to unlock the door it is positioned in front of.

        Returns
        -------
        bool
            True if the agent is carrying a correct key (of the same color as the door), False otherwise.
        """
        return (
            isinstance(self.carrying, Key)
            and self.carrying.color == self.front_cell.color
        )

    def _is_valid_move_forward(self):
        """
        Check if the agent can move forward.

        Returns
        -------
        bool
            True if the agent can move forward, False otherwise.
        """
        return self._is_empty_cell() or self._is_open_door()

    def _get_move_forward_feedback(self):
        """
        Return the feedback for the move forward action.

        Returns
        -------
        str
            The feedback for the move forward action with respect to the object in the cell that the agent is facing.
        """
        if self._is_closed_door():
            return "You can't move forward here. The door in front of you is closed."
        if self._is_locked_door():
            return "You can't move forward here. The door in front of you is locked."
        if self._is_wall():
            return "You can't move forward while you're facing the wall."
        if self._is_obstacle():
            return (
                "You can't move forward here. "
                + f"There is an obstacle in the form of a {self.front_cell.type} blocking the way."
            )

    def _is_valid_toggle(self):
        """
        Check if the agent can toggle the object in front of it.

        Returns
        -------
        bool
            True if the agent can toggle the object in front of it, False otherwise."""
        return self.front_cell.toggle()

    def _get_toggle_feedback(self):
        """
        Return the feedback for the toggle action.

        Returns
        -------
        str
            The feedback for the toggle action with respect to the object in the cell that the agent is facing.
        """
        if self._is_empty_cell():
            return "There is nothing to toggle in front of you."
        if self._is_open_door():
            return "You can't toggle an already open door."
        if self._is_locked_door() and not self._is_carrying_correct_key():
            return "You can't toggle a locked door without the correct key."
        if self._is_wall():
            return "You can't toggle the wall."
        if self._is_obstacle():
            return f"You can't toggle {self.front_cell.type}s"

    def _is_valid_pickup(self):
        """
        Check if the agent can pick up the object in front of it.

        Returns
        -------
        bool
            True if the agent can pick up the object in front of it, False otherwise.
        """
        return self.front_cell.can_pickup()

    def _get_pickup_feedback(self):
        """
        Return the feedback for the pickup action.

        Returns
        -------
        str
            The feedback for the pickup action with respect to the object in the cell that the agent is facing.
        """
        if self._is_empty_cell():
            return "There is nothing to pick up in front of you."
        if self._is_door():
            return "You can't pick up a door."
        if self._is_wall():
            return "You can't pick up the wall."
        if self._is_carrying():
            return "You can't pick up another object while you're already carrying one."

    def _is_valid_drop(self):
        """
        Check if the agent can drop an object it is carrying.

        Returns
        -------
        bool
            True if the agent can drop an object it is carrying, False otherwise.
        """
        return self._is_carrying() and self._is_empty_cell()

    def _get_drop_feedback(self):
        """
        Return the feedback for the drop action.

        Returns
        -------
        str
            The feedback for the drop action with respect to the object the agent is carrying and the cell that the agent is facing.
        """
        if not self._is_carrying():
            return "You can't drop an object while you're not carrying anything."
        if self._is_wall():
            return "You can't drop an object while you're facing the wall."
        if self._is_door():
            return "You can't drop an object while you're facing a door."
        if self._is_obstacle():
            return (
                "You can't drop an object in front of you. "
                + f"There is already a {self.front_cell.type} there."
            )

    def _get_rule_feedback(self):
        """
        Return the rule violation feedback for the action taken by the agent.

        Returns
        -------
        str
            The feedback for the action taken by the agent.
        """
        if (
            self.action == self.env.actions.forward
            and not self._is_valid_move_forward()
        ):
            return self._get_move_forward_feedback()
        if self.action == self.env.actions.toggle and not self._is_valid_toggle():
            return self._get_toggle_feedback()
        if self.action == self.env.actions.pickup and not self._is_valid_pickup():
            return self._get_pickup_feedback()
        if self.action == self.env.actions.drop and not self._is_valid_drop():
            return self._get_drop_feedback()
        return ""
