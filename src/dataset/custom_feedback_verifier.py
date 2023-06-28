from abc import ABC, abstractmethod

from minigrid.core.world_object import Box, Door, Key, Wall
from minigrid.envs.babyai.core.verifier import (
    AfterInstr,
    AndInstr,
    BeforeInstr,
    GoToInstr,
    ObjDesc,
    OpenInstr,
    PickupInstr,
    PutNextInstr,
    SeqInstr,
    pos_next_to,
)


class Feedback(ABC):
    """
    Super class for generating feedback for actions on BabyAI environments.
    """

    env = None
    action = None
    front_pos = None
    front_cell = None
    carrying = None

    @abstractmethod
    def verify_feedback(self, env, action):
        """
        Verify the feedback for the action taken by the agent.

        Parameters
        ----------
        env : MiniGridEnv
            The environment which to verify an action against. MiniGridEnv is a subclass of gym.Env.
        action : int
            The action to verify.

        Raises
        ------
        NotImplementedError
            Raised when not overriden by a derived class
        """
        raise NotImplementedError

    def _is_empty_cell(self):
        """
        Check if the agent is positioned in front of an empty cell.

        Returns
        -------
        bool
            True if the agent is positioned in front of an empty cell, False otherwise.
        """
        return self.front_cell is None

    def _is_wall(self):
        """
        Check if the agent is positioned in front of a wall.

        Returns
        -------
        bool
            True if the agent is positioned in front of a wall, False otherwise.
        """
        return isinstance(self.front_cell, Wall)

    def _is_door(self):
        """
        Check if the agent is positioned in front of a door.

        Returns
        -------
        bool
            True if the agent is positioned in front of a door, False otherwise.
        """
        return isinstance(self.front_cell, Door)

    def _is_open_door(self):
        """
        Check if the agent is positioned in front of an open door.

        Returns
        -------
        bool
            True if the agent is positioned in front of an open door, False otherwise.
        """

        return self._is_door() and self.front_cell.is_open


class RuleFeedback(Feedback):
    """
    Sub class for generating rule feedback for actions on BabyAI environments.
    """

    def _is_obstacle(self):
        """
        Check if there is an obstacle object in front of the agent.

        Returns
        -------
        bool
            True if the object in front of the agent is an obstacle (other than a closed/locked door or wall), False otherwise.
        """
        return not self.front_cell.can_overlap() and not (
            self._is_closed_door() or self._is_locked_door() or self._is_wall()
        )

    def _is_closed_door(self):
        """
        Check if the agent is positioned in front of a closed door.

        Returns
        -------
        bool
            True if the agent is positioned in front of an cloed door, False otherwise.
        """
        if self._is_door():
            return not self.front_cell.is_open and not self.front_cell.is_locked
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
            return self.front_cell.is_locked
        return False

    def _is_box(self):
        """
        Check if there is a box object in front of the agent.

        Returns
        -------
        bool
            True if the object in front of the agent is a box, False otherwise.
        """
        if isinstance(self.front_cell, Box):
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
        if self._is_locked_door():
            return "You can't move forward here as the door in front of you is locked."
        if self._is_closed_door():
            return "You can't move forward here as the door in front of you is closed."
        if self._is_wall():
            return "You can't move forward while you're facing the wall."
        if self._is_obstacle():
            return (
                "You can't move forward here "
                + f"as there is an obstacle in the form of a {self.front_cell.type} blocking the way."
            )
        return "No feedback available."

    def _is_valid_toggle(self):
        """
        Check if the agent can toggle the object in front of it.

        Returns
        -------
        bool
            True if the agent can toggle the object in front of it, False otherwise."""
        if self.front_cell:
            return (
                (self._is_locked_door() and self._is_carrying_correct_key())
                or self._is_closed_door()
                or self._is_box()
            )
        return False

    def _get_toggle_feedback(self):
        """
        Return the feedback for the toggle action.

        Returns
        -------
        str
            The feedback for the toggle action with respect to the object in the cell that the agent is facing.
        """
        if self._is_empty_cell():
            return "There is nothing to open in front of you."
        if self._is_open_door():
            return "You just closed an already open door."
        if self._is_locked_door() and not self._is_carrying_correct_key():
            return "You can't open a locked door without a key of the same color as the door."
        if self._is_wall():
            return "You can't open the wall."
        if self._is_obstacle():
            return f"You can't open {self.front_cell.type}s."
        return "No feedback available."

    def _is_valid_pickup(self):
        """
        Check if the agent can pick up the object in front of it.

        Returns
        -------
        bool
            True if the agent can pick up the object in front of it, False otherwise.
        """
        if self.front_cell and not self.carrying:
            return self.front_cell.can_pickup()
        return False

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
            return "You can't pick up doors."
        if self._is_wall():
            return "You can't pick up the wall."
        if self._is_carrying():
            return "You can't pick up another object while you're already carrying one."
        return "No feedback available."

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
                "You can't drop an object on top of another object, and "
                + f"there is already a {self.front_cell.type} in front of you."
            )
        return "No feedback available."

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
        return "No feedback available."

    def verify_feedback(self, env, action):
        """
        Verify the feedback for the action taken by the agent.

        Raises
        ------
        NotImplementedError
            Raised when not overriden by a derived class
        """
        self.env = env
        self.front_pos = self.env.front_pos
        self.front_cell = self.env.grid.get(*self.front_pos)
        self.carrying = self.env.carrying
        self.action = action

        return self._get_rule_feedback()


class TaskFeedback(Feedback):
    def __init__(self, env, test_mode=False):
        self.env = env
        self.tasks = self._get_tasks()
        self.subtasks = self._get_subtasks()
        self.agent_pos = self.env.agent_pos
        if test_mode:
            self.pop_from = -1
        else:
            self.pop_from = 0
        # self.prev_goal_objects = None

    # METHODS FOR DECOMPOSING TASKS INTO SUBTASKS

    def _task_is_sequence(self):
        return isinstance(self.env.instrs, SeqInstr)

    # Instructions for AfterInst are sequences linked by inst_a 'after you' inst_b
    def _task_is_after(self):
        return isinstance(self.env.instrs, AfterInstr)

    # Instructions for BeforeInst are sequences linked by inst_a ', then' inst_b
    def _task_is_before(self):
        return isinstance(self.env.instrs, BeforeInstr)

    def _task_is_and(self, instrs):
        return isinstance(instrs, AndInstr)

    def _task_is_goto(self, instrs):
        return isinstance(instrs, GoToInstr)

    def _task_is_open(self, instrs):
        return isinstance(instrs, OpenInstr)

    def _task_is_unlock(self, instrs):
        door_pos = instrs.desc.obj_poss[0]
        door = self.env.grid.get(*door_pos)
        return self._task_is_open(instrs) and door.is_locked

    def _task_is_pickup(self, instrs):
        return isinstance(instrs, PickupInstr)

    def _task_is_putnext(self, instrs):
        return isinstance(instrs, PutNextInstr)

    # THIS DECIDES THE ORDER IN WHICH FEEDBACK IS PROVIDED, HOWEVER THE ORDER OF
    # 'AND' SUBTASKS SHOULD BE ALLOWED TO BE ARBITRARY
    def _decompose_and_instrs(self, instrs):
        if self._task_is_and(instrs):
            return instrs.instr_a, instrs.instr_b
        return [instrs]

    def _get_tasks(self):
        if self._task_is_before():
            return [
                *self._decompose_and_instrs(self.env.instrs.instr_a),
                *self._decompose_and_instrs(self.env.instrs.instr_b),
            ]
        if self._task_is_after():
            return [
                *self._decompose_and_instrs(self.env.instrs.instr_b),
                *self._decompose_and_instrs(self.env.instrs.instr_a),
            ]
        if self._task_is_and(self.env.instrs):
            return [*self._decompose_and_instrs(self.env.instrs)]
        return [self.env.instrs]

    def _decompose_open_instrs(self, instrs):
        return GoToInstr(instrs.desc), instrs

    def _decompose_unlock_instrs(self, instrs):
        goto_key_instrs = GoToInstr(ObjDesc("key", instrs.desc.color))
        goto_key_instrs.reset_verifier(self.env)
        pickup_key_instrs = PickupInstr(ObjDesc("key", instrs.desc.color))
        pickup_key_instrs.reset_verifier(self.env)
        return (
            goto_key_instrs,
            pickup_key_instrs,
            GoToInstr(instrs.desc),
            instrs,
        )

    def _decompose_pickup_instrs(self, instrs):
        return GoToInstr(instrs.desc), instrs

    def _decompose_putnext_instrs(self, instrs):
        return (
            GoToInstr(instrs.desc_move),
            PickupInstr(instrs.desc_move),
            instrs,
        )

    def _get_subtasks(self):
        subtasks = []
        for task in self.tasks:
            if self._task_is_goto(task):
                subtasks.append(task)
            if self._task_is_open(task):
                if self._task_is_unlock(task):
                    subtasks.extend(self._decompose_unlock_instrs(task))
                else:
                    subtasks.extend(self._decompose_open_instrs(task))
            if self._task_is_pickup(task):
                subtasks.extend(self._decompose_pickup_instrs(task))
            if self._task_is_putnext(task):
                subtasks.extend(self._decompose_putnext_instrs(task))
        return subtasks

    # METHODS FOR GENERATING FEEDBACK FOR EACH SUBTASK

    def _is_goal(self, current_obj, goal_obj):
        return current_obj in goal_obj.obj_set

    def _is_next_to_goal(self, goal_poss, current_pos):
        for pos in goal_poss:
            if pos_next_to(pos, current_pos):
                return True
        return False

    def _has_multiple_goals(self, goal_obj):
        return len(goal_obj.obj_set) > 1

    def _get_article(self, goal_obj):
        if self._has_multiple_goals(goal_obj):
            return "a"
        return "the"

    def _get_goto_type(self, goal_obj):
        if goal_obj.type == "door":
            return "door"
        return "object"

    def _get_completion_level(self):
        if not self.subtasks:
            return ""
        return "a part of "

    def _get_goto_feedback(self, instrs):
        goal_obj = instrs.desc
        if not (self._is_wall() or self._is_empty_cell()):
            if self._is_goal(self.front_cell, goal_obj):
                self.subtasks.pop(self.pop_from)
                return f"You've completed {self._get_completion_level()}your task by going to {self._get_article(goal_obj)} correct {self._get_goto_type(goal_obj)}."
        return "No feedback available."

    def _get_open_feedback(self, instrs):
        goal_obj = instrs.desc
        if not (self._is_wall() or self._is_empty_cell()):
            if self._is_goal(self.front_cell, goal_obj):
                if self._is_open_door():
                    self.subtasks.pop(self.pop_from)
                    return f"You've completed {self._get_completion_level()}your task by opening {self._get_article(goal_obj)} correct door."
        return "No feedback available."

    def _get_pickup_feedback(self, instrs):
        goal_obj = instrs.desc
        if self._is_goal(self.carrying, goal_obj):
            # if self._is_goal_pickup(self.carrying, goal_obj):
            self.subtasks.pop(self.pop_from)
            return f"You've completed {self._get_completion_level()}your task by picking up {self._get_article(goal_obj)} correct object."
        return "No feedback available."

    def _get_putnext_feedback(self, instrs):
        goal_obj_1 = instrs.desc_move
        goal_obj_2 = instrs.desc_fixed
        if self._is_goal(self.front_cell, goal_obj_1):
            if self._is_next_to_goal(goal_obj_2.obj_poss, self.front_pos):
                self.subtasks.pop(self.pop_from)
                return f"You've completed {self._get_completion_level()}your task by putting {self._get_article(goal_obj_1)} correct move object next to {self._get_article(goal_obj_2)} correct {'fixed object' if self._get_goto_type(goal_obj_2) == 'object' else self._get_goto_type(goal_obj_2)}."
        return "No feedback available."

    def _get_task_feedback(self):
        try:
            current_subtask = self.subtasks[self.pop_from]
        except IndexError:
            return "No feedback available."
        if (
            self.action == self.env.actions.left
            or self.action == self.env.actions.right
            or self.action == self.env.actions.forward
        ) and self._task_is_goto(current_subtask):
            return self._get_goto_feedback(current_subtask)
        if self.action == self.env.actions.toggle and self._task_is_open(
            current_subtask
        ):
            return self._get_open_feedback(current_subtask)
        if self.action == self.env.actions.pickup and self._task_is_pickup(
            current_subtask
        ):
            return self._get_pickup_feedback(current_subtask)
        if self.action == self.env.actions.drop and self._task_is_putnext(
            current_subtask
        ):
            return self._get_putnext_feedback(current_subtask)
        return "No feedback available."

    def verify_feedback(self, env, action):
        self.env = env
        self.action = action
        self.front_pos = self.env.front_pos
        self.front_cell = self.env.grid.get(*self.front_pos)
        self.carrying = self.env.carrying

        return self._get_task_feedback()
