import unittest

import gymnasium as gym

from src.dataset.custom_feedback_verifier import RuleFeedback, TaskFeedback


class TestCustomRuleFeedbackVerifier(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("BabyAI-MiniBossLevel-v0")

    def test_invalid_forward_wall(self):
        action = 2
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You can't move forward while you're facing the wall."
        )

    def test_invalid_pickup_wall(self):
        action = 3
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You can't pick up the wall."
        )

    def test_invalid_toggle_wall(self):
        action = 5
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You can't open the wall."
        )

    def test_invalid_forward_obstacle(self):
        action = 2
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        self.env.step(1)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You can't move forward here as there is an obstacle in the form of a ball blocking the way."
        )

    def test_invalid_toggle_obstacle(self):
        action = 5
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        self.env.step(1)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You can't open balls."
        )

    def test_valid_pickup(self):
        action = 3
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1]:
            self.env.step(step)
        assert feedback_verifier.verify_feedback(self.env, action) == ""

    def test_valid_forward_empty(self):
        action = 2
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3]:
            self.env.step(step)
        assert feedback_verifier.verify_feedback(self.env, action) == ""

    def test_invalid_toggle_empty(self):
        action = 5
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "There is nothing to open in front of you."
        )

    def test_invalid_drop_wall(self):
        action = 4
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3, 1]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You can't drop an object while you're facing the wall."
        )

    def test_invalid_drop_obstacle(self):
        action = 4
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3, 2]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You can't drop an object on top of another object, and there is already a key in front of you."
        )

    def test_invalid_pickup_carrying(self):
        action = 3
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3, 2]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You can't pick up another object while you're already carrying one."
        )

    def test_valid_drop(self):
        action = 4
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3, 2, 0]:
            self.env.step(step)
        assert feedback_verifier.verify_feedback(self.env, action) == ""

    def test_invalid_pickup_empty(self):
        action = 3
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3, 2, 0, 4, 0]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "There is nothing to pick up in front of you."
        )

    def test_invalid_drop_not_carrying(self):
        action = 4
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3, 2, 0, 4, 0]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You can't drop an object while you're not carrying anything."
        )

    def test_invalid_pickup_door(self):
        action = 3
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3, 2, 0, 4, 0, 0]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You can't pick up doors."
        )

    def test_invalid_drop_door(self):
        action = 4
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3, 2, 1]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You can't drop an object while you're facing a door."
        )

    def test_invalid_forward_closed_door(self):
        action = 2
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3, 2, 1]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You can't move forward here as the door in front of you is closed."
        )

    def test_valid_toggle_closed_door(self):
        action = 5
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3, 2, 1]:
            self.env.step(step)
        assert feedback_verifier.verify_feedback(self.env, action) == ""

    def test_valid_forward_open_door(self):
        action = 2
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3, 2, 1, 5]:
            self.env.step(step)
        assert feedback_verifier.verify_feedback(self.env, action) == ""

    def test_invalid_toggle_locked_door_wrong_key(self):
        action = 5
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [
            1,
            1,
            3,
            2,
            0,
            0,
            4,
            0,
            5,
            2,
            3,
            1,
            1,
            2,
            2,
            2,
            1,
            2,
            5,
            2,
            2,
            2,
            1,
            2,
            2,
        ]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You can't open a locked door without the correct key."
        )

    def test_invalid_toggle_locked_door_no_key(self):
        action = 5
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3, 2, 0, 0, 4, 1, 2, 2, 1, 2, 5, 2, 2, 2, 1, 2, 2]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You can't open a locked door without the correct key."
        )

    def test_invalid_forward_locked_door(self):
        action = 2
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3, 2, 0, 0, 4, 1, 2, 2, 1, 2, 5, 2, 2, 2, 1, 2, 2]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You can't move forward here as the door in front of you is locked."
        )

    def test_valid_toggle_locked_door(self):
        action = 5
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3, 2, 1, 1, 4, 0, 0, 3, 0, 2, 2, 1, 2, 5, 2, 2, 2, 1, 2, 2]:
            self.env.step(step)
        assert feedback_verifier.verify_feedback(self.env, action) == ""

    def test_valid_toggle_box(self):
        action = 5
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback()
        for step in [1, 1, 3, 2, 0, 0, 4, 1, 2, 2, 1, 2, 5, 2, 2, 1, 2]:
            self.env.step(step)
        assert feedback_verifier.verify_feedback(self.env, action) == ""


class TestCustomTaskGoToFeedbackVerifier(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("BabyAI-GoToRedBall-v0")
        self.env.reset(seed=1)

    def test_goto_wrong_color_type(self):
        action = 0
        for step in [2, action]:
            self.env.step(step)
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        assert feedback_verifier.verify_feedback(self.env, action) == ""

    # Only relevant if including 'partial success' feedback
    def test_goto_wrong_color(self):
        action = 2
        for step in [1, 2, 2, 2, 1, action]:
            self.env.step(step)
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        assert feedback_verifier.verify_feedback(self.env, action) == ""

    # Only relevant if including 'partial success' feedback
    def test_goto_wrong_type(self):
        action = 0
        for step in [2, 2, 0, 2, 2, 0]:
            self.env.step(step)
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        assert feedback_verifier.verify_feedback(self.env, action) == ""

    def test_goto_wrong_not_facing(self):
        action = 2
        for step in [0, 2, 0, 2, 1, action]:
            self.env.step(step)
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        assert feedback_verifier.verify_feedback(self.env, action) == ""

    def test_goto_success_color(self):
        action = 0
        for step in [0, 2, 0, 2, 1, 2, 0]:
            self.env.step(step)
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You've gone to the correct object."
        )

    def test_goto_multiple_goals_success(self):
        action = 1
        self.env.reset(seed=8)
        for step in [2, 2, action]:
            self.env.step(step)
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        assert (
            feedback_verifier.verify_feedback(self.env, action)
            == "You've gone to a correct object."
        )


class TestCustomTaskOpenFeedbackVerifier(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("BabyAI-OpenDoor-v0")
        self.env.reset(seed=5)
        self.action = 5

    # Only relevant if including 'partial success' feedback
    def test_open_color_wrong_door(self):
        for step in [2, 1, self.action]:
            self.env.step(step)
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        assert feedback_verifier.verify_feedback(self.env, self.action) == ""

    def test_open_color_success(self):
        for step in [2, 0, 2, 2, 2, 2, 1, self.action]:
            self.env.step(step)
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        assert (
            feedback_verifier.verify_feedback(self.env, self.action)
            == "You've opened the correct door."
        )

    def test_open_location_wrong_door(self):
        self.env.reset(seed=2)
        for step in [2, 0, 2, 2, 1, 5, self.action]:
            self.env.step(step)
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        assert feedback_verifier.verify_feedback(self.env, self.action) == ""

    def test_open_location_success(self):
        self.env.reset(seed=2)
        for step in [1, self.action]:
            self.env.step(step)
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        assert (
            feedback_verifier.verify_feedback(self.env, self.action)
            == "You've opened the correct door."
        )

    def test_open_multiple_goals_success(self):
        self.env.reset(seed=1)
        for step in [2, 2, 2, 2, 2, self.action]:
            self.env.step(step)
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        assert (
            feedback_verifier.verify_feedback(self.env, self.action)
            == "You've opened a correct door."
        )


class TestCustomTaskPickupFeedbackVerifier(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("BabyAI-PickupLoc-v0")
        self.env.reset(seed=2)
        self.action = 3

    def test_pickup_wrong_color_type(self):
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        for step in [2, self.action]:
            self.env.step(step)
        assert feedback_verifier.verify_feedback(self.env, self.action) == ""

    def test_pickup_wrong_color(self):
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        for step in [1, 2, self.action]:
            self.env.step(step)
        assert feedback_verifier.verify_feedback(self.env, self.action) == ""

    def test_pickup_wrong_type(self):
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        for step in [0, 2, 1, 2, 2, self.action]:
            self.env.step(step)
        assert feedback_verifier.verify_feedback(self.env, self.action) == ""

    def test_pickup_color_success(self):
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        for step in [1, 1, 2, 2, 1, self.action]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, self.action)
            == "You've picked up the correct object."
        )

    def test_pickup_location_success(self):
        self.env.reset(seed=30)
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        for step in [2, 0, self.action]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, self.action)
            == "You've picked up the correct object."
        )

    def test_pickup_multiple_goals_success(self):
        self.env.reset(seed=18)
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        for step in [2, 2, 0, 2, 1, self.action]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, self.action)
            == "You've picked up a correct object."
        )


class TestCustomTaskPutnextFeedbackVerifier(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("BabyAI-PutNextS5N2-v0")
        self.env.reset(seed=14)
        self.action = 4

    def test_putnext_wrong_no_neighbour(self):
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        for step in [0, 2, 0, 3, 2, 1, self.action]:
            self.env.step(step)
        assert feedback_verifier.verify_feedback(self.env, self.action) == ""

    def test_putnext_wrong_color(self):
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        for step in [0, 2, 0, 3, 2, 0, 2, self.action]:
            self.env.step(step)
        assert feedback_verifier.verify_feedback(self.env, self.action) == ""

    def test_putnext_wrong_type(self):
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        for step in [0, 2, 0, 3, 2, self.action]:
            self.env.step(step)
        assert feedback_verifier.verify_feedback(self.env, self.action) == ""

    def test_putnext_success(self):
        feedback_verifier = TaskFeedback(self.env, test_mode=True)
        for step in [0, 2, 0, 3, 0, 2, self.action]:
            self.env.step(step)
        assert (
            feedback_verifier.verify_feedback(self.env, self.action)
            == "You've put the correct object next to the correct object."
        )


class TestCustomTaskSequenceFeedbackVerifier(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("BabyAI-SynthSeq-v0")

    def test_after_sequence_success(self):
        self.env.reset(seed=0)
        feedback_verifier = TaskFeedback(self.env)

        # "put the red box next to a grey door and open a purple door after you pick up a key and go to the yellow box"

        # after you

        # pick up a key and go to the yellow box"

        # "go to a key"
        for step in [1, 2]:
            self.env.step(step)

        assert (
            feedback_verifier.verify_feedback(self.env, 2)
            == "You've gone to a correct object."
        )

        # "pick up a key"
        for step in [3]:
            self.env.step(step)

        assert (
            feedback_verifier.verify_feedback(self.env, 3)
            == "You've picked up a correct object."
        )

        # "go to the yellow box"
        for step in [
            1,
            2,
            2,
            2,
            2,
            0,
            2,
            1,
            5,
            2,
            2,
            0,
            2,
            2,
            2,
            1,
            2,
            0,
            5,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            0,
            2,
            5,
            2,
            2,
            2,
            2,
            0,
            2,
            2,
            2,
        ]:
            self.env.step(step)

        assert (
            feedback_verifier.verify_feedback(self.env, 2)
            == "You've gone to the correct object."
        )

        # "put the red box next to a grey door"

        # "go to the red box"
        for step in [
            0,
            2,
            2,
            0,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            0,
            2,
            2,
            2,
            0,
            2,
            2,
            1,
            5,
            2,
            2,
            2,
            0,
            5,
            2,
            2,
            2,
            2,
            4,
            0,
        ]:
            self.env.step(step)

        assert (
            feedback_verifier.verify_feedback(self.env, 2)
            == "You've gone to the correct object."
        )

        # "pick up the red box"
        for step in [3]:
            self.env.step(step)

        assert (
            feedback_verifier.verify_feedback(self.env, 3)
            == "You've picked up the correct object."
        )

        # "go to a grey door"
        for step in [0, 2, 2, 2, 2, 1, 2, 2, 2, 2, 0, 2, 2, 1, 2, 2, 2, 1, 2, 2]:
            self.env.step(step)

        assert (
            feedback_verifier.verify_feedback(self.env, 2)
            == "You've gone to a correct door."
        )

        # "put next to grey door"
        for step in [
            1,
            1,
            2,
            2,
            1,
            1,
            2,
            4,
        ]:
            self.env.step(step)

        assert (
            feedback_verifier.verify_feedback(self.env, 4)
            == "You've put the correct object next to a correct door."
        )

        # "and open a purple door"

        # "go to a purple door"
        for step in [
            1,
            1,
            2,
            2,
            2,
            2,
        ]:
            self.env.step(step)

        assert (
            feedback_verifier.verify_feedback(self.env, 2)
            == "You've gone to a correct door."
        )

        # "open a purple door"
        for step in [
            5,
            5,
        ]:
            self.env.step(step)

        assert (
            feedback_verifier.verify_feedback(self.env, 5)
            == "You've opened a correct door."
        )


if __name__ == "__main__":
    unittest.main()
