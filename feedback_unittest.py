import unittest

import gymnasium as gym

from src.dataset.custom_feedback_verifier import RuleFeedback


class TestCustomRuleFeedbackVerifier(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("BabyAI-MiniBossLevel-v0")

    def test_invalid_forward_wall(self):
        action = 2
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback(self.env, action)
        assert (
            feedback_verifier.verify_feedback()
            == "You can't move forward while you're facing the wall."
        )

    def test_invalid_pickup_wall(self):
        action = 3
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback(self.env, action)
        assert feedback_verifier.verify_feedback() == "You can't pick up the wall."

    def test_invalid_toggle_wall(self):
        action = 5
        self.env.reset(seed=16)
        feedback_verifier = RuleFeedback(self.env, action)
        assert feedback_verifier.verify_feedback() == "You can't toggle the wall"

    def test_invalid_forward_obstacle(self):
        action = 2
        self.env.reset(seed=16)
        self.env.step(1)
        feedback_verifier = RuleFeedback(self.env, action)
        assert (
            feedback_verifier.verify_feedback()
            == "You can't move forward here. There is an obstacle in the form of a ball blocking the way."
        )

    def test_invalid_toggle_obstacle(self):
        action = 5
        self.env.reset(seed=16)
        self.env.step(1)
        feedback_verifier = RuleFeedback(self.env, action)
        assert feedback_verifier.verify_feedback() == "You can't toggle balls."

    def test_valid_pickup(self):
        action = 3
        self.env.reset(seed=16)
        for step in [1, 1]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert feedback_verifier.verify_feedback() == ""

    def test_valid_forward_empty(self):
        action = 2
        self.env.reset(seed=16)
        for step in [1, 1, 3]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert feedback_verifier.verify_feedback() == ""

    def test_invalid_toggle_empty(self):
        action = 5
        self.env.reset(seed=16)
        for step in [1, 1, 3]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert (
            feedback_verifier.verify_feedback()
            == "There is nothing to toggle in front of you."
        )

    def test_invalid_drop_wall(self):
        action = 4
        self.env.reset(seed=16)
        for step in [1, 1, 3, 1]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert (
            feedback_verifier.verify_feedback()
            == "You can't drop an object while you're facing the wall."
        )

    def test_invalid_drop_obstacle(self):
        action = 4
        self.env.reset(seed=16)
        for step in [1, 1, 3, 2]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert (
            feedback_verifier.verify_feedback()
            == "You can't drop an object in front of you. There is already a key there."
        )

    def test_invalid_pickup_carrying(self):
        action = 3
        self.env.reset(seed=16)
        for step in [1, 1, 3, 2]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert (
            feedback_verifier.verify_feedback()
            == "You can't pick up another object while you're already carrying one."
        )

    def test_valid_drop(self):
        action = 4
        self.env.reset(seed=16)
        for step in [1, 1, 3, 2, 0]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert feedback_verifier.verify_feedback() == ""

    def test_invalid_pickup_empty(self):
        action = 3
        self.env.reset(seed=16)
        for step in [1, 1, 3, 2, 0, 4, 0]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert (
            feedback_verifier.verify_feedback()
            == "There is nothing to pick up in front of you."
        )

    def test_invalid_pickup_door(self):
        action = 3
        self.env.reset(seed=16)
        for step in [1, 1, 3, 2, 0, 4, 0, 0]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert feedback_verifier.verify_feedback() == "You can't pick up a door."

    def test_invalid_drop_not_carrying(self):
        action = 4
        self.env.reset(seed=16)
        for step in [1, 1, 3, 2, 0, 4, 0]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert (
            feedback_verifier.verify_feedback()
            == "You can't drop an object while you're not carrying anything."
        )

    def test_invalid_drop_door(self):
        action = 4
        self.env.reset(seed=16)
        for step in [1, 1, 3, 2, 1]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert (
            feedback_verifier.verify_feedback()
            == "You can't drop an object while you're facing a door."
        )

    def test_invalid_forward_closed_door(self):
        action = 2
        self.env.reset(seed=16)
        for step in [1, 1, 3, 2, 1]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert (
            feedback_verifier.verify_feedback()
            == "You can't move forward here. The door in front of you is closed."
        )

    def test_valid_toggle_closed_door(self):
        action = 5
        self.env.reset(seed=16)
        for step in [1, 1, 3, 2, 1]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert feedback_verifier.verify_feedback() == ""

    def test_valid_forward_open_door(self):
        action = 2
        self.env.reset(seed=16)
        for step in [1, 1, 3, 2, 1, 5]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert feedback_verifier.verify_feedback() == ""

    def test_invalid_toggle_locked_door_wrong_key(self):
        action = 5
        self.env.reset(seed=16)
        for step in [1, 1, 3, 2, 1, 5, 2, 3, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert (
            feedback_verifier.verify_feedback()
            == "You can't toggle a locked door without the correct key."
        )

    def test_invalid_toggle_locked_door_no_key(self):
        action = 5
        self.env.reset(seed=16)
        for step in [1, 1, 3, 2, 0, 0, 4, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert (
            feedback_verifier.verify_feedback()
            == "You can't toggle a locked door without the correct key."
        )

    def test_invalid_forward_locked_door(self):
        action = 2
        self.env.reset(seed=16)
        for step in [1, 1, 3, 2, 0, 0, 4, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert (
            feedback_verifier.verify_feedback()
            == "You can't move forward here. The door in front of you is locked."
        )

    def test_valid_toggle_locked_door(self):
        action = 5
        self.env.reset(seed=16)
        for step in [1, 1, 3, 2, 1, 1, 4, 0, 0, 3, 0, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert feedback_verifier.verify_feedback() == ""

    def test_valid_toggle_box(self):
        action = 5
        self.env.reset(seed=16)
        for step in [1, 1, 3, 2, 0, 0, 4, 1, 1, 2, 2, 1, 2, 2, 2, 1, 2]:
            self.env.step(step)
        feedback_verifier = RuleFeedback(self.env, action)
        assert (
            feedback_verifier.verify_feedback()
            == "You can't toggle an already open door."
        )


if __name__ == "__main__":
    unittest.main()
