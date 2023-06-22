import os

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt

plt.rcParams.update({"axes.titlesize": "x-small"})
plt.rcParams.update({"axes.labelsize": "x-small"})

from minigrid.wrappers import RGBImgObsWrapper

from src.dataset.custom_feedback_verifier import RuleFeedback, TaskFeedback

DEFAULT_HARD_ACTIONS = [
    2,
    1,
    2,
    2,
    2,
    2,
    # try to move forward into the wall
    2,
    # try to open the wall
    5,
    1,
    2,
    2,
    0,
    2,
    # try to pick up the door
    3,
    5,
    2,
    2,
    2,
    2,
    2,
    2,
    0,
    2,
    # try to open empty cell
    5,
    2,
    5,
    2,
    2,
    2,
    0,
    2,
    2,
    2,
    # try to move forward into obstacle
    2,
    # try to open obstacle
    5,
    3,
    2,
    # try to open door with wrong key
    5,
    1,
    1,
    2,
    2,
    2,
    2,
    1,
    2,
    # try to open an already open door
    5,
    5,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    2,
    2,
    # try to pickup while carrying
    3,
    # try to drop on top of another object
    0,
    # try to drop on wall
    4,
    1,
    1,
    4,
    0,
    0,
    3,
    # try to pickup wall
    1,
    3,
    0,
    0,
    2,
    2,
    0,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    0,
    2,
    2,
    2,
    2,
    5,
    4,
    1,
    1,
    2,
    2,
    2,
    2,
    1,
    2,
    2,
    2,
    2,
    2,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    0,
    2,
    1,
    2,
    # try to drop on top of box
    4,
    5,
    # try to open an open box
    5,
    4,
    0,
    0,
    2,
    0,
    2,
    3,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    4,
    3,
    2,
    4,
    0,
    2,
    0,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    0,
    2,
    2,
    1,
    0,
    2,
]
DEFAULT_EASY_ACTIONS = [2, 2, 2, 2, 2, 2, 5, 0, 3, 1, 1, 2, 5, 3, 2, 2, 1, 0, 2, 1]

DEFAULT_HARD_CONFIG = "BabyAI-BossLevel-v0"
DEFAULT_EASY_CONFIG = "BabyAI-GoToRedBall-v0"

DEFAULT_HARD_SEED = 103
DEFAULT_EASY_SEED = 0

ACTION_DICT = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle (=open)",
}


class Demo:
    def __init__(self, config, seed, actions, demo_mode, output_dir):
        self.config = config
        self.seed = seed
        self.actions = actions
        self.demo_mode = demo_mode
        self.output_dir_root = output_dir.split("/")[0]
        if not os.path.exists(self.output_dir_root):
            print(self.output_dir_root)
            os.mkdir(self.output_dir_root)
        self.output_dir_sub = os.path.join(self.output_dir_root, demo_mode)
        if not os.path.exists(self.output_dir_sub):
            print(self.output_dir_sub)
            os.mkdir(self.output_dir_sub)
        if "ood" in demo_mode:
            self.output_dir_sub_sub = os.path.join(
                self.output_dir_sub, output_dir.split("/")[-1]
            )
            if not os.path.exists(self.output_dir_sub_sub):
                print(self.output_dir_sub_sub)
                os.mkdir(self.output_dir_sub_sub)
            self.output_dir = self.output_dir_sub_sub
        else:
            self.output_dir = self.output_dir_sub
        self.mission = None
        self.env = self._instantiate_rgb_env(config, seed)

    def _format_mission(self):
        if ", then " in self.mission:
            mission_split = self.mission.split(", then ")
            return f"{mission_split[0]},\nthen {mission_split[1]}"
        if " after you " in self.mission:
            mission_split = self.mission.split(" after you ")
            return f"{mission_split[0]}\nafter you {mission_split[1]}"
        else:
            return self.mission

    def _format_feedback(self, feedback, action):
        rule_feedback, task_feedback = feedback
        return (
            f"action: {ACTION_DICT[action]}\n"
            + f"RF: {rule_feedback if rule_feedback != 'No feedback available.' else '--'}\n"
            + f"TF: {task_feedback if task_feedback != 'No feedback available.' else '--'}"
        )

    def _get_image_name(self, i):
        if self.demo_mode in ["ood_seeds", "in_domain_seeds"]:
            return f"{self.config}_seed-{self.seed}.png"
        if i:
            return (
                f"step_{i+1 if i >= 99 else (f'0{i+1}' if i >= 9 else f'00{i+1}')}.png"
            )
        return "step_000.png"

    def _plot_and_save_obs(self, frame, i=None, feedback=None):
        _, ax = plt.subplots(1)
        plt.imshow(frame)
        # plt.show()
        # _ = plt.imshow(frame)
        mission_formatted = self._format_mission()
        feedback_formatted = (
            self._format_feedback(feedback, self.actions[i]) if feedback else None
        )
        title = feedback_formatted if feedback else None
        ax.set(title=title, xlabel=mission_formatted, xticks=[], yticks=[])
        image_name = self._get_image_name(i)
        plt.savefig(os.path.join(self.output_dir, image_name))
        plt.close()

    def _instantiate_rgb_env(self, config, seed):
        env = gym.make(config, render_mode="rgb_array")
        env.reset(seed=seed)
        frame = env.render()
        self.mission = env.instrs.surface(env)
        self._plot_and_save_obs(frame)
        return env

    def interact_with_env(self):
        rule_feedback_generator = RuleFeedback()
        task_feedback_generator = TaskFeedback(self.env)
        for i, action in enumerate(self.actions):
            rule_feedback = rule_feedback_generator.verify_feedback(self.env, action)
            self.env.step(action)
            frame = self.env.render()
            task_feedback = task_feedback_generator.verify_feedback(self.env, action)
            self._plot_and_save_obs(frame, feedback=(rule_feedback, task_feedback), i=i)

    def make_demo_video_from_images(self):
        video_name = "demo.mp4"
        image_list = [
            image for image in os.listdir(self.output_dir) if image.endswith(".png")
        ]
        image_list.sort()
        frame = cv2.imread(os.path.join(self.output_dir, image_list[0]))
        height, width, _ = frame.shape
        video = cv2.VideoWriter(
            os.path.join(self.output_dir, video_name),
            cv2.VideoWriter_fourcc(*"mp4v"),
            1,
            (width, height),
        )

        for image in image_list:
            video.write(cv2.imread(os.path.join(self.output_dir, image)))

        cv2.destroyAllWindows()
        video.release()

    def make_demo_video(self):
        self.interact_with_env()
        self.make_demo_video_from_images()
