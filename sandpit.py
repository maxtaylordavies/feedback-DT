# import seaborn_image as isns
import matplotlib.pyplot as plt

plt.rcParams.update({"axes.titlesize": "small"})
plt.rcParams.update({"axes.labelsize": "x-small"})

import gymnasium as gym
from minigrid.envs.babyai.core.verifier import (
    SeqInstr,
    AfterInstr,
    AndInstr,
    BeforeInstr,
)
from minigrid.wrappers import RGBImgObsWrapper
from src.dataset.custom_feedback_verifier import RuleFeedback, TaskFeedback


# def get_instrs_to_check(instrs):
#     instrs_list = []
#     if isinstance(env.instrs, SeqInstr):
#         if isinstance(env.instrs.instr_a, AndInstr):
#             instrs_list.extend([env.instrs.instr_a.instr_a, env.instrs.instr_a.instr_b])
#         else:
#             instrs_list.append(env.instrs.instr_a)
#         if isinstance(env.instrs.instr_b, AndInstr):
#             instrs_list.extend([env.instrs.instr_b.instr_a, env.instrs.instr_b.instr_b])
#         else:
#             instrs_list.append(env.instrs.instr_b)
#         return instrs_list
#     return [instrs]


# train_seeds = {}
# test_seeds = {}
# unseen_properties = [{"color": "blue", "type": "ball"}]
configuration = "BabyAI-BossLevel-v0"
# env = gym.make(configuration)
# train_seeds[configuration] = []
# test_seeds[configuration] = []

# for seed in range(100):
#     env.reset(seed=seed)
#     instrs_to_check = get_instrs_to_check(env.instrs)
#     goal_desc = None
#     goal_desc_move = None
#     goal_desc_fixed = None
#     test_seed = False
#     for instrs in instrs_to_check:
#         try:
#             goal_desc = instrs.desc
#         except AttributeError:
#             goal_desc_move = instrs.desc_move
#             goal_desc_fixed = instrs.desc_fixed
#         for uns_prop in unseen_properties:
#             if (
#                 (
#                     goal_desc
#                     and uns_prop["color"] == goal_desc.color
#                     and uns_prop["type"] == goal_desc.type
#                 )
#                 or (
#                     goal_desc_move
#                     and uns_prop["color"] == goal_desc_move.color
#                     and uns_prop["type"] == goal_desc_move.type
#                 )
#                 or (
#                     goal_desc_fixed
#                     and uns_prop["color"] == goal_desc_fixed.color
#                     and uns_prop["type"] == goal_desc_fixed.type
#                 )
#             ):
#                 test_seed = True
#     if test_seed:
#         test_seeds[configuration].append(seed)
#     else:
#         train_seeds[configuration].append(seed)

# print(train_seeds)
# print(test_seeds)
# print(set(train_seeds[configuration]).intersection(set(test_seeds[configuration])))


def format_mission(mission):
    if ", then " in mission:
        mission = mission.split(", then ")
        mission = f"{mission[0]},\nthen {mission[1]}"
    if " after you " in mission:
        mission = mission.split(" after you ")
        mission = f"{mission[0]}\nafter you {mission[1]}"
    return mission


# env = RGBImgObsWrapper(gym.make(configuration))
# for seed in range(100, 200):
#     obs, _ = env.reset(seed=seed)
#     mission = env.instrs.surface(env)
#     if (
#         isinstance(env.instrs, (BeforeInstr, AfterInstr))
#         and "put" in mission
#         and "open" in mission
#     ):
#         mission_formatted = format_mission(mission)
#         fig, ax = plt.subplots(1)
#         plt.imshow(obs["image"])
#         ax.set(xlabel=mission_formatted, title=seed, xticks=[], yticks=[])
#         plt.show()

# USE 103 (81 grey door can't be opened)


# actions = [2, 2, 2, 2, 2, 3, 5, 1, 2, 0, 2, 3, 5, 2, 2, 2, 2, 2, 2, 5]
actions = [2, 1, 2, 2, 2, 2, 1, 2, 2, 0, 5, 5]
action_dict = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle (=open)",
}
env = RGBImgObsWrapper(gym.make(configuration))
obs, _ = env.reset(seed=103)
mission = env.instrs.surface(env)
mission_formatted = format_mission(mission)
fig, ax = plt.subplots(1)
plt.imshow(obs["image"])
ax.set(xlabel=mission_formatted, xticks=[], yticks=[])
plt.savefig("start.png")
plt.show()
# rule_feedback_generator = RuleFeedback()
# task_feedback_generator = TaskFeedback(env)
for i, action in enumerate(actions):
    # rule_feedback = rule_feedback_generator.verify_feedback(env, action)
    print(f"Toggle action: {action == env.actions.toggle}")
    output = env.step(action)
    front_cell = env.grid.get(*env.front_pos)
    if front_cell and front_cell.type == "door":
        # print(f"Can be toggled: {front_cell.toggle(env, env.front_pos)}")
        print(f"Door open: {front_cell.is_open}")
    # task_feedback = task_feedback_generator.verify_feedback(env, action)
    # feedback = (
    #     f"action: {action_dict[action]}\n"
    #     + f"RF: {rule_feedback if rule_feedback else '--'}\n"
    #     + f"TF: {task_feedback if task_feedback else '--'}"
    # )
    fig, ax = plt.subplots(1)
    plt.imshow(output[0]["image"])
    # ax.set(title=feedback, xlabel=mission_formatted, xticks=[], yticks=[])
    plt.savefig(f"step_{i}.png")
    plt.show()
