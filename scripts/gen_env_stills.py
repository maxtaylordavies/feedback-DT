from src.utils.demos import Demo
from src.utils.argparsing import get_args
import json
import os

if __name__ == "__main__":
    args = get_args()
    demo_mode = args["demo"]
    seed_path = demo_mode + ".json"
    seeds_checked = 0
    seeds_used = 0
    with open(seed_path, "r") as f:
        seeds = json.load(f)
    for level, configs in seeds.items():
        for conf in configs.keys():
            if demo_mode == "ood_seeds":
                for ood_type in seeds[level][conf].keys():
                    for seed in seeds[level][conf][ood_type]:
                        seeds_checked += 1
                        subdir = ood_type
                        output_dir = os.path.join(args["output_dir"], subdir)
                        try:
                            Demo(conf, seed, [], args["demo"], output_dir)
                            seeds_used += 1
                        except:
                            continue
            else:
                for seed in seeds[level][conf]:
                    seeds_checked += 1
                    output_dir = args["output_dir"]
                    try:
                        Demo(
                            conf,
                            seed,
                            [],
                            args["demo"],
                            output_dir,
                        )
                        seeds_used += 1
                    except:
                        continue
    print(f"Seeds checked: {seeds_checked}")
    print(f"Seeds used: {seeds_used}")
