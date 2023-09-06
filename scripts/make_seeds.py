from src.dataset.seeds import SeedFinder

if __name__ == "__main__":
    seed_finder = SeedFinder(n_train_seeds_required=1280)
    seed_finder.find_seeds()
