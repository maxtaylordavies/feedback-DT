from src.dataset.seeds import SeedFinder

if __name__ == "__main__":
    seed_finder = SeedFinder()
    seed_finder.save_in_domain_seeds()
    in_domain_seeds = seed_finder.load_in_domain_seeds()
    seed_finder.save_ood_seeds()
    ood_seeds = seed_finder.load_ood_seeds()
