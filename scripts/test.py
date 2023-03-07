import minari
from src.train import create_collator_and_model

minari_dataset = minari.load_dataset("BabyAI-GoToRedBallGrey-v0_10-eps_incl-timeout")
collator, model = create_collator_and_model(minari_dataset)

print(collator.state_dim)
print(collator.act_size)
