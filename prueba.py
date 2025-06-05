from src.train import Trainer

trainer = Trainer(data_dir="data/raw")
trainer.train(epochs=10)