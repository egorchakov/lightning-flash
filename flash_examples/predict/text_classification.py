# import our libraries
import torch
from pytorch_lightning import _logger as log, Trainer

from flash.core.data import download_data
from flash.text import TextClassificationData

# 1 Load finetuned model
model = torch.load("text_classification_model.pt")

# 2.1 Perform inference from list of sequences
predictions = model.predict([
    "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
    "The worst movie in the history of cinema. I don't know if it was trying to be funny or sad, poignant or droll, but the end result was unwatchable.",
    "Well , I come from Bulgaria where it 's almost impossible to have a tornado but my imagination tells me to be "
    "very , very afraid"
    "!!!This guy (Devon Sawa) has done a great job with this movie!I don't know exactly how old he was but he didn't act like a child (WELL DONE)!",
])
print(predictions)

# 5.2 Or perform inference from `.csv` file
datamodule = TextClassificationData.from_file(
    predict_file="data/imdb/test.csv",
    input="review",
)
predictions = Trainer().predict(model, datamodule=datamodule)
print(predictions)