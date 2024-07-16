import torch
from e2_tts_pytorch import E2TTS, DurationPredictor

from torch.optim import Adam
from datasets import load_dataset

from e2_tts_pytorch.trainer import (
    HFDataset,
    E2Trainer
)

duration_predictor = DurationPredictor(
    transformer = dict(
        dim = 80,
        depth = 2,
    )
)

e2tts = E2TTS(
    duration_predictor = duration_predictor,
    immiscible=True,
    transformer = dict(
        dim = 80,
        depth = 4,
        skip_connect_type = 'concat'
    ),
)

train_dataset = HFDataset(load_dataset("blanchon/ears_dataset_sentences", split="train"))

optimizer = Adam(e2tts.parameters(), lr=1e-4)

trainer = E2Trainer(
    e2tts,
    optimizer,
    checkpoint_path = 'e2tts.pt',
    log_file = 'e2tts.txt'
)

epochs = 10
batch_size = 8
grad_accumulation_steps = 1

trainer.train(train_dataset, epochs, batch_size, grad_accumulation_steps, save_step=1000)

from vocos import Vocos

vocoder = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1")
e2tts.sample(torch.tensor(train_dataset[0]["audio"]["array"])[None], text=["This is a test."], vocoder=vocoder)
