import os
import torch
import time
from gtts import gTTS, gTTSError
from TTS.config import load_config
from TTS.utils.audio import AudioProcessor
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


dataset = load_dataset("mozilla-foundation/common_voice_8_0", "de", split='train')

# Print some examples
for example in dataset[:5]:
    print(example['sentence'])
    print(example['audio'])

class CommonVoiceDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['sentence']
        audio = self.dataset[idx]['audio']
        return text, audio

common_voice_dataset = CommonVoiceDataset(dataset)
dataloader = DataLoader(common_voice_dataset, batch_size=32, shuffle=True)

with open('config.json', 'w') as config_file:
    config_file.write(config_data)

# Load the configuration file
config = load_config("config.json")

# Initialize the audio processor
ap = AudioProcessor(**config.audio)

# Define optimizer and criterion
num_epochs = config.train['epochs']
learning_rate = config.train['learning_rate']
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()  # Example loss function, replace with appropriate one

for epoch in range(num_epochs):
    for data in dataloader:
        texts, audios = data
        # Your fine-tuning logic here
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, audios)  # Define your loss function
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

    # Save model checkpoint
    if epoch % config.train['save_interval'] == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

# Using gTTS for Text-to-Speech Conversion
test_sentences = [
    "यह एक उदाहरण है।",
    "हम गुणवत्ता का मूल्यांकन कर रहे हैं।"
]

for sentence in test_sentences:
    try:
        tts = gTTS(text=sentence, lang='hi')
        tts.save(f"{sentence[:10]}.mp3")
        print(f"Saved: {sentence[:10]}.mp3")
    except gTTSError as e:
        print(f"Error: {str(e)}")
