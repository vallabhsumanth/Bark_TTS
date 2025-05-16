from transformers import AutoProcessor, AutoModel
import scipy.io.wavfile as wavfile
import numpy as np
import torch
from datetime import datetime  # For time

# Load processor and model
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")

# Define voice preset
voice_preset = "v2/en_speaker_6"

# Prepare input text with throat clearing
text_prompt = "Hello, my name is Suno.[laughs]"
inputs = processor(
    text=[text_prompt],
    voice_preset=voice_preset,
    return_tensors="pt",
)

# Move inputs to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate speech
speech_values = model.generate(
    **inputs,
    do_sample=True
)

# Convert to numpy array and normalize
audio_array = speech_values.cpu().numpy().squeeze()
audio_array = audio_array / np.max(np.abs(audio_array))  # Normalize to [-1, 1]

# Save as WAV file
sample_rate = 24000  # Bark's native sample rate
filename = "output_audio_speaker6.wav"
wavfile.write(filename, sample_rate, audio_array.astype(np.float32))

# Print only the time
current_time = datetime.now().strftime("%M:%S")
print(f"Audio saved at: {current_time}")
