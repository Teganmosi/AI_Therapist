# -*- coding: utf-8 -*-
"""tts and llm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NKjgiL-tRhFYKPzOb_LLJ0vtPKepOUyZ
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install llama-index-llms-huggingface

# Commented out IPython magic to ensure Python compatibility.
# %pip install llama-index-embeddings-langchain

!pip install pypdf
!pip install python-dotenv
!pip install -q transformers einops accelerate langchain bitsandbytes
!pip install sentence_transformers
!pip install llama-index
!pip install nltk

from google.colab import drive
drive.mount('/content/drive')

from mental_health import create_llm

!pip install git+https://github.com/suno-ai/bark.git
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from IPython.display import Audio
import nltk  # we'll use this to split into sentences
import numpy as np

from bark.generation import (
    generate_text_semantic,
    preload_models,
)

from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE


preload_models()

# Create the query engine using the create_llm functio

query_engine = create_llm()

SPEAKER = "v2/en_speaker_6"
silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

# Use the query engine to get a response from the LLM
response = query_engine.query("What is cyclothymic disorder?")

print(response)

# Convert the response to a string and use it as the script for the text-to-speech model
script = str(response).replace("\n", " ").strip()

import nltk
nltk.download('punkt')

sentences = nltk.sent_tokenize(script)

SPEAKER = "v2/en_speaker_6"
silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

pieces = []
for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=SPEAKER)
    pieces += [audio_array, silence.copy()]

Audio(np.concatenate(pieces), rate=SAMPLE_RATE)

from IPython.display import display, Audio
import numpy as np

while True:
    query = input()
    # Get the response from the language model
    response = query_engine.query(query)
    # Convert the response to a string
    script = str(response).replace("\n", " ").strip()

    # Split the script into sentences
    sentences = nltk.sent_tokenize(script)

    # Generate the audio for each sentence and add a short silence after each one
    pieces = []
    for sentence in sentences:
        audio_array = generate_audio(sentence, history_prompt=SPEAKER)
        pieces += [audio_array, silence.copy()]

    # Concatenate the audio pieces and create an audio player widget
    audio = np.concatenate(pieces)
    display(Audio(audio, rate=SAMPLE_RATE))

