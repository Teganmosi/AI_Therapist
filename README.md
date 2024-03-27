**AI THERAPIST**

This repository demonstrates a system that seamlessly integrates speech-to-text, large language model querying, and text-to-speech capabilities to create a conversational interface for an AI therapist. Users can pose questions verbally and receive comprehensive audio responses, facilitating a more natural and intuitive human-computer interaction experience.

Key Features:

Real-Time Speech Transcription: Transforms audio input into text using the highly efficient Whisper-large-v3 speech-to-text model.
LLM Integration: Leverages a user-defined LLM for comprehensive response generation, enabling access to a vast knowledge base and diverse response styles.
Natural Audio Responses: Synthesizes spoken responses using advanced text-to-speech models for a more engaging and human-like interaction.
Getting Started:

Installation:

Clone this repository.
Install required libraries: pip install -r requirements.txt
Note: Additional setup may be needed for libraries specified via URLs.
Usage:

Execute the script: python Three_implemented.ipynb (or relevant filename)
Record your query upon prompt.
Receive transcribed text, LLM response, and generated audio output.
Flexibility and Customization:

Adapt the LLM to your needs by replacing the create_llm function with your preferred language model's query interface.
Future Enhancements:

Robust Error Handling: Incorporate mechanisms for graceful error recovery and meaningful user feedback.
Refined Text Processing: Explore advanced techniques for improved accuracy and natural language understanding.
Contributions Welcome!

We invite you to participate in this ongoing development effort. Explore, experiment, and contribute to create an even more robust and engaging audio-based question answering experience.

The mental_health.py file contains the code used to train the large language model

The tts_and_llm.py file contains the code to run the text to speech and large language model alone

The stt_and_llm.py file contains the code to run the speech to text an large language model alone

The STT-->LLM-->TTS file contains the code to run all three models together in one notebook

The Data file contains the link to the google drive folder where the data used to fine tune the model is saved
