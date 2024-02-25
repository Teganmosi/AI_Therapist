# Import necessary libraries
from pydub import AudioSegment
from IPython.display import Javascript, Audio
from google.colab import output
from base64 import b64decode
import io
import torch
import nltk
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from mental_health import create_llm
from bark.generation import preload_models
from bark import generate_audio, SAMPLE_RATE

# Load the speech-to-text model and processor
model_id = "openai/whisper-large-v3"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
stt_model.to(device)
stt_processor = AutoProcessor.from_pretrained(model_id)

# Create the ASR pipeline
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=stt_model,
    tokenizer=stt_processor.tokenizer,
    feature_extractor=stt_processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Create the query engine using the create_llm function
query_engine = create_llm()

# Preload models for text-to-speech
preload_models()

# Define the JavaScript code for recording audio
RECORD = """
const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader()
  reader.onloadend = e => resolve(e.srcElement.result)
  reader.readAsDataURL(blob)
})
var record = time => new Promise(async resolve => {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  chunks = []
  recorder.ondataavailable = e => chunks.push(e.data)
  recorder.start()
  await sleep(time)
  recorder.onstop = async ()=>{
    blob = new Blob(chunks)
    text = await b2text(blob)
    resolve(text)
  }
  recorder.stop()
})
"""

# Define the function to start recording
def record(sec=3):
    display(Javascript(RECORD))
    s = output.eval_js('record(%d)' % (sec*1000))
    b = b64decode(s.split(',')[1])
    return b  # Now `b` is a bytes object containing the raw audio data

# Define the function to convert audio to .wav format
def convert_webm_to_wav(audio_data):
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
    wav_data = io.BytesIO()
    audio_segment.export(wav_data, format="wav")
    wav_data.seek(0)
    return wav_data.read()

# Define the function to process audio input
def process_audio_input(audio_input):
    # Convert the audio input to text
    audio = convert_webm_to_wav(audio_input)
    stt_result = asr_pipe(audio)
    query = stt_result['text']  # Access 'text' key directly

    # Get the response from the language model
    response = query_engine.query(query)

    # Convert the response to a string and use it as the script for the text-to-speech model
    script = str(response).replace("\n", " ").strip()

    # Split the script into sentences
    sentences = nltk.sent_tokenize(script)

    # Generate the audio for each sentence and concatenate them
    SPEAKER = "v2/en_speaker_6"
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence
    pieces = []
    for sentence in sentences:
        audio_array = generate_audio(sentence, history_prompt=SPEAKER)
        pieces += [audio_array, silence.copy()]

    # Return the concatenated audio
    return Audio(np.concatenate(pieces), rate=SAMPLE_RATE)

# Now you can get the audio input, process it, and play the results like this:
audio_input = record(5)  # Record 5 seconds of audio
audio_output = process_audio_input(audio_input)
audio_output
