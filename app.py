import os
import scipy
import nltk
import tempfile
import numpy as np
from bark.generation import preload_models, SAMPLE_RATE
from bark import generate_audio
from scipy.io import wavfile

import gradio as gr
nltk.download('punkt')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
preload_models()



def generate_audio_from_text(text,language_prompt,speaker_prompt):
    if language_prompt == "english":
        if speaker_prompt=="speaker 1":
            history_prompt = "v2/en_speaker_0"
        elif speaker_prompt=="speaker 2":
          history_prompt = "v2/en_speaker_1"
        elif speaker_prompt=="speaker 3":
          history_prompt = "v2/en_speaker_2"
        elif speaker_prompt=="speaker 4":
          history_prompt = "v2/en_speaker_3"
        elif speaker_prompt=="speaker 5":
          history_prompt = "v2/en_speaker_4"
        elif speaker_prompt=="speaker 6":
          history_prompt = "v2/en_speaker_5"
        elif speaker_prompt=="speaker 7":
          history_prompt = "v2/en_speaker_6"
        elif speaker_prompt=="speaker 8":
          history_prompt = "v2/en_speaker_7"
        elif speaker_prompt=="speaker 9":
          history_prompt = "v2/en_speaker_8"
        elif speaker_prompt=="speaker 10":
          history_prompt = "v2/en_speaker_9"
        else:
            history_prompt = "v2/en_speaker_9"

    elif language_prompt == "french":
        if speaker_prompt=="speaker 1":
          history_prompt = "v2/fr_speaker_0"
        elif speaker_prompt=="speaker 2":
          history_prompt = "v2/fr_speaker_1"
        elif speaker_prompt=="speaker 3":
          history_prompt = "v2/fr_speaker_2"
        elif speaker_prompt=="speaker 4":
          history_prompt = "v2/fr_speaker_3"
        elif speaker_prompt=="speaker 5":
          history_prompt = "v2/fr_speaker_4"
        elif speaker_prompt=="speaker 6":
          history_prompt = "v2/fr_speaker_5"
        elif speaker_prompt=="speaker 7":
          history_prompt = "v2/fr_speaker_6"
        elif speaker_prompt=="speaker 8":
          history_prompt = "v2/fr_speaker_7"
        elif speaker_prompt=="speaker 9":
          history_prompt = "v2/fr_speaker_8"
        elif speaker_prompt=="speaker 10":
          history_prompt = "v2/fr_speaker_9"
        else:
            history_prompt = "v2/fr_speaker_9"

    elif language_prompt =="german":
        if speaker_prompt=="speaker 1":
          history_prompt = "v2/de_speaker_0"
        elif speaker_prompt=="speaker 2":
          history_prompt = "v2/de_speaker_1"
        elif speaker_prompt=="speaker 3":
          history_prompt = "v2/de_speaker_2"
        elif speaker_prompt=="speaker 4":
          history_prompt = "v2/de_speaker_3"
        elif speaker_prompt=="speaker 5":
          history_prompt = "v2/de_speaker_4"
        elif speaker_prompt=="speaker 6":
          history_prompt = "v2/de_speaker_5"
        elif speaker_prompt=="speaker 7":
          history_prompt = "v2/de_speaker_6"
        elif speaker_prompt=="speaker 8":
          history_prompt = "v2/de_speaker_7"
        elif speaker_prompt=="speaker 9":
          history_prompt = "v2/de_speaker_8"
        elif speaker_prompt=="speaker 10":
          history_prompt = "v2/de_speaker_9"
        else:
            history_prompt = "v2/de_speaker_9"

    elif language_prompt =="hindi":
        if speaker_prompt=="speaker 1":
          history_prompt = "v2/hi_speaker_0"
        elif speaker_prompt=="speaker 2":
          history_prompt = "v2/hi_speaker_1"
        elif speaker_prompt=="speaker 3":
          history_prompt = "v2/hi_speaker_2"
        elif speaker_prompt=="speaker 4":
          history_prompt = "v2/hi_speaker_3"
        elif speaker_prompt=="speaker 5":
          history_prompt = "v2/hi_speaker_4"
        elif speaker_prompt=="speaker 6":
          history_prompt = "v2/hi_speaker_5"
        elif speaker_prompt=="speaker 7":
          history_prompt = "v2/hi_speaker_6"
        elif speaker_prompt=="speaker 8":
          history_prompt = "v2/hi_speaker_7"
        elif speaker_prompt=="speaker 9":
          history_prompt = "v2/hi_speaker_8"
        elif speaker_prompt=="speaker 10":
          history_prompt = "v2/hi_speaker_9"
        else:
            history_prompt = "v2/hi_speaker_9"

    elif language_prompt =="chinese":
        if speaker_prompt=="speaker 1":
          history_prompt = "v2/zh_speaker_0"
        elif speaker_prompt=="speaker 2":
          history_prompt = "v2/zh_speaker_1"
        elif speaker_prompt=="speaker 3":
          history_prompt = "v2/zh_speaker_2"
        elif speaker_prompt=="speaker 4":
          history_prompt = "v2/zh_speaker_3"
        elif speaker_prompt=="speaker 5":
          history_prompt = "v2/zh_speaker_4"
        elif speaker_prompt=="speaker 6":
          history_prompt = "v2/zh_speaker_5"
        elif speaker_prompt=="speaker 7":
          history_prompt = "v2/zh_speaker_6"
        elif speaker_prompt=="speaker 8":
          history_prompt = "v2/zh_speaker_7"
        elif speaker_prompt=="speaker 9":
          history_prompt = "v2/zh_speaker_8"
        elif speaker_prompt=="speaker 10":
          history_prompt = "v2/zh_speaker_9"
        else:
            history_prompt = "v2/zh_speaker_9"

    elif language_prompt =="italian":
        if speaker_prompt=="speaker 1":
          history_prompt = "v2/it_speaker_0"
        elif speaker_prompt=="speaker 2":
          history_prompt = "v2/it_speaker_1"
        elif speaker_prompt=="speaker 3":
          history_prompt = "v2/it_speaker_2"
        elif speaker_prompt=="speaker 4":
          history_prompt = "v2/it_speaker_3"
        elif speaker_prompt=="speaker 5":
          history_prompt = "v2/it_speaker_4"
        elif speaker_prompt=="speaker 6":
          history_prompt = "v2/it_speaker_5"
        elif speaker_prompt=="speaker 7":
          history_prompt = "v2/it_speaker_6"
        elif speaker_prompt=="speaker 8":
          history_prompt = "v2/it_speaker_7"
        elif speaker_prompt=="speaker 9":
          history_prompt = "v2/it_speaker_8"
        elif speaker_prompt=="speaker 10":
          history_prompt = "v2/it_speaker_9"
        else:
            history_prompt = "v2/it_speaker_9"
            
    elif language_prompt =="japanese":
        if speaker_prompt=="speaker 1":
          history_prompt = "v2/ja_speaker_0"
        elif speaker_prompt=="speaker 2":
          history_prompt = "v2/ja_speaker_1"
        elif speaker_prompt=="speaker 3":
          history_prompt = "v2/ja_speaker_2"
        elif speaker_prompt=="speaker 4":
          history_prompt = "v2/ja_speaker_3"
        elif speaker_prompt=="speaker 5":
          history_prompt = "v2/ja_speaker_4"
        elif speaker_prompt=="speaker 6":
          history_prompt = "v2/ja_speaker_5"
        elif speaker_prompt=="speaker 7":
          history_prompt = "v2/ja_speaker_6"
        elif speaker_prompt=="speaker 8":
          history_prompt = "v2/ja_speaker_7"
        elif speaker_prompt=="speaker 9":
          history_prompt = "v2/ja_speaker_8"
        elif speaker_prompt=="speaker 10":
          history_prompt = "v2/ja_speaker_9"
        else:
            history_prompt = "v2/ja_speaker_9"
    else:
        raise ValueError("Invalid language or gender selection")

    sentences = nltk.sent_tokenize(text)
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

    pieces = []
    for sentence in sentences:
        audio_array = generate_audio(sentence, history_prompt=history_prompt)
        pieces += [audio_array]

    # Concatenate the audio pieces
    final_audio = np.concatenate(pieces)

    # Save the audio to a WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        wavfile.write(temp_wav.name, SAMPLE_RATE, final_audio)

    # Return the saved audio file
    return temp_wav.name

# Define lists of language and gender options
language_options = [
    "english",
    "french",
    "german",
    "hindi",
    "chinese",
    "italian",
    "japanese",


]

speaker_options=[
    "speaker 1",
    "speaker 2",
    "speaker 3",
    "speaker 4",
    "speaker 5",
    "speaker 6",
    "speaker 7",
    "speaker 8",
    "speaker 9",
    "speaker 10",
]
# Create a Gradio interface with text input and dropdown menus for language and gender
iface = gr.Interface(
    fn=generate_audio_from_text,
    inputs=[
        gr.Textbox(text="Enter text to convert to speech:"),
        gr.Dropdown(choices=language_options, label="Select language:"),
        gr.Dropdown(choices=speaker_options, label="Select speaker:"),
    ],
    outputs=gr.outputs.File(label="Download WAV File"),
    title="Text-to-Speech App by Badar Masood",
    timeout=300,
)

# Launch the Gradio app with sharing enabled
iface.launch(debug=True, enable_queue=True)
