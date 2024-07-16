# TODO List:
## 1. Optimize imports
## 2. Export images from pptx
## 3. Delete unnecessary libs from env
## 4. Add subtitle sync from mp3 + txt to srt
## 5. Test multitreading

import scipy.io.wavfile as wavfile
import io
import gradio as gr
import time
import yaml
import torch
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
from Models import *
from Utils import *
from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
import torch
import glob
import gc
import re

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random

random.seed(0)

import numpy as np

np.random.seed(0)

# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
from models import *
from utils import *
from text_utils import TextCleaner
from txtsplit import txtsplit
from pydub import AudioSegment
import os
from pptx import Presentation
import zipfile
import os

textclenaer = TextCleaner()

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load phonemizer
import phonemizer

global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

config = yaml.safe_load(open("Models/LibriTTS/config.yml"))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert

BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu')
params = params_whole['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict

            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
    clamp=False
)


def inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                         embedding=bert_dur,
                         embedding_scale=embedding_scale,
                         features=ref_s,  # reference from the same speaker as the embedding
                         num_steps=diffusion_steps).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en,
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr,
                            F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()[..., :-50]  # weird pulse at the end of the model, need to be fixed later


def LFinference(text, s_prev, ref_s, alpha=0.3, beta=0.7, t=0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    ps = ps.replace('``', '"')
    ps = ps.replace("''", '"')

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                         embedding=bert_dur,
                         embedding_scale=embedding_scale,
                         features=ref_s,  # reference from the same speaker as the embedding
                         num_steps=diffusion_steps).squeeze(1)

        if s_prev is not None:
            # convex combination of previous and current style
            s_pred = t * s_prev + (1 - t) * s_pred

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        s_pred = torch.cat([ref, s], dim=-1)

        d = model.predictor.text_encoder(d_en,
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr,
                            F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()[...,
           :-100], s_pred  # weird pulse at the end of the model, need to be fixed later


def STinference(text, ref_s, ref_text, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    ref_text = ref_text.strip()
    ps = global_phonemizer.phonemize([ref_text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    ref_tokens = textclenaer(ps)
    ref_tokens.insert(0, 0)
    ref_tokens = torch.LongTensor(ref_tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        ref_input_lengths = torch.LongTensor([ref_tokens.shape[-1]]).to(device)
        ref_text_mask = length_to_mask(ref_input_lengths).to(device)
        ref_bert_dur = model.bert(ref_tokens, attention_mask=(~ref_text_mask).int())
        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                         embedding=bert_dur,
                         embedding_scale=embedding_scale,
                         features=ref_s,  # reference from the same speaker as the embedding
                         num_steps=diffusion_steps).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en,
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr,
                            F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()[..., :-50]  # weird pulse at the end of the model, need to be fixed later


def split_text(text, max_length=50):
    words = text.split()
    chunks = []
    current_chunk = []

    for i in range(words):
        word = words[i]
        if len(current_chunk) + len(word) + 1 <= max_length:
            current_chunk.append(word)
        else:
            current_chunk.append("for")
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def generate_recursively(audio_file, directory, alpha, beta, diffusion_steps, embedding_scale, file_encoding="utf-8"):
    # Use glob to find all .txt files recursively
    txt_files = glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True)

    txt_files.sort(key=natural_keys)
    # Save a concatenated text file
    with open(directory + "/concatenated.txt", "w", encoding=file_encoding) as file:
        for txt_file in txt_files:
            with open(txt_file, "r", encoding=file_encoding) as f:
                file.write(f.read() + "\n")

    create_subtitle_files(directory + "/concatenated.txt", directory + "/subtitle.srt", directory + "/subtitle.sbv")

    # Create generated_voices directory if it doesn't exist
    if not os.path.exists(directory + "/generated_voices"):
        os.makedirs(directory + "/generated_voices")

    print("Generating speech for all text files...")
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding=file_encoding) as file:
                content = file.read()
                content = content.replace("-", " ")
                output = generate_speech(audio_file, content, alpha, beta, diffusion_steps, embedding_scale)

                if output:
                    # Get the output file name with convert txt to mp3 and adding generated_voices to the path
                    output_file = txt_file.replace(".txt", ".mp3").replace(directory + "/", directory + "/generated_voices/")

                    # Save the generated speech as an MP3 file
                    wav_file = "temp.wav"
                    wavfile.write(wav_file, *output[0])
                    audio = AudioSegment.from_file(wav_file)
                    audio.export(output_file, format="mp3")

                    # Delete the .wav file
                    os.remove(wav_file)

                    # Delete the .txt file
                    os.remove(txt_file)


        except Exception as e:
            print(f"Error reading {txt_file}: {e}")

        gc.collect()

        print("Audio generated")

    # Concatenate the generated files
    generated_files = glob.glob(os.path.join(directory + "/generated_voices", '**', '*.mp3'), recursive=True)
    # Sort the generated files
    print(generated_files)
    generated_files.sort(key=natural_keys)
    print(generated_files)
    audio = AudioSegment.from_file(generated_files[0])
    for file in generated_files[1:]:
        audio += AudioSegment.from_file(file)

    # Export the concatenated audio
    audio.export(directory + "/concatenated.mp3", format="mp3")


def gen_from_text(audio_file, text, alpha, beta, diffusion_steps, embedding_scale):
    voice, message = generate_speech(audio_file, text, alpha, beta, diffusion_steps, embedding_scale)

    # Save the generated speech as an MP3 file
    wav_file = "temp.wav"
    wavfile.write(wav_file, *voice)
    audio = AudioSegment.from_file(wav_file)
    audio.export("output.mp3", format="mp3")

    # Delete the .wav file
    os.remove(wav_file)

    # Zip the single file
    with zipfile.ZipFile("output.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write("output.mp3")

    return "output.zip", "output.mp3", message

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def parse_generate(audio_file, text_input_type, text_input, text_file, zip_file, pptx_inp,
                   alpha, beta, diffusion_steps, embedding_scale):
    # Clear the memory
    gc.collect()

    output_dir = "extracted"

    ## Delete the generated files if they exist
    if os.path.exists("output.zip"):
        os.remove("output.zip")
    if os.path.exists("generated.zip"):
        os.remove("generated.zip")


    # Make sure the audio file is a WAV file
    if audio_file is not None:
        audio_file = convert_to_wav(audio_file)

        if text_input_type == "Plain Text":
            result = gen_from_text(audio_file, text_input, alpha, beta, diffusion_steps, embedding_scale)

        if text_input_type == "TXT File":
            with open(text_file, "r") as f:
                text_input = f.read()
            result = gen_from_text(audio_file, text_input, alpha, beta, diffusion_steps, embedding_scale)

        elif text_input_type == "ZIP File":

            # Extract zip file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(output_dir)

            # Generate speech recursively
            generate_recursively(audio_file, output_dir, alpha, beta, diffusion_steps, embedding_scale)

            print("Generated all files")
            # Zip the generated files
            zip_output = "generated.zip"
            with zipfile.ZipFile(zip_output, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk the directory
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        # Create the full path of the file
                        file_path = os.path.join(root, file)
                        # Add file to the zip file, preserving the directory structure
                        zipf.write(file_path, os.path.relpath(file_path, output_dir))

            # Delete the extracted directory recursively without error
            os.system(f"rm -rf {output_dir} || true")

            result = zip_output, None, "Success: Speech generated successfully."


        elif text_input_type == "PowerPoint File":
            output_dir = "extracted_notes"
            zip_file_path = extract_notes(pptx_inp, output_dir, "notes.zip")

            output_dir = "extracted"

            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Create output/images directory if it doesn't exist
            if not os.path.exists(output_dir + "/images"):
                os.makedirs(output_dir + "/images")
            # Convert the PowerPoint slides to images
            os.system(f"soffice --headless --convert-to pdf {pptx_inp} --outdir {os.path.dirname(pptx_inp)}")
            os.system(
                f"pdftoppm -r 150 -jpeg -jpegopt quality=100 {pptx_inp.replace('.pptx', '.pdf')} {output_dir + '/images/slide'}")

            # Delete the PDF file
            os.remove(pptx_inp.replace('.pptx', '.pdf'))

            result = parse_generate(audio_file, "ZIP File", None, None, zip_file_path, None, alpha, beta, diffusion_steps,
                                  embedding_scale)

        # Delete the generated files if they exist
        if os.path.exists("output.mp3"):
           os.remove("output.mp3")
        if os.path.exists("notes.zip"):
           os.remove("notes.zip")
        if os.path.exists("extracted_notes"):
           # delete recursively and force delete
           os.system("rm -rf extracted_notes || true")
        return result

    else:
        return None, None, "Error: Audio file is missing."


# Insert here 3


def convert_to_wav(input_file, output_file=None):
    # Determine the output file name if not provided
    if not output_file:
        base, ext = os.path.splitext(input_file)
        output_file = base + ".wav"

    # Check if the input file is a WAV file
    if not input_file.endswith(".wav"):
        # Load the audio file
        audio = AudioSegment.from_file(input_file)

        # Export as WAV
        audio.export(output_file, format="wav")

    return output_file


# Insert here 1


# Insert here 2


def update_input_fields(input_type):
    if input_type == "Plain Text":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=True), gr.update(visible=False)
    elif input_type == "TXT File":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(
            visible=False), gr.update(visible=True), gr.update(visible=False)
    elif input_type == "ZIP File":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=True)
    elif input_type == "PowerPoint File":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=True), gr.update(visible=False), gr.update(visible=True)


# iface = gr.Interface(
#    fn=generate_speech,
#    inputs=[
#        gr.Audio(label="Upload a reference audio file", type="filepath"),
#        txt_radio,
#        txt_box,
#        txt_inp,
#        zip_inp,
#        gr.Slider(minimum=0.0, maximum=1.0, value=0.3, label="Alpha"),
#        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, label="Beta"),
#        gr.Slider(minimum=1, maximum=10, value=5, label="Diffusion Steps"),
#        gr.Slider(minimum=0.0, maximum=3.0, value=1.0, label="Embedding Scale"),
#    ],
#    outputs=[gr.Audio(label="Synthesized Audio"), gr.Textbox(label="Message")],
#    title="StyleTTS2 Voice Cloning",
#    description="Upload a reference wav audio file and enter text to synthesize speech."
# )

# Make upper part of the interface as block
with gr.Blocks() as iface:
    with gr.Row():
        # define inputs
        with gr.Column():
            audio_file = gr.Audio(label="Upload a reference audio file", type="filepath")
            txt_radio = gr.Radio(["Plain Text", "TXT File", "ZIP File", "PowerPoint File"],
                                 label="Input Type")
            txt_box = gr.Textbox(lines=2, placeholder="Enter text to synthesize", label="Text to Synthesize",
                                 visible=True)
            txt_inp = gr.File(label="Upload a TXT file", type="filepath", file_types=[".txt"], visible=False)
            zip_inp = gr.File(label="Upload a ZIP file", type="filepath", file_types=[".zip"], visible=False)
            pptx_inp = gr.File(label="Upload a PowerPoint file to generate from notes", type="filepath",
                               file_types=[".pptx"])

            with gr.Row():
                with gr.Column():
                    clear_button = gr.ClearButton(value="Clear")
                with gr.Column():
                    submit_button = gr.Button(value="Submit")

            alpha_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, label="Alpha")
            beta_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, label="Beta")
            diffusion_slider = gr.Slider(minimum=1, maximum=10, value=5, label="Diffusion Steps")
            embedding_slider = gr.Slider(minimum=0.0, maximum=3.0, value=1.0, label="Embedding Scale")

        with gr.Column():
            # define outputs
            audio_output = gr.Audio(label="Synthesized Audio", visible=True)
            zip_output = gr.File(label="Download ZIP file", visible=False)
            message_output = gr.Textbox(label="Message")

    txt_radio.change(update_input_fields, inputs=[txt_radio],
                     outputs=[txt_box, txt_inp, zip_inp, pptx_inp, audio_output, zip_output])

    submit_button.click(parse_generate,
                        inputs=[audio_file, txt_radio, txt_box, txt_inp, zip_inp, pptx_inp, alpha_slider, beta_slider,
                                diffusion_slider, embedding_slider],
                        outputs=[zip_output, audio_output, message_output])

if __name__ == "__main__":
    iface.launch( server_port=7861, share=True) #server_name="0.0.0.0",