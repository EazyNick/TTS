import torch
from TTS.api import TTS

# Check if CUDA is installed
if torch.cuda.is_available():
    print("CUDA installed successfully\n")
    device = "cuda"  # Use GPU if available
else:
    print("CUDA not properly installed. Using CPU instead.\n")
    device = "cpu"  # Fallback to CPU

# Print available TTS models
view_models = input("View models? [y/n]\n")
if view_models == "y":
    tts_manager = TTS().list_models()
    all_models = tts_manager.list_models()
    print("TTS models:\n", all_models, "\n", sep="")

# Prompt model selection
model = input("Enter model:\n")
# for example, tts_models/multilingual/multi-dataset/xtts_v2

# Example voice cloning with selected model
tts = TTS(model, progress_bar=True).to(device)
tts.tts_to_file(
    "정찰다녀오겠습니다. 출동 준비 완료! 신속히! 크기가 전부는 아니잖아요.", 
    speaker_wav="train-audio.wav", 
    language="ko", 
    file_path="output.wav"
)

# from TTS.tts.utils.text.phonemizers.ko_kr_phonemizer import KO_KR_Phonemizer

# phonemizer = KO_KR_Phonemizer()
# result = phonemizer.phonemize("안녕하께세요.")
# print(result)