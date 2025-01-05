import os
import subprocess
import librosa
import noisereduce as nr
import soundfile as sf
import zipfile
import shutil
from yt_dlp import YoutubeDL
import whisper


class YouTubeDownloader:
    @staticmethod
    def download_audio(url, output_path="input.wav"):
        options = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [
                {'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'},
            ],
        }
        with YoutubeDL(options) as ydl:
            ydl.download([url])
        print(f"Audio downloaded to: {output_path}")


class WhisperTranscriber:
    def __init__(self, model_name="small"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)
        print(f"Transcription completed: {result['text'][:300]}")
        return result


class AudioProcessor:
    @staticmethod
    def split_audio(input_file, segments, output_dir="wavs", target_sample_rate=22050):
        os.makedirs(output_dir, exist_ok=True)
        for i, r in enumerate(segments):
            start, end = r["start"], r["end"]
            output_file = os.path.join(output_dir, f"audio{i+1}.wav")
            ffmpeg_command = [
                "ffmpeg", "-y", "-i", input_file, "-ss", str(start), "-to", str(end),
                "-ar", str(target_sample_rate),  # 샘플 레이트 설정
                "-hide_banner", "-loglevel", "error", output_file
            ]
            subprocess.run(ffmpeg_command, check=True)
        print(f"Audio split into {output_dir} with sample rate {target_sample_rate}")

    @staticmethod
    def reduce_noise(input_dir="wavs", output_dir="processed_wavs"):
        os.makedirs(output_dir, exist_ok=True)
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".wav"):
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, file_name)
                y, sr = librosa.load(input_path, sr=None)
                noise_sample = y[:int(sr * 0.5)]
                cleaned_audio = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
                sf.write(output_path, cleaned_audio, sr)
        print(f"Noise reduced audio saved to: {output_dir}")


class MetadataGenerator:
    @staticmethod
    def generate_metadata(segments, output_file="metadata.txt"):
        with open(output_file, "w", encoding="utf-8") as f:
            for i, r in enumerate(segments):
                f.write(f"audio{i+1}|{r['text'].strip()}|{r['text'].strip()}\n")
        print(f"Metadata saved to: {output_file}")


class DatasetCompressor:
    @staticmethod
    def compress_dataset(files_to_zip, output_zip="dataset.zip", output_dir="dataset"):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define the path for the zip file within the output directory
        output_zip_path = os.path.join(output_dir, output_zip)

        # Create the zip file at the specified location
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_or_dir in files_to_zip:
                if os.path.isdir(file_or_dir):
                    for root, _, files in os.walk(file_or_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, start=os.path.dirname(file_or_dir))
                            zipf.write(file_path, arcname)
                else:
                    zipf.write(file_or_dir, os.path.basename(file_or_dir))
        
        print(f"Dataset compressed to: {output_zip_path}")


def main():
    # Step 1: Download audio from YouTube
    youtube_url = 'https://www.youtube.com/watch?v=2zEJ5runkSs'
    audio = "kato5"
    # YouTubeDownloader.download_audio(youtube_url, audio)

    # Step 2: Transcribe audio using Whisper
    audio_path = "kato5.wav"
    transcriber = WhisperTranscriber(model_name="small")
    transcription_result = transcriber.transcribe(audio_path)

    # Step 3: Split audio into segments
    AudioProcessor.split_audio(audio_path, transcription_result['segments'], output_dir="wavs")

    # Step 4: Reduce noise in audio files
    AudioProcessor.reduce_noise(input_dir="wavs", output_dir="processed_wavs")

    # Step 5: Generate metadata
    MetadataGenerator.generate_metadata(transcription_result['segments'], output_file="metadata.txt")

    # Step 6: Compress dataset
    files_to_zip = ["processed_wavs", "metadata.txt"]
    DatasetCompressor.compress_dataset(files_to_zip, output_zip="kato5_noisex.zip", output_dir="tts-dataset")


if __name__ == "__main__":
    main()

# import os
# from yt_dlp import YoutubeDL

# url = 'https://www.youtube.com/watch?v=Ucd5JSlr7EM'
# output_path = './input'

# options = {
#     'format': 'bestaudio/best',
#     'outtmpl': output_path,
#     'postprocessors': [
#         {'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'},
#     ],
# }

# with YoutubeDL(options) as ydl:
#     ydl.download([url])


# import whisper

# model = whisper.load_model("small")

# result = model.transcribe(r"C:\Users\User\Desktop\python\TTS\input.wav")

# print(result["text"][:300])

# print(result["segments"][:3])

# print(len(result["text"]))

# for r in result['segments']:
#     print(f'[{r["start"]:.2f} --> {r["end"]:.2f}] {r["text"]}')

# import os
# import subprocess

# # 오디오 파일 경로와 출력 디렉토리 설정
# input_file = "input.wav"
# output_dir = "wavs"
# os.makedirs(output_dir, exist_ok=True)  # 출력 디렉토리 생성

# # Segments를 처리하며 FFmpeg 실행
# for i, r in enumerate(result['segments']):
#     start = r["start"]
#     end = r["end"]
#     output_file = os.path.join(output_dir, f"audio{i+1}.wav")
    
#     # FFmpeg 명령어
#     ffmpeg_command = [
#         "ffmpeg",
#         "-y",  # 기존 파일 덮어쓰기
#         "-i", input_file,
#         "-ss", str(start),
#         "-to", str(end),
#         "-hide_banner",
#         "-loglevel", "error",
#         output_file
#     ]
    
#     # FFmpeg 실행
#     subprocess.run(ffmpeg_command, check=True)

# print("오디오 파일 분할 완료!")

# # 배경음 제거
# import librosa
# import noisereduce as nr
# import soundfile as sf
# import os

# processed_dir = "processed_wavs"  # 배경음 제거 파일 저장 디렉토리
# os.makedirs(processed_dir, exist_ok=True)  # 배경음 제거 파일 저장 디렉토리 생성

# # 분할된 파일에 대해 배경음 제거 수행
# for file_name in os.listdir(output_dir):
#     if file_name.endswith(".wav"):  # WAV 파일만 처리
#         input_path = os.path.join(output_dir, file_name)
#         output_path = os.path.join(processed_dir, file_name)

#         # 오디오 파일 로드
#         y, sr = librosa.load(input_path, sr=None)

#         # 배경 소음 샘플 추출 (초기 0.5초를 배경 소음으로 가정)
#         noise_sample = y[:int(sr * 0.5)]

#         # 노이즈 제거
#         cleaned_audio = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)

#         # 결과 저장
#         sf.write(output_path, cleaned_audio, sr)

# print(f"배경음 제거 완료! 결과 저장 경로: {processed_dir}")


# # metadata 만들기
# with open("metadata.txt", "w", encoding="utf-8") as f:
#     for i, r in enumerate(result['segments']):
#         f.write(f"audio{i+1}|{r['text'].strip()}|{r['text'].strip()}\n")

# # 압축
# import os
# import shutil
# import zipfile

# # 출력 디렉토리 생성
# output_dir = "tts-dataset"
# os.makedirs(output_dir, exist_ok=True)  # 디렉토리 생성, 이미 있으면 무시

# # 압축할 폴더와 파일
# files_to_zip = ["processed_wavs", "metadata.txt"]
# zip_file_path = "yangazi.zip"

# # ZIP 파일 생성
# with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#     for file_or_dir in files_to_zip:
#         if os.path.isdir(file_or_dir):  # 디렉토리일 경우
#             for root, _, files in os.walk(file_or_dir):
#                 for file in files:
#                     file_path = os.path.join(root, file)
#                     arcname = os.path.relpath(file_path, start=os.path.dirname(file_or_dir))
#                     zipf.write(file_path, arcname)
#         else:  # 파일일 경우
#             zipf.write(file_or_dir, os.path.basename(file_or_dir))

# print(f"ZIP 파일 생성 완료: {zip_file_path}")

# # ZIP 파일을 대상 디렉토리로 복사
# shutil.copy(zip_file_path, output_dir)
# print(f"ZIP 파일 복사 완료: {output_dir}")