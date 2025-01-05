# import os
# import subprocess
# from IPython.display import Audio

# # 학습된 모델 및 설정 파일 경로
# test_ckpt = r"C:\Users\User\Desktop\python\TTS\run\training\GPT_XTTS_v2.0_kato1_FT-January-04-2025_04+20PM-dbf1a08a"
# test_config = r"C:\Users\User\Desktop\python\TTS\run\training\GPT_XTTS_v2.0_kato1_FT-January-04-2025_04+20PM-dbf1a08a\config.json"

# # 생성할 텍스트
# input_text = "안녕하세요. 이 문장은 텍스트를 음성으로 변환하기 위한 예제입니다."
# speaker_wav = r"C:\Users\User\Desktop\python\TTS\tts-dataset\kato1\wavs\audio1.wav"  # speaker reference to be used in training test sentences

# output_wav_path = "out.wav"  # 출력 경로
# tokenizer_file = r"C:\Users\User\Desktop\python\TTS\run\training\XTTS_v2.0_original_model_files\vocab.json"

# # TTS 명령 실행
# command = [
#     "tts",
#     "--text", input_text,
#     "--model_path", test_ckpt,
#     "--config_path", test_config,
#     "--out_path", output_wav_path,
#     "--language_idx", "ko", 
#     "--speaker_wav", speaker_wav,  # 화자 정보 추가
# ]
# print(f"Running command: {' '.join(command)}")
# subprocess.run(command, check=True)

# # 생성된 음성을 Jupyter Notebook에서 재생
# Audio(output_wav_path)

# #cmd에 입력
# & tts --text "오늘 나랑 니케팝업 가기로 했잖아" `
#     --model_path "C:/Users/User/Desktop/python/TTS/run/training/GPT_XTTS_v2.0_kato1_FT-January-05-2025_02+45PM-dbf1a08a" `
#     --config_path "C:/Users/User/Desktop/python/TTS/run/training/GPT_XTTS_v2.0_kato1_FT-January-05-2025_02+45PM-dbf1a08a/config.json" `
#     --out_path "out.wav" `
#     --language_idx "ko" `
#     --speaker_wav "C:/Users/User/Desktop/python/TTS/tts-dataset/kato1/wavs/audio1.wav"

# from gtts import gTTS
# from TTS.api import TTS
# import torch
# import os

# def create_reference_audio(text="안녕하세요. 저는 한국어 음성 안내 시스템입니다. 깨끗한 음성으로 안내해드리도록 하겠습니다.", 
#                         filename="reference_voice.wav"):
#     """
#     gTTS를 사용하여 참조용 한국어 음성 파일을 생성합니다.
#     """
#     try:
#         # MP3로 먼저 생성 (gTTS는 기본적으로 MP3만 지원)
#         mp3_filename = "temp_reference.mp3"
#         tts = gTTS(text=text, lang='ko', slow=False)
#         tts.save(mp3_filename)
        
#         # MP3를 WAV로 변환
#         from pydub import AudioSegment
#         audio = AudioSegment.from_mp3(mp3_filename)
#         audio.export(filename, format="wav")
        
#         # 임시 MP3 파일 삭제
#         os.remove(mp3_filename)
        
#         print(f"참조용 음성이 생성되었습니다: {filename}")
#         return filename
    
#     except Exception as e:
#         print(f"참조용 음성 생성 중 오류 발생: {str(e)}")
#         return None

# def generate_korean_speech(text, speaker_wav, output_path="output.wav"):
#     """
#     XTTS v2를 사용하여 한국어 텍스트를 음성으로 변환합니다.
#     """
#     try:
#         # XTTS v2 모델 로드
#         tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", 
#                 progress_bar=True, 
#                 gpu=torch.cuda.is_available())
        
#         # 음성 생성
#         tts.tts_to_file(text=text,
#                     file_path=output_path,
#                     speaker_wav=speaker_wav,
#                     language="ko")
        
#         print(f"XTTS 음성이 생성되었습니다: {output_path}")
#         return output_path
    
#     except Exception as e:
#         print(f"XTTS 음성 생성 중 오류 발생: {str(e)}")
#         return None

# def main():
#     # 1. 먼저 참조용 음성 생성
#     reference_wav = create_reference_audio()
    
#     if not reference_wav:
#         print("참조용 음성 생성에 실패했습니다.")
#         return
    
#     # 2. XTTS로 실제 변환하고 싶은 텍스트 설정
#     test_text = "안녕하세요, 반갑습니다. 오늘도 좋은 하루 보내세요. 이 음성은 XTTS v2를 사용하여 생성된 한국어 음성입니다."
    
#     # 3. XTTS를 사용하여 고품질 음성 생성
#     output_file = generate_korean_speech(test_text, reference_wav)
    
#     if output_file:
#         print("모든 과정이 성공적으로 완료되었습니다.")

# if __name__ == "__main__":
#     main()




