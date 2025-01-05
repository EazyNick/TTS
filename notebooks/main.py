import os
# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.tts.configs.shared_configs import BaseDatasetConfig

output_path = "tts_train_dir"
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
dataset_config = BaseDatasetConfig(
    formatter="kss", meta_file_train="metadata.csv", path=os.path.join(output_path, "timo")
)

# GlowTTSConfig: all model related values for training, validating and testing.
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
config = GlowTTSConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=100,
    text_cleaner="phoneme_cleaners", # "phoneme_cleaners"
    use_phonemes=True,
    phoneme_language="ko-kr",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    save_step=1000,
)

from TTS.utils.audio import AudioProcessor
ap = AudioProcessor.init_from_config(config)
# Modify sample rate if for a custom audio dataset:
# ap.sample_rate = 48000  

from TTS.tts.utils.text.tokenizer import TTSTokenizer
tokenizer, config = TTSTokenizer.init_from_config(config)

# Tokenizer 어휘 확인
print("Tokenizer Vocabulary:", tokenizer.characters.vocab)

from TTS.tts.datasets import load_tts_samples
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=0.1, # config.eval_split_size
)

from TTS.tts.models.glow_tts import GlowTTS
model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

from trainer import Trainer, TrainerArgs
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

def main():
    # Your existing training setup code here
    trainer.fit()

if __name__ == '__main__':

    from TTS.tts.utils.text.korean.phonemizer import korean_text_to_phonemes
    # 텍스트 -> Phonemizer -> Tokenizer ID 확인
    text = "정찰다녀오겠습니다."
    phonemes = korean_text_to_phonemes(text)
    print("Phonemes:", phonemes)

    # Phonemizer 결과가 Tokenizer 어휘에 포함되는지 확인
    missing_chars = [ch for ch in phonemes if ch not in tokenizer.characters.vocab]
    if missing_chars:
        print("Missing Characters in Vocabulary:", missing_chars)
    else:
        print("All Phonemes are in the Vocabulary.")


    # import multiprocessing
    # multiprocessing.freeze_support()
    # main()

# --------------------------------------------------------------------------

# import os
# # BaseDatasetConfig: defines name, formatter and path of the dataset.
# from TTS.tts.configs.shared_configs import BaseDatasetConfig

# output_path = "tts_train_dir"
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
    
# dataset_config = BaseDatasetConfig(
#     formatter="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "LJSpeech-1.1/")
# )

# # GlowTTSConfig: all model related values for training, validating and testing.
# from TTS.tts.configs.glow_tts_config import GlowTTSConfig
# config = GlowTTSConfig(
#     batch_size=32,
#     eval_batch_size=16,
#     num_loader_workers=4,
#     num_eval_loader_workers=4,
#     run_eval=True,
#     test_delay_epochs=-1,
#     epochs=100,
#     text_cleaner="phoneme_cleaners", # "phoneme_cleaners"
#     use_phonemes=True,
#     phoneme_language="en-us",
#     phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
#     print_step=25,
#     print_eval=False,
#     mixed_precision=True,
#     output_path=output_path,
#     datasets=[dataset_config],
#     save_step=1000,
# )

# from TTS.utils.audio import AudioProcessor
# ap = AudioProcessor.init_from_config(config)
# # Modify sample rate if for a custom audio dataset:
# # ap.sample_rate = 48000  

# from TTS.tts.utils.text.tokenizer import TTSTokenizer
# tokenizer, config = TTSTokenizer.init_from_config(config)

# from TTS.tts.datasets import load_tts_samples
# train_samples, eval_samples = load_tts_samples(
#     dataset_config,
#     eval_split=True,
#     eval_split_max_size=config.eval_split_max_size,
#     eval_split_size=config.eval_split_size, # config.eval_split_size
# )

# from TTS.tts.models.glow_tts import GlowTTS
# model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

# from trainer import Trainer, TrainerArgs
# trainer = Trainer(
#     TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
# )

# def main():
#     # Your existing training setup code here
#     trainer.fit()

# if __name__ == '__main__':
#     import multiprocessing
#     multiprocessing.freeze_support()
#     main()



