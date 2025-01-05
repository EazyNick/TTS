import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# tensorboard --logdir=run/training/ --host=127.0.0.1 --port=6006

# Logging parameters
RUN_NAME = "GPT_XTTS_v2.0_kato1_FT"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# Set here the path that the checkpoints will be saved. Default: ./run/training/
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
START_WITH_EVAL = True  # if True it will star with evaluation
BATCH_SIZE = 2  # set here the batch size
GRAD_ACUMM_STEPS = 126  # set here the grad accumulation steps
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.

# Define here the dataset that you want to use for the fine-tuning on.
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="kato1",
    path=r"C:\Users\User\Desktop\python\TTS\tts-dataset\kato1",
    meta_file_train=r"C:\Users\User\Desktop\python\TTS\tts-dataset\kato1\metadata.txt",
    language="ko",
)

# Add here the configs of the datasets
DATASETS_CONFIG_LIST = [config_dataset]

# Define the path where XTTS v2.0.1 files will be downloaded
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


# DVAE files
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

# Set the path to the downloaded files
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

# download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)


# Download XTTS v2.0 checkpoint if needed
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file

# download XTTS v2.0 files if needed
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )

# Training sentences generations
SPEAKER_REFERENCE = [
    r"C:\Users\User\Desktop\python\TTS\tts-dataset\kato1\wavs\audio1.wav"  # speaker reference to be used in training test sentences
]
LANGUAGE = config_dataset.language

def main():
    
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=220500,  # 10 secs
        min_conditioning_length=22050,  # 1 secs
        debug_loading_failures=False,
        max_wav_length=286125,  # ~13 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        epochs=64,
        log_model_step=1000,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="StepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"step_size": 50, "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "엑스스폴릿인데 뭔가 충돌돼서 자꾸 꺼지는 것 같아요.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "오늘 나랑 니케팝업 가기로 했잖아",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "나 오늘을 위해서 새로운 의상까지 구입했는데, 벌써 까먹은 거야?",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "자기 잠깐만 기다려봐, 빨리 갈아입고 올게",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "아 자기야 왜 이렇게 와앙~ 오늘 나랑 데이트하기로 한 거 잊었어?!",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "내가 카페에도 올렸는데, 쪽지 보내기 기능이 차단이 됐다는 거예요",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "왜 쪽지 보내기가 차단이 된 거예요 선생님? 이러면서 제가 문의를 써서 올렸어요",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "여러분들이 보고 싶은 컨텐츠나 혹은 같이 했으면 좋겠는 성우분이 계시다면 꼭",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "아무리 황량하고 따분해도 다른 아름다운 곳보다 고향에서 살고 싶어 하는 게 사람이에요. 세상에 집보다 좋은 곳은 없거든요.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "바보들은 심장이 있어도 그걸로 무엇을 해야 하는지 몰라요.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "살아 있는 존재라면 누구나 위험 앞에서 두려움을 느껴. 진정한 용기란 두려움에도 불구하고 위험에 맞서는 것인데, 너는 이미 그런 용기를 충분히 갖고 있어.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "너는 작가가 되지 않아도, 배우가 되지 않아도, 그저 너이기에 사랑스럽고 완전한 존재란다. 다른 무엇이 되려고 애쓰지 않아도 좋아. 너는 그저 너, 너다운 너이기만 하면 된단다.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "소설을 쓰며 사는 삶보다 소설처럼 살아가는 삶이 훨씬 더 재미있을 거예요.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "저는 매 순간 제가 행복하다는 사실을 온전히 느껴요. 아무리 속상한 일이 생겨도 그 사실을 잊지 않을 거예요.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "저는 인간에게 가장 필요한 자질은 상상력이라고 생각합니다. 상상력이 있어야 타인을 이해할 수 있고, 그래야 친절할 수도, 남을 이해할 수도, 또 동정할 수도 있으니까요.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "자신에게 찾아오는 기회를 붙잡을 의지만 있다면 세상은 행복으로 가득 차 있고, 가볼 곳도 많아요. 비법은 바로 유연한 마음이에요.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "정말 소중한 것은 커다란 기쁨이 아니에요. 사소한 곳에서 얻는 기쁨이 더 중요해요.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "큰 시련이 닥쳤을 때만 인격이 필요한 게 아니에요. 위기에 대처하거나, 치명적인 비극에 맞서는 건 누구나 할 수 있지만, 그날그날의 사소한 불운들을 웃음으로 넘기는 일은 '정신력'이 없다면 불가능한 일이에요. 제가 키워나가야 할 게 바로 이런 종류의 인격이에요.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "눈이 보이지 않으면 눈이 보이는 코끼리와 살을 맞대고 걸으면 되고, 다리가 불편하면 다리가 튼튼한 코끼리에게 기대서 걸으면 돼. 같이 있으면 그런 건 큰 문제가 아니야.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "누구든 너를 좋아하게 되면, 네가 누구인지 알아볼 수 있어. 아마 처음에는 호기심으로 너를 관찰하겠지. 하지만 점 너를 좋아하게 되어서 너를 눈여겨보게 되고, 네가 가까이 있을 때는 어떤 냄새가 나는지 알게 될 거고, 네가 걸을 때는 어떤 소리가 나는지에도 귀 기울이게 될 거야. 그게 바로 너야.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # 확인용 코드: 텍스트 길이 95자 초과 데이터 출력
    MAX_LENGTH = 95
    print(f"Checking for text exceeding {MAX_LENGTH} characters...")
    for sample in train_samples:
        text = sample["text"]
        if len(text) > MAX_LENGTH:
            print(f"[Warning] Text exceeds {MAX_LENGTH} characters: {text} (Length: {len(text)})")

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

if __name__ == "__main__":
    main()
