# 此脚本包含批量重采样至16kHz的功能。
import torchaudio
import os

# 这个应该不用在意visqol和visqol.pb2的位置，build之后应该自动能找到。
# 但也有可能环境情况特殊。
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2


def eval(i, operate_path):
    # 如果不是16kHz，批量重采样至16kHz：
    target_sr = 16000

    for root, dirs, files in os.walk(operate_path):
        for file in files:
            if file.endswith(".wav"):
                # print(file)
                wav, sr = torchaudio.load(os.path.join(root, file), frame_offset=0, num_frames=-1,
                                        normalize=True, channels_first=True)
                if sr != target_sr:
                    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
                    torchaudio.save(os.path.join(root, file), wav, target_sr, channels_first=True)

    # 计算所传入的第i个wav文件的对比评分：
    config = visqol_config_pb2.VisqolConfig()

    config.audio.sample_rate = 16000
    config.options.use_speech_scoring = True
    svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"

    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)

    api = visqol_lib_py.VisqolApi()

    api.Create(config)

    reference = os.path.join(operate_path, "gt_audio_"+str(i)+".wav")
    degraded = os.path.join(operate_path, "gen_audio_"+str(i)+".wav")
    
    similarity_result = api.Measure(reference, degraded)

    return similarity_result.moslqo
