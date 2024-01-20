# 此脚本包含批量重采样至16kHz的功能。
# visqol的api要求输入的2个音频是双精度浮点数double列表。
# 用soundfile读取出来的就是这种类型！
import torchaudio
import os
import soundfile as sf

# 这个需要在visqol文件夹内执行 pip install . ，执行其setup.py文件，从而才能生成可以导入的module
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

    reference, sr1 = sf.read(os.path.join(operate_path, "gt_audio_"+str(i)+".wav"))
    degraded, sr2 = sf.read(os.path.join(operate_path, "gen_audio_"+str(i)+".wav"))

    print(reference.shape, reference.dtype, sr1)
    print(degraded.shape, degraded.dtype, sr2)
    
    similarity_result = api.Measure(reference, degraded)

    return similarity_result.moslqo
