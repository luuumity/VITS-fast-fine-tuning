# 此脚本包含批量重采样至16kHz的功能。
# 最终通过命令行（子进程）方式实现。
# 但应该需要pip install .，使用setup.py之后才能导入这些模块。
# 但如果只用到一个visqol_lib_py.so文件的话，我也许可以直接copy这一个文件，因为Colab上安装环境有些问题。
import torchaudio
import os
#import numpy
import soundfile as sf
import subprocess
import re

# 需要pip install .，使用setup.py之后才能导入这些模块。
from visqol import visqol_lib_py
# from visqol.pb2 import visqol_config_pb2
# from visqol.pb2 import similarity_result_pb2

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
    reference_file = os.path.join(operate_path, "gt_audio_"+str(i)+".wav")
    degraded_file = os.path.join(operate_path, "gen_audio_"+str(i)+".wav")

    # 仅用于获取采样率。
    sr1 = sf.info(reference_file).samplerate
    sr2 = sf.info(degraded_file).samplerate
    
    if sr1 == 16000 and sr2 == 16000:
		# 使用visqol的绝对路径
        visqol_dir = "/Users/sunjiayi/Downloads/visqol/"
        visqol_path = visqol_dir + "bazel-bin/visqol"
		# 确定所用模型的绝对路径
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
        model_path = os.path.join(
            os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
        result = subprocess.run([visqol_path,"--use_speech_mode","--reference_file",reference_file,"--degraded_file",degraded_file,"--similarity_to_quality_model",model_path], 
                                capture_output=True, text=True)
		# 使用正则表达式剥去命令行的输出，获得浮点数。
        matches = re.findall(r"(\d+\.\d+)", result.stdout)
        if matches:
            float_mark = float(matches[0])
        return float_mark

if __name__=="__main__":
    mark = eval(1,"./")
    print(mark)
    print(type(mark))
    # mark = (float)(mark)
    # print(mark, type(mark))
