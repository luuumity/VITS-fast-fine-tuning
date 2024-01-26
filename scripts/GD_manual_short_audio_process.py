import os
import json
import torchaudio
import argparse
import torch
import numpy as np

with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
    hps = json.load(f)
target_sr = hps['data']['sampling_rate']

parent_dir = "./custom_character_voice/"
speaker_names = list(os.walk(parent_dir))[0][1]

for speaker in speaker_names:
    for root, dirs, files in os.walk(parent_dir + speaker):
        for file in files:
            if file.endswith(".wav"):
                # print(file)
                wav, sr = torchaudio.load(os.path.join(root, file), frame_offset=0, num_frames=-1,
                                        normalize=True, channels_first=True)
                # 我和原音频处理脚本就差了这么一句话：
                wav = wav.mean(dim=0).unsqueeze(0)
                # 但如果我第一次跑这个脚本没有加这句话，后面再加上也没有用。因为后面的if判断不满足，这个wav并不会被保存！！！
                if sr != target_sr:
                    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
                    torchaudio.save(os.path.join(root, file), wav, target_sr, channels_first=True)

# 运行完后可以用!file命令查看采样率是否为json中的目标采样率。

print("修改格式中：")
print(speaker_names[0])

speaker_annos = []           
i = 1         

with open("short_character_raw.txt", 'r') as f:
    line=f.readline()
    while line:
        # 去掉原本回车符：（需要）
        line = line.strip()
        text = "[GD]"+line+"[GD]"
        save_path = parent_dir + speaker_names[0] + "/" + speaker_names[0] + f"_{i}.wav"
        speaker_annos.append(save_path + "|" + speaker_names[0] + "|" + text)
        # 每次打印检查：
        print(speaker_annos[i-1])
        i+=1
        # 必须用这个才能读取下一行！否则会陷入死循环！一直在读取第一行。
        line=f.readline()

with open("short_character_anno.txt", 'w', encoding='utf-8') as f:
    for line in speaker_annos:
        # 需要手动补上换行符，竟然不会自动换行！
        # 也是因为f.write()，又不是f.writelines()，那个line只是我们自己取的变量名。
        f.write(line + "\n")
