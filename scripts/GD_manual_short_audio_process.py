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
                if sr != target_sr:
                    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
                    torchaudio.save(os.path.join(root, file), wav, target_sr, channels_first=True)

# 运行完后可以用!file命令查看采样率是否为json中的目标采样率。

speaker_annos = []           
i = 1         

with open("short_character_anno.txt", 'r', encoding='utf-8') as f:
    line=file.readline()
    while line:
        text = "[GD]"+line+"[GD]"
        save_path = parent_dir + speaker + "/" + f"{speaker}_{i}.wav"
        speaker_annos.append(save_path + "|" + speaker + "|" + text)

with open("short_character_anno.txt", 'w', encoding='utf-8') as f:
    for line in speaker_annos:
        f.write(line)