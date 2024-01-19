from torch.utils.data import DataLoader

import utils

from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler
)

hps = utils.get_hparams()
symbols = hps['symbols']
collate_fn = TextAudioSpeakerCollate()

eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data, symbols)
eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False,
    batch_size=hps.train.batch_size, pin_memory=True,
    drop_last=False, collate_fn=collate_fn)

for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
    print(batch_idx)
    print(x.shape)
    print(x_lengths)
    print(spec.shape)
    print(spec_lengths)
    print(y.shape)
    print(y_lengths)
    print(speakers)

# 看来这个循环就只会执行一次，所以那个break并不起到关键因素。
# 起到关键因素的是截断到第几个，就是选取几个句子作为评估输出。

# 0 —— 现在是第一批次batch？
# torch.Size([14, 205]) —— 这个14经常出现，eval集里确实只有14个句子。14x205的矩阵，有14行。用于生产eval音频，每次截取1行即可。
# tensor([199, 197, 205, 157, 155, 139, 155, 115,  87, 109, 129,  95,  81,  55]) x_lengths有14个元素，是每个句子的长度。每次也截取1个用于infer。
# torch.Size([14, 513, 838])
# tensor([838, 800, 706, 632, 578, 562, 514, 475, 393, 378, 364, 308, 303, 187])
# torch.Size([14, 1, 214532])
# tensor([214532, 204941, 180847, 161943, 148022, 144024, 131801, 121680, 100622,
#          96976,  93316,  78881,  77653,  47981])
# tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) —— speaker id 都是0，因为只有一个说话人。