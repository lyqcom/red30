# 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""wdsr eval script"""
import argparse
import glob
import os
import cv2
import io
import time
import numpy as np
import mindspore.dataset as ds
from mindspore import Tensor, context, ops, nn
from mindspore.common import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.data.dataset import Dataset
from src.model import REDNet30
from src.metrics import get_PSNR_SSIM
import PIL.Image as pil_image

device_id = int(os.getenv('DEVICE_ID', '1'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id, save_graphs=False)
context.set_context(max_call_depth=10000)

def normalize(data):
    return data / 255.

def eval_net():
    """eval"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default="./ckpt/RED_5-20_18.ckpt")
    parser.add_argument('--images_dir', type=str, default="/disk2/lihan/lihan/RED30/")
    parser.add_argument('--data_test', type=str, default="BSD200")
    parser.add_argument('--jpeg_quality', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=50)
    opt = parser.parse_args()

    fourteen = (opt.data_test == "fourteen")
    # eval_dataset = Dataset(opt.images_dir, opt.data_test, opt.patch_size, opt.jpeg_quality, test=True, fourteen=fourteen)
    # eval_de_dataset = ds.GeneratorDataset(eval_dataset, ["input", "label"], shuffle=False)
    # eval_de_dataset = eval_de_dataset.batch(1, drop_remainder=True)
    # eval_loader = eval_de_dataset.create_dict_iterator(output_numpy=True)
    net_m = REDNet30()
    if opt.ckpt_path:
        param_dict = load_checkpoint(opt.ckpt_path)
        load_param_into_net(net_m, param_dict)
    net_m.set_train(False)
    print('load mindspore RED30 net successfully.')
    # num_imgs = eval_de_dataset.get_dataset_size()
    # psnrs = np.zeros((num_imgs, 1))
    # ssims = np.zeros((num_imgs, 1))
    file_source = glob.glob(os.path.join(opt.images_dir, opt.data_test, '*jpg'))
    # file_source = glob.glob(os.path.join(opt.images_dir, opt.data_test, '*bmp'))
    file_source.sort()
    psnr_test = 0
    ssim_test = 0
    for f in file_source:
        # img = cv2.imread(f)
        # if img.shape[0] % 2 != 0:
        #     img = img[0:img.shape[0]-1, :, :]
        # if img.shape[1] % 2 != 0:
        #     img = img[:, 0:img.shape[1]-1, :]
        # img = pil_image.fromarray(img)
        # buffer = io.BytesIO()
        # img.save(buffer, format='jpeg', quality=opt.jpeg_quality)
        # img = pil_image.open(buffer)
        # img = normalize(np.float32(img))
        # img = np.transpose(img, axes=[2, 0, 1])
        # img = np.expand_dims(img, 0)
        # source = Tensor(img, dtype=mstype.float32)
        # noisy_img = source
        # out = ops.clip_by_value(noisy_img - net_m(noisy_img),
        #                         Tensor(0., mstype.float32), Tensor(1., mstype.float32))

        img = cv2.imread(f)
        if img.shape[0] % 2 != 0:
            img = img[0:img.shape[0]-1, :, :]
        if img.shape[1] % 2 != 0:
            img = img[:, 0:img.shape[1]-1, :]
        img = normalize(np.float32(img))
        img = np.transpose(img, axes=[2, 0, 1])
        img = np.expand_dims(img, 0)
        # print(img.shape)
        source = Tensor(img, dtype=mstype.float32)
        noise = np.random.standard_normal(size=source.shape) * (opt.jpeg_quality / 255.0)
        noise = Tensor(noise, dtype=mstype.float32)
        noisy_img = source + noise
        out = ops.clip_by_value(noisy_img - net_m(noisy_img),
                                Tensor(0., mstype.float32), Tensor(1., mstype.float32))
        # psnr, ssim = get_PSNR_SSIM(out, source, 3.)
        psnr = nn.PSNR()(out, source)
        ssim = nn.SSIM()(out, source)
        psnr_test += psnr
        ssim_test += ssim
    psnr_test = psnr_test / len(file_source)
    ssim_test = ssim_test / len(file_source)
    print("psnr:", psnr_test)
    print("ssim:", ssim_test)

if __name__ == '__main__':
    time_start = time.time()
    print("Start eval function!")
    eval_net()
    time_end = time.time()
    print('eval_time: %f' % (time_end - time_start))
