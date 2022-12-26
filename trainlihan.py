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
"""wdsr train script"""
import argparse
import os
from mindspore import context
from mindspore import dataset as ds
import mindspore.nn as nn
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from src.data.dataset import Dataset
from src.model import REDNet30
import time


def train_net():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='REDNet30', help='REDNet10, REDNet20, REDNet30')
    parser.add_argument('--images_dir', type=str, default='/disk2/lihan/lihan/RED30/')
    parser.add_argument('--name', type=str, default='BSD300')
    parser.add_argument('--outputs_dir', type=str, default='./ckpt/')
    parser.add_argument('--jpeg_quality', type=int, default=30)
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=90)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    # parser.add_argument('--use_fast_loader', action='store_true')
    parser.add_argument('--ckpt_save_interval', type=int, default=10)
    parser.add_argument('--ckpt_save_max', type=int, default=5)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--loss_scale', type=float, default=1024.0)
    parser.add_argument('--init_loss_scale', type=float, default=65536.)

    opt = parser.parse_args()
    set_seed(1)
    device_id = int(os.getenv('DEVICE_ID', '0'))
    rank_id = int(os.getenv('RANK_ID', '0'))
    device_num = int(os.getenv('RANK_SIZE', '1'))
    # if distribute:
    if device_num > 1:
        init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          device_num=device_num, gradients_mean=True)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=device_id)
    train_dataset = Dataset(opt.images_dir, opt.name, opt.patch_size, opt.jpeg_quality)
    train_de_dataset = ds.GeneratorDataset(train_dataset, ["input", "label"], num_shards=device_num,
                                           shard_id=rank_id, shuffle=True)
    train_de_dataset = train_de_dataset.batch(opt.batch_size, drop_remainder=True)
    step_size = train_de_dataset.get_dataset_size()
    net_m = REDNet30()
    print("Init RED30 net successfully")


    lr = []
    for i in range(0, opt.num_epochs):
        cur_lr = opt.lr
        lr.extend([cur_lr] * step_size)
    temp = nn.Adam(net_m.trainable_params(), learning_rate=lr, loss_scale=opt.loss_scale)
    loss = nn.MSELoss()
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=opt.init_loss_scale, \
             scale_factor=2, scale_window=1000)
    model = Model(net_m, loss_fn=loss, optimizer=temp, loss_scale_manager=loss_scale_manager)
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=opt.ckpt_save_interval * step_size,
                                 keep_checkpoint_max=opt.ckpt_save_max)
    ckpt_cb = ModelCheckpoint(prefix="RED", directory=opt.outputs_dir, config=config_ck)
    cb += [ckpt_cb]
    model.train(opt.num_epochs, train_de_dataset, callbacks=cb, dataset_sink_mode=True)


if __name__ == "__main__":
    time_start = time.time()
    train_net()
    time_end = time.time()
    print('train_time: %f' % (time_end - time_start))

