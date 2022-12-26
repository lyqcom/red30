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
""""srdata"""
import os
import glob
import random
import pickle
import imageio
from src.data import common
from PIL import ImageFile
import io
import numpy as np
import PIL.Image as pil_image

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset:
    def __init__(self, images_dir, name, patch_size, jpeg_quality, n_channels=3, test=False, fourteen=False):
        self.patch_size = patch_size
        self.jpeg_quality = jpeg_quality
        self.n_channels = n_channels
        self.test = test
        self.fourteen = fourteen
        self.name = name
        self._set_filesystem(images_dir)

    def _set_filesystem(self, images_dir):
        self.images_dir = os.path.join(images_dir, self.name)
        names = sorted(glob.glob(os.path.join(self.images_dir, '*')))
        self.images = names

    def __getitem__(self, idx):
        label = pil_image.open(self.images[idx]).convert('RGB')
        # if self.test:
        #     label = pil_image.open(self.images[idx]).convert('L')
        # else:
        #     label = pil_image.open(self.images[idx]).convert('RGB')

        # randomly crop patch from training set
        if not self.fourteen:
            crop_x = random.randint(0, label.width - self.patch_size)
            crop_y = random.randint(0, label.height - self.patch_size)
            label = label.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))

        # additive jpeg noise
        # buffer = io.BytesIO()
        # label.save(buffer, format='jpeg', quality=self.jpeg_quality)
        # input = pil_image.open(buffer)
        # input = np.array(input).astype(np.float32)
        # label = np.array(label).astype(np.float32)
        # input = np.transpose(input, axes=[2, 0, 1])
        # label = np.transpose(label, axes=[2, 0, 1])

        label = np.array(label)
        noise = np.random.standard_normal(size=label.shape) * (self.jpeg_quality)
        # noise = np.random.standard_normal(size=label.shape) * (self.jpeg_quality/255.0)
        input = label + noise

        noise = np.array(noise).astype(np.float32)
        input = np.array(input).astype(np.float32)
        label = np.array(label).astype(np.float32)
        noise = np.transpose(noise, axes=[2, 0, 1])
        input = np.transpose(input, axes=[2, 0, 1])
        label = np.transpose(label, axes=[2, 0, 1])
        #
        # img = np.array(img).astype(np.float32)
        # img = np.transpose(img, axes=[2, 0, 1])

        # normalization
        input /= 255.0
        noise /= 255.0
        label /= 255.0
        # return input, label
        return input, noise

    def __len__(self):
        return len(self.images)

    def _get_index(self, idx):
        return idx % len(self.images)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f = self.images[idx]
        f = imageio.imread(f)
        return f





