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
"""metrics"""
import math
import numpy as np
import cv2
import mindspore as ms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def quantize(img, rgb_range):
    """quantize image range to 0-255"""
    pixel_range = 255 / rgb_range
    img = np.multiply(img, pixel_range)
    img = np.clip(img, 0, 255)
    img = np.round(img) / pixel_range
    return img


def calc_psnr(sr, hr, scale, rgb_range):
    """calculate psnr"""
    hr = np.float32(hr)
    sr = np.float32(sr)
    diff = (sr - hr) / rgb_range
    gray_coeffs = np.array([65.738, 129.057, 25.064]).reshape((1, 3, 1, 1)) / 256
    diff = np.multiply(diff, gray_coeffs).sum(1)
    if hr.size == 1:
        return 0
    if scale != 1:
        shave = scale
    else:
        shave = scale + 6
    if scale == 1:
        valid = diff
    else:
        valid = diff[..., shave:-shave, shave:-shave]
    mse = np.mean(pow(valid, 2))
    return -10 * math.log10(mse)


def rgb2ycbcr(img, y_only=True):
    """from rgb space to ycbcr space"""
    img.astype(np.float32)
    if y_only:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    return rlt


def calc_ssim(img1, img2, scale):
    """calculate ssim value"""
    def ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    border = 0
    if scale != 1:
        border = scale
    else:
        border = scale + 6
    img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 256.0 + 16.0
    img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 256.0 + 16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h - border, border:w - border]
    img2_y = img2_y[border:h - border, border:w - border]
    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    if img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for _ in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        if img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def get_PSNR_SSIM(original_clean_image, result_image, data_range):
    """compute the PSNR and SSIM between the original image and the result image(get by learning)"""
    # convert to numpy array
    if not isinstance(original_clean_image, np.ndarray):
        if isinstance(original_clean_image, ms.Tensor):
            original_clean_image = original_clean_image.asnumpy()
    if not isinstance(result_image, np.ndarray):
        if isinstance(result_image, ms.Tensor):
            result_image = result_image.asnumpy()
    image_num = original_clean_image.shape[0]
    PSNR = []
    SSIM = []
    for i in range(image_num):
        orginal_clean_image_ = np.transpose(original_clean_image[i, :, :, :], axes=[1, 2, 0])
        result_image_ = np.transpose(result_image[i, :, :, :], axes=[1, 2, 0])

        # print(np.array(orginal_clean_image_, dtype=np.float16).shape)
        SSIM.append(structural_similarity(np.array(orginal_clean_image_, dtype=np.float16),
                                          np.array(result_image_, dtype=np.float16),
                                          data_range=1, multichannel=True))

        PSNR.append(peak_signal_noise_ratio(np.array(orginal_clean_image_, dtype=np.float16),
                                            np.array(result_image_, dtype=np.float16),
                                            data_range=data_range))



    return np.mean(PSNR), np.mean(SSIM)
