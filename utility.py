import logging, os, math, re, ntpath, sys, subprocess, datetime, time, shlex, json
import numpy as np
from PIL import Image
import ntpath
import cv2
import math
import copy
import torch.nn as nn
import torch

THREAD_NUM=4

class ImageSplitter:
    # key points:
    # Boarder padding and over-lapping img splitting to avoid the instability of edge value
    # Thanks Waifu2x's autorh nagadomi for suggestions (https://github.com/nagadomi/waifu2x/issues/238)

    def __init__(self, patch_size, scale_factor, stride):
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.stride = stride
        self.height = 0
        self.width = 0

    def split_img_tensor(self, img_tensor):
        # resize image and convert them into tensor
        batch, channel, height, width = img_tensor.size()
        self.height = height
        self.width = width

        side = min(height, width, self.patch_size)
        delta = self.patch_size - side
        Z = torch.zeros([batch, channel, height+delta, width+delta])
        Z[:, :, delta//2:height+delta//2, delta//2:width+delta//2] = img_tensor
        batch, channel, new_height, new_width = Z.size()

        patch_box = []

        # split image into over-lapping pieces
        for i in range(0, new_height, self.stride):
            for j in range(0, new_width, self.stride):
                x = min(new_height, i + self.patch_size)
                y = min(new_width, j + self.patch_size)
                part = Z[:, :, x-self.patch_size:x, y-self.patch_size:y]

                patch_box.append(part)
        patch_tensor = torch.cat(patch_box, dim=0)
        return patch_tensor

    def merge_img_tensor(self, list_img_tensor):
        img_tensors = copy.copy(list_img_tensor)

        patch_size = self.patch_size * self.scale_factor
        stride = self.stride * self.scale_factor
        height = self.height * self.scale_factor
        width = self.width * self.scale_factor
        side = min(height, width, patch_size)
        delta = patch_size - side
        new_height = delta + height
        new_width = delta + width
        out = torch.zeros((1, 3, new_height, new_width))
        mask = torch.zeros((1, 3, new_height, new_width))

        for i in range(0, new_height, stride):
            for j in range(0, new_width, stride):
                x = min(new_height, i + patch_size)
                y = min(new_width, j + patch_size)
                mask_patch = torch.zeros((1, 3, new_height, new_width))
                out_patch = torch.zeros((1, 3, new_height, new_width))
                mask_patch[:, :, (x - patch_size):x, (y - patch_size):y] = 1.0
                out_patch[:, :, (x - patch_size):x, (y - patch_size):y] = img_tensors.pop(0)
                mask = mask + mask_patch
                out = out + out_patch

        out = out / mask

        out = out[:, :, delta//2:new_height - delta//2, delta//2:new_width - delta//2]

        return out






def encode_JPG(input_np, quality=95):
    enc = cv2.imencode('.jpg',input_np, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tostring()
    return enc

def encode_JPG2000(input_np, quality=95):
    enc = cv2.imencode('.jp2',input_np, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tostring()
    return enc

def encode_PNG(input_np, compression=6):
    enc = cv2.imencode('.png',input_np, [int(cv2.IMWRITE_PNG_COMPRESSION), compression])[1].tostring()
    return enc

def decode_IMG(input_enc):
    nparr = np.frombuffer(input_enc, np.uint8)
    dec = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return dec

def resize_DIV2K(targetloc, path, scale, interpolation, upscale):
    with Image.open(path) as img:
        width, height = img.size

    filename = ntpath.basename(path)
    filename, _ = os.path.splitext(filename)

    command = [ 'ffmpeg',
                '-loglevel', 'fatal',
                '-i', '{}'.format(path),
                '-threads', str(THREAD_NUM),
                '-vf', 'scale=%d:%d'%(width * scale, height * scale),
                '-sws_flags', '%s'%(interpolation),
                '{}/{}.png'.format(targetloc, filename) ]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if (err): print('error',err); return None;

def write_frame(imageloc, videoloc, t_w, t_h, interpolation, extract_fps):
    command = ['ffmpeg',
               '-loglevel', 'fatal',
               '-i', videoloc,
               '-threads', str(THREAD_NUM),
               '-vf', 'scale=%d:%d, fps=%f'%(t_w, t_h, extract_fps),
               '-sws_flags', '%s'%(interpolation),
               '{}/%d.png'.format(imageloc)]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print('error',err); return None;

def write_frame_noscale(imageloc, videoloc, extract_fps):
    command = ['ffmpeg',
               '-loglevel', 'fatal',
               '-i', videoloc,
               '-threads', str(THREAD_NUM),
               '-vf', 'fps=%f'%(extract_fps),
               '{}/%d.png'.format(imageloc)]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print('error',err); return None;

def get_video_info(fileloc) :
    command = ['ffprobe',
               '-v', 'fatal',
               '-show_entries', 'stream=width,height,r_frame_rate,duration',
               '-of', 'default=noprint_wrappers=1:nokey=1',
               fileloc, '-sexagesimal']
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print(err)
    out = out.decode().split('\n')

    cmd = "ffprobe -v quiet -print_format json -show_streams"
    args = shlex.split(cmd)
    args.append(fileloc)
    ffprobeOutput = subprocess.check_output(args).decode('utf-8')
    ffprobeOutput = json.loads(ffprobeOutput)
    bitrate = int(ffprobeOutput['streams'][0]['bit_rate']) / 1000

    return {'file' : fileloc,
            'width': int(out[0]),
            'height' : int(out[1]),
            'fps': float(out[2].split('/')[0])/float(out[2].split('/')[1]),
            'duration' : out[3],
            'bitrate': bitrate}

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

class videoInfo:
    def __init__(self, fps, duration, quality):
        self.fps = fps
        self.duration = duration
        self.quality = quality

class timer():
    def __init__(self):
        self.acc = 0
        self.total_time = 0
        self.tic()

    def tic(self):
        self.t0 = time.perf_counter()

    def toc(self):
        return time.perf_counter() - self.t0

    def toc_total_sum(self):
        elapsed_time = time.perf_counter() - self.t0
        return self.total_time + elapsed_time

    def toc_total_add(self):
        elapsed_time = time.perf_counter() - self.t0
        self.total_time += elapsed_time
        return elapsed_time

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

    def add_total(self, time):
        self.total_time += time

def get_psnr(pred, gt, max_value=1.0):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    mse = np.mean((pred-gt)**2)
    if mse == 0:
        return 100
    else:
        return 20*math.log10(max_value/math.sqrt(mse))

def gpu_psnr(pred, gt, max_value=255.0):
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return 100
    else:
        return 20*math.log10(max_value/torch.sqrt(mse))

def print_progress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def get_logger(save_dir, save_name):
    Logger = logging.getLogger(save_name)
    Logger.setLevel(logging.INFO)
    Logger.propagate = False

    filePath = os.path.join(save_dir, save_name)
    if os.path.exists(filePath):
        os.remove(filePath)

    fileHandler = logging.FileHandler(filePath)
    logFormatter = logging.Formatter('%(message)s')
    fileHandler.setFormatter(logFormatter)
    Logger.addHandler(fileHandler)

    return Logger

def get_logger_stream(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('\r||----%(name)s----||----%(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    
    return logger

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')