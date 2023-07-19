# First, Install these packages:
# !pip install git+https://github.com/ternaus/iglovikov_helper_functions
# !pip install cloths_segmentation  > /dev/null

from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from cloths_segmentation.pre_trained_models import create_model
import albumentations as albu
import numpy as np
import torch
import os
import urllib.request
from matplotlib.pyplot import imshow
from PIL import Image
# from urllib.parse import urlparse
from datetime import datetime
import requests
from requests import session
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from typing import Dict, Any
import argparse
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm
from datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images
import cv2
import urllib.request
from matplotlib.pyplot import imshow
from PIL import Image
from urllib.parse import urlparse
from torch import tensor
import torch
from torch.nn import init
from torch.nn.utils.spectral_norm import spectral_norm
from fastapi import Response
import warnings
warnings.filterwarnings('ignore')
from pydantic import BaseModel
from pydantic import ValidationError
from fastapi import FastAPI
import json
import torchvision.transforms as transforms
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import multiprocessing
import random
import time
import signal

app = FastAPI()

image_names = []
class Args(BaseModel):
    name: str
    batch_size: int
    workers: int
    load_height: int
    load_width: int
    shuffle: str
    dataset_dir: str
    dataset_mode: str
    dataset_list: str
    checkpoint_dir: str
    save_dir: str
    display_freq: int
    seg_checkpoint: str
    gmm_checkpoint: str
    alias_checkpoint: str
    semantic_nc: int
    init_type: str 
    init_variance: float
    grid_size: int
    norm_G: str
    ngf: int
    num_upsampling_layers: str

async def cropping(image_file, contoured):
    img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    dst = cv2.Canny(gray, 0, 150)
    MIN_CONTOUR_AREA = 200
    # image = cv2.bitwise_not(dst)
    img_thresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    Contours, imgContours = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in Contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            [X, Y, W, H] = cv2.boundingRect(contour)
            # creating the bounding box around the object
            box=cv2.rectangle(img, (X, Y), (X + W, Y + H), (0,0,0), 2)
    img4 = Image.open(image_file)
    box = (X, Y, X+W, Y+H)
    img2 = img4.crop(box)
    img2.save(contoured)
    return 

async def image_resize(image_file, contoured):
    img_to_resize = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    desired_width = 768
    desired_height = 1024
    resized_img = cv2.resize(img_to_resize, (desired_width, desired_height), interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(contoured, resized_img)

async def create_mask(image_file, mask_path):
    model = create_model("Unet_2020-10-30")
    model.eval()
    image = load_rgb(image_file)
    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
    with torch.no_grad():
        prediction = model(x)[0][0]

        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = unpad(mask, pads)

    dst = cv2.addWeighted(image, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0)
    imshow(dst)
    imshow(np.hstack([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * 255, dst]))
    imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * 255)
    cv2.imwrite('masked-colored.jpg',cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0))

    """ convert to greyscale """
    im_gray = cv2.imread('masked-colored.jpg', cv2.IMREAD_GRAYSCALE)
    thresh = 127
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(mask_path, im_bw)

async def cloth_resize_and_mask_generation(url):  

    image_path = 'datasets/test/cloth/100.jpg'
    contoured_image = 'datasets/test/cloth/100.jpg'
    mask_address = 'datasets/test/cloth-mask/100.jpg'

    """ header """
    retry_strategy = Retry(total = 4, backoff_factor = 1, status_forcelist=[429, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    adapter.max_retries.respect_retry_after_header = False

    session = requests.session()
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        data_image = session.get(url,timeout=50, headers={'x_test12':'true'})

        with open(image_path, 'wb') as obj:
            obj.write(data_image.content)
    except:
       print("error in retrieving the url")
       result_list = []
       print("image status: FAILED TO SAVE")
       return result_list
    session.close()

    """ First Cropping the input image """
    await cropping(image_path, contoured_image)

    """ Resize the image using the new width and height """
    await image_resize(image_path, contoured_image)

    """ create the mask with UNet """
    await create_mask(image_path, mask_address)

    return print("image status: OK")

def segment_function(parse_agnostic, pose, c, cm, seg, gauss, up, opt):
    parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
    pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
    c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
    cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
    seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size()).cuda()), dim=1)

    parse_pred_down = seg(seg_input)
    parse_pred = gauss(up(parse_pred_down))
    parse_pred = parse_pred.argmax(dim=1)[:, None]

    parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float).cuda()
    parse_old.scatter_(1, parse_pred, 1.0)

    labels = {
        0:  ['background',  [0]],
        1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
        2:  ['upper',       [3]],
        3:  ['hair',        [1]],
        4:  ['left_arm',    [5]],
        5:  ['right_arm',   [6]],
        6:  ['noise',       [12]]
    }

    parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float).cuda()
    for j in range(len(labels)):
        for label in labels[j][1]:
            parse[:, j] += parse_old[:, label]
    return parse

def gmm_function(img_agnostic, parse, pose, c, gmm, cm):
    agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
    parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
    pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
    c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
    gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

    _, warped_grid = gmm(gmm_input, c_gmm)
    warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
    warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')
    return warped_c, warped_cm

def alias_function(parse, warped_cm, img_agnostic, pose, warped_c, alias, img_names, c_names, opt):
    misalign_mask = parse[:, 2:3] - warped_cm
    misalign_mask[misalign_mask < 0.0] = 0.0
    parse_div = torch.cat((parse, misalign_mask), dim=1)
    parse_div[:, 2:3] -= misalign_mask

    output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

    unpaired_names = []
    for img_name, c_name in zip(img_names, c_names):
        unpaired_names.append('{}_{}'.format(str(random.getrandbits(128)), c_name))
    save_images(output, unpaired_names, os.path.join(opt.save_dir, opt.name))
    return unpaired_names[0]

def test(opt, seg, gmm, alias):
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss.cuda()

    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)
    image_names = []

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            img_names = inputs['img_name']
            c_names = inputs['c_name']['unpaired']

            img_agnostic = inputs['img_agnostic'].cuda()
            parse_agnostic = inputs['parse_agnostic'].cuda()
            pose = inputs['pose'].cuda()
            c = inputs['cloth']['unpaired'].cuda()
            cm = inputs['cloth_mask']['unpaired'].cuda()

            # Part 1. Segmentation generation
            parse = segment_function(parse_agnostic, pose, c, cm, seg, gauss, up, opt)
            warped_c, warped_cm = gmm_function(img_agnostic, parse, pose, c, gmm, cm)
            result = alias_function(parse, warped_cm, img_agnostic, pose, warped_c, alias, img_names, c_names, opt)
            # print("result==========>", result)
            image_names.append(result) 

            if (i + 1) % opt.display_freq == 0:
                print("step: {}".format(i + 1))
        print("========== finished clothes synthesis ========== ")
        return image_names


def process1_send_function(opt, seg, gmm, alias, conn):
    events = test(opt, seg, gmm, alias)
    conn.send(events)

def handler(signum, frame):
    print('Signal handler called with signal', signum)
    raise IOError("Couldn't open device!")

@app.post('/test_api')
def test_api(data: Dict[Any, Any] = None):
    print("data:::::::::::::::::::::::::::::::", data)
    postData = {"url": data["url"]}
    try:
        response = requests.post("http://192.168.1.54:9090/virtual_try_on", json=postData)
        json_out = json.loads(response.text)
        return json_out
    except Exception:
        return {'sucess': False}

@app.post('/virtual_try_on', response_class=Response)
async def tooba_try_on(data: Dict[Any, Any] = None):

    """ input image resize and generation """
    url = data["url"]
    await cloth_resize_and_mask_generation(url)

    print("STAAAAAAAAAAAAAAAAAAAARTED")
    data = {'name': 'images',
        'batch_size': 1,
        'workers': 1,
        'load_height': 1024,
        'load_width': 768,
        'shuffle': 'store_true',
        'dataset_dir': './datasets/',
        'dataset_mode': 'test',
        'dataset_list': 'test_pairs.txt',
        'checkpoint_dir': './checkpoints/',
        'save_dir': '/opt/',
        'display_freq': 1,
        'seg_checkpoint': 'seg_final.pth',
        'gmm_checkpoint': 'gmm_final.pth',
        'alias_checkpoint': 'alias_final.pth',
        'semantic_nc': 13,
        'init_type': 'xavier',  # choices=['normal', 'xavier','xavier_uniform', 'kaiming', 'orthogonal', 'none']
        'init_variance': 0.02,
        'grid_size': 5,
        'norm_G': 'spectralaliasinstance',
        'ngf': 64,
        'num_upsampling_layers': 'most' } # choices=['normal', 'more', 'most'] 

    opt = Args(**data)
    if not os.path.exists(os.path.join(opt.save_dir, opt.name)):
        os.makedirs(os.path.join(opt.save_dir, opt.name))

    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13
    load_checkpoint(seg, os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
    load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))
    load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))
    seg.cuda().eval()
    gmm.cuda().eval()
    alias.cuda().eval()
    
    pipe_list = []
    recieve_end, send_end = multiprocessing.Pipe(False)
    process_1 = multiprocessing.Process(target=process1_send_function, args=(opt, seg, gmm, alias, send_end))
    pipe_list.append(recieve_end)
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(22)
    process_1.start()
    signal.alarm(0) 

    print(f'{datetime.now()} {process_1.name} started')
    # process_1.join()
    if recieve_end.poll(22):
        result_list = [x.recv() for x in pipe_list]
    else:
        print("No data available after {0} seconds...".format(3))
        print(f'{datetime.now()} {process_1.name} terminated')
        result_list = []

    process_1.join()

    print(result_list)
    return JSONResponse(content=jsonable_encoder(result_list))

# def tensor_to_image(tensor):
#     tensor = tensor*255
#     tensor = np.array(tensor, dtype=np.uint8)
#     if np.ndim(tensor)>3:
#         assert tensor.shape[0] == 1
#         tensor = tensor[0]
#     return PIL.Image.fromarray(tensor)

# def to_tensor(img):
#     image1= PIL.Image.open(img)
#     # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
#     transform = transforms.Compose([transforms.ToTensor()])
#     tensor = transform(image1)
    # return json.dumps(tensor.tolist())

# def to_tensor():
#     transform = transforms.Compose([transforms.ToTensor()])
#     image1= cv2.imread('results/tryon_result/00013_100.jpg')
#     image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

#     image1 = Image.open('results/tryon_result/00013_100.jpg')
#     image2= cv2.imread('results/tryon_result/00263_100.jpg')
#     image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

#     image3= cv2.imread('results/tryon_result/01147_100.jpg')
#     image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

#     tensor = transform(image1)
#     tensor2 = transform(image2)
#     tensor3 = transform(image3)
    
    # json_file = [{'image1': str(tensor)}, {'image2': str(tensor2)}, {'image3': str(tensor3)}]
    # json_file = [{'image0': json.dumps(tensor.tolist())}, {'image1': json.dumps(tensor2.tolist())}, {'image2':json.dumps(tensor3.tolist())}]

    # print(json_file)
    # return json_file


