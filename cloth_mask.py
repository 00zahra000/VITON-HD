""" 
CODE FOR READING 1 IMAGE. RESIZING AND CREATING MASK
"""

# First, Install these packages:
# !pip install git+https://github.com/ternaus/iglovikov_helper_functions
# !pip install cloths_segmentation  > /dev/null

import cv2
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from cloths_segmentation.pre_trained_models import create_model
import albumentations as albu
import numpy as np
import torch
import os
import urllib.request
import argparse
from matplotlib.pyplot import imshow

from PIL import Image, ImageEnhance
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')
import sys
import requests
import urllib.parse
from requests import session
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from PIL import Image, ImageDraw
import time
import matplotlib.pyplot as plt
import urllib.request
import torchvision.transforms as T
import urllib.request
from urllib.error import HTTPError

""" header """
retry_strategy = Retry(
    total = 4,
    backoff_factor = 1, 
    status_forcelist=[429, 502, 503, 504]
)

adapter = HTTPAdapter(max_retries=retry_strategy)
adapter.max_retries.respect_retry_after_header = False

session = requests.session()
session.mount('http://', adapter)
session.mount('https://', adapter)
start_time = time.monotonic()

image_path = 'datasets/test/cloth/100.jpg'
contoured_image = 'datasets/test/cloth/100.jpg'

# image_path = 'preprocess/100.jpg'
# contoured_image = 'preprocess/cropped.jpg'

def main(url):
  try:
    data_image = session.get(url,timeout=50, headers={'x_test12':'true'})
    with open(image_path, 'wb') as obj:
      obj.write(data_image.content)
  except Exception as e:
    print(type(e))
    print(e)
  
  session.close()

  """ First Cropping the input image """
  
  img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
# # Adjust the brightness and contrast
# # Adjusts the brightness by adding 10 to each pixel value
#   brightness = 5
# # Adjusts the contrast by scaling the pixel values by 2.3
#   contrast = 1.2
#   image2 = cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness)
#   cv2.imwrite("preprocess/contrast.jpg", image2)
#   cv2.imwrite(image_path, image2)
# gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) 
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  cv2.imwrite("preprocess/gray.jpg", gray)
  dst = cv2.Canny(gray, 0, 150)
  cv2.imwrite("preprocess/dst.jpg", dst)
  MIN_CONTOUR_AREA = 200
  image = cv2.bitwise_not(dst)
  img_thresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
  Contours, imgContours = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  for contour in Contours:
      if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
          [X, Y, W, H] = cv2.boundingRect(contour)
          # creating the bounding box around the object
          box=cv2.rectangle(img, (X, Y), (X + W, Y + H), (0,0,0), 2)
  img4 = Image.open(image_path)
  box = (X, Y, X+W, Y+H)
  img2 = img4.crop(box)
  img2.save(contoured_image)

  """ TEST: counting the white pixels of the original vs cropped resized image"""

  # # to rescale the image, choose the width that you would like
  # # then scale the height by a factor of width / height

  # img_cropped = cv2.imread(contoured_image, cv2.IMREAD_UNCHANGED)        # cv2.IMREAD_UNCHANGED or -1.
  # width = 768
  # # height = int(img_cropped.shape[0] * width / img_cropped.shape[1])
  # height = 1024
  # dsize = (width, height)
  # resized_img = cv2.resize(img_cropped, dsize, interpolation = cv2.INTER_CUBIC)
  # cv2.imwrite("preprocess/cropped-resized.jpg", resized_img)
  # cv2.imwrite("datasets/test/cloth/100.jpg", resized_img)

  # count_white_cropped = cv2.imread("preprocess/cropped-resized.jpg", cv2.IMREAD_GRAYSCALE)
  # n_white_pix = np.sum(count_white_cropped == 255)
  # print(f" **** white pixels of the  resized cropped image: {n_white_pix} **** ")

  # # Get the current width and height of the image
  # height, width = img.shape[:2]

  # # Find the ratio of the desired width and height to the current width and height
  # width_ratio = desired_width / width
  # height_ratio = desired_height / height

  # # Use the smallest ratio to resize the image while preserving the aspect ratio
  # ratio = min(width_ratio, height_ratio)
  # new_width = int(width * ratio)
  # new_height = int(height * ratio)

  """ Resize the image using the new width and height """

  img_to_resize = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
  desired_width = 768
  desired_height = 1024
  resized_img = cv2.resize(img_to_resize, (desired_width, desired_height), interpolation = cv2.INTER_NEAREST)
  cv2.imwrite("preprocess/cropped.jpg", resized_img)


  """ create the mask with UNet """

  model = create_model("Unet_2020-10-30")
  model.eval()
  image = load_rgb(image_path)
  transform = albu.Compose([albu.Normalize(p=1)], p=1)
  padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
  x = transform(image=padded_image)["image"]
  x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
  with torch.no_grad():
    prediction = model(x)[0][0]

    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)

  dst = cv2.addWeighted(image, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0)
  # /When you do, you will discover that it is returning the input as a list. (That's because you used .split() - the return is a list.) Your convert() function "works" because it expects a list. However, check50 tests by calling with a string, like this:
  # imshow(np.hstack([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * 255, dst]))
  # imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * 255)
  cv2.imwrite('masked-colored.jpg', cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0))

  """ convert to greyscale """

  im_gray = cv2.imread('masked-colored.jpg', cv2.IMREAD_GRAYSCALE)
  thresh = 127
  im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
  (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  cv2.imwrite('datasets/test/cloth-mask/100.jpg', im_bw)
  cv2.destroyAllWindows()
  print('Completed!')

if __name__ == '__main__':
    URL = sys.argv[1]
    main(URL)

# # url = 'https://dkstatics-public.digikala.com/digikala-products/7fbf1be4c409cfac0680c2877b75be07672efb4e_1664213195.jpg?x-oss-process=image/resize,m_lfit,h_600,w_600/quality,q_90'
# # url = 'https://dkstatics-public.digikala.com/digikala-products/39f0f975b8bcc0b6b0fcf65b85d1b93f17e14566_1634383584.jpg?x-oss-process=image/resize,m_lfit,h_600,w_600/quality,q_90'
