from __future__ import print_function
from __future__ import division

import tarfile
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from PIL import Image
import scipy.misc

execfile('tsahelper.py')

labels = pd.read_csv('stage1_labels.csv')

a = pd.DataFrame(labels.Id.str.split('_').tolist(),
             columns = ['File','Zone'])

labels = a.join(labels.Probability)
labels.Zone = labels.Zone.map(lambda x: x.lstrip('Zone'))
labels.Zone = pd.to_numeric(labels.Zone)
labels = labels.sort_values(by=['File','Zone'])

labels.File = labels.File.astype('str')
# In[6]:

Zone1Threats = pd.Series(labels.File[(labels.Zone==1) & (labels.Probability==1)].astype('str'))
# Zone2Threats = pd.Series(labels.File[(labels.Zone==2) & (labels.Probability==1)])
# Zone3Threats = pd.Series(labels.File[(labels.Zone==3) & (labels.Probability==1)])
# Zone4Threats = pd.Series(labels.File[(labels.Zone==4) & (labels.Probability==1)])
# Zone5Threats = pd.Series(labels.File[(labels.Zone==5) & (labels.Probability==1)])
# Zone6Threats = pd.Series(labels.File[(labels.Zone==6) & (labels.Probability==1)])
# Zone7Threats = pd.Series(labels.File[(labels.Zone==7) & (labels.Probability==1)])
# Zone8Threats = pd.Series(labels.File[(labels.Zone==8) & (labels.Probability==1)])
# Zone9Threats = pd.Series(labels.File[(labels.Zone==9) & (labels.Probability==1)])
# Zone10Threats = pd.Series(labels.File[(labels.Zone==10) & (labels.Probability==1)])
# Zone11Threats = pd.Series(labels.File[(labels.Zone==11) & (labels.Probability==1)])
# Zone12Threats = pd.Series(labels.File[(labels.Zone==12) & (labels.Probability==1)])
# Zone13Threats = pd.Series(labels.File[(labels.Zone==13) & (labels.Probability==1)])
# Zone14Threats = pd.Series(labels.File[(labels.Zone==14) & (labels.Probability==1)])
# Zone15Threats = pd.Series(labels.File[(labels.Zone==15) & (labels.Probability==1)])
# Zone16Threats = pd.Series(labels.File[(labels.Zone==16) & (labels.Probability==1)])
# Zone17Threats = pd.Series(labels.File[(labels.Zone==17) & (labels.Probability==1)])


def get_single_image(infile, nth_image):
    # read in the aps file, it comes in as shape(512, 620, 16)
    img = read_data(infile)

    # transpose so that the slice is the first dimension shape(16, 620, 512)
    img = img.transpose()

    return np.flipud(img[nth_image])

APS_FILE_NAME = '3de88afea1b8fd356b119c5b44dcb47e.aps'
SLICE = 0
an_img = get_single_image(APS_FILE_NAME, 0)

# fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
#
# axarr[0].imshow(an_img, cmap=COLORMAP)
# plt.subplot(122)
# plt.hist(an_img.flatten(), bins=256, color='c')
# plt.xlabel("Raw Scan Pixel Value")
# plt.ylabel("Frequency")
# plt.show()
Zone1Threats = Zone1Threats.values
SLICE = 0

for i in Zone1Threats:
    file_name = 'stage1_aps/' + i + '.aps'

    an_img = get_single_image(file_name, SLICE)

    file_name = file_name.replace('stage1_aps/', '')
    file_to_save = 'zone1-threat-images/' + file_name.replace('.aps', '') + '_Slice' + '_' + str(SLICE) + '.jpeg'
    scipy.misc.imsave(file_to_save, an_img, dpi = 1000)



