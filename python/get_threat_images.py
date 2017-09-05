from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
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

Zone1Threats = Zone1Threats.values
SLICE = 0

for i in Zone1Threats:
    file_name = 'stage1_aps/' + i + '.aps'

    an_img = get_single_image(file_name, SLICE)
    img_rescaled = convert_to_grayscale(an_img)
    img_high_contrast = spread_spectrum(img_rescaled)
    # masked_img = roi(img_high_contrast, zone_slice_list[0][0])
    # cropped_img = crop(masked_img, zone_crop_list[0][0])
    normalized_img = normalize(img_high_contrast)

    file_name = file_name.replace('stage1_aps/', '')
    file_to_save = 'zone1-threat-images/' + file_name.replace('.aps', '') + '_Slice' + '_' + str(SLICE) + '.png'
    scipy.misc.imsave(file_to_save, normalized_img)



