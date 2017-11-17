import pandas as pd
import random
import glob
import os

labels = pd.read_csv('stage1_labels.csv')
a = pd.DataFrame(labels.Id.str.split('_').tolist(),
             columns = ['File','Zone'])

labels = a.join(labels.Probability)
labels.Zone = labels.Zone.map(lambda x: x.lstrip('Zone'))
labels.Zone = pd.to_numeric(labels.Zone)
labels = labels.sort_values(by=['File','Zone'])

zone_num = 12
zone_folder = 'z' + str(zone_num) + '/'

Zone_Pos = labels.File[(labels.Zone == zone_num) & (labels.Probability == 1)]
Zone_Neg = labels.File[(labels.Zone == zone_num) & (labels.Probability == 0)]

files = os.listdir(zone_folder)

ZP = list()
ZN = list()
for zp in Zone_Pos:
    for f in files:
        if zp in f.split('_')[0]:
            ZP.append(zp)

for zn in Zone_Neg:
    for f in files:
        if zn in f.split('_')[0]:
            ZN.append(zn)

pos_sample = ZP
neg_sample = ZN #random.sample(list(ZN), 2*len(pos_sample))

#SUBJECT_LIST = pos_sample + neg_sample
pos_train = random.sample(pos_sample, round(0.8 * len(pos_sample)))
pos_val = list(set(pos_sample) - set(pos_train))

neg_train = random.sample(neg_sample, round(0.8 * len(neg_sample)))
neg_val = list(set(neg_sample) - set(neg_train))

for i in files:
    if i.split('_')[0] in pos_train:
         os.rename(zone_folder + i,
                   'data_zone' + str(zone_num) + '/train/threat/' + i)
    elif i.split('_')[0] in pos_val:
        os.rename(zone_folder + i,
                  'data_zone' + str(zone_num) + '/val/threat/' + i)
    if i.split('_')[0] in neg_train:
         os.rename(zone_folder + i,
                   'data_zone' + str(zone_num) + '/train/safe/' + i)
    elif i.split('_')[0] in neg_val:
        os.rename(zone_folder + i,
                  'data_zone' + str(zone_num) + '/val/safe/' + i)





#safe_files = glob.glob('Safe/' + '*')
#safe_files[:] = [s.replace('Safe\\', '') for s in safe_files]
#
# threat_files = glob.glob('Threats/' + '*')
# threat_files[:] = [s.replace('Threats\\', '') for s in threat_files]
#
# safe_to_use = random.sample(safe_files, len(threat_files))
# safe_train = random.sample(safe_to_use, round(0.8 * len(safe_to_use)))
# safe_val = list(set(safe_to_use) - set(safe_train))
#
# threat_train = random.sample(threat_files, round(0.8 * len(threat_files)))
# threat_val = list(set(threat_files) - set(threat_train))

#
# for sf in safe_train:
#     os.rename('Safe/' + sf, 'data_zone' + str(zone_num) + '/train/safe/' + sf)
# for sf in safe_val:
#     os.rename('Safe/' + sf, 'data_zone' + str(zone_num) + '/val/safe/' + sf)
# for sf in threat_train:
#     os.rename('Threats/' + sf, 'data_zone' + str(zone_num) + '/train/threat/' + sf)
# for sf in threat_val:
#     os.rename('Threats/' + sf, 'data_zone' + str(zone_num) + '/val/threat/' + sf)

# SUBJECT_LIST = pos_sample + neg_sample
# random.shuffle(SUBJECT_LIST)


