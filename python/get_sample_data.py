import pandas as pd
import random

labels = pd.read_csv('stage1_labels.csv')
a = pd.DataFrame(labels.Id.str.split('_').tolist(),
             columns = ['File','Zone'])

labels = a.join(labels.Probability)
labels.Zone = labels.Zone.map(lambda x: x.lstrip('Zone'))
labels.Zone = pd.to_numeric(labels.Zone)
labels = labels.sort_values(by=['File','Zone'])


Zone1_Pos = labels.File[(labels.Zone==1) & (labels.Probability==1)]
Zone1_Neg = labels.File[(labels.Zone==1) & (labels.Probability==0)]

pos_sample = random.sample(list(Zone1_Pos), 100)
neg_sample = random.sample(list(Zone1_Neg), 400)

SUBJECT_LIST = pos_sample + neg_sample
random.shuffle(SUBJECT_LIST)

