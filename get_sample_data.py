import pandas as pd
import random

labels = pd.read_csv('stage1_labels.csv')
a = pd.DataFrame(labels.Id.str.split('_').tolist(),
             columns = ['File','Zone'])

labels = a.join(labels.Probability)
labels.Zone = labels.Zone.map(lambda x: x.lstrip('Zone'))
labels.Zone = pd.to_numeric(labels.Zone)
labels = labels.sort_values(by=['File','Zone'])


Zone5_Pos = labels.File[(labels.Zone==5) & (labels.Probability==1)]
Zone5_Neg = labels.File[(labels.Zone==5) & (labels.Probability==0)]

pos_sample = random.sample(list(Zone5_Pos), len(Zone5_Pos))
neg_sample = random.sample(list(Zone5_Neg), len(Zone5_Pos))

SUBJECT_LIST = pos_sample + neg_sample
random.shuffle(SUBJECT_LIST)

