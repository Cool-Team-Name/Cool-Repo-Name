
# coding: utf-8

# In[1]:

import tarfile
import pandas as pd
import numpy as np


# In[2]:

files = tarfile.open('stage1_aps.tar.gz', mode='r:gz')

file_list = files.getnames()
del file_list[0]
file_list = [i.strip('./') for i in file_list]

labels = pd.read_csv('stage1_labels.csv')

a = pd.DataFrame(labels.Id.str.split('_').tolist(),
             columns = ['File','Zone'])

labels = a.join(labels.Probability)
labels.Zone = labels.Zone.map(lambda x: x.lstrip('Zone'))
labels.Zone = pd.to_numeric(labels.Zone)
labels = labels.sort_values(by=['File','Zone'])


# In[6]:

Zone1Threats = pd.Series(labels.File[(labels.Zone==1) & (labels.Probability==1)])
Zone2Threats = pd.Series(labels.File[(labels.Zone==2) & (labels.Probability==1)])
Zone3Threats = pd.Series(labels.File[(labels.Zone==3) & (labels.Probability==1)])
Zone4Threats = pd.Series(labels.File[(labels.Zone==4) & (labels.Probability==1)])
Zone5Threats = pd.Series(labels.File[(labels.Zone==5) & (labels.Probability==1)])
Zone6Threats = pd.Series(labels.File[(labels.Zone==6) & (labels.Probability==1)])
Zone7Threats = pd.Series(labels.File[(labels.Zone==7) & (labels.Probability==1)])
Zone8Threats = pd.Series(labels.File[(labels.Zone==8) & (labels.Probability==1)])
Zone9Threats = pd.Series(labels.File[(labels.Zone==9) & (labels.Probability==1)])
Zone10Threats = pd.Series(labels.File[(labels.Zone==10) & (labels.Probability==1)])
Zone11Threats = pd.Series(labels.File[(labels.Zone==11) & (labels.Probability==1)])
Zone12Threats = pd.Series(labels.File[(labels.Zone==12) & (labels.Probability==1)])
Zone13Threats = pd.Series(labels.File[(labels.Zone==13) & (labels.Probability==1)])
Zone14Threats = pd.Series(labels.File[(labels.Zone==14) & (labels.Probability==1)])
Zone15Threats = pd.Series(labels.File[(labels.Zone==15) & (labels.Probability==1)])
Zone16Threats = pd.Series(labels.File[(labels.Zone==16) & (labels.Probability==1)])
Zone17Threats = pd.Series(labels.File[(labels.Zone==17) & (labels.Probability==1)])


# In[7]:

Zone1Safe = pd.Series(labels.File[(labels.Zone==1) & (labels.Probability==0)])
Zone2Safe = pd.Series(labels.File[(labels.Zone==2) & (labels.Probability==0)])
Zone3Safe = pd.Series(labels.File[(labels.Zone==3) & (labels.Probability==0)])
Zone4Safe = pd.Series(labels.File[(labels.Zone==4) & (labels.Probability==0)])
Zone5Safe = pd.Series(labels.File[(labels.Zone==5) & (labels.Probability==0)])
Zone6Safe = pd.Series(labels.File[(labels.Zone==6) & (labels.Probability==0)])
Zone7Safe = pd.Series(labels.File[(labels.Zone==7) & (labels.Probability==0)])
Zone8Safe = pd.Series(labels.File[(labels.Zone==8) & (labels.Probability==0)])
Zone9Safe = pd.Series(labels.File[(labels.Zone==9) & (labels.Probability==0)])
Zone10Safe = pd.Series(labels.File[(labels.Zone==10) & (labels.Probability==0)])
Zone11Safe = pd.Series(labels.File[(labels.Zone==11) & (labels.Probability==0)])
Zone12Safe = pd.Series(labels.File[(labels.Zone==12) & (labels.Probability==0)])
Zone13Safe = pd.Series(labels.File[(labels.Zone==13) & (labels.Probability==0)])
Zone14Safe = pd.Series(labels.File[(labels.Zone==14) & (labels.Probability==0)])
Zone15Safe = pd.Series(labels.File[(labels.Zone==15) & (labels.Probability==0)])
Zone16Safe = pd.Series(labels.File[(labels.Zone==16) & (labels.Probability==0)])
Zone17Safe = pd.Series(labels.File[(labels.Zone==17) & (labels.Probability==0)])

