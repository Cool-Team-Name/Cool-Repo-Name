import pandas as pd
import numpy as np
import glob
import os
#import nx_itertools
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator


#execfile('preprocessing_pipeline.py')

#this generates the file names corresponding to what we should submit
sample_sub = pd.read_csv('stage1_sample_submission.csv')

a = pd.DataFrame(sample_sub.Id.str.split('_').tolist(),
             columns = ['File','Zone'])

submission_subjects = list(set(a.File))

files = os.listdir('Submissions/')

# subjects = list()
# for s in files:
#     subjects.append(s.split('_')[0])
#
# unique_subjects = set(subjects)
# len(unique_subjects.intersection(set(submission_subjects)))

zone_num = 17
zone_text = 'Zone' + str(zone_num)

zone_files = list()
for s in files:
    zn1 = s.split('_')[1]
    zn2 = zn1.split('.')[0]
    if zn2 == zone_text:
        zone_files.append(s)
        os.rename('Submissions/' + s, 'submission_subjects_zone' + str(zone_num) + '/safe/' + s)

num_files = len(os.listdir('submission_subjects_zone' + str(zone_num) + '/safe/'))

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))

model.add(Conv2D(32, (3, 3), data_format="channels_first"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.load_weights('zone' + str(zone_num) + '_model.h5')

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory('submission_subjects_zone'+str(zone_num),
                                                        target_size=(150, 150),
                                                        batch_size=1,
                                                        shuffle=False)

predictions = model.predict_generator(validation_generator, val_samples=validation_generator.samples)

namies = validation_generator.filenames
ids = list()
for id in namies:
    ids.append(id[5:].split('.')[0])


zone_df = pd.DataFrame(
    {
        'Id': ids,
        'Probability': np.concatenate(predictions),
        'Zone': [zone_num] * num_files
    })

#full_zone_df = pd.merge(zone_df, sample_sub, how='right', on='Id')
#sample_sub = pd.DataFrame(sample_sub)
sample_sub['Zone'] = [s.split('Zone')[1] for s in sample_sub.Id]
sample_sub_remaining = sample_sub[(sample_sub['Zone'] == str(zone_num))]
sample_sub_remaining = sample_sub_remaining[np.logical_not(sample_sub_remaining['Id'].isin(zone_df['Id']))]

zone_df = pd.concat([zone_df, sample_sub_remaining])
del zone_df['Zone']

zone_df.to_csv("zone" + str(zone_num) + "_submission.csv", sep = ",", index = False)

