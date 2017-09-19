import pandas as pd
import numpy as np
import glob
import os
import nx_itertools

execfile('preprocessing_pipeline.py')

#this generates the file names corresponding to what we should submit
sample_sub = pd.read_csv('stage1_sample_submission.csv')

a = pd.DataFrame(sample_sub.Id.str.split('_').tolist(),
             columns = ['File','Zone'])

submission_subjects = list(set(a.File))

#instantiate the model
model_zone1 = alexnet(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE)

#it took 69 iterations to train the model, may be different for someone else's run
#read in the model file. this was created during the train_conv_net() call
model_zone1.load(MODEL_PATH + 'tsa-alexnet-v0.1-lr-0.001-250-250-tz-1' + '-69')

#this should just return the prepared data we're supposed to predict on
def submission_pipeline(filename, path):
    preprocessed_tz_scans = []
    feature_batch = []
    # Load a batch of preprocessed tz scans
    preprocessed_tz_scans = np.load(os.path.join(path, filename))

    # Shuffle to randomize for input into the model
    np.random.shuffle(preprocessed_tz_scans)

    # separate features and labels
    for example_list in preprocessed_tz_scans:
        for example in example_list:
            feature_batch.append(example[0])

    feature_batch = np.asarray(feature_batch, dtype=np.float32)

    return feature_batch


PREPROCESSED_DATA_FOLDER = 'submission_preprocessed/'
submission_files = glob.glob(PREPROCESSED_DATA_FOLDER + '*')
submission_files.sort(key=os.path.getmtime)
submission_files[:] = [s.replace('submission_preprocessed\\', '') for s in submission_files]

#we have the file names, just need to grab the actual data
for j, test_f_in in enumerate(submission_files):
    if j == 0:
        val_features = submission_pipeline(test_f_in, PREPROCESSED_DATA_FOLDER)
    else:
        tmp_feature_batch = submission_pipeline(test_f_in, PREPROCESSED_DATA_FOLDER)
        val_features = np.concatenate((tmp_feature_batch, val_features), axis=0)

val_features = val_features.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)


#predict
pred_zone1 = model.predict(X=val_features)
#probability of a threat should be the second column
pred_zone1 = pred_zone1[:,1]

#there are 11 predictions for each
submission_subjects_repeated = list(nx_itertools.chain.from_iterable(nx_itertools.repeat(x, 11) for x in submission_subjects))
submission_subjects_repeated_z1 = [s + '_Zone1' for s in submission_subjects_repeated]

zone1_df = pd.DataFrame(
    {
        'Id': submission_subjects_repeated_z1,
        'Probability': pred_zone1
    })

zone1_df = zone1_df[['Id', 'Probability']]
zone1_submission = pd.DataFrame({'Probability': zone1_df.groupby(by = 'Id')['Probability'].max()}).reset_index()

#left like this until we have a model for this zone
zone2_df = pd.DataFrame(
    {
        'Id': [s.replace('Zone1', 'Zone2') for s in zone1_submission.Id],
        'Probability': [0.5] * 100
    })

#do the same for other zones
zone3_df = zone2_df.copy(); zone3_df.Id = [s.replace('Zone2', 'Zone3') for s in zone3_df.Id]
zone4_df = zone2_df.copy(); zone4_df.Id = [s.replace('Zone2', 'Zone4') for s in zone4_df.Id]
zone5_df = zone2_df.copy(); zone5_df.Id = [s.replace('Zone2', 'Zone5') for s in zone5_df.Id]
zone6_df = zone2_df.copy(); zone6_df.Id = [s.replace('Zone2', 'Zone6') for s in zone6_df.Id]
zone7_df = zone2_df.copy(); zone7_df.Id = [s.replace('Zone2', 'Zone7') for s in zone7_df.Id]
zone8_df = zone2_df.copy(); zone8_df.Id = [s.replace('Zone2', 'Zone8') for s in zone8_df.Id]
zone9_df = zone2_df.copy(); zone9_df.Id = [s.replace('Zone2', 'Zone9') for s in zone9_df.Id]
zone10_df = zone2_df.copy(); zone10_df.Id = [s.replace('Zone2', 'Zone10') for s in zone10_df.Id]
zone11_df = zone2_df.copy(); zone11_df.Id = [s.replace('Zone2', 'Zone11') for s in zone11_df.Id]
zone12_df = zone2_df.copy(); zone12_df.Id = [s.replace('Zone2', 'Zone12') for s in zone12_df.Id]
zone13_df = zone2_df.copy(); zone13_df.Id = [s.replace('Zone2', 'Zone13') for s in zone13_df.Id]
zone14_df = zone2_df.copy(); zone14_df.Id = [s.replace('Zone2', 'Zone14') for s in zone14_df.Id]
zone15_df = zone2_df.copy(); zone15_df.Id = [s.replace('Zone2', 'Zone15') for s in zone15_df.Id]
zone16_df = zone2_df.copy(); zone16_df.Id = [s.replace('Zone2', 'Zone16') for s in zone16_df.Id]
zone17_df = zone2_df.copy();zone17_df.Id = [s.replace('Zone2', 'Zone17') for s in zone17_df.Id]

all_zones_submission = pd.concat([zone1_submission, zone2_df, zone3_df, zone4_df, zone5_df, zone6_df, zone7_df, zone8_df,
                                  zone9_df, zone10_df, zone11_df, zone12_df, zone13_df, zone14_df, zone15_df, zone16_df, zone17_df])

all_zones_submission.to_csv("tsa_submission.csv", sep = ",", index = False)