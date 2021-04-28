import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('wine_dataset.csv', sep=';', dtype=np.float64)
features = tpot_data.drop('quality', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['quality'], random_state=None)


import tpot
AutoML = tpot.TPOTClassifier(generations=5,population_size=100,
         offspring_size=None,mutation_rate=0.9,crossover_rate=0.1,
         scoring=None,cv=5,subsample=1.0,n_jobs=1,
         max_time_mins=None,max_eval_time_mins=5,random_state=None,
         config_dict=None,template=None,warm_start=False,
         memory=None,use_dask=False,periodic_checkpoint_folder=None,
         early_stop=None,verbosity=2,
         disable_update_check=False)
AutoML.fit(training_features,training_target)