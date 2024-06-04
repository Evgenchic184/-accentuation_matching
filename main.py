import os

import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import Counter

from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import PrecisionRecallDisplay, accuracy_score, precision_recall_curve, average_precision_score

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from itertools import product

from sklearn.metrics import average_precision_score

def squared_subtraction(x, y):
  return (x - y) ** 2


def squared_sum(x, y):
  return (x + y) ** 2
  

target_columns = 'group'
col_stats = ['параноидальность', 'эпилептоидность', 'гипертимность', 'истероидность',
       'шизоидность', 'психоастения', 'сезитивность', 'гипотимность',
       'коформность', 'неустойчивость', 'астения', 'лабильность',
       'циклоидность']
def collect_targer(x, y, target_col=target_columns):
  return x[target_col] == y[target_col]

def create_dataset(data_all,
                   col_stats=col_stats,
                   target_column=target_columns,
                   prod_index=None,
                   distance_funct=None):

  data = data_all[col_stats]

  distance_data = np.array(
      [
          distance_funct(data.loc[x[0]], data.loc[x[1]]) for x in prod_index
      ]
  )

  target = np.array(
      [
          collect_targer(data_all.loc[x[0]], data_all.loc[x[1]]) for x in prod_index
      ]
  )
  dataset = np.concatenate([distance_data, target.reshape(-1, 1)], axis=1)
  dataset = pd.DataFrame(data=dataset, columns=col_stats + [target_columns], index=pd.MultiIndex.from_tuples(prod_index))
  return dataset



def main():

	data_all = pd.read_excel('accentuation_data.xlsx')

	data_all = data_all[~data_all[target_columns].isna()]
	data = data_all[col_stats]
	prod_index = list((x, y) for x in data.index.values for y in data.index.values if x != y)


	dataset = create_dataset(data_all, distance_funct=squared_subtraction, prod_index=prod_index)
	log_reg_pipeline = Pipeline([('SS', StandardScaler())])

	x = dataset[col_stats]
	y = dataset[target_columns]

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

	x_train_transformed = log_reg_pipeline.fit_transform(x_train)
	x_test_transformed = log_reg_pipeline.transform(x_test)


	logreg = LogisticRegression(solver='saga', max_iter=20)
	logreg.fit(x_train_transformed, y_train)
	
	data_all = pd.read_excel('accentuation_data.xlsx')
	
	data_all_predict = data_all[col_stats]
	vect = data_all_predict.iloc[0, :]
	
	data_all_predict = data_all_predict.apply(lambda x: squared_subtraction(x, vect), axis=1)
	print(data_all_predict)
	data_all_transformed = log_reg_pipeline.transform(data_all_predict)
	data_all['similarity'] = logreg.predict_proba(data_all_transformed)[:, 1].round(2)
	
	data_all.to_excel('predicted.xlsx', index=False)
	
	print(pd.read_excel('predicted.xlsx'))
	print(average_precision_score(y_score=logreg.predict_proba(x_test_transformed)[:, 1], y_true=y_test))


if __name__ == '__main__':
	main()
