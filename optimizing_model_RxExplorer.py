"""
Please read the license file.
RxExplorer.ML (version 1.0) was designed to help identify new chemical reactions
Modified: 31.05.2023
"""
#Import libraries
import argparse
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from category_encoders import TargetEncoder
from sklearn.metrics import make_scorer, matthews_corrcoef


#Optional arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--seed', type=int, action='store', default=1, help='Random seed value.')
parser.add_argument('-c', '--combos_file', type=str, action='store', default='all_combos_filtered_ite_9.txt', help='The filename of the searching space.')
parser.add_argument('-r', '--training_data', type=str, action='store', default='training_data.txt', help='The filename of the training data.')
parser.add_argument('-m', '--max_features', nargs='+', type=int, default=[2, 3, 5], help='Number of features to consider when training')
parser.add_argument('-d', '--max_depth', nargs='+', type=int, default=[2, 3], help='Maximum depth of the tree.')
parser.add_argument('-e', '--n_estimators', nargs='+', type=int, default=[5, 10, 25], help='Number of trees.')
parser.add_argument('-j', '--jobs', type=int, action='store', default=-1, help='Number of parallel jobs when optimising hyperparameters and calculate proximity matrix.')
parser.add_argument('-n', '--select_reactions', type=int, action='store', default=5, help='Number of experiments to be included in batch.')
parser.add_argument('-k', '--k_value', type=int, action='store', default=4, help='k value of the chaos equation')
parser.add_argument('-t', '--threshold', type=int, action='store', default=0.4, help='Select the threshold value of similarity matrix')
args = parser.parse_args()

print('    ____         ______              __                          __  ___ __  ')
print('   / __ \ _  __ / ____/_  __ ____   / /____   _____ ___   _____ /  |/  // /  ')
print('  / /_/ /| |/_// __/  | |/_// __ \ / // __ \ / ___// _ \ / ___// /|_/ // /   ')
print(' / _, _/_>  < / /___ _>  < / /_/ // // /_/ // /   /  __// /_  / /  / // /___ ')
print('/_/ |_|/_/|_|/_____//_/|_|/ .___//_/ \____//_/    \___//_/(_)/_/  /_//_____/ ')
print('                         /_/                                                 ')
print('                                                                             ')
print('Version 1.0 | Powered by DigiChem Lab in collaboration with IBM Research Zürich')
print('                                                                             ')
print('Welcome!')
print('Let me work out what is the best experiment for you to run...')

seed = args.seed

# Remove any leading/trailing whitespace and split the lines into columns
search_space = pd.read_csv(args.combos_file, sep='\t')
prior_score = search_space.iloc[:, -1]
X_test = search_space.iloc[:, :-1]
X_train = pd.read_csv(args.training_data, sep='\t')
Y_train = X_train.iloc[:, -1]
X_train = X_train.drop(X_train.columns[-1], axis=1)
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()


def target_encode (X_train, X_test, Y_train, search_space, categorial_cols, min_sample_leaf=20, smoothing=10):
    '''
         This function will encode categorical variables with TargetEncode
         :X_train: {dataframe} with all variables for training the model
         :X_test: {dataframe} with all variables for potential experiments
         :Y_train: {dataframe} with the actual values of the experiments
         :search_space: {dataframe} with all the possible combinations, the whole search space
         :categorical_cols: {list} with all names of categorical variables
         :min_samples_leaf: {int} for regularization of the weighted average between category mean and global mean
         :smoothing: {float} to balance categorical average vs. prior. Higher value means stronger regularization and must be bigger than 0

         :return:
         :X_train_encoded: {array} with encoded training and test sets
         :X_test_encoded: {array} with encoded all combinations file
         :search_space_encoded: {array} with encoded all combinations file to concatenate data
     '''
    encoder = TargetEncoder(cols=categorical_cols, min_samples_leaf=min_sample_leaf, smoothing=smoothing)
    X_train_encoded = encoder.fit_transform(X_train, Y_train)
    
    import joblib
    joblib.dump(encoder, 'target_encoder.joblib')
    
    X_test_encoded = encoder.transform(X_test)
    return X_train_encoded, X_test_encoded
X_train_encoded, X_test_encoded = target_encode(X_train, X_test, Y_train, search_space, categorial_cols=categorical_cols)
print('Encoding...done!')


X_test_encoded.to_csv('ite_10/encoded_search_space_10_0.4.txt', index=False, sep='\t')
X_train_encoded.to_csv('ite_10/encoded_training_data_10_0.4.txt', index=False, sep='\t')


# Hyperparameter search of random forest classifier
def grid_search_random_forest(X_train_encoded, Y_train, max_features, max_depth, n_estimators):
    '''
        This function will run a grid search to find the best model hyperparameters according to a 10-fold CV loop.
        :X_train_encoded: {dataframe} with all variables for training the model
        :Y_train: {dataframe} with all labels
        :max_features: {list} with number of features for model training. Default= 2, 3, 5]. Customizable through optional arguments
        :max_depth: {list} for depth of the tree. Default=[2, 3]. Customizable through optional arguments
        :n_estimators: {list} for number of tree in the forest. Default=[5, 10, 25]. Customizable through optional arguments

        :return {dict}:
        :best_params: best hyperparameters found in grid search
        :avg_mcc: average MCC value according to the 10-fold CV loop for the best estimator.
    '''

    model = RandomForestClassifier(random_state=seed)
    kf = StratifiedKFold(n_splits=9, shuffle=True, random_state=seed)
    scoring = make_scorer(matthews_corrcoef)

    param = {'max_features': max_features,
             'max_depth': max_depth,
             'n_estimators': n_estimators}

    grid = GridSearchCV(estimator=model, scoring=scoring, param_grid=param, cv=kf, n_jobs=-1)
    grid_result = grid.fit(X_train_encoded, Y_train.values)

    best_params = grid.best_params_
    avg_mcc = np.mean(grid_result.cv_results_['mean_test_score'])
    print(avg_mcc)
    with open('ite_10/best_params_10_0.4.txt', 'w') as f:
        f.write(str(best_params))

    with open('ite_10/avg_mcc_10_0.4.txt', 'w') as f:
        f.write(str(avg_mcc))

    return {'best_params': best_params, 'avg_mcc': avg_mcc}


model_opt = grid_search_random_forest(X_train_encoded, Y_train, max_features=args.max_features, max_depth=args.max_depth, n_estimators=args.n_estimators)
print('Hyperparameters... found!')

'''
#Example input: python RxExplorerML.py --max_features 2 4 6 --max_depth 6 8 --n_estimators 30 40
'''

def build_and_predict(X_train_encoded, Y_train, X_test_encoded):
    '''
    This function builds a random forest classifier with the best hyperparameters found in the grid search, trains it and makes predictions. It returns the class 1 probabilities and predicted label.

    :param X_train_encoded: {dataframe} with all descriptors to train the model
    :param Y_train: {dataframe} with all labels
    :param X_test_encoded: {dataframe} with all descriptors of potential experiments

    :return:
    class_1_prob: {ndarray} class 1 probabilities for the test data
    model_best: {ndarray} model with best hyperparameters to train data
    '''

    model_best = RandomForestClassifier(**model_opt['best_params'], random_state=seed)
    model_best.fit(X_train_encoded, Y_train)
    class_1_prob = model_best.predict_proba(X_test_encoded)
    class_prob_training = model_best.predict_proba(X_train_encoded)
    feature_importance = model_best.feature_importances_

    with open('ite_10/feature_importance_ite_10_0.4.txt', 'w') as f:
        for i in range(len(feature_importance)):
            f.write(str(feature_importance[i]) + '\n')
    return class_1_prob, model_best, class_prob_training


class_1_prob, model_best, class_prob_training = build_and_predict(X_train_encoded, Y_train, X_test_encoded)
print('Model and predictions... done!')

import joblib
joblib.dump(model_best, 'rx_model.joblib')

df_class = pd.DataFrame(class_1_prob)
df1_class = pd.DataFrame(class_prob_training)

df_class.to_csv('ite_10/Predicted_prob_search_space_10_0.4.txt', index=None, sep='\t')
df1_class.to_csv('ite_10/Predicted_prob_training_data_10_0.4.txt', index=None, sep='\t')

#Generation of random numbers
def random_value_in_interval(nums: 'prob_values') -> list:
    """
    This function creates a list of random numbers with 3 decimals, according to a reference value. Range of values is according to a predefined bin.

    :param nums: {list} with the predicted probabilities for label '1', i.e., 'productive' reaction.

    :return: {list} with noisy probability values.
    """

    random_nums = []
    for num in nums:
        if (num >= 0).any() and (num < 0.05).any():
            random_nums.append(round(random.uniform(0, 0.049), 4))
        elif (num >= 0.05).any() and (num < 0.15).any():
            random_nums.append(round(random.uniform(0.05, 0.149), 4))
        elif (num >= 0.15).any() and (num < 0.25).any():
            random_nums.append(round(random.uniform(0.15, 0.249), 4))
        elif (num >= 0.25).any() and (num < 0.35).any():
            random_nums.append(round(random.uniform(0.25, 0.349), 4))
        elif (num >= 0.35).any() and (num < 0.45).any():
            random_nums.append(round(random.uniform(0.35, 0.449), 4))
        elif (num >= 0.45).any() and (num < 0.55).any():
            random_nums.append(round(random.uniform(0.45, 0.549), 4))
        elif (num >= 0.55).any() and (num < 0.65).any():
            random_nums.append(round(random.uniform(0.55, 0.649), 4))
        elif (num >= 0.65).any() and (num < 0.75).any():
            random_nums.append(round(random.uniform(0.65, 0.749), 4))
        elif (num >= 0.75).any() and (num < 0.85).any():
            random_nums.append(round(random.uniform(0.75, 0.849), 4))
        elif (num >= 0.85).any() and (num < 0.95).any():
            random_nums.append(round(random.uniform(0.85, 0.949), 4))
        elif (num >= 0.95).any() and (num < 1.0).any():
            random_nums.append(round(random.uniform(0.95, 0.999), 4))
        elif (num == 1.0).any():
            random_nums.append(1.000)
        else:
            raise ValueError("probability value must be in the [0, 1] range")
    return random_nums

nums = class_1_prob
random_nums = random_value_in_interval(nums)
random_value_pred_prob = pd.DataFrame(random_nums)


def scoring_function (search_space, k=args.k_value):

    '''
    Implementation of the logistic map function
    :k: {int} value
    :df_encoded: {dataframe} of search space, categorical columns and the scoring value encoded
    :df_sorted: {dataframe} with search space sorted based on the Scoring value. Sorted in descending order
    :df_sorted_encoded: {dataframe} with encoded df_sorted.
    '''

    eq = lambda x: k * x * (1-x)
    search_space['Prior_score_chaos'] = prior_score.apply(eq)
    search_space['Prelim_score'] = random_value_pred_prob.apply(eq)
    search_space['Scoring'] = search_space['Prior_score_chaos'] * search_space['Prelim_score']
    search_space['Scoring'] = search_space['Scoring'].round(7)
    df_encoded = pd.concat([X_test_encoded, search_space['Scoring']], axis=1)
    df_dict = pd.concat([X_test, search_space['Scoring']], axis=1)
    search_space = search_space.drop(columns=['Prior_score', 'Prior_score_chaos', 'Prelim_score'], inplace=True)
    df_sorted = df_encoded.sort_values(by='Scoring', ascending=False)
    df_dictionary = df_dict.sort_values(by='Scoring', ascending=False)
    df_sorted_encoded = df_sorted.iloc[:, :-1]
    return df_encoded, df_sorted, df_sorted_encoded, df_dictionary

df_encoded, df_sorted, df_sorted_encoded, df_dictionary = scoring_function(search_space)


#Rename Scoring and save new search_space values
search_space.rename(columns={'Scoring': 'Prior_score'}, inplace=True)
search_space.to_csv('all_combos_10_0.4.txt', index=None, sep='\t')
df_sorted.to_csv('ite_10/encoded_sorted_search_space_10_0.4.txt', index=False, sep='\t')
print('search_space_ite10: ', len(search_space))

#Reset indexes
df1_reset = df_sorted_encoded.reset_index()
df_reset = pd.DataFrame(df1_reset)
df_reset.drop(columns=['index'], inplace=True)
print('reset :', df_reset)
df2_reset = df_sorted.reset_index()
df2_reset = pd.DataFrame(df2_reset)
df2_reset.drop(columns=['index'], inplace=True)

df1_dict = df_dictionary.reset_index()
df_dict = pd.DataFrame(df1_dict)
df_dict.drop(columns=['index'], inplace=True)


# Similarity Matrix
def proximityMatrix(model_best, df_reset, normalize=True):
    terminals = model_best.apply(df_reset)
    nTrees = terminals.shape[1]

    a = terminals[:, 0]
    proxMat = 1*np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:, i]
        proxMat += 1*np.equal.outer(a, a)

    if normalize:
        proxMat = proxMat / nTrees

    return proxMat

#Selection of following iterations
def select_batch(proxMat, n_examples=args.select_reactions, threshold=args.threshold):
    n_samples = proxMat.shape[0]
    # Initialize the batch with the first example
    batch = [0]
    for i in range(1, n_samples):
        is_dissimilar = True
        for j in batch:
            if proxMat[i, j] >= threshold:
                is_dissimilar = False
                break
        if is_dissimilar:
            batch.append(i)
        if len(batch) == n_examples:
            break

    # If we couldn't find enough dissimilar examples, just return the ones we have
    if len(batch) < n_examples:
        print(f"Warning: could only find {len(batch)} dissimilar examples.")

    return batch

proxMat = proximityMatrix(model_best, df_reset)
selected_indices = select_batch(proxMat)
selected_examples = df_reset.iloc[selected_indices, :]
selected_scoring = df2_reset.iloc[selected_indices, -1]

print('selected indices :', selected_indices)
print(selected_examples)

selected_examples.to_csv('ite_10/encoded_complete_iteration_10_0.4.txt', sep='\t')
selected_examples.to_csv('ite_10/encoded_iteration_10_0.4.txt', index=None, sep='\t')
selected_scoring.to_csv('ite_10/Scoring_ite_10_0.4.txt', index=None, sep='\t')

#Extraction of index value
index_number = selected_examples.index
index_list = tuple(index_number)

# Create an empty dictionary
dictionary = {}
for index, row in df_dict.iterrows():
    dictionary[index] = row.to_dict()

selected_rows = [dictionary[index] for index in index_list]
new_reactions = pd.DataFrame(selected_rows)
Scoring = pd.read_csv('ite_10/Scoring_ite_10_0.4.txt', sep='\t')
new_reactions['Scoring'] = Scoring

# Appending new iterations to existing reactions performed
df1 = pd.concat([X_train, new_reactions], ignore_index=True)
print(new_reactions)


# To save all files
new_reactions.to_csv('ite_10/iteration_10_scoring_0.4.txt', index=False, sep='\t')
new_reactions.iloc[:, :-1].to_csv('iteration_10_0.4.txt', index=False, sep='\t')
new_reactions.iloc[:, :-1].to_csv('ite_10/iteration_10_0.4.txt', index=False, sep='\t')

#To read dataframe and remove the reactions that have been selected from search space

# Merge the two DataFrames based on all columns
merged_df = pd.merge(search_space, new_reactions, how='outer', indicator=True)

# Filter out the rows present in the second DataFrame
filtered_df1 = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
filtered_df1.drop('Scoring', axis=1, inplace=True)

# Display the filtered DataFrame
print("Filtered DataFrame:")
print(filtered_df1)


filtered_df1.to_csv('all_combos_filtered_ite_10_0.4.txt', index=None, sep='\t')
filtered_df1.to_csv('all_combos_filtered.txt', index=None, sep='\t')




