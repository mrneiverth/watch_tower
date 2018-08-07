import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import tensorflow as tf
import random
from collections import Counter
# from sklearn.metrics import roc_curve, auc, average_precision_score

path = 'steam-200k.csv'
#path = 'steam-200k.csv'
df = pd.read_csv(path, header = None, names = ['UserID', 'Game', 'Action', 'Hours', 'Not Needed'])
df.head()
df['Hours_Played'] = df['Hours'].astype('float32')
df.loc[(df['Action'] == 'purchase') & (df['Hours'] == 1.0), 'Hours_Played'] = 0


# ordena por ID, Games, Horas jogadas
df.UserID = df.UserID.astype('int')
df = df.sort_values(['UserID', 'Game', 'Hours_Played'])

#dropa as linhas duplicadas
clean_df = df.drop_duplicates(['UserID', 'Game'], keep = 'last').drop(['Action', 'Hours', 'Not Needed'], axis = 1)

n_users = len(clean_df.UserID.unique())
n_games = len(clean_df.Game.unique())

print('Sao {0} usuarios e um total de {1} jogos'.format(n_users, n_games))

user_counter = Counter()
for user in clean_df.UserID.tolist():
    user_counter[user] +=1

game_counter = Counter()
for game in clean_df.Game.tolist():
    game_counter[game] += 1

# ?
# for i in range(n_users):
#     if(user_counter[i] != 0):
#         print ("i: {0} UC: {1}".format(i, user_counter[i]))
#     else:
#         print user_counter[i]

user2idx = {user: i for i, user in enumerate(clean_df.UserID.unique())}
idx2user = {i: user for user, i in user2idx.items()}

# for k, v in user2idx.iteritems():
#     print k, v

# enum_user ::: id_user
# for k, v in idx2user.iteritems():
#     print k, v

game2idx = {game: i for i, game in enumerate(clean_df.Game.unique())}
#just games
# for k, v in game2idx.iteritems():
#     print k, v

idx2game = {i: game for game, i in game2idx.items()}
#enum_games :::: games_name
# for k, v in idx2game.iteritems():
#     print k, v


# usuarios e jogos to idx
#user_idx <type 'numpy.ndarray'>
user_idx = clean_df['UserID'].apply(lambda x: user2idx[x]).values
game_idx = clean_df['gameIdx'] = clean_df['Game'].apply(lambda x: game2idx[x]).values
hours = clean_df['Hours_Played'].values
# arr = np.array(user2idx)
# print arr.tolist()

zero_matrix = np.zeros(shape = (n_users, n_games))
user_game_pref = zero_matrix.copy()

# fill the matrix will preferences (bought)
user_game_pref[user_idx, game_idx] = 1
# arr = np.array(user_game_pref)
# print arr.tolist()
user_game_interactions = zero_matrix.copy()
user_game_interactions[user_idx, game_idx] = hours + 1
# arr = np.array(user_game_interactions)
# print arr.tolist()


#----------VALIDACAO-------------
k = 5

# purchase_counts for each user
purchase_counts = np.apply_along_axis(np.bincount, 1, user_game_pref.astype(int))

#find users who purchase 2 * k games
buyers_idx = np.where(purchase_counts[:, 1] >= 2 * k)[0]
print('{0} usuarios compraram {1} ou mais jogos'.format(len(buyers_idx), 2 * k))

# 10% do dataset pra validacaoo e 10% pra teste
test_frac = 0.2
test_users_idx = np.random.choice(buyers_idx, size = int(np.ceil(len(buyers_idx) * test_frac)), replace = False)
val_users_idx = test_users_idx[:int(len(test_users_idx) / 2)]
test_users_idx = test_users_idx[int(len(test_users_idx) / 2):]
# arr = np.array(test_users_idx)
# print arr.tolist()

# A function used to mask the preferences data from training matrix
def data_process(dat, train, test, user_idx, k):
    for user in user_idx:
        purchases = np.where(dat[user, :] == 1)[0]
        #print("purchases: {0}".format(np.where(dat[user, :] == 1)[0]))
        mask = np.random.choice(purchases, size = k, replace = False)
        #print("mask: {0}".format(np.where(dat[user, :] == 1)[0]))

        train[user, mask] = 0
        test[user, mask] = dat[user, mask]
    return train, test

train_matrix = user_game_pref.copy()
test_matrix = zero_matrix.copy()
val_matrix = zero_matrix.copy()

# mask the train matrix and create the validation and test matrices
train_matrix, val_matrix = data_process(user_game_pref, train_matrix, val_matrix, val_users_idx, k)
train_matrix, test_matrix = data_process(user_game_pref, train_matrix, test_matrix, test_users_idx, k)
# arr = np.array(train_matrix)
# print arr.tolist()


# You can see the test matrix preferences are masked in the train matrix
test_matrix[test_users_idx[0], test_matrix[test_users_idx[0], :].nonzero()[0]]
#array([ 1.,  1.,  1.,  1.,  1.])
train_matrix[test_users_idx[0], test_matrix[test_users_idx[0], :].nonzero()[0]]
#array([ 0.,  0.,  0.,  0.,  0.])
