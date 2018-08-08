import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import random
from collections import Counter
# from sklearn.metrics import roc_curve, auc, average_precision_score

path = 'steam_original.csv'
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

# adicionado 1 aos jogos que o usuario tem preferencia (comprados)
user_game_pref[user_idx, game_idx] = 1
# arr = np.array(user_game_pref)
# print arr.tolist()
user_game_interactions = zero_matrix.copy()
# matriz de horas jogadas, sendo o minimo 1 (pq ele comprou)
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


# #------tensorflow----------
tf.reset_default_graph() # Create a new graphs

 # preference matrix
pref = tf.placeholder(tf.float32, (n_users, n_games))
# hours played matrix
interactions = tf.placeholder(tf.float32, (n_users, n_games))
users_idx = tf.placeholder(tf.int32, (None))

# number of latent features to be extracted
n_features = 30

# X matrix represents the user latent preferences with a shape of user x latent features
X = tf.Variable(tf.truncated_normal([n_users, n_features], mean = 0, stddev = 0.05))

# Y matrix represents the game latent features with a shape of game x latent features
Y = tf.Variable(tf.truncated_normal([n_games, n_features], mean = 0, stddev = 0.05))

# Here's the initilization of the confidence parameter
conf_alpha = tf.Variable(tf.random_uniform([1], 0, 1))


# initialize a user bias vector
user_bias = tf.Variable(tf.truncated_normal([n_users, 1], stddev = 0.2))

# Concatenate the vector to the user matrix
# Due to how matrix algebra works, we also need to add a column of ones to make sure
# the resulting calculation will take into account the item biases.
X_plus_bias = tf.concat([X,
                         #tf.convert_to_tensor(user_bias, dtype = tf.float32),
                         user_bias,
                         tf.ones((n_users, 1), dtype = tf.float32)], axis = 1)


# initialize the item bias vector
item_bias = tf.Variable(tf.truncated_normal([n_games, 1], stddev = 0.2))

# Cocatenate the vector to the game matrix
# Also, adds a column one for the same reason stated above.
Y_plus_bias = tf.concat([Y,
                         tf.ones((n_games, 1), dtype = tf.float32),
                         item_bias],
                         axis = 1)
# Here, we finally multiply the matrices together to estimate the predicted preferences
pred_pref = tf.matmul(X_plus_bias, Y_plus_bias, transpose_b=True)

# Construct the confidence matrix with the hours played and alpha paramter
conf = 1 + conf_alpha * interactions

cost = tf.reduce_sum(tf.multiply(conf, tf.square(tf.subtract(pref, pred_pref))))
l2_sqr = tf.nn.l2_loss(X) + tf.nn.l2_loss(Y) + tf.nn.l2_loss(user_bias) + tf.nn.l2_loss(item_bias)
lambda_c = 0.01
loss = cost + lambda_c * l2_sqr
lr = 0.05
optimize = tf.train.AdagradOptimizer(learning_rate = lr).minimize(loss)

# This is a function that help to calculate the top k precision
def top_k_precision(pred, mat, k, user_idx):
    precisions = []

    for user in user_idx:
        # found the top recommendation from the predictions
        rec = np.argsort(-pred[user, :])
        # arr = np.array(rec)
        # print ("user: {0} rec: {1}".format(user, arr.tolist()))
        top_k = rec[:k]
        labels = mat[user, :].nonzero()[0]
        # arr = np.array(labels)
        # print ("labels: {0}".format(arr.tolist()))
        # print "\n"
        # calculate the precisions from actual labels
        precision = len(set(top_k) & set(labels)) / float(k)
        precisions.append(precision)
    return np.mean(precisions)


iterations = 70
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        sess.run(optimize, feed_dict = {pref: train_matrix, interactions: user_game_interactions})

        if i % 10 == 0:
            mod_loss = sess.run(loss, feed_dict = {pref: train_matrix, interactions: user_game_interactions})
            mod_pred = pred_pref.eval()
            train_precision = top_k_precision(mod_pred, train_matrix, k, val_users_idx)
            val_precision = top_k_precision(mod_pred, val_matrix, k, val_users_idx)
            print('Ite: {0}'.format(i),
                  'Training Loss {:.2f}'.format(mod_loss),
                  'Train Precision: {:.3f}'.format(train_precision),
                  'Validation Precision: {:.3f}'.format(val_precision)
                )

    rec = pred_pref.eval()
    test_precision = top_k_precision(rec, test_matrix, k, test_users_idx)
    print type(test_precision)
    print('\n')
    print('test_precision {:.3f}'.format(test_precision))


# #--------------Exemplos ---------------

n_examples = 5

while True:
    arr = np.array(test_users_idx)
    print arr.tolist()

    ans = input("1. Para exemplos random\n2. Para entrar com um user_id\n")

    if ans==1:
      users = np.random.choice(test_users_idx, size = n_examples, replace = False)

      arr = np.array(users)
      print ("enums:\n {0}".format(arr.tolist()))

      rec_games = np.argsort(-rec)
      for user in users:
          print("Enum: {0} steam_id: {1}".format(user, idx2user[user]))
          # print('Usuario: {0}'.format(idx2user[user]))
          purchase_history = np.where(train_matrix[user, :] != 0)[0]
          recommendations = rec_games[user, :]

          new_recommendations = recommendations[~np.in1d(recommendations, purchase_history)][:k]

          print('Nos recomendamos:')
          print(', '.join([idx2game[game] for game in new_recommendations]))
          print('\n')
          print('Os jogos que o usuario {0} comprou sao:'.format(idx2user[user]))
          print(', '.join([idx2game[game] for game in np.where(test_matrix[user, :] != 0)[0]]))
          print('\n')
          print('Precision de {0}'.format(len(set(new_recommendations) & set(np.where(test_matrix[user, :] != 0)[0])) / float(k)))
          #print('--------------------------------------\n')
    # elif ans==2:
    #     ans_id = input("Digite o enum: ")
    #
    #     # present = np.isin(ans_id,test_users_idx)
    #     # if(present == False):
    #     #     print "esse enum nao esta na lista de test_users_idx"
    #     #     continue
    #     #
    #     # users = np.where(test_users_idx == ans_id)
    #     users = np.random.choice(test_users_idx, size = 219, replace = False)
    #
    #
    #     # print np.prod(test_users_idx.shape)
    #
    #     rec_games = np.argsort(-rec)
    #
    #     for user in users:
    #         if(user == ans_id):
    #             print("Enum: {0} idx2user(steam_id): {1}".format(user, idx2user[user]))
    #             # print('Usuario: {0}'.format(idx2user[user]))
    #             purchase_history = np.where(train_matrix[user, :] != 0)[0]
    #             recommendations = rec_games[user, :]
    #
    #             new_recommendations = recommendations[~np.in1d(recommendations, purchase_history)][:k]
    #
    #             print('Recomendacao:')
    #             print(', '.join([idx2game[game] for game in new_recommendations]))
    #             print('\n')
    #             print('Os jogos que o usuario {0} comprou sao:'.format(idx2user[user]))
    #             print(', '.join([idx2game[game] for game in np.where(test_matrix[user, :] != 0)[0]]))
    #             print('\n')
    #             #print('Precision de {0}'.format(len(set(new_recommendations) & set(np.where(test_matrix[user, :] != 0)[0])) / float(k)))
    #             #print('--------------------------------------')
    #             #print('\n')
    elif ans != (1):
        print("\n Opcao invalida")
