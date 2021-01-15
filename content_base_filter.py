from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import math
import numpy as np
import pickle
import json

import read_data as rd

X0 = rd.items.values
X_train_counts = X0[:, -19:]

category = np.array(['unknown ', 'Action ', 'Adventure ',
                     'Animation ', 'Children\'s ', 'Comedy ', 'Crime ', 'Documentary ', 'Drama ', 'Fantasy ',
                     'Film-Noir ', 'Horror ', 'Musical ', 'Mystery ', 'Romance ', 'Sci-Fi ', 'Thriller ', 'War ', 'Western'])

# lấy thể loại phim


def get_category(array):
    arr_category = []
    for id in array:
        category_item = category.dot(X_train_counts[id]).strip()
        arr_category.append(category_item)
    return np.array(arr_category)


# tfidf
transformer = TfidfTransformer(smooth_idf=True, norm='l2')
tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()

# print(tfidf[[0, 1, 2, 3, 4], :])


def get_items_rated_by_user(rate_matrix, user_id):
    """
    in each line of rate_matrix, we have infor: user_id, item_id, rating (scores), time_stamp
    we care about the first three values
    return (item_ids, scores) rated by user user_id
    """
    y = rate_matrix[:, 0]  # all users
    # item indices rated by user_id
    # we need to +1 to user_id since in the rate_matrix, id starts from 1
    # while index in python starts from 0
    ids = np.where(y == user_id + 1)[0]
    item_ids = rate_matrix[ids, 1] - 1  # index starts from 0
    scores = rate_matrix[ids, 2]
    return (item_ids, scores)


# Tìm mô hình cho mỗi user
d = tfidf.shape[1]  # data dimension
W = np.zeros((d, rd.n_users))
b = np.zeros((1, rd.n_users))


filename = 'models/user_model_'


for n in range(rd.n_users):
    ids, scores = get_items_rated_by_user(rd.rate_train, n)
    clf = Ridge(alpha=0.01, fit_intercept=True)
    Xhat = tfidf[ids, :]

    clf.fit(Xhat, scores)

    # lưu model
    tuple_objects = (clf, Xhat, scores)
    pickle.dump(tuple_objects, open(filename + str(n+1) + '.pkl', 'wb'))

    W[:, n] = clf.coef_
    b[0, n] = clf.intercept_

# predicted scores
# Yhat = tfidf.dot(W) + b
# # Ví dụ với với user có id=10
# n = 0
# np.set_printoptions(precision=2)  # 2 digits after .
# ids, scores = get_items_rated_by_user(rd.rate_test, n)

# Yhat[n, ids]

# Đánh giá mô hình


# def evaluate(Yhat, rates, W, b):
#     se = 0
#     cnt = 0
#     for n in range(rd.n_users):
#         ids, scores_truth = get_items_rated_by_user(rates, n)
#         scores_pred = Yhat[ids, n]
#         e = scores_truth - scores_pred
#         se += (e*e).sum(axis=0)
#         cnt += e.size
#     return np.sqrt(se/cnt)


# print('RMSE for training:', evaluate(Yhat, rd.rate_train, W, b))
# print('RMSE for test    :', evaluate(Yhat, rd.rate_test, W, b))


n = 10

ids, ratings = get_items_rated_by_user(rd.rate_test, n-1)

movie_name = rd.items.values[ids, 1]
category_list = get_category(ids)


pickled_model, pickled_Xhat, pickled_scores = pickle.load(
    open(filename + str(n) + '.pkl', 'rb'))


predict = pickled_model.predict(tfidf[ids, :])

# hiển thị theo dataframe pandas
table_user_item = pd.DataFrame(
    {'Rated movies ids': ids+[1], "Name movie": movie_name, 'category': category_list, 'Predicted ratings': predict})

# Sort theo predict rating
table_sorted = table_user_item.sort_values(
    by='Predicted ratings', ascending=False)

print(table_sorted)

result = table_sorted.to_json(orient='records')
parsed = json.loads(result)
print(json.dumps(parsed, indent=2))
