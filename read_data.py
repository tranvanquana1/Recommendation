# import libraries
import pandas as pd
import numpy as np

# Đặt tên cột cho mỗi cột trong file CSV và đọc với thư viện pandas
# ĐỌc file user
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|',
                    names=u_cols, encoding='latin-1')

# Đọc file rating
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t',
                      names=r_cols, encoding='latin-1')

# Dữ liệu train
ratings_train = pd.read_csv(
    'ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
rate_train = ratings_train.values

# Dữ liệu test
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t',
                           names=r_cols, encoding='latin-1')
rate_test = ratings_test.values

# Đọc file items
i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
                    encoding='latin-1')

# Tính toán số user và số item
n_users = users.shape[0]
n_items = items.shape[0]

# Hiện thị nội dung các tập dữ liệu


def main():

    # User data
    print("\nUser Data :")
    print("shape : ", users.shape)
    print(users.head())
    print("number users: ", n_users)

    # Rating data
    print("\nRating Data :")
    print("shape : ", ratings.shape)
    print(ratings.head())

    # Items data
    print("\nItem Data :")
    print("shape : ", items.shape)
    print(items.head())
    print("number Items: ", n_items)


if __name__ == "__main__":
    main()
