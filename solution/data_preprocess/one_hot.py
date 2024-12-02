from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
token_vectors = encoder.fit_transform(tokens)  # tokens is a list of unique words
