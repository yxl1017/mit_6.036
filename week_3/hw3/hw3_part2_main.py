import pdb
import numpy as np
import code_for_hw3_part2 as hw3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]

# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
print('auto data and labels shape', auto_data.shape, auto_labels.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------
features2 =  [('cylinders', hw3.one_hot),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]

# for learner in (hw3.perceptron, hw3.averaged_perceptron):
#     for T in (1, 10, 50):
#         for feature_set in (features, features2):
#             data, labels = hw3.auto_data_and_labels(auto_data_all, feature_set)
#             score = hw3.xval_learning_alg(learner, T, data, labels, 10)
#             print(f'learner = {learner} T = {T} feature set = {feature_set}: {score}')
#             print()

# data, labels = hw3.auto_data_and_labels(auto_data_all, features2)
# th, th0 = hw3.averaged_perceptron(data, labels, params={'T': 10})
# print(th, th0)

#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

# for learner in (hw3.perceptron, hw3.averaged_perceptron):
#     for T in (1, 10, 50):
#         score = hw3.xval_learning_alg(learner, T, review_bow_data, review_labels, 10)
#         print(f'learner = {learner} T = {T} : {score}')
#         print()
# th, th0 = hw3.averaged_perceptron(review_bow_data, review_labels, params={'T': 10})
# largest_indices = np.argpartition(th.flatten(), -10)[-10:]
# rev_dic = hw3.reverse_dict(dictionary)

# li = []
# for ele in largest_indices:
#     li.append(rev_dic[ele])
# print(li)

# th, th0 = hw3.averaged_perceptron(review_bow_data, review_labels, params={'T': 10})
# smallest_indices = np.argpartition(th.flatten(), 10)[:10]
# rev_dic = hw3.reverse_dict(dictionary)

# li = []
# for ele in smallest_indices:
#     li.append(rev_dic[ele])
# print(li)
#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[2]["images"]
d1 = mnist_data_all[4]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    li = []
    for i in range(x.shape[0]):
        li.append(x[i])
    return np.array(li).reshape(-1, x.shape[0])

def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    n_sample, m, n = x.shape
    r_ave = np.mean(x[0], axis=1).reshape(m, 1)
    for i in range(1, n_sample):
        ans = np.mean(x[i], axis=1).reshape(m, 1)
        r_ave = np.append(r_ave, ans, axis=1)
    return r_ave


def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    n_sample, m, n = x.shape
    c_ave = np.mean(x[0], axis=0).reshape(n, 1)
    for i in range(1, n_sample):
        ans = np.mean(x[i], axis=0).reshape(n, 1)
        c_ave = np.append(c_ave, ans, axis=1)
    return c_ave


def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    n_sample, m, n = x.shape
    top = x[:, :m//2, :]
    bottom = x[:, m//2:, :]
    ave_top = np.mean(row_average_features(top), axis=0).reshape(-1, n_sample)
    ave_bottom = np.mean(row_average_features(bottom), axis=0).reshape(-1, n_sample)
    return np.concatenate((ave_top, ave_bottom), axis=0)

# use this function to evaluate accuracy
acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)

#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# There are 77 images of 0, stored in d0: a list with len = 77
data_0 = hw3.cv(d0[0].flatten())
for i in range(1, len(d0)):
    data_0 = np.concatenate((data_0, hw3.cv(d0[i].flatten())), axis=1)

acc_0 = hw3.get_classification_accuracy(data_0, y0)

def digit_data(di):
    D = hw3.cv(di[0].flatten())
    for i in range(1, len(di)):
        D = np.concatenate((D, hw3.cv(di[i].flatten())), axis=1)
    return D
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    