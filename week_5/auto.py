import numpy as np
import code_for_hw5 as hw5

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw5.load_auto_data('auto-mpg-regression.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw5.standard and hw5.one_hot.
# 'name' is not numeric and would need a different encoding.
features1 = [('cylinders', hw5.standard),
            ('displacement', hw5.standard),
            ('horsepower', hw5.standard),
            ('weight', hw5.standard),
            ('acceleration', hw5.standard),
            ('origin', hw5.one_hot)]

features2 = [('cylinders', hw5.one_hot),
            ('displacement', hw5.standard),
            ('horsepower', hw5.standard),
            ('weight', hw5.standard),
            ('acceleration', hw5.standard),
            ('origin', hw5.one_hot)]

# Construct the standard data and label arrays
#auto_data[0] has the features for choice features1
#auto_data[1] has the features for choice features2
#The labels for both are the same, and are in auto_values
auto_data = [0, 0]
auto_values = 0
auto_data[0], auto_values = hw5.auto_data_and_values(auto_data_all, features1)
auto_data[1], _ = hw5.auto_data_and_values(auto_data_all, features2)

#standardize the y-values
auto_values, mu, sigma = hw5.std_y(auto_values)

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------     
        
#Your code for cross-validation goes here
# =============================================================================
# min_rmse = 1 # because I know some are less than 1, the choice of 1 is arbitrary
# for f in (0, 1):
#     for order in (1, 2, 3):
#         # X is k by n matrix
#         X = hw5.make_polynomial_feature_fun(order)(auto_data[f])
#         y = auto_values
#         if order in (1, 2):
#             for lam in np.arange(0.01, 0.11, 0.01):
#                 rmse = hw5.xval_learning_alg(X, y, lam, 10)
#                 # vector, mu, sigma = hw5.std_y(y)
#                 # rmse_scale = rmse*sigma
#                 #print(f'feature={f} order={order} lambda={lam}: {rmse}')
#                 if rmse < min_rmse:
#                     min_f = f
#                     min_order = order
#                     min_lam = lam
#                     min_rmse = rmse
#             
#         else:
#             for lam in range(0, 201, 20):
#                 rmse = hw5.xval_learning_alg(X, y, lam, 10)
#                 # vector, mu, sigma = hw5.std_y(y)
#                 # rmse_scale = rmse*sigma
#                 #print(f'feature={f} order={order} lambda={lam}: {rmse}')
#                 if rmse < min_rmse:
#                     min_f = f
#                     min_order = order
#                     min_lam = lam
#                     min_rmse = rmse
# print(min_f, min_order, min_lam, min_rmse)
# =============================================================================
min_rmse = 7
X = hw5.make_polynomial_feature_fun(3)(auto_data[0])
for lam in range(0, 201, 20):
    rmse = hw5.xval_learning_alg(X, auto_values, lam, 10)
    rmse = rmse*sigma
    print(f'{lam}: {rmse}')
    if rmse < min_rmse:
        min_lam = lam
        min_rmse = rmse
print(min_lam, min_rmse)
#Make sure to scale the RMSE values returned by xval_learning_alg by sigma,
#as mentioned in the lab, in order to get accurate RMSE values on the dataset

