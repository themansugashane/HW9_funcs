#!/usr/bin/env python
# coding: utf-8

# ## Week 9 Assignment - DATASCI200 Introduction to Data Science Programming, UC Berkeley MIDS

# Write code in this Jupyter Notebook to solve the following problems. Please upload this **Notebook** with your solutions to your GitHub repository and gradescope.
# 
# Assignment due date: 11:59PM PT the night before the Week 11 Live Session.

# ## Objectives
# 
# - Understand different ways to manipulate and perform calculations on Numpy arrays
# - Using numpy to find the max values in each column and row, calculating the mean of a matrix
# - Demonstarting matrix value comparisons and slicing of values
# - Calculations using numpy matrixes

# ## General Guidelines:
# 
# - All calculations need to be done in the functions (that includes any formatting of the output)
# - Name your functions exactly as written in the problem statement
# - Please have your functions return the answer rather than printing it inside the function
# - Do not make a separate input() statement.  The functions will be passed the input as shown in the examples
# - The examples given are samples of how we will test/grade your code. Please ensure your functions output the same information
# - Answer format is graded - please match the examples
# - Docstrings and comments in your code are strongly suggested but won't be graded
# - In each code block, do NOT delete the ### comment at the top of a cell (it's needed for the auto-grading!)
#   - Do NOT print or put other statements in the grading cells. The autograder will fail - if this happens please delete those statments and re-submit 
#   - You will get 70 points from the autograder for this assignment and 30 points will be hidden. That is, passing all of the visible tests will give you 70 points. Make sure you are meeting the requirements of the problem to get the other 30 points 
#   - The 30 'hidden' points are the same questions run on a different matrix - make sure your code is generalizable to work on different matrixes!
#   - Full autograder tests and results are on gradescope.
#   - You may upload and run the autograder as many times as needed in your time window to get full points
#   - The assignment needs to be named HW_Unit_09.ipynb to be graded from the autograder!
#   - The examples given are samples of how we will test/grade your code. Please ensure your code outputs the same information.
#     - In addition to the given example, the autograder will test other examples
#     - Each autograder test tells you what input it is using
#   - Once complete, the autograder will show each tests, if that test is passed or failed, and your total score
#   - The autograder fails for a couple of reasons:
#     - Your code crashes with that input (for example: `Test Failed: string index out of range`)
#     - Your code output does not match the 'correct' output (for example: `Test Failed: '1 2 3 2 1' != '1 4 6 4 1'`

# ### Instructions
# 
# In this assignment, we will work with NumPy. For each question below, use NumPy functionality to execute the logic of your answer. <br>(**Do not use ```for``` or ```while``` loops!**)
# 
# The first cell (below) makes two matrixes we will use for this assignment. 
# - The first matrix named `features` is a 1000 row x 10 column matrix that contains a sample of our data. 
#   - Each row is a one sample of the data and each column is the weight of that feature for that sample
#   - The matrix has been 'normalized' to have positive integer values on a scale of 0-100
# - The second matrix named `weights` is a 1 row by 10 column matrix consisting of the weights from the algorithm 
#   - Each column corresponds to the 'weight' of that feature in the calculation of a prediction result
#   - The weights are in the range -1 to +1 and indicate the 'correlation' between that feature and the prediciton result with a negative weight being a negative correlation

# In[ ]:


import numpy as np
np.random.seed(25)

# Creating the matrixes to use for the problem
features = np.random.randint(0, 100, 10000).reshape(1000,10)
weights = np.array([ 0.24540611,  0.31467517, -0.07656614, -0.16161533, -0.09064962,
                    -0.00315615,  0.09609595,  0.2517064 ,  0.01100181, -0.38629211])


# ***
# 
# ##### Data Quality checks
# 
# The first few questions below are some checks to access the data quality of the arrays. Since the arrays were made above, we already know their composition but its good practise to check the data that they contain. 

# **1)** 
# 
# In the function `get_shape` below, return the shape of a given matrix. (5 points)

# In[4]:


# Q9-1 Grading Tag:
def get_shape(ar):
    return ar.shape


# In[ ]:


# Should return: (1000, 10)

get_shape(features)


# **2)** 
# 
# In the function `check_min_max` below, return the minimum value and maximum value of an inputted matrix as a tuple (minimum_value, maximum_value) (5 points)

# In[5]:


# Q9-2 Grading Tag:
def check_min_max(ar):
   return (np.min(ar), np.max(ar))


# In[ ]:


# Should return: (0, 99)

check_min_max(features)


# **3)** 
# 
# In the function `count_nan` below, return the count of the number of NaN values (NaN = 'Not a Number' or Null values) of an inputted matrix (5 points)

# In[6]:


# Q9-3 Grading Tag:
def count_nan(ar):
   return np.isnan(ar).sum()


# In[ ]:


# Should return: 0

count_nan(weights)


# ***
# 
# ##### Outlier Identification
# 
# The next questions below are functions to identify outliers. A good practise is to identify outliers, as outliers are sometimes the most interesting cases to look into but also might cause problems with a machine learning algorithm.

# **4)** 
# 
# In the function `get_median` below, return the median value for each column of an inputted matrix (that is the median value - column-wise) (5 points)

# In[7]:


# Q9-4 Grading Tag:
def get_median(ar):
   return np.median(ar, axis=0)


# In[ ]:


# Should return: array([50. , 49. , 50. , 49. , 50. , 52. , 49. , 49. , 49. , 50.5])

get_median(features)


# **5)** 
# 
# In the function `feat_minimum` below, return the row index that contains the minimum value in a given matrix. (10 points)
# 
# For example - in the matrix below, the minimum value is 2 and that appears in rows 1 and 3 so the function would return the row indexes `array([1,3])`:
# 
# ```
# test_matrix = np.array([[ 4, 62, 90],
#                         [ 2, 89, 31],
#                         [84, 45,  3],
#                         [24, 54,  2]])
# 
# feat_minimum(test_matrix) -> array([1,3])
# ```

# In[8]:


# Q9-5 Grading Tag:
def feat_minimum(ar):
   return np.where(ar == np.min(ar))


# In[ ]:


# Should return: 
# array([ 12,  14,  15,  24,  44,  51,  57,  65,  73,  80,  83,  93,  94,
#         99, 106, 106, 111, 112, 114, 126, 172, 186, 202, 209, 224, 230,
#        239, 259, 264, 270, 281, 284, 297, 299, 305, 306, 308, 364, 367,
#        367, 370, 383, 390, 396, 417, 427, 437, 444, 462, 478, 480, 490,
#        500, 529, 539, 542, 552, 553, 579, 595, 612, 618, 619, 627, 636,
#        643, 646, 667, 679, 681, 719, 720, 720, 725, 732, 735, 741, 743,
#        775, 775, 807, 807, 811, 823, 834, 847, 852, 863, 865, 870, 871,
#        874, 903, 907, 921, 932, 963, 971, 987, 990, 992, 993], dtype=int64)

feat_minimum(features)


# **6)** 
# 
# In the function `feat_maximum` below, return the row index with the maximum value for the `feat_num` feature. For example, if the function was passed 3 for the `feat_num`, it would find the row index with the maximum value for the fourth feature (column 3). (10 points) 
# 
# (Hint: Build on what you made in #5 above!)
# 
# More in-depth example: in the matrix below, for feature 1 (passed as `feat_num`) the maximum value is 89 and that appears in rows 1 and 2 so the function would return the row indexes `array([1,2])`:
# 
# ```
# test_matrix = np.array([[ 4, 62,  9],
#                         [ 2, 89, 31],
#                         [84, 89,  3],
#                         [89, 54,  2]])
# 
# feat_maximum(test_matrix, 1) -> array([1,2])
# ```

# In[9]:


# Q9-6 Grading Tag:
def feat_maximum(ar, feat_num):
   return np.where(ar[:, feat_num] == np.max(ar[:, feat_num]))


# In[ ]:


# Should return: 
# array([ 14,  94, 120, 161, 168, 483, 634, 650, 662, 699, 938, 997], dtype=int64)

feat_maximum(features, 3)


# **7)** 
# 
# In the function `outlier_high` below, return the count of the number of values at or above 90 in a matrix. (5 points)

# In[10]:


# Q9-7 Grading Tag:
def outlier_high(ar):
   return np.sum(ar >= 90)


# In[ ]:


# Should return 990

outlier_high(features)


# **8)** 
# 
# In the function `outlier_low` below, return the count of the number of values at or below 10 in a matrix. (5 points)

# In[11]:


# Q9-8 Grading Tag:
def outlier_low(ar):
   return np.sum(ar <= 10)


# In[ ]:


# Should return 1043

outlier_low(features)


# ***
# 
# ##### Prediction
# 
# The next questions below are functions to use the matrixes made above to calculate a prediction result. This is a simplified example of what some machine learning algorithms do to provide a prediction.

# **9)** 
# 
# In the function `calc_features` below, multiply each row of the `features` matrix with the `weights` matrix return this new matrix and name it `calced`. (5 points)

# In[12]:


# Q9-9 Grading Tag:
def calc_features(np_feat, np_weight):
   return np_feat * np_weight


# In[ ]:


# Expected first couple of lines:
# array([[  0.98162444,  19.50986054,  -6.8909526 , ...,  12.58532   ,
#           0.08801448, -10.81617908],
#        [  0.98162444,  28.00609013,  -2.37355034, ...,  22.1501632 ,
#           0.60509955,  -1.15887633],  ...

calced = calc_features(features, weights)
calced


# **10)** 
# 
# In the function `matrix_score` below, sum the values of the `calced` matrix (from 9 above) row-wise to get a single weighted 'score' for each row return this new matrix and name it `score` (10 points)

# In[13]:


# Q9-10 Grading Tag:
def matrix_score(np_calc):
   return np.sum(np_calc, axis=1)


# In[ ]:


# Expected first couple of lines:
# array([ 1.16594614e+01,  3.71336413e+01,  4.33563146e+01,  1.62602648e+01,
#         2.84698142e+01,  2.60448311e+01, -4.23597988e+00,  4.25279392e+01, ...


score = matrix_score(calced)
score


# **11)** 
# 
# In the function `score_mean` below, find and return the mean of the of the entire matrix `score` (5 points)

# In[14]:


# Q9-11 Grading Tag:
def score_mean(np_score):
   return np.mean(np_score)


# In[ ]:


# Should return 9.73170897548
score_mean(score)


# **12)** 
# 
# In the function `classify` below, using the matrix `score`, 'classify' each record based on if they are above/below the mean as follows: <br>
# - [0] below mean <br>
# - [1] at or above mean <br>
#     
# Make a new matrix called `prediction` that has the values 0 or 1 based on the `score` matrix from 10 above. (10 points)
# 
# Hint: You can use the function you made above in #11 inside this function!

# In[15]:


# Q9-12 Grading Tag:
def classify(np_score):
   mean_score = score_mean(np_score)
   return (np_score >= mean_score).astype(int)


# In[ ]:


# Expected first couple of lines:
# array([1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
#        0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, ...


prediction = classify(score)
prediction


# **13)** 
# 
# In the function `perc_ones` below, using the `prediction` matrix, return the percentage of predictions classified as a 1 (10 points)

# In[16]:


# Q9-13 Grading Tag:
def perc_ones(np_pred):
   return np.mean(np_pred)


# In[ ]:


# Should return: 0.505

perc_ones(prediction) 


# ***
# 
# ##### Evaluation
# 
# The next question makes a function to evaluate how well our algorithm's predictions performed. 
# 
# Run the code below to make the `truth` matrix. This matrix is 1x1000 and is the ground truth on if a result should be 0 or 1. 

# In[ ]:


# Run this to make the truth matrix

np.random.seed(1234)
truth = np.random.randint(0, 2, 1000)


# **14)** 
# 
# In the function `num_correct` below, compare the `truth` matrix to our prediction matrix `prediction`. Return the number of correct predictions in a tuple of two values: (correct_zero_prediction, correct_one_prediction). (10 points)
# 
# A correct_zero_prediction is where our algorithm predicts a 0 and the truth matrix is a 0. (In machine learning, we call this result a 'True Negative')
# 
# A correct_one_prediction is where our algorithm predicts a 1 and the truth matrix is a 1. (In machine learning, we call this result a 'True Positive')

# In[24]:


# Q9-14 Grading Tag:
import numpy as np

def num_correct(np_truth, np_pred):
    
    correct_zero_prediction = np.sum((np_pred == 0) & (np_truth == 0))
    correct_one_prediction = np.sum((np_pred == 1) & (np_truth == 1))
    return (correct_zero_prediction, correct_one_prediction)


num_correct(truth, prediction)


# In[ ]:


# Should return: (233, 263)


