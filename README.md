# MIT 6.036 MITOpenLearningLibrary
Introduction to Machine Learning
course website: https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/course/

My solutions to some exercise and homework in this open course (taken in April 2023), cannot guarentee the correctness.

## known issues:
### week_8: 
hw8 comes with pre-written code and we are asked to run it and answer questions, I have no idea why everytime I ran it for question 3A), the following error message came up:

Cell In[1], line 80, in run_keras(X_train, y_train, X_val, y_val, X_test, y_test, layers, epochs, split, verbose)
     78 # Evaluate the model on validation data, if any
     79 if X_val is not None or split > 0:
---> 80     val_acc, val_loss = history.values['epoch_val_acc'][-1], history.values['epoch_val_loss'][-1]
     81     print ("\nLoss on validation set:"  + str(val_loss) + " Accuracy on validation set: " + str(val_acc))
     82 else:

IndexError: list index out of range

The same issue persisted when I tried to run the jupyter notebook from this link: https://github.com/elahea2020/6.036/blob/df86454f45f308815301c0783ac793f211c8dc1f/HW8/hw8/HW8.ipynb
