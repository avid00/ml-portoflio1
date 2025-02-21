# Log for Random Forest
### Student id: C00313459
### Name: Amisha Das

## Introduction
Using the Sloan Digital Sky Survey (SDSS) data, perform regression (prediction) task using random forest model to predict redshift values and then compare the results.

Later on, I performs randomizedsearchcv to see if I can make the model more accurate.

## Log work
1. added sdss data
2. dividing dat into 80-20 ratio 
3. LESSON: blobs are used for artifical and neat, gaussian structured data.
4. Added two functions:
    4.1. Classification (classifies data according to features)
    4.2. Regression (predicts redshift)
5. Trying model with classification task first
6. results with 89% accuracy. Added a comparative scatter plot to see differences between true and predicted results.
7. NEXT STEP: to make model better from 89%, trying to increase no. of trees.
8. going to try hyperparamtere tuning to improve model performance and evaluate previous performance
9.  testing only 20 random combinations instead of all 54 to reduce time
10. increasing no. of trees with n_estimators
11. depth of trees made 10,20 and None to reduce overfitting 
NOTE: Took around taking a very long time (8 hours estimated!) to grid_search to finish with 3 folds (243 fits)
LESSON: RF is computationally expensive. maybe reducing option from 3 to 2 would help or testing on ly a subset.
12. using randomizedsearchcv instead of GridSearchCV : 

grid_search = GridSearchCV(RandomForestClassifier(random_state=42),param_grid, cv=3, n_jobs=-1, verbose=2) #TODO: maybe add more jobs
param_grid = {
    'n_estimators': [100,300,500], #number of trees in rf
    'max_depth': [10,20,None], #mx depth of trees
    'min_samples_split': [2,5,10], #minimum samples per splitting 
    'min_samples_leaf': [1,2,4] #min samplesper leaf
}

EDITS:
    Using randomizedsearchcv
    'None' as well to remove deep trees
    cross validation from 3 to 2
    reducing n_estimators

12. new code:
param_dist = {
    'n_estimators': [50, 100, 300],  # Smaller range for efficiency
    'max_depth': [10, 20],  # Prevent deep trees for faster results
    'min_samples_split': [2, 5, 10],  # Control overfitting
    'min_samples_leaf': [1, 2, 4]  # Keep best minimum leaf options
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,  # Test only 20 random combinations instead of all 54
    cv=2,  # Faster cross-validation
    n_jobs=-1,  # Use all CPU cores
    verbose=2,
    random_state=42  # Ensures reproducibility
)
difference between gridsearchcv and randomizedsearchcv
(ADD TABLE)

13. retraining model with new paramtres:
best_params = {'n_estimators': 300, 
               'min_samples_split': 5, 
               'min_samples_leaf': 1, 
               'max_depth': 20}

14. Results didn't improve much at all

    
## Future Work
increase accuracy using other methods.
try regression
Learn how to assign GPU for randomizedsearchcv


## Appendix and Acknolwedgments
- Chatgpt for findign data set, ideas and debugging especially with randomizedsearchcv
- SDSS for data set
- https://www.kaggle.com/code/ktrinh/sdss-classification-with-random-forests-99-2/notebook
