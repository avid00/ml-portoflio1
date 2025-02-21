# Log for Support Vector Machine
### Student id: C00313459
### Name: Amisha Das

## Introduction

## Log work
1. Error:
    from mlxtend.plotting import plot_decision_regions
    module mlxten dnot found
    
    Solution: !pip install mlxtend

2. plt for svm_pca was taking too long, so i edited the parameters to reduce number of X_train_pca samples.

    changed from: 
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    changed to:
    X_train_pca_small = X_train[:2000]  
    y_train_small = y_train[:2000]

    using only 2000 points
3.
   error:
    -->  plot_decision_regions(X_train, y_train.to_numpy(), clf=svm_pca, legend=2)
    ValueError: Filler values must be provided when X has more than 2 training features.
    
    fix:
    Convert y_train to a NumPy array as it still a pandas series
    y_train_small_np = y_train_small.to_numpy()

4. for classificaion report:
    error:
    'undefinemetricwarning' because label 0 has no correctly predicted samples.
    precision and recall doesnt matter when no positives exist
    
    changing from:
    print(classification_report(y_train_small_filtered, y_pred_pca))
    
    changing to:
    print(classification_report(y_train_small_filtered, y_pred_pca, zero_division=0))
    zero_division=0 replaces NaN with 0.

## Future Work
1. Compare this result of same data with Random Forest

## Appendix and Acknolwedgments
- Chatgpt for debugging and function correction
- https://www.kaggle.com/code/farazrahman/predicting-star-galaxy-quasar-with-svm
