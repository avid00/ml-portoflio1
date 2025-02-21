# Log for Naive Bayes
### Student id: C00313459
### Name: Amisha Das

## Introduction
This notebook dlevs into 2 datasets:
1. Diabetes datase- about a patient's vitals and then a binary column where 0 signifies no diabetes and 1 signifies diabetes positive.
2. Major Atmospheric Gamma Imaging Cherenkov (MAGIC) telescope  - binary classification set to classify an event as gamma ray signal (g) or background noise from hardons (h)

## Log Work
### Gaussian Naive Bayes (GNB)
1. Added count-based diabetes data.
2. added LabelEncoder() to convert categories into numbers
3.  removed:
    magic_df.columns = feature_names
    
    added: 
    if isinstance(magic_df, pd.DataFrame):
        magic_df.columns = feature_names
    else:
        print("Error: magic_df is not a DataFrame. Check file reading.")

### GNB with PCA
1. tried increasing model accuracy with PCA.
2. removed: redundant fit_transform() 
3. results are negligible. Can be worked on for future work

### Multinomial NB
1. added MultinomialNB and KBinsDiscretizer to sklearn import
2. Tried working with .data file but it wouldn't work.
3. adding code to check if there is error:
   if isinstance(magic_df, pd.DataFrame):
    magic_df.columns = feature_names
else:
    print("Error: magic_df is not a DataFrame. Check file reading.")
4. Converted .data file to .csv to work better with pandas
5. 

## Extra work
1. Added jiblib.dump to export model pkl file.

## Future Work
1. add bernoulliNB
2. increase accuracy of multionomialNB

## Appendix and Acknolwedgments
- https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
- Chatgpt for findign data set, ideas and debugging
- https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
- 
