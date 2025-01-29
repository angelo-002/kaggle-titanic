Kaggle Titanic Solution

This project uses a Mixture of Experts approach to predict the survival of passengers on the Titanic, leveraging various machine learning classifiers like Random Forest and Gradient Boosting. 
The model splits the data into specialized "experts" based on passenger features such as class, gender, family size, and age, and a gating network decides which expert to trust for each prediction.

Key Features:

-    Data preprocessing: Feature engineering, missing value imputation, and categorical encoding.
-    Mixture of Experts: A unique ensemble model approach combining multiple classifiers.
-    Kaggle Submission: Predicts survival and generates a submission CSV file, including PassengerId and Survived.
-    Visualization: 2D bar chart showing survival prediction distribution.
