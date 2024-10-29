[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Dz6Kd_jy)
# Programming Assignment 2
**DATA 259**  
**Fall 2024**

## Introduction
For questions on this and other assignments, you may need to write Python code to do data analysis, provide only an explanation in prose, or do both. In this course, we will be using Quarto documents to author assignment submissions and reports to develop our skills in reproducible research tools. Use a code chunk for any code you write in solving problems, and use Markdown for your explanation of what you did. We will not count your answer as correct if we cannot see it/it is written in a code comment.

We recommend one code chunk for each question, but feel free to add more as you see fit. In general, within the Markdown, you should explain the approach you took in your code at a high level in addition to explicitly answering any questions that were asked. It is up to you to decide what code-based analyses, if any, are appropriate for a particular problem.

When you are finished, please render your document to a PDF and upload your assignment in Gradescope. Make sure to select the areas of the page corresponding to the questions on the assignment outline. It is much easier for the graders to give you feedback this way, and you will therefore get your homework assignments back faster. If there is a lot of excess output, either revisit your code to make sure you are not printing excessively, or delete the pages with excess output from the PDF before submitting.

---

## Problem 1: Pre-processing
Machine learning starts with data. For this assignment, we will be using the UCI income dataset “income.csv”. This is data collected from the US Census in 1994. It contains demographic data paired with information about whether the person in the entry has an annual income over $50,000. While this data is of questionable use for solving data science problems (consider why), it is commonly used as a benchmark for machine learning tasks. For more details, please read the UCI [income dataset documentation](https://archive.ics.uci.edu/dataset/20/census+income).

1. **Pre-process the data**: Before we use this data, we’ll need to do some pre-processing. First, you need to decide how to handle missing values. Investigate the presence of missing values in your features, specify how each is to be handled for the rest of this exercise, and carry out any pre-processing you need.
   
2. **Handling categorical variables**: In addition to making sure that the numeric data is properly handled, there are several categorical variables in this data set. For example, `workclass` contains information about what sector each person is employed in. If we encode this by saying that `State-gov = 0`, `Self-emp-not-inc = 1`, etc., certain kinds of models (e.g., logistic regression) will treat this as a statement that there is an ordering of these variables. The way to avoid enforcing arbitrary structure on your dataset is by using indicator or dummy variables. Use pandas’ `get_dummies()` function to convert all columns containing categorical variables into properly encoded columns.

3. **Downsides of dummy variables**: Name any potential downside(s) of the creation of dummy variables for categorical features.

4. **Handling outliers**: Finally, an essential part of data cleaning and exploratory data analysis is to investigate the presence of outliers before proceeding to modeling. How do you suggest checking for outliers in this data set? How do you propose to handle them?

---

## Problem 2: Classifiers
1. This is technically enough processing to build a model. Let’s start by using the Python library Scikit-learn (`sklearn`) to create a classification model. Here, we take a look at how a Random Forest Classifier can be trained to predict whether a person has an income above $50,000. Carefully examine and modify the code below based on the comment:

    ```python
    # change df_dummy to the DataFrame with dummy coded columns you created in 3.1.1
    X = df_dummy.drop("income_>50K", axis=1)
    y = df_dummy["income_>50K"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    def platform_preprocess(X_train, X_test):
        # preprocess data
        scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)
        return scaled_X_train, scaled_X_test

    def platform_train_process(X_train, y_train):
        # model selection and training
        parameters_for_testing = {
            "n_estimators": [100,150,200],
            "max_features": [3,4],
        }
        model = RandomForestClassifier()
        kfold = KFold(n_splits=10, random_state=None)
        grid_cv = GridSearchCV(estimator=model, param_grid=parameters_for_testing, scoring='accuracy', cv=kfold)
        result = grid_cv.fit(X_train, y_train)
        print("Best: {} using {}".format(result.best_score_, result.best_params_))

        # model training
        tuned_model = RandomForestClassifier(n_estimators=result.best_params_['n_estimators'], max_features=result.best_params_['max_features'])
        tuned_model.fit(X_train, y_train)

        return tuned_model

    def platform_test_model(model, X_test, y_test):
        # prediction on test data (benchmark)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    pp_X_train, pp_X_test = platform_preprocess(X_train, X_test)
    # train
    model = platform_train_process(pp_X_train, y_train)

    # test
    accuracy = platform_test_model(model, pp_X_test, y_test)
    print("Acc: " + str(accuracy))
    ```

2. **Train-test split**: What does the function `train_test_split()` do and why should we use it? Report the train and test accuracy that were printed and interpret them in a few sentences.

3. **F1 Score**: Accuracy as a performance metric has some significant drawbacks when it comes to handling imbalanced class distribution, and a better measure in this regard is F1-score. Using `sklearn.metrics` module, test how the trained classifier performs on the F1 score metric.

4. **Testing another classifier**: There are different kinds of models you can use (e.g., decision tree, logistic regression, neural network) to build a classifier. `sklearn` implements many of the most popular model classes. Using the previous template, test the performance of another classifier on this dataset. Pick one other model class and train a supervised learning model you believe to be reasonably generalizable using the same data. If you’ve never built models before, feel encouraged to do some searching through the documentation or other tutorials to see how to build other models. Note that you may need to apply some transformations to the data depending on the model class you choose. Report the performance metrics and comment on how it compares to the Random Forest classifier above.

---

## Problem 3: Regression
The learning process for predicting real-valued dependent variables instead of categorical ones is called Regression. For regression analysis, we will be investigating a dataset `insurance.csv` that records various patient-related information, like age, sex, BMI, etc. We want to use these features to accurately predict individual medical costs billed by their health insurance.

1. **Pre-processing for regression**: As with any machine learning problem, we begin with data cleaning and exploratory data analysis. Perform the necessary steps before we can train a model for the prediction task. Do we need to create dummy variables for all categorical features in this dataset? Is there an alternative encoding of categorical features that you can apply?

2. **Visualizations**: Provide three visualizations, each of which illustrates an interesting relationship within the dataset. For example, what impact does whether a patient is a smoker have on the insurance? For the classification task, we had more data available at our disposal. However, for the current regression task, we have an additional smaller dataset. So, rather than splitting the training data further to extract a validation set, apply cross-validation.

3. **Baseline model**: Train a baseline Linear Regression model on this dataset. Along with the cross-validation, what performance metric would you choose to evaluate your model? How does your model perform on the test data based on this metric? Although linear regression is a simple and easily interpretable model, you may want to improve the baseline performance with a better model.

4. **Improving performance**: Attempt to fit a regression model that yields a better fit on our dataset than the baseline when compared on the same performance metric as the baseline model. Report the performance of this new trained and fine-tuned model on your test data.
