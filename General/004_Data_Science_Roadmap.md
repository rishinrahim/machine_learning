# Data Science Roadmap

[source](https://www.reddit.com/r/learnmachinelearning/comments/lbz4md/the_ultimate_blueprint_to_getting_a_job_in_data/)

The ultimate blueprint to getting a job in data science

## Software Engineering

Not all Data Scientist roles will grill you on the time complexity of an algorithm, but all of these roles will expect you to write code. Data Science isn’t one job, but a collection of jobs that attracts talent from a variety of industries, including the software engineering world. As such you’re competing with guys that know the ins and outs of writing efficient code and I would recommend spending at least 1–2 hours a day in the lead-up to your interview practicing the following concepts:

- Arrays
- Hash Tables
- Linked Lists
- Two-Pointer based algorithms
- String algorithms (interviewers LOVE these)
- Binary Search
- Divide and Conquer Algorithms
- Sorting Algorithms
- Dynamic Programming Recursion

**DO NOT LEARN THE ALGORITHMS OFF BY HEART**. This approach is useless, because the interviewer can question you on any variation of the algorithm and you will be lost. Instead learn the strategy behind how each algorithm works. Learn what computational and spatial complexity are, and learn why they are so fundamental to building efficient code.

## Applied Statistics

Data science has an implicit dependence on applied statistics, and how implicit that will be depends on the role you’ve applied for. Where do we use applied statistics? It pops up just about anywhere where we need to organize, interpret and derive insights from data.

- Descriptive statistics (What distribution does my data follow, what are the modes of the distribution, the expectation, the variance)
- Probability theory (Given my data follows a Binomial distribution, what is the probability of observing 5 paying customers in 10 click-through events)
- Hypothesis testing (forming the basis of any question on A/B testing, T-tests, anova, chi-squared tests, etc).
- Regression (Is the relationship between my variables linear, what are potential sources of bias, what are the assumptions behind the ordinary least squares solution)
- Bayesian Inference (What are some advantages/disadvantages vs frequentist methods)
- Introduction to Probability and Statistics, an open course
- Machine Learning: A Bayesian and Optimization Perspective by Sergios Theodoridis.

The way you’re going to remember this stuff isn’t through memorization, you need to solve as many problems as you can get your hands on. Glassdoor is a great repo for the sorts of applied stats questions typically asked in interviews. The most challenging interview I had by far was with G-Research, but I really enjoyed studying for the exam, and their sample exam papers were fantastic resources when it came to testing how far I was getting in my applied statistics revision.

## Machine Learning

Now we come to the beast, the buzzword of our millennial era, and a topic so broad that it can be easy to get so lost in revision that you want to give up. The applied statistics part of this study guide will give you a very very strong foundation to get started with machine learning (which is basically just applied applied statistics written in fancy linear algebra), but there are certain key concepts that came up over and over again during my interviews. Here is a (by no means exhaustive) set of concepts organized by topic:

- **Metrics** — Classification Confusion Matrices, Accuracy, Precision, Recall, Sensitivity F1 Score TPR, TNR, FPR, FNR Type I and Type II errors AUC-ROC Curves, Regression Total sum of squares, explained sum of squares, residual sum of squares Coefficient of determination and its adjusted form AIC and BIC Advantages and disadvantages of RMSE, MSE, MAE, MAPE
- **Bias-Variance Tradeoff, Over/Under-Fitting K Nearest Neighbors** algorithm and the choice of k in bias-variance trade-off , **Random Forests,** The asymptotic property Curse of dimensionality Model Selection **K-Fold Cross Validation,** **L1 and L2 Regularization** , **Bayesian Optimization**
- **Sampling** Dealing with class imbalance when training classification models, **SMOTE** for generating pseudo observations for an underrepresented class ,Class imbalance in the independent variables Sampling methods, Sources of sampling bias, Measuring Sampling Error
- **Hypothesis Testing** This really comes under under applied statistics, but I cannot stress enough the importance of learning about statistical power. It’s enormously important in **A/B testing.**
- **Regression Models,** **Ordinary Linear Regression**, its assumptions, estimator derivation and limitations are covered in significant detail in the sources cited in the applied statistics section. Other regression models you should be familiar with are:
    - Deep Neural Networks for Regression
    - Random Forest Regression
    - XGBoost Regression
    - Time Series Regression (ARIMA/SARIMA)
    - Bayesian Linear Regression
    - Gaussian Process Regression
- **Clustering Algorithm**s :  K-Means, Hierarchical Clustering, Dirichlet Process Mixture Models
- **Classification Models** : Logistic Regression (Most important one, revise well) Multiple Regression XGBoost, Classification, Support Vector Machines

It’s a lot, but much of the content will be trivial if your applied statistics foundation is strong enough. I would recommend knowing the ins and outs of at least three different classification/regression/clustering methods, because the interviewer could always (and has previously) asked “what other methods could we have used, what are some advantages/disadvantages”? This is a small subset of the machine learning knowledge in the world, but if you know these important examples, the interviews will flow a lot more smoothly.

## Data Wrangling, Manipulation

**“What are some of the steps for data wrangling and data cleaning before applying machine learning algorithms”?**

We are given a new dataset, the first thing you’ll need to prove is that you can perform an exploratory data analysis (EDA). Before you learn anything realize that there is one path to success in data wrangling: Pandas. The Pandas IDE, when used correctly, is the most powerful tool in a data scientists toolbox. The best way to learn how to use Pandas for data manipulation is to download many, many datasets and learn how to do the following set of tasks as confidently as you making your morning cup of coffee.

One of my interviews involved downloading a dataset, cleaning it, visualizing it, performing feature selection, building and evaluating a model all in one hour. It was a crazy hard task, and I felt overwhelmed at times, but I made sure I had practiced building model pipelines for weeks before actually attempting the interview, so I knew I could find my way if I got lost.

Advice: The only way to get good at all this is to practice, and the **Kaggle community** has an incredible wealth of knowledge on mastering EDAs and model pipeline building. I would check out some of the top ranking notebooks on some of the projects out there. Download some example datasets and build your own notebooks, get familiar with the Pandas syntax.

**Data Organization** There are three sure things in life: death, taxes and getting asked to merge datasets, and **perform groupby** and apply tasks on said **merged datasets**. Pandas is INCREDIBLY versatile at this, so please **practice practice practice.**

**Data Profiling** : This involves getting a feel for the “meta” characteristics of the dataset, such as the shape and description of numerical, categorical and date-time features in the data. You should always be seeking to address a set of questions like “how many observations do I have”, “what does the distribution of each feature look like”, “what do the features mean”. This kind of profiling early on can help you reject non-relevant features from the outset, such as categorical features with thousands of levels (names, unique identifiers) and mean less work for you and your machine later on (work smart, not hard, or something woke like that).

## Visualization

Here you are asking yourself **“what does the distribution of my features even look like?”**. A word of advice, if you didn’t learn about boxplots in the applied statistics part of the study guide, then here is where I stress you learn about them, because you need to learn how to identify outliers visually and we can discuss how to deal with them later on.

**Histograms** and **kernel density estimation** plots are extremely useful tools when looking at properties of the distributions of each feature. We can then ask “what does the relationship between my features look like”, in which case Python has a package called **seaborn** containing very nifty tools like **pairplot** and a visually satisfying **heatmap** for **correlation** plots. Handling Null Values, Syntax Errors and Duplicate Rows/Columns Missing values are a sure thing in any dataset, and arise due to a multitude of different factors, each contributing to bias in their own unique way. There is a whole field of study on how best to **deal with missing values** (and I once had an interview where I was expected to know individual methods for missing value imputation in much detail). Check out this primer on ways of handling null values.

Syntax errors typically arise when our dataset contains information that has been manually input, such as through a form. This could lead us to erroneously conclude that a categorical feature has many more levels than are actually present, because “Hot”, ‘hOt”, “hot/n” are all considered unique levels. Check out this primer on **handling dirty text data**. Finally, duplicate columns are of no use to anyone, and having duplicate rows could lead to **overrepresentation bias**, so it’s worth dealing with them early on.

**Standardization** or **Normalization** Depending on the dataset you’re working with and the machine learning method you decide to use, it may be useful to standardize or normalize your data so that different scales of different variables don’t negatively impact the performance of your model. 

There’s a lot here to go through, but honestly it wasn’t as much the “memorize everything” mentality that helped me insofar as it was the confidence building that learning as much as I could instilled in me. I must have failed so many interviews before the formula “clicked” and I realized that all of these things aren’t esoteric concepts that only the elite can master, they’re just tools that you use to build incredible models and derive insights from data.

**Resources**

[Awesome Deep Learning for Natural Language Processing (NLP)](Data%20Science%20Roadmap%20680be36e47b34222b922f90a11c0e7f8/Awesome%20Deep%20Learning%20for%20Natural%20Language%20Process%20ebbe08d20a4b4dd19ad8b3df67bdaaba.md)