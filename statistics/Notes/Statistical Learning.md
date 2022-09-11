# Statistical Learning

### **What is Statistical Learning ?**

$$Y = f(X) + ε$$

**Y** : Output Variable/Dependent Variable/Response Variable

**X** : Independent variable or predictors

**ε** : Error - random error, irreducible error or reducible error

**f** : Systematic information that X provides about Y  

Statistical Learning are set of approaches for estimating f 

The accuracy of Y^ as a prediction of Y depends on two quantities : Reducible error and irreducible error.

$$E(Y - Y')^2 = E[F(X) + ε- f'(X)]^2 = [f(X) - f'(X)]^2 + var(ε)$$

*f^*  will not be an accurate estimate of f. This inaccuracy will introduce some errors and this error is called reducible error.

Y is also a function of ε, which by definition, cannot be predicted using X. Therefore, variability of ε also affects the accuracy. This is known as irreducible error

---

### **Why estimate f ?**

Two reasons :

1. **Prediction**
    - Set of inputs X available but output Y cannot be easily obtained
    - we can predict Y using :

        $$Y' = f(X)$$

f is treated as black box. Prediction is not concerned with exact form of f

**2.  Inference**

- How is Y affected when X changes . Understand the relationship between X and Y
- Which Predictors are associated with the response ?
- What is the relationship between response and each predictor ?
- Can the relationship between Y and X can be summarized using linear equation or is the relationship more complex ?

f cannot be treated as black box

### How do we estimate f ?

Apply a Statistical learning method to the training data in order to estimate **f** .  Most Statistical methods can be characterized as Parametric or Non Parametric.

1. **Parametric**
    - We make an assumption about f or shape of eg: f is linear. Once we assume, the problem of estimating f is simplified
    - After Model has been selected, we use the training data to fit or train the model. Most common approach is ordinary least squares

Reduce the problem of estimating f down to estimating a set of parameters , This significantly reduces computation time, Fewer parameter estimates also means that less observations are required to accurately estimate, more interpretable, although there are non-parametric methods that are also highly interpretable (e.g. decision trees)

Disadvantage : Model usually does not match with the true unknown form of **f,** Often more heavily influenced by outliers, particularly when compared to tree-based non-parametric approaches

**2. Non Parametric**

- We don't make any explicit assumptions about f
- They seek an estimate of f as close to the true form of unknown f.

Require very large number of observation in order to get accurate estimation. Overfitting

Overfitting : 100 % accurate for train data but yields inaccurate estimates for new data

### Accuracy vs Interpretability

Why would you choose a restrictive method instead of a flexible method ? 

Restrictive Models are much more interpretable . Where inference is the goal, Linear Model may be a better approach. Complicated estimates of f is difficult to understand how any individual predictor is associated with response.

Higher the flexibility, More difficult to interpret

### Accessing Model Accuracy

Selecting the best model can be the most challenging parts of performing statistical learning

**Measuring the quality of fit**

- Quantify the closeness of prediction with actual.
- In regression, the most popular method to quantify is MSE ( Mean squared Error)
- There is no guarantee that Model with the lowest MSE in training will yield the lowest test MSE.
- When there is a small train MSE but large test MSE, it is called Overfitting
- Cross validation is method to estimate test MSE from training data

**Bias-Variance tradeoff**

The expected test MSE for a given value X' can always be decomposed into the sum of three fundamental quantities :  variance of f(X'), the squared bias of f(X') and the variance of the error term. (proof is beyond the scope now)

$$E(y'-f(X'))^2 = Var(f(X')+ [Bias(f(X')]^2+Var(e)$$

**Variance** refers to the amount by which f would change if we estimated it using a different training data set. ideally the estimate for f should not vary much between training sets

if a method has high variance, then small changes in the training data can have large changes in f

In general, more flexible models have high variance

**Bias** refers to the error that is introduced when we approximate a highly complicated model with a simpler model

 

The relative rate of change of variance and bias will determine whether test MSE will increase or decrease. As we increase the flexibility of the model, bias tends to decrease faster than the variance increases. However at some point, increasing flexibility will have little impact on bias but starts to increase the variance significantly, when this happens MSE increases

The relationship between variance, bias and test MSE is referred as bias-variance tradeoff

### Classification Setting

Training error rate :

$$1/nΣ(I(y≠y'))$$

I is referred to Indicator variable that equals 1 if y ≠ y' and 0 if y==y'

Bayes classifier assigns each observation to most likely class , given its predictor values. Bayes classifier produces the lower error rate called Bayes error rate. Bayes classifier serves as the unattainable golden standard against which to compare other methods

KNN applies Bayes rule. The choice of K has a drastic effect on the classifier obtained. As K grows, The flexibility decreases and produces a decision boundary close to linear and this will result in high bias.

Choosing the correct level of flexibility is crucial for success of any statistical model. Bias variance tradeoff makes it a difficult task

Exercise solutions: 

[https://www.kaggle.com/lmorgan95/islr-statistical-learning-ch-2-solutions](https://www.kaggle.com/lmorgan95/islr-statistical-learning-ch-2-solutions)