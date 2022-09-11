# Likelihood

Likelihood is different for discrete and continuous random variable

### **Discrete Random Variables**

- Suppose that you have a stochastic process that takes discrete values ( eg. Outcomes of tossing of coin 10 times)
- In such cases, we calculate probability of particular set of outcomes by making assumptions about the underlying stochastic process (  eg Probability of landing Heads is $p$ and that coin tosses are independent)
- Denote the observed outcomes by $O$ and set of parameters that describe the stochastic process as $\theta$,  , then when we speak about probability we mean $P(O|\theta)$
- However, when we model real life stochastic process, we often do not know $\theta$, we simple observe  $O$ and then our goal is to estimate  $\theta$
- We know that given a value of  $\theta$ the probability of observing $O$ is  $P(O|\theta)$  Thus, a 'natural' estimation process is to choose that value of $\theta$ that would maximize the probability that we would actually observe  $O$.
- Find the parameter values  $\theta$ that maximize the following function :

$$L(\theta|O) = P(O|\theta)$$

- **L(θ|O) is called the likelihood function. Notice that by definition the likelihood function is conditioned on the observed O and that it is a function of the unknown parameters θ.**

### Continuous Random Variable

- In the continuous case the situation is similar with one important difference. We can no longer talk about the probability that we observed O given θ because in the continuous case P(O|θ)=0.
- Denote the probability density function (pdf) associated with the outcomes  $O$ as:  $f(O|\theta)$. Thus, in the continuous case we estimate θ given observed outcomes  $O$ by maximizing the following function:

$$L(\theta|O) = f(O|\theta)$$