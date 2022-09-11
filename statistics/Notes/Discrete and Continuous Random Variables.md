# Discrete and Continuous Random Variables

- A **Variable** is a quantity whose value changes
- A **Discrete variable** is a variable whose value is obtained by counting . eg: Number of students in a class, number of marbles in a jar
- A **Continuous Variable** is a variable whose value is obtained by measuring eg: Height of students in a class
- A **random variable** is a variable whose value is determined by random chance
    - A random variable is denoted by a capital letter
    - Probability distribution of a random variable X tells what the possible values of X are and how probabilities are assigned to those values
    - Random variable can be discrete or continuous
- A **Discrete random variable**  X has countable number of possible values.
    - To graph the probability distribution of X, we construct probability histogram
- A C**ontinuous random variable** X takes all values in a given interval of values:
    - The Probability distribution of a continuous random variable is shown by a density curve
    - The Probability that X is between an interval of numbers  is the area under the density curve between the interval points
    - The probability that X is exactly equal to a number is Zero.


## Key statistics

### Expected Value ($E_x$)

- If you have a collection of numbers, $a_1..a_n$, then the average of the numbers is a representation of the collection. No concider a collection of random variables $X$, then the average of the random variables is called the **Expected Value**
- To understand the concept behind $E_X$, consider a discrete random variable with range $R_X={x_1,x_2,x_3,...}$ This random variable is a result of random experiment. Suppose that we repeat this experiment a very large number of times $N$, and that the trials are independent. Let $N1$ be the number of times we observe $x1$, $N2$ be the number of times we observe $x2$, ...., $Nk$ be the number of times we observe $x_k$, and so on. Since $P(X=x_k)=P_X(x_k)$, we expect that

$$P_X(x_1)≈N_1/N$$
$$P_X(x_2)≈N_2/N$$
$$P_X(x_k)≈N_k/N$$

In other words, we have $N_k≈N*P_X(x_k)$. Now, if we take the average of the observed values of $X$, we obtain

$$Average = (N1x1+N2x2+N3x3+...)/N $$
$$= (NP_X(x_1)x1+NP_X(x_2)x2+...)/N $$
$$= P_X(x_1)x1+P_X(x_2)x2+...$$
$$= \sum P_X(x_i)xi$$



>
Thus, the intuition behind $E_X$ is that if you repeat the random experiment independently NN times and take the average of the observed data, the average gets closer and closer to $E_X$ as $N$ gets larger and larger.


### Variance