# Day 2: Method of Central Tendency

- Among the first #statistics computed for the continuous variables in a new dataset. These measures indicate where most values in a distribution fall .
- The three most common measures of central tendency are the *mean*, the *median*, and the *mode*. One’s choice of mean, median, or mode can dramatically change the interpretation of the data.
- The **mean** is the arithmetic average (the sum of the scores divided by the number of scores). While the mean is the most common measure of a dataset’s center, there are situations when it cannot be used.
- Mean is not an appropriate summary measure for every dataset because it is sensitive to extreme values (outliers), so can be misleading for Skewed data.
-  **Trimmed Mean** also known as Winsorized Mean is calculated by discarding certain percentage of extreme values in a distribution and calculating mean of the remaining values.
- When the data are nominal (i.e., when the data are categories rather than values), you must use the **mode** (Most frequently occurring value) to summarize the center
- **Median** is the best option when data are ordinal. The median of a distribution of scores is the value at the **50th percentile**, which means that half of the scores are below this value and half are above it.
- To summarize interval or ratio data, in general, you should use the mean. however, if the data set contains one or more outliers, you should use the median.
- If **mean and median are close to each other**, and the most common ranges cluster around the mean , then we can conclude that data is <b> Normal and Symmetrical </b>
- A **mean lower than the median** is typical of **left-skewed data** because the extreme lower values pull the mean down, whereas they do not have the same effect on the median
- A **mean higher than a median** is common for **right-skewed data** because the extreme higher values pull the mean up but do not have the same effect on the median.

### Preformance Analysis and Percentiles

- In real world environments, performance gets attention when it is poor and has a negative impact on the business and users. But how can we identify performance issues quickly to prevent negative effects?
- We cannot alert on every slow transaction, since there are always some. The industry has come up with a solution called Automatic Baselining. Baselining calculates out the “normal” performance and only alerts us when an application slows down or produces more errors than us
- Most approaches rely on averages and standard deviations. This approach assumes that the response times are distributed over a bell curve. Averages , in this case, are ineffective because they are too simplistic and one-dimensional.
- A percentile gives a much better sense of real world performance, it tells me at which part of the curve I am looking at and how many transactions are represented by that metric.
- For exactly that reason percentiles are perfect for automatic baselining. If the 50th percentile moves from 500ms to 600ms I know that 50% of my transactions suffered a 20% performance degradation. You need to react to that.
- Percentile-based alerts do not suffer from false positives, are a lot less volatile and don’t miss any important performance degradations! Consequently, a baselining approach that uses percentiles does not require a lot of tuning variables to work effectively.

Reference :
1. Why Averages Suck and Percentiles are Great 
