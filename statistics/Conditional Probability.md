# Conditional Probability  

If $A$ and $B$ are two events in a sample space SS, then the **conditional probability of A given B** is defined as: $P(A|B)=P(A∩B)P(B), when P(B)>0$

>Here is the intuition behind the formula. When we know that $B$ has occurred, every outcome that is outside $B$ should be discarded. Thus, our sample space is reduced to the set $B$, Now the only way that $A$ can happen is when the outcome belongs to the set $A∩B$. We divide $P(A∩B)$ by $P(B)$, so that the conditional probability of the new sample space becomes $1$, i.e., $P(B|B)=P(B∩B)/P(B)=1$.

## Independent Events
- Two events $A$ and $B$ are independent if and only if $P(A∩B)=P(A)P(B)$
- If two events $A$ and $B$ are independent and $P(B)≠0, then P(A|B)=P(A)$
- To summarize, we can say "independence means we can multiply the probabilities of events to obtain the probability of their intersection", or equivalently, "independence means that conditional probability of one event given another is the same as the original (prior) probability".

## Disjoint Events
- One common mistake is to confuse independence and being disjoint. These are completely different concepts.:
-  When two events $A$ and $B$ are disjoint it means that if one of them occurs, the other one cannot occur, i.e., $A∩B=∅$. Thus, event $A$ usually gives a lot of information about event $B$ which means that they cannot be independent. 
	
## law of total probability

- The total probability rule (also called the Law of Total Probability) breaks up probability calculations into distinct parts. It’s used to find the probability of an event, A, when you don’t know enough about A’s probabilities to calculate it directly. Instead, you take a related event, B, and use that to calculate the probability for A.
- Suppose $B_1,B_2,B_3..B_k$  are mutually exclusive and exhaustive events in a sample space $S$. The probability for $A$ can be written :

$$P(A) = P(A∩B) + P(A∩Bc)$$


>Bayes’ theorem is one of the most common applications of conditional probabilities. Bayes' theorem is the law of probability governing the strength of evidence - the rule saying how much to revise our probabilities when we learn a new fact or observe new evidence.

