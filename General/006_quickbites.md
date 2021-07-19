# QuickBites

An interview guide on common Machine Learning concepts, best practices, definitions, and theory.

### Contents

1. Model Scoring Metrics
2. Parameter Sharing
3. k-Fold Cross Validation
4. Python Data Types
5. Improving Model Performance
6. Computer Vision Models
7. Attention and its Variants
8. Handling Class Imbalance
9. Computer Vision Glossary
10. Vanilla Backpropagation
11. Regularization
12. References

---

## Model Scoring Metrics

- **Classification Accuracy**

    $$accuracy = \frac{\text{\# Correct Prediction}}{\text{\# Total predictions}}$$

    - Used when the number of positive and negative classes are almost equal
    - May be misleading → if model gets an accuracy of 90%, and has 90% positive instances and 10% negative instances, it doesn't mean model is good if it fails on all negative instances
    - Problem arises when the cost of a misclassification is very high (fatal disease prediction)
- **Log Loss**

    $$\text{LogLoss} = \frac{-1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N} y_{ij} * log(p_{ij})$$

    where $y_{ij}$ indicates whether sample $i$ belongs to class $j$, and $p_{ij}$ is the probability of sample $i$ belonging to class $j$

    - Log loss has no upper bound and ranges from $[0, \infty)$
    - Scores near $0$ indicate high accuracies while scores away from $0$  indicate lower accuracies
- **Confusion Matrix**

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b29929bb-6f61-4b10-92b5-b9c541e0972d/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b29929bb-6f61-4b10-92b5-b9c541e0972d/Untitled.png)

    - Mostly used for binary classification models

    ```coffeescript
    **Name       Pred  Truth**
    TP          Y      Y
    TN          N      N
    FP          Y      N
    FN          N      Y
    ```

    $$\text{accuracy} = \frac{TP + TN}{TP+TN+FP+FN}$$

    - Components of Confusion Matrix form the basis for other metrics
- **Precision**

    $$\text{precision} = \frac{TP}{TP + FP}$$

    - Number of correct positive predictions over total positive predictions
    - Used for binary classifications tasks
- **Recall**

    $$\text{recall} = \frac{TP}{TP+FN}$$

    - Number of correct positive predictions over all samples to be identified as positive
    - Used for binary classification tasks
- **F1 Score**

    $$\text{F1} = 2 * \frac{1}{\frac{1}{precision} + \frac{1}{recall}}$$

    - Harmonic mean of the precision and recall
    - Lies between $[0, 1]$
    - Shows how precise (how many instances correctly classified) and robust (it did not misclassify a significant number of instances) the model is
        - High precision + low recall → high accuracy
            - Large number of difficult instances missed
        - Higher F1 score → better performing model
- **Receiver Operator Characteristic Area Under Curve (ROC AUC)**

    $$\text{Sensitivity | TP Rate} = \frac{TP}{TP + FN} \\ \text{Specificity | TN Rate} = \frac{TN}{TN + FP} \\ ~~~\text{False Positive Rate} = \frac{FP}{TN + FP}$$

    - Most widely used for binary classification
    - AUC is the probability that the classifier will rank a randomly chosen postive example higher than a random chosen negative example
    - FPR, TPR are in the range $[0, 1]$ and are computed at various classification thresolds
    - AUC is the area under the ROC curve that goes through these various (FPR, TPR) points

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f8cfaac3-16d5-4429-aac4-cc659d4c4ee4/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f8cfaac3-16d5-4429-aac4-cc659d4c4ee4/Untitled.png)

- **Mean Absolute Error**

    $$MSE = \frac{1}{N}\sum_{i=1}^{N}|(y_i - \hat{y_i})|$$

    - Average of the difference between the original values and predicted values
        - Measure of how far the predictions are from actual output
    - No indication of direction of error
- **Mean Squared Error**

    $$MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y_i})^2$$

    - Average of the square of the difference between original and predicted values
    - Easier to compute gradients with MSE compared to MAE
        - Effects of larger errors become more pronounced than smaller errors
        - Model can focus on minimising the larger errors
- **BLEU Score**

    $$\text{BLEU} = \text{BP} \cdot \exp \bigg( \sum_{i=1}^{N} w_i \ln p_i\bigg)$$

    $$\exp \bigg( \sum_{i=1}^{N} w_i \log p_i \bigg) = \prod_{i=1}^{N} \exp \big( w_i \log p_i \big) \\
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~= \prod_{i=1}^{N} \Big[ \exp \big( \log p_i \big) \Big]^{w_i} \\
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~= \prod_{i=1}^{N} {p_i}^{w_i} \in [0,1]$$

    - Bilingual Evaluation Understudy → a way to learn and mimic a "senior actor" (the grouth truth)
    - A away of quantitatively evaluating quality of sequence transduction predictions
    - Ranges from $[0, 1]$ where $1$ means exact same sequence while $0$ means no match
    - Checks if each token in grouth truth sequence occurs at least once in the prediction
    - $\text{Precision}$ calculates the average of the sum of the binary presences of the tokens in the prediction
        - $BP$ is the brevity penalty that penalises short machine translations

        ```coffeescript
        Reference: "I am outside of the office"
          Machine: "am am am am am am"

        precision = 1 + 1 + 1 + 1 + 1 + 1 / 6 = 1
        modified precision = 1 + 0 + 0 + 0 + 0 + 0 / 6 = 1/
        ```

    - Modified precision uses the frequency occurence of unique n-grams and adds them together
        - Takes the clipped count → minimum of special frequency count and original count
    - Better variants include BLEU over `n-grams` that calculates BLEU over the tuples

## Weight Sharing

**Advantages**

1. Use the same feature detector across all sections of the input data
2. Reduce the number of weights in a convolutional layer → reduced training time
3. Allows model to learn features that are agnostic to the region of the input being considered since it learns a more general mapping instead of a region-specific one
- **Convolutional Neural Networks**
    - Weight value within filters are learnable during training
        - Filter used on a single 2D plane contains a weight that is shared across all filters used across the same plane

    $$X = \Bigg[\begin{matrix}
    x_{11} & x_{12} & x_{13}\\
    x_{21} & x_{22} & x_{23}\\x_{31} & x_{32} & x_{33}
    \end{matrix}\Bigg] \\ W =\Bigg[\begin{matrix}
    w_{11} & w_{12}\\
    w_{21} & w_{22}
    \end{matrix}\Bigg] \\ W*X = \Bigg[\begin{matrix}
    W\cdot(x_{11}, x_{12}, x_{21}, x_{22}) & W\cdot(x_{12}, x_{13}, x_{22}, x_{23})\\
    W\cdot(x_{21}, x_{22}, x_{31}, x_{32}) & W\cdot(x_{22}, x_{23}, x_{32}, x_{33})
    \end{matrix}\Bigg] \\ \text{where the filter $W$ is used for all regions of the image}$$

    - When the image is transformed or translated, the output of the feature map will be shifted by the same amount → invariant of any affine transformations on input images
    - Parameter sharing occurs when a feature map is generated from the result of the convolution between a filter and the input data from a unit within a plane in the layer
        - All units within this layer plane share the same weights
- **Recurrent Neural Networks**
    - If a new network had to be learned for each time step and feed the output of the first model to the second, it would end up l;ike a regular feed-forward network
    - For sequences, `"A B C D E"` and `"X Y A B C D F"`, `"A B C D"` is the common part that occurs at different time steps
    - By sharing parameters, the model only learns what the part means once instead of having to learn it for every time step it could possibly occur in the model

**Drawbacks of Parameter Sharing**

- Since the same transformation is applied at each timestep/region, the model needs to learn something so general that it can be applied to the whole input
- TensorFlow does not allow recurrence in its computation graph, only feed-forward. An alternative is to create the graphs before training

## k-Fold Cross Validation

**Types of Cross Validation**

1. **K-fold** → $k-1$ folds for training, $1$ for testing repeated $k$ times 
2. **Stratified** → ensure each fold has an equal proportion of observations from a given class
3. **Leave One Out** → leave one instance out of the dataset ($n-1$ for training) repeated $\times n$ 
- Used to estimate the skill of a ML model on unseen data
    - Using a limited sample to estimate general performance on other unseen data
- Less bias is involved and users are less optimistic of the model's skills

## Python Data Types

- `str` → text
- `int`, `float`, `complex` → numeric
- `list`, `tuple`, `range` → sequence
- `dict` → mapping
- `set` → set
- `bool` → boolean

All objects in Python have an ***identity***, ***type***, and ***value***.

- Identity never changes after creation (it's like the object's location in memory)
- Type never changes after creation → dictates operations and values object supports
- Value can change → objects with changeable values are *mutable*, *immutable* otherwise

---

Mutability is determined by an object's type

- **Mutable**: `list`, `dict`, `set`, and user-defined classes
- **Immutable**: `int`, `float`, `decimal`, `bool`, `string`, `tuple`, `range`

If variables have the same `id`, they are pointing to the same object in memory

Mutability of containers only imply the identities of the contained objects

## Improving Model Performance

- Using Dropout
    - Prevents models from overfitting (sometimes called a Regularisation techniques)
    - Shuts off random neurons in a layer to prevent very complex functions from being learned
    - Each neuron has a set probability (hyper-parameter) of being turned off
    - Used only for deep neural networks
- Using more training data
    - Models no longer rely on weak correlations and underlying assumptions
    - Leads to more accurate models
- Handling anomalies
    - Outliers and missing values affect the accuracy of models → leads to biased model
    - For continuous missing values, use mean, mode, median of data to impute (k-NN Imputation)
    - For outliers, deletion, transformation, and binning can be performed
    - Best to perform EDA to see what treatments to perform on dataset
- Early Stopping
    - Saving the final checkpoints and stopping training the model performance has reached its peak → a plateau in loss and accuracy landscapes
    - Libraries like `keras` and `tensorflow` allow for early stopping techinques
- Trying other models
    - Some models may be not be the best for certain problems
    - Worth training other unrelated models on the dataset and checking performance metrics

## Popular Models

- **Long Short Term Memory Network**
    - Have 4 gating networks to control flow of information within the cell
    - The cell state line is a conveyor belt that lets information flow through the whole chain
        - Very easy for information to flow through it unimpeaded → LSTM can control what gets added and taken away from this cell state
    1. **Forget Gate** (`sigmoid` layer) decides what to throw away from the cell state
        - $0$ means remove this completely while $1$ means keep everything
    2. **Input Gate** (`sigmoid` + `tanh` layer) decides what value to update the cell state with
    3. **Update Gate** takes this information and performs the cell state update
    4. **Output Gate** (`sigmoid` + `tanh`) outputs a filtered version of the cell state (hidden state $h$)
    - Minimises impact of vanishing gradient problem that was seen in RNNs while having longer-term dependencies (what they were designed for)
    - Variants include GRUs that combine the forget and input gates into the update network while also merging the cell state and hidden state
- **Convolutional Neural Network (C)**
    - Processes data with grid-like topology (like pixels in space)
    - Creates spatial features from the input data
    - Properties for reducing parameter count
        1. Sparse interactions/connections between layers → no need fully connected network
        2. Parameter Sharing → filter is shared for all regions in the input
        3. Equivariant Representation $f(g(x)) = g(f(x))$ → changes to the image has no effect on the prediction
    - 4 main operations: Convolution, Activation, Pooling, Fully Connected

    ---

    **Convolutional Layer**

    $$(f * g)(t) = \int_{-\infty}^{\infty}f(\tau) \cdot g(t - \tau)d\tau \\ (f * g)(t) = \int_{0}^{t}f(\tau) \cdot g(t - \tau)d\tau$$

    - Image is convolved using filters / kernels → applied on image using sliding window approach
    - Sum of the element-wise product is stored in a convolved matrix
    - Depth of the filter is the same as the depth of the volume/image (ie. # channels in an image)
    - Initial convolutions are for detecting edges while deeper convolutions are for higher-level features

    ---

    **Activation**

    - Non-linear activation functions for learning
    - Leaky ReLU is used over ReLU to avoid dying neurons

    $$\text{ReLU(MaxPool(Conv(M))) = MaxPool(ReLU(Conv(M)))}$$

    ---

    **Pooling Layer**

    - Downsampling method to reduce dimensionality of input feature maps after convolution
    - Has many types: global, average, max
        - Max Pooling works better in practice
    - Reduces chances of overfitting due to fewer parameters being passed down
- **Residual Net (C)**
    - Implements Additive Skip Connections in individual blocks
    - Allows for training of deeper networks compared to deeper pure CNNs
- **YOLO family (OD)**
    - Image is passed only once through the FCNN
    - Splits image into a $m \times m$ grid
    - For each grid, generates $B$ bounding boxes and their class probabilities
    - To clean the bounding boxes to find the final bounding box, it uses **Non-max Suppression** algorithm and **Intersection Over Union** (based on the confidences/class probabilities)
        - Select box with highest probability → gets IOU with all other boxes → remove the box with highest IOU → repeat until one box remains
- **R-CNN (OD)**
    1. Proposes a bunch of bounding boxes and checks if there's object inside one by one
        - Creates region proposals via "selective search" → clusters adjacent pixels by colour, texture, and intensity
    2. Warps the region into a square size and passes the features into a custom implementation of *AlexNet* *2012*
    3. Final layer contains a SVM to see if there's an object inside and what object it is
    4. Runs Regression on the region proposals to generate tighter bounding boxes
- **Fast R-CNN (OD)**
    1. **ROI Pooling** → pools together the overlapping regions from all proposals
        - For the same region, there were many bounding box proposals that overlapped
        - Runs CNN on the image just once
    2. Combines all models from pure R-CNN into just one model
        - Predicts $(x, y, w, b)$ at one shot instead of going through 3 separate networks
- **Faster R-CNN (OD)**
    - Reuses the CNN results for region proposal instead of preforming selective search
    - Has an Attention mechanism over Fast R-CNN in the Region Proposal Network
    - Performs OD in two stages
        1. Get ROIs using **Region Proposal Network** protocol with scores
        2. For each ROI, perform classification on the content inside
- **Mask R-CNN (Seg)**
    - Extends Faster R-CNN for pixel-level image segmentation
        - Bounding boxes are from OD while shaded masks are from SS
    - A FCNN is added on top of the CNN from the Faster R-CNN
        - Applies binary mask that says whether a pixel belongs to an object
    - The stride in **ROI Pooling** is quantized → for a $17 \times 17$, ROI Pooling with stride $2$ results in an output of $7 \times 7$ → only $14 \times 14$ is considered resulting in loss of information (misalignment)
        - Simply put, the regions were misaligned from the original input
    - **ROI Align** is used in Mask R-CNN → stride is not quantized
        - ROI Align is better than ROI Pooling
    - For a given bounding box of size $m \times m$, there are $k$ object classes that could exist inside
        - For each type $k$, a binary mask is created → a loss of $km^2$ is incurred
- **U-Net (Seg)**
    - Sequence of convolution, down-sampling, and up-sampling layers that share information between one another
    - The left half is the contracting path and the right half is the expansive path

    $$\text{Contracting path follows} \\ Conv(Conv(Pooling(Dropout))) \\ \text{Expansive path follow} \\ Conv(Concat(Conv(Conv)))$$

    - Performs localisation per pixel to segment the image
    - Variants include Attentive Convolutions

## Attention and its Variants

- **Vaswani Attention**

    $$\text{Attention(K, Q, V)} = \text{softmax}\Big(\frac{Q \cdot V^T}{\sqrt{d_k}}\Big) \cdot V$$

    - Inspired by information retrieval to map a set of queries to the most appropriate values in the "database" memory
    - Scaled Dot-products minimises vanishing gradients
        - Without scaling, dot products increase in magnitude, pushing the Softmax values to region where the gradients are very small during Backpropagation
    - Multiple heads are used to increase predictive power of the Transformer as each head comes up with its own internal representation of the input data
- **Soft Attention**
    - Popularised for image captioning systems
    - Weights are learned and placed "softly" over multiple patches over the input image
    - Models are differentiable and smooth
    - Expensive when the source input is very large
- **Hard Attention**
    - Only selects one patch of the image to attend to at a time
    - Fewer calculations needed at inference time
    - Model is non-differentiable and requires complex training techniques like variance reduction (a process to increase precision of models)
- **Global Attention**
    - Similar to Soft Attention
    - Makes use of encoder and decoder outputs from the current timestep only
        - Easier to implement with vectorised libraries like `keras`
- **Local Attention**
    - Mix of Hard and Soft Attention
    - Extends Hard Attention by making it differentiable

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b0970d34-2a24-49e4-8972-d5a601d4a745/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b0970d34-2a24-49e4-8972-d5a601d4a745/Untitled.png)

## Handling Class Imbalance

- Getting more data
- Changing performance metrics
- Resampling dataset
- Generating synthetic samples
- Trying different algorithms
- Try penalised models (penalise misclassification)

## Glossary

- **Occlusion**
    - When an object is covered by another object in a scene
    - Important to have occluded samples to prevent model overfitting on features that are highly distinctive
- **Segmentation**
    - Semantic: Process of classifying each pixel to a distinct class
    - Instance: Process of identifying a particular instance of a class object (one step above Semantic Segmentation)
- **Histogram of Oriented Gradients (HOG)**
    - Use of feature descriptors to extract useful parts of an image and discard the rest
    - Calculates horizontal and vertical component of the gradient's magnitude and direction of each pixel and then organises the information into a 9-bin histogram to determine shifts in the data
    - Block normalisation can be used to make the model more optimal and less biased
    - Used in AR/VR
- **Ablation Study**
    - Removing features one by one to see how much they contribute to overall predictions
    - Performed in research papers as a way to study feature dependence and importance
- **Backbone**
    - All object detection models are made of a head, neck, and backbone
    - Backbone is the base classification model that the detection model is based on
- **Localisation**
    - Process of finding the location of an object in an image using $(x, y, w, h)$ coordinates
- **Skip Connections**
    - Routing of data into other layers by "skipping" the subsequent layer
    - Alternative path for gradients to flow during Backpropagation → helps converge faster
    - Allows for training deeper convolutional networks
    - Variants include Highway Networks where the skip connections are passed through parametric gates as done in LSTMs
    - Two types: ***Addition*** and ***Concatenation***
    - **Addition**
    - **Concatenation**
    - **Short Skips**
    - **Long Skips**
- **Vanishing Gradients Problem**
    - During Backpropagation, as gradients are sent back through the layers via partial differentiation, they become increasingly smaller and tend to zero
    - Parameters in earlier layers will barely get updated
    - To combat this, LSTMs and GRUs became the norm over RNNs (also because of the longer-term dependencies in memory)
    - A version of this is the **Exploding Gradients Problem** where gradients get too large during Backpropagation if gradients $> 1$ → causes parameters to $NaN$
- **Dying ReLU Problem**
    - Dead ReLU outputs the same value of zero for any input → usually happens when a large negative bias term is learned
        - No role in discriminating between inputs
    - Network is unlikely to recover once this state reached
        - Function gradient at $0$ is $0$ during Backpropagation → no parameters altered
    - $tanh$ and $sigmoid$ face similar issues as values saturate to zero → though, a small gradient always exists that allows for gradual recovery
- **Quantization**
    - Using lower precision data types (FP16 vs. FP32)
        - Little loss in generality and overall prediction quality
    - Saves storage space when deployed on the cloud or on edge
    - Faster inference speeds as the computations are not that heavy

## Vanilla Backpropagation

A technique to optimise the parameters in the network. We take the loss after prediction and propagate the error backwards through each layer. We make incremental changes to the parameters using Gradient Descent.

1. First we feed-forward

    $$z = W \cdot x + b \\ a = \sigma(z) \\ L = C(y, \hat{y})$$

2. Then we backpropagate to get the gradients $\text{w.r.t}$ to loss $L$ using chain rule

    $$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \times \frac{\partial a}{\partial z} \times \frac{\partial z}{\partial w}$$

3. We update the parameters using the gradients via gradient descent

    $$W := W - \alpha \cdot \frac{\partial{L}}{\partial{W}} \\ b := b - \alpha \cdot \frac{\partial{L}}{\partial{b}}$$

- Variations include **Backpropagation Through Time** (BPTT) for recurrent architectures
- If the parameter update is done after the running through all the samples, then it's pure G**radient Descent**. If parameter update is done after one sample or a subset of all samples, then it's **Minibatch Stochastic Gradient Descent**
    - If dataset is large, GD might take too long
    - SGD converges faster → error function is not minimised as well as in GD

## Regularization

- A technique to reduce bias in a network
- Introduces a penalty heavily weighting features → forces network to create flexible model
- **L1**

    $$L = \sum_{i=1}^{N}(y_i - \hat{y_i}) + \lambda\sum^{p}_{j=1}|\beta_j|$$

    - Also called ***Lasso Regularisation***
        - **L**east **A**bsolute **S**hrinkage and **S**election **O**perator
    - Adds absolute value of magnitude of coefficient as penalty term to loss
        - If $\lambda =0$, it's usual Ordinary Least Squares
        - If $\lambda \neq 0$, coefficients will become zero → model will underfit
- **L2**

    $$L = \sum_{i=1}^{N}(y_i - \hat{y_i}) + \lambda\sum^{p}_{j=1}\beta_j^2$$

    - Also called **Ridge Regularisation**
    - Adds squared value of magnitude of coefficients as penalty term to loss
        - If $\lambda = 0$, it's usual Ordinary Least Squares
        - If $\lambda \neq 0$, it adds extra weight causing model to underfit
- Lasso helps with feature selection by shrinking the less important features to zero (removes some)
- Ridge helps minimise model overfitting by adding extra weights that cause underfitting

---

# References

1. [https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234](https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234)
2. [https://www.kaggle.com/dansbecker/what-is-log-loss](https://www.kaggle.com/dansbecker/what-is-log-loss)
3. [https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
4. [https://towardsdatascience.com/understanding-parameter-sharing-or-weights-replication-within-convolutional-neural-networks-cc26db7b645a](https://towardsdatascience.com/understanding-parameter-sharing-or-weights-replication-within-convolutional-neural-networks-cc26db7b645a)
5. [https://datascience.stackexchange.com/questions/26755/cnn-how-does-backpropagation-with-weight-sharing-work-exactly](https://datascience.stackexchange.com/questions/26755/cnn-how-does-backpropagation-with-weight-sharing-work-exactly)
6. [https://stackoverflow.com/questions/47865034/recurrent-nns-whats-the-point-of-parameter-sharing-doesnt-padding-do-the-tri](https://stackoverflow.com/questions/47865034/recurrent-nns-whats-the-point-of-parameter-sharing-doesnt-padding-do-the-tri)
7. [https://machinelearningmastery.com/k-fold-cross-validation/](https://machinelearningmastery.com/k-fold-cross-validation/)
8. [https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85](https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85)
9. [https://towardsdatascience.com/https-towardsdatascience-com-python-basics-mutable-vs-immutable-objects-829a0cb1530a](https://towardsdatascience.com/https-towardsdatascience-com-python-basics-mutable-vs-immutable-objects-829a0cb1530a)
10. [https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c](https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c)
11. [https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd](https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd)
12. [https://datascience.stackexchange.com/questions/36450/what-is-the-difference-between-gradient-descent-and-stochastic-gradient-descent](https://datascience.stackexchange.com/questions/36450/what-is-the-difference-between-gradient-descent-and-stochastic-gradient-descent)
13. [https://blog.roboflow.com/glossary/](https://blog.roboflow.com/glossary/)
14. [https://datascience.stackexchange.com/questions/52015/what-is-the-difference-between-semantic-segmentation-object-detection-and-insta](https://datascience.stackexchange.com/questions/52015/what-is-the-difference-between-semantic-segmentation-object-detection-and-insta)
15. [https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)
16. [https://www.analyticsvidhya.com/blog/2015/12/improve-machine-learning-results/](https://www.analyticsvidhya.com/blog/2015/12/improve-machine-learning-results/)
17. [https://theaisummer.com/skip-connections/](https://theaisummer.com/skip-connections/)
18. [https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
19. [https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/anova/#ANOVA](https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/anova/#ANOVA)
20. [https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006](https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006)
21. [https://alittlepain833.medium.com/simple-understanding-of-mask-rcnn-134b5b330e95](https://alittlepain833.medium.com/simple-understanding-of-mask-rcnn-134b5b330e95)
22. [https://rishtech.substack.com/p/vit](https://rishtech.substack.com/p/vit)
23. [https://machinelearningmastery.com/global-attention-for-encoder-decoder-recurrent-neural-networks/](https://machinelearningmastery.com/global-attention-for-encoder-decoder-recurrent-neural-networks/)
24. [https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#soft-vs-hard-attention](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#soft-vs-hard-attention)
25. [https://www.youtube.com/watch?v=0vt05rQqk_I](https://www.youtube.com/watch?v=0vt05rQqk_I)
26. [https://www.youtube.com/watch?v=m8pOnJxOcqY](https://www.youtube.com/watch?v=m8pOnJxOcqY)
27. [https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks](https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks)
28. [https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
29. [https://leimao.github.io/blog/BLEU-Score/](https://leimao.github.io/blog/BLEU-Score/)
30. [https://www.aclweb.org/anthology/P02-1040/](https://www.aclweb.org/anthology/P02-1040/)