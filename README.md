# Machine Learning By Prof. Andrew Ng :star2::star2::star2::star2::star:

This page continas all my coursera machine learning courses and resources :book: by [Prof. Andrew Ng](http://www.andrewng.org/) :man:

# Table of Contents
1. [Breif Intro](#breif-intro)
2. [Video lectures Index](#video-lectures-index)
3. [Programming Exercise Tutorials](#programming-exercise-tutorials)
4. [Programming Exercise Test Cases](#programming-exercise-test-cases)
5. [Useful Resources](#useful-resources)
6. [Schedule](#schedule)
7. [Extra Information](#extra-information)
8. [Online E-Books](#online-e-books)
9. [Aditional Information](#aditional-information)

## Breif Intro

The most of the course talking about **hypothesis function** and minimising **cost funtions**

### Hypothesis
A hypothesis is a certain function that we believe (or hope) is similar to the true function, the target function that we want to model. In context of email spam classification, it would be the rule we came up with that allows us to separate spam from non-spam emails.

### Cost Function
The cost function or **Sum of Squeared Errors(SSE)** is a measure of how far away our hypothesis is from the optimal hypothesis. The closer our hypothesis matches the training examples, the smaller the value of the cost function. Theoretically, we would like J(θ)=0

### Gradient Descent
Gradient descent is an iterative minimization method. The gradient of the error function always shows in the direction of the steepest ascent of the error function. Thus, we can start with a random weight vector and subsequently follow the
negative gradient (using a learning rate alpha)

#### Differnce between cost function and gradient descent functions
<table>
    <th> Cost Function </th>
    <th> Gradient Descent </th>
    <tr VALIGN=TOP>
    <td> <pre> <code>
    function J = computeCostMulti(X, y, theta)
        m = length(y); % number of training examples
        J = 0;
        predictions =  X*theta;
        sqerrors = (predictions - y).^2;
        J = 1/(2*m)* sum(sqerrors);
    end
    </code> </pre> </td>
    <td> <pre> <code>
    function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)    
        m = length(y); % number of training examples
        J_history = zeros(num_iters, 1);
        for iter = 1:num_iters
            predictions =  X * theta;
            updates = X' * (predictions - y);
            <b>theta = theta - alpha * (1/m) * updates;
            J_history(iter) = computeCostMulti(X, y, theta);</b>
        end
    end
    </code> </pre> </td>
    </tr>
</table>

### Bias and Variance
When we discuss prediction models, prediction errors can be decomposed into two main subcomponents we care about: error due to "bias" and error due to "variance". There is a tradeoff between a model's ability to minimize bias and variance. Understanding these two types of error can help us diagnose model results and avoid the mistake of over- or under-fitting.

Source: http://scott.fortmann-roe.com/docs/BiasVariance.html

### Hypotheis and Cost Function Table

| Algorithem 	| Hypothesis Function 	| Cost Function 	| Gradient Descent 	|
|--------------------------------------------	|-----------------------------------------------------------------------	|-------------------------------------------------------------------------------	|---------------------------------------------------------------------------------------	|
| Linear Regression 	| ![linear_regression_hypothesis](/extra/img/linear_hypothesis.gif) 	| ![linear_regression_cost](/extra/img/linear_cost.gif) 	|  	|
| Linear Regression with Multiple variables 	| ![linear_regression_hypothesis](/extra/img/linear_hypothesis.gif) 	| ![linear_regression_cost](/extra/img/linear_cost.gif) 	| ![linear_regression_multi_var_gradient](/extra/img/linear_multi_var_gradient.gif) 	|
| Logistic Regression 	| ![logistic_regression_hypothesis](/extra/img/logistic_hypothesis.gif) 	| ![logistic_regression_cost](/extra/img/logistic_cost.gif) 	| ![logistic_regression_gradient](/extra/img/logistic_gradient.gif) 	|
| Logistic Regression with Multiple Variable 	|  	| ![logistic_regression_multi_var_cost](/extra/img/logistic_multi_var_cost.gif) 	| ![logistic_regression_multi_var_gradient](/extra/img/logistic_multi_var_gradient.gif) 	|
| Nural Networks 	|  	| ![nural_cost](/extra/img/nural_cost.gif) 	|  	|                                                                                      |

### Regression with Pictures
[Linear Regression](http://adit.io/posts/2016-02-20-Linear-Regression-in-Pictures.html)
[Logistic Regression](http://adit.io/posts/2016-03-13-Logistic-Regression.html#non-linear-classification)

## Video lectures Index
[https://class.coursera.org/ml/lecture/preview](https://class.coursera.org/ml/lecture/preview)

## Programming Exercise Tutorials
[https://www.coursera.org/learn/machine-learning/discussions/all/threads/m0ZdvjSrEeWddiIAC9pDDA](https://www.coursera.org/learn/machine-learning/discussions/all/threads/m0ZdvjSrEeWddiIAC9pDDA)

## Programming Exercise Test Cases
[https://www.coursera.org/learn/machine-learning/discussions/all/threads/0SxufTSrEeWPACIACw4G5w](https://www.coursera.org/learn/machine-learning/discussions/all/threads/0SxufTSrEeWPACIACw4G5w)

## Useful Resources
[https://www.coursera.org/learn/machine-learning/resources/NrY2G](https://www.coursera.org/learn/machine-learning/resources/NrY2G)

## Schedule:
### Week 1 - Due 07/16/17:
- Welcome - [pdf](/home/week-1/lectures/pdf/Lecture1.pdf) - [ppt](/home//week-1/lectures/ppt/Lecture1.pptx)
- Linear regression with one variable - [pdf](/home/week-1/lectures/pdf/Lecture2.pdf) - [ppt](/home/week-1/lectures/ppt/Lecture2.pptx)
- Linear Algebra review (Optional) - [pdf](/home/week-1/lectures/pdf/Lecture3.pdf) - [ppt](/home/week-1/lectures/ppt/Lecture3.pptx)
- [Lecture Notes](/home/week-1/lectures/notes.pdf)
- [Errata](/home/week-1/errata.pdf)

### Week 2 - Due 07/23/17:
- Linear regression with multiple variables - [pdf](/home/week-2/lectures/pdf/Lecture4.pdf) - [ppt](/home/week-2/lectures/ppt/Lecture4.pptx)
- Octave tutorial [pdf](/home/week-2/lectures/pdf/Lecture5.pdf)
- Programming Exercise 1: Linear Regression - [pdf](/home/week-2/exercises/machine-learning-ex1/ex1.pdf) - [Problem](/home/week-2/exercises/machine-learning-ex1.zip) - [Solution](/home/week-2/exercises/machine-learning-ex1/ex1/)
- [Lecture Notes](/home/week-2/lectures/notes.pdf)
- [Errata](/home/week-2/errata.pdf)
- [Program Exercise Notes](/home/week-2/exercises/Programming%20Ex.1.pdf)

### Week 3 - Due 07/30/17:
- Logistic regression - [pdf](/home/week-2/lectures/pdf/Lecture4.pdf) - [ppt](/home/week-2/lectures/ppt/Lecture4.pptx)
- Regularization - [pdf](/home/week-2/lectures/pdf/Lecture4.pdf) - [ppt](/home/week-2/lectures/ppt/Lecture4.pptx)
- Programming Exercise 2: Logistic Regression - [pdf](/home/week-3/exercises/machine-learning-ex2/ex2.pdf) - [Problem](home/week-3/exercises/machine-learning-ex2.zip) - [Solution](/home/week-3/exercises/machine-learning-ex2/ex2)
- [Lecture Notes](/home/week-3/lectures/notes.pdf)
- [Errata](/home/week-3/errata.pdf)
- [Program Exercise Notes](/home/week-3/exercises/Programming%20Ex.2.pdf)

### Week 4 - Due 08/06/17:
- Neural Networks: Representation - [pdf](/home/week-4/lectures/pdf/Lecture8.pdf) - [ppt](/home/week-4/lectures/ppt/Lecture8.pptx)
- Programming Exercise 3: Multi-class Classification and Neural Networks - [pdf](/home/week-4/exercises/machine-learning-ex3/ex3.pdf) - [Problem](/home/week-4/exercises/machine-learning-ex3.zip) - [Solution](/home/week-4/exercises/machine-learning-ex3/ex3)
- [Lecture Notes](/home/week-4/lectures/notes.pdf)
- [Errata](/home/week-4/errata.pdf)
- [Program Exercise Notes](/home/week-4/exercises/Programming%20Ex.3.pdf)

### Week 5 - Due 08/13/17:
- Neural Networks: Learning - [pdf](/home/week-5/lectures/pdf/Lecture9.pdf) - [ppt](/home/week-5/lectures/ppt/Lecture9.pptx)
- Programming Exercise 4: Neural Networks Learning - [pdf](/home/week-5/exercises/machine-learning-ex4/ex4.pdf) - [Problem](/home/week-5/exercises/machine-learning-ex4.zip) - [Solution](/home/week-5/exercises/machine-learning-ex4/ex4)
- [Lecture Notes](/home/week-5/lectures/notes.pdf)
- [Errata](/home/week-5/errata.pdf)
- [Program Exercise Notes](/home/week-4/exercises/Programming%20Ex.4.pdf)

### Week 6 - Due 08/20/17:
- Advice for applying machine learning - [pdf](/home/week-6/lectures/pdf/Lecture10.pdf) - [ppt](/home/week-6/lectures/ppt/Lecture10.pptx)
- Machine learning system design - [pdf](/home/week-6/lectures/pdf/Lecture11.pdf) - [ppt](/home/week-6/lectures/ppt/Lecture11.pptx)
- Programming Exercise 5: Regularized Linear Regression and Bias v.s. Variance - [pdf](/home/week-6/exercises/machine-learning-ex5/ex5.pdf) - [Problem](/home/week-6/exercises/machine-learning-ex5.zip) - [Solution](/home/week-6/exercises/machine-learning-ex5/ex5)
- [Lecture Notes](/home/week-6/lectures/notes.pdf)
- [Errata](/home/week-6/errata.pdf)
- [Program Exercise Notes](/home/week-6/exercises/Programming%20Ex.5.pdf)

### Week 7 - Due 08/27/17:
- Support vector machines - [pdf](/home/week-7/lectures/pdf/Lecture12.pdf) - [ppt](/home/week-7/lectures/ppt/Lecture12.pptx)
- Programming Exercise 6: Support Vector Machines - [pdf](/home/week-7/exercises/machine-learning-ex6/ex6.pdf) - [Problem](/home/week-7/exercises/machine-learning-ex6.zip) - [Solution](/home/week-7/exercises/machine-learning-ex6/ex6)
- [Lecture Notes](/home/week-7/lectures/notes.pdf)
- [Errata](/home/week-7/errata.pdf)
- [Program Exercise Notes](/home/week-7/exercises/Programming%20Ex.6.pdf)

### Week 8 - Due 09/03/17:
- Clustering - [pdf](/home/week-8/lectures/pdf/Lecture13.pdf) - [ppt](/home/week-8/lectures/ppt/Lecture13.ppt)
- Dimensionality reduction - [pdf](/home/week-8/lectures/pdf/Lecture14.pdf) - [ppt](/home/week-8/lectures/ppt/Lecture14.ppt)
- Programming Exercise 7: K-means Clustering and Principal Component Analysis - [pdf](/home/week-8/exercises/machine-learning-ex7/ex7.pdf) - [Problems](/home/week-8/exercises/machine-learning-ex7.zip) - [Solution](/home/week-8/exercises/machine-learning-ex7/ex7)
- [Lecture Notes](/home/week-8/lectures/notes.pdf)
- [Errata](/home/week-8/errata.pdf)
- [Program Exercise Notes](/home/week-8/exercises/Programming%20Ex.7.pdf)

### Week 9 - Due 09/10/17:
- Anomaly Detection - [pdf](/home/week-9/lectures/pdf/Lecture15.pdf) - [ppt](/home/week-9/lectures/ppt/Lecture15.ppt)
- Recommender Systems  - [pdf](/home/week-9/lectures/pdf/Lecture16.pdf) - [ppt](/home/week-9/lectures/ppt/Lecture16.ppt)
- Programming Exercise 8: Anomaly Detection and Recommender Systems - [pdf](/home/week-9/exercises/machine-learning-ex8/ex8.pdf) - [Problems](/home/week-9/exercises/machine-learning-ex8.zip) - [Solution](/home/week-9/exercises/machine-learning-ex8/ex8)
- [Lecture Notes](/home/week-9/lectures/notes.pdf)
- [Errata](/home/week-9/errata.pdf)
- [Program Exercise Notes](/home/week-9/exercises/Programming%20Ex.8.pdf)

### Week 10 - Due 09/17/17:
- Large scale machine learning - [pdf](/home/week-10/lectures/pdf/Lecture17.pdf) - [ppt](/home/week-11/lectures/ppt/Lecture17.ppt)
- [Lecture Notes](/home/week-10/lectures/notes.pdf)

### Week 11 - Due 09/24/17:
- Application example: Photo OCR - [pdf](/home/week-11/lectures/pdf/Lecture18.pdf) - [ppt](/home/week-11/lectures/ppt/Lecture18.ppt)

## Extra Information

- [Linear Algebra Review and Reference Zico Kolter](/extra/cs229-linalg.pdf)
- [CS229 Lecture notes](/extra/cs229-notes1.pdf)
- [CS229 Problems](/extra/cs229-prob.pdf)
- [Financial time series forecasting with machine learning techniques](/extra/machine%20learning%20stocks.pdf)
- [Octave Examples](/extra/octave_session.m)

## Online E Books

- [Introduction to Machine Learning by Nils J. Nilsson](robotics.stanford.edu/~nilsson/MLBOOK.pdf)
- [Introduction to Machine Learning by Alex Smola and S.V.N. Vishwanathan](http://alex.smola.org/drafts/thebook.pdf)
- [Introduction to Data Science by Jeffrey Stanton](http://surface.syr.edu/cgi/viewcontent.cgi?article=1165&context=istpub)
- [Bayesian Reasoning and Machine Learning by David Barber](http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.Online)
- [Understanding Machine Learning, © 2014 by Shai Shalev-Shwartz and Shai Ben-David](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/copy.html)
- [Elements of Statistical Learning, by Hastie, Tibshirani, and Friedman](http://statweb.stanford.edu/~tibs/ElemStatLearn/)
- [Pattern Recognition and Machine Learning, by Christopher M. Bishop](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)


## Aditional Information

## :boom: Course Status :point_down:
![coursera_course_completion](/extra/img/coursera_course_completion.png)

### Links
- [What are the top 10 problems in deep learning for 2017?](https://www.quora.com/What-are-the-top-10-problems-in-deep-learning-for-2017)
- [When will the deep learning bubble burst?](https://www.quora.com/When-will-the-deep-learning-bubble-burst)

### Statistics Models

- HMM - [Hidden Markov Model](https://en.wikipedia.org/wiki/Hidden_Markov_model)
- CRFs - [Conditional Random Fields](https://en.wikipedia.org/wiki/Conditional_random_field)
- LSI - [Latent Semantic Indexing](https://www.searchenginejournal.com/what-is-latent-semantic-indexing-seo-defined/21642/)
- MRF - [Markov Random Fields](https://en.wikipedia.org/wiki/Markov_random_field)

### NLP forums

- SIGIR - [Special Interest Group on Information Retrieval](http://sigir.org/)
- ACL - [Association for Computational Linguistics](https://www.aclweb.org/portal/)
- NAACL - [The North American Chapter of the Association for Computational Linguistics](http://naacl.org/)
- EMNLP - [Empirical Methods in Natural Language Processing](http://emnlp2017.net/)
- NIPS - [Neural Information Processing Systems](https://nips.cc/)