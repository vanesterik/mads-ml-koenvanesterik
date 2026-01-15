---
layout: post
title: "RNNs Assignment Report"
tags: machine-learning report
---

|               |                                       |
| ------------- | ------------------------------------- |
| **Date**      | 8th of January 2026                   |
| **Student**   | Koen van Esterik                      |
| **Institute** | HAN University of Applied Sciences    |
| **Course**    | Deep Learning & Deployment (2025 P1A) |
| **Subject**   | Recurrent Neural Networks (RNNs)      |

## Introduction

The core of this research is to investigate the extent to which hyperparameter tuning - specifically epochs, hidden layers, depth, and dropout - affects the accuracy of a Gated Recurrent Unit (GRU) model when performing supervised classification on the **SmartWatch Gestures Dataset**.

This research is significant because high-accuracy gesture recognition is a cornerstone of modern wearable technology. By refining these models, we can make wearable devices more accessible and intuitive for a broader range of users, including those with varying motor abilities. While this report is primarily intended for personal skill development in deep learning and neural network optimization, it addresses the fundamental challenge of balancing model complexity with generalization. The guiding hypothesis for this study is that by systematically increasing the capacity and training duration of the model, a classification accuracy exceeding **0.90** can be achieved.

The scope is limited to a standard GRU architecture provided by the PyTorch framework, focusing on acceleration data (3-axis) for 20 distinct gesture classes.

## Methodology

The research follows a quantitative experimental design.

- **Dataset & Sampling**: The SmartWatch Gestures Dataset was used, containing acceleration data from a wrist-worn device. The data was partitioned using an 80/20 train-test split to ensure a robust evaluation of the model’s ability to generalize to unseen data.
- **Optimization Strategy**: A Bayesian Optimization approach was employed via the hyperopt library. Unlike a random or grid search, this strategy uses the results of previous iterations to "smartly" navigate the hyperparameter space, focusing on regions that yield the best performance.
- **Tools & Frameworks**:
  - **PyTorch**: For model construction and training.
  - **Hyperopt**: For the Bayesian search of the hyperparameter space.
  - **MLFlow**: Used for experiment tracking and visualising the relationships between parameters and performance metrics.

## Results

The experiment successfully validated the hypothesis, with the tuned model significantly surpassing the 0.90 accuracy threshold.

- **Peak Performance**: The highest recorded accuracy was 0.9954, achieved with a configuration of 256 hidden units, 2 layers, and a dropout rate of approximately 0.06.
- **Key Drivers of Accuracy**: The most substantial performance "jump" occurred when increasing the training duration to 20 epochs, which moved the default model from roughly 0.41 to over 0.97 accuracy.
- **The Overfitting Trade-off**: While accuracy was near-perfect, the results revealed a 25x increase in overfitting in the highest-performing model. The training loss (0.0012) was significantly lower than the test loss (0.0308).

Below a summary of the key results:

| Model | State         | Epochs | Hidden Size | Num Layers | Dropout | Accuracy |
| ----- | ------------- | ------ | ----------- | ---------- | ------- | -------- |
| GRU   | Default       | 3      | 64          | 1          | 0.1     | 0.4097   |
| GRU   | Default       | 10     | 64          | 1          | 0.1     | 0.8925   |
| GRU   | Default       | 20     | 64          | 1          | 0.1     | 0.9742   |
| GRU   | Tuned (Final) | 20     | 256         | 2          | 0.06    | 0.9955   |

## Discussion

The findings demonstrate that while increasing model complexity (hidden units and layers) and training time (epochs) directly correlates with higher training accuracy, it also creates a high risk of the model "memorizing" the training data. This is evident in the 25-fold difference between training and testing loss.

In the context of wearable technology, these results have practical implications: a model with 0.99 accuracy provides a seamless user experience where the device correctly interprets "imperfect" or slightly deviated gestures. This increases accessibility, as the user does not need to perform a gesture with absolute precision to trigger a command.

However, the results also highlight a theoretical ceiling: there is a "sweet spot" where complexity provides enough capacity to learn the gestures, yet regularization (like dropout) must be high enough to prevent the model from becoming too rigid. Future research should focus on this delicate balance, in order to explore adjusted GRU architecture or other models could maintain this 0.99 accuracy while reducing the observed overfitting.

## Conclusion

This research concludes that hypertuning a GRU model using a Bayesian approach is highly effective for gesture classification, resulting in a near-perfect accuracy of **99.5%**. The initial hypothesis was confirmed, as the model easily cleared the 0.90 benchmark.

For practitioners, the main takeaway is that **Epoch count** is a primary lever for performance in this specific dataset, but **Dropout** must be carefully managed to ensure the model remains useful in real-world, "unseen" scenarios.
