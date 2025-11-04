---
layout: post
title: "CNNs Assignment Report"
tags: machine-learning report
---

|               |                                       |
| ------------- | ------------------------------------- |
| **Date**      | 12th of October 2025                  |
| **Student**   | Koen van Esterik                      |
| **Institute** | HAN University of Applied Sciences    |
| **Course**    | Deep Learning & Deployment (2025 P1A) |
| **Subject**   | Convolutional Neural Networks         |

## Introduction

For the lectures of Deep Learning and Deployment, we are tasked with exploring the intricacies of **convolutional neural networks**, their architectures, and their applications. This report aims to summarize the findings from the experiments conducted during the assignment.

You can find the the code for this assignment in the [notebook]({{ "/notebooks/neural-networks/notebook.ipynb" | relative_url }}). A detailed log of the experiments, observations, and conclusions drawn during the assignment is included as appendix to this report.

All experiments are done on a convolutional neural network architecture, which was inspired by lecture material and is based on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. The task of the convolutional neural network is to classify images of clothing into one of ten categories. The dataset consists of 60,000 training images and 10,000 test images, each being a 28x28 grayscale image.

The dataset is balanced, meaning that each class has an equal number of images. Therefore we are using accuracy as the primary metric to evaluate the performance of the neural network.

The design of the convolutional neural network architecture is adjustable based on hyperparameters such as the number of convolutional layers, the number of filters per layer, kernel sizes, etc. This allows for experimentation with different architectures to find the optimal configuration for the task at hand.

Multiple experiments were conducted, each focusing on different aspects of convolutional neural networks, such as hyperparameter tuning, architecture variations, etc. This report will detail the methodology, results, and conclusions drawn from these experiments.

## Method

The instructions for this assignment are as follows:

1. Define a hypothesis related to hyper tuning of the convolutional neural network.
2. Design an experiment based on the hypothesis.
3. Run the experiment.
4. Analyze the results and draw conclusions.
5. Repeat steps 1-4 to explore more aspects of convolutional neural networks.

## Results

Various experiments were conducted to explore the effects of different hyperparameters and architectural choices on the performance of the convolutional neural network. The results of these experiments are summarized in the journal entries included in the appendix.

Generally speaking, the following observations were made:

- Adding dropout layers did not improve the model's performance and often led to decreased accuracy.
- Incorporating batch normalization layers significantly improved the model's accuracy and training stability.
- Increasing the number of convolutional layers without batch normalization led to overfitting and reduced accuracy.
- Increasing the number of convolutional layers with batch normalization improved performance, but there was a point of diminishing returns.
- The optimal architecture found during the experiments consisted of 3 convolutional layers with batch normalization, achieving an accuracy of approximately 85.5% on the test set.

## Discussion

The experiments conducted during this assignment provided valuable insights into the design and optimization of convolutional neural networks. The findings highlight the importance of architectural choices and hyperparameter tuning in achieving optimal performance.

The experiments were well designed and executed, allowing for a systematic exploration of different configurations. However, there are always opportunities for further improvement and exploration. Future work could involve exploring additional hyperparameters, such as learning rate schedules.

Also I kept bumping into the situation of spatial dimensions becoming too small after multiple max-pooling layers. This required careful consideration of the number of max-pooling layers used in conjunction with the number of convolutional layers. This could be further investigated to find optimal strategies for maintaining sufficient spatial dimensions while still benefiting from the advantages of deeper architectures.

## Conclusion

In conclusion, this assignment provided a comprehensive exploration of convolutional neural networks and their optimization. The experiments conducted demonstrated the impact of various architectural choices and hyperparameters on model performance. The findings underscore the importance of batch normalization in improving training stability and accuracy, while also highlighting the potential pitfalls of overfitting with deeper architectures.

## Appendix

### Journal Entries

#### 21/10/2025, 15:58:04 Determine baseline to measure accuracy

**Objective:**
Establish a baseline accuracy for the convolutional neural network on the Fashion-MNIST dataset.

**Methodology:**
Train CNN architecture with following specifications:

```text
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
CNN                                      --
├─Sequential: 1-1                        --
│    └─Conv2d: 2-1                       1,280
│    └─ReLU: 2-2                         --
│    └─MaxPool2d: 2-3                    --
│    └─Conv2d: 2-4                       147,584
│    └─ReLU: 2-5                         --
│    └─MaxPool2d: 2-6                    --
│    └─Conv2d: 2-7                       147,584
│    └─ReLU: 2-8                         --
│    └─MaxPool2d: 2-9                    --
├─Sequential: 1-2                        --
│    └─Flatten: 2-10                     --
│    └─Linear: 2-11                      16,512
│    └─ReLU: 2-12                        --
│    └─Linear: 2-13                      1,290
=================================================================
Total params: 314,250
Trainable params: 314,250
Non-trainable params: 0
=================================================================
```

**Results:**

| Metric          | Value              |
| --------------- | ------------------ |
| Loss/train      | 0.5591158866882324 |
| Loss/test       | 0.5238780838251114 |
| Metric/Accuracy | 0.8053125          |
| Learning_rate   | 0.001              |

#### 21/10/2025, 16:17:56 Effect of adding dropout layer before output layer

**Objective:**
Investigate the impact of adding a dropout layer before the output layer on the model's performance.

**Hypothesis:**
Adding a dropout layer will help prevent overfitting and improve the model's generalization, leading to higher accuracy on the test set.

**Methodology:**
Modify the CNN architecture to include a dropout layer with a dropout rate of 0.5 before the output layer.

**Results:**

| Metric          | Value              |
| --------------- | ------------------ |
| Loss/train      | 0.6401745080947876 |
| Loss/test       | 0.5426638874411583 |
| Metric/Accuracy | 0.78796875         |
| Learning_rate   | 0.001              |

**Evaluation:**
The results do not support the hypothesis. Adding the dropout layer resulted in a decrease in accuracy on the test set compared to the baseline model without dropout.

***Conclusion:**
The dropout layer may have hindered the model's ability to learn effectively from the training data.

#### 21/10/2025, 17:58:10 Effect of adding dropout layer after last convolutional layer

**Objective:**
Investigate the impact of adding a dropout layer after the last convolutional layer on the model's performance.

**Hypothesis:**
Adding a dropout layer after the last convolutional layer will help prevent overfitting and improve the model's generalization, leading to higher accuracy on the test set.

**Methodology:**
Modify the CNN architecture to include a dropout layer with a dropout rate of 0.5 after the last convolutional layer.

**Results:**

| Metric          | Value              |
| --------------- | ------------------ |
| Loss/train      | 0.6530442237854004 |
| Loss/test       | 0.5801776033639908 |
| Metric/Accuracy | 0.7796875          |
| Learning_rate   | 0.001              |

**Evaluation:**
The results do not support the hypothesis. Adding the dropout layer resulted in a decrease in accuracy on the test set compared to the baseline model without dropout.

**Conclusion:**
The dropout layer may have hindered the model's ability to learn effectively from the training data.

#### 21/10/2025, 18:02:35 Effect of decreasing dropout rate to 0.3 after last convolutional layer

**Objective:**
Investigate the impact of decreasing the dropout rate to 0.3 after the last convolutional layer on the model's performance.

**Hypothesis:**
Decreasing the dropout rate will allow more information to flow through the network, potentially improving accuracy on the test set.

**Methodology:**
Modify the CNN architecture to include a dropout layer with a dropout rate of 0.3 after the last convolutional layer.

**Results:**

| Metric          | Value              |
| --------------- | ------------------ |
| Loss/train      | 0.6161350607872009 |
| Loss/test       | 0.5603274443745613 |
| Metric/Accuracy | 0.783125           |
| Learning_rate   | 0.001              |

**Evaluation:**
The results support the hypothesis. Decreasing the dropout rate improved the model's accuracy on the test set compared to the previous configuration with a dropout rate of 0.5.

**Conclusion:**
Lowering the dropout rate allowed the model to retain more information, leading to better performance on the test set. But the accuracy is still lower than the baseline model without dropout.

#### 21/10/2025, 18:05:15 Effect of adding batch normalization after each convolutional layer

**Objective:**
Investigate the impact of adding batch normalization after each convolutional layer on the model's performance.

**Hypothesis:**
Adding batch normalization will stabilize and accelerate training, leading to higher accuracy on the test set.

**Methodology:**
Modify the CNN architecture to include batch normalization layers after each convolutional layer.

```text
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
CNN                                      --
├─ModuleList: 1-1                        --
│    └─Conv2d: 2-1                       1,280
│    └─BatchNorm2d: 2-2                  256
│    └─ReLU: 2-3                         --
│    └─MaxPool2d: 2-4                    --
│    └─Conv2d: 2-5                       147,584
│    └─BatchNorm2d: 2-6                  256
│    └─ReLU: 2-7                         --
│    └─MaxPool2d: 2-8                    --
│    └─Conv2d: 2-9                       147,584
│    └─BatchNorm2d: 2-10                 256
│    └─ReLU: 2-11                        --
│    └─MaxPool2d: 2-12                   --
├─Sequential: 1-2                        --
│    └─Flatten: 2-13                     --
│    └─Linear: 2-14                      16,512
│    └─ReLU: 2-15                        --
│    └─Dropout: 2-16                     --
│    └─Linear: 2-17                      1,290
=================================================================
Total params: 315,018
Trainable params: 315,018
Non-trainable params: 0
=================================================================
```

**Results:**

| Metric          | Value               |
| --------------- | ------------------- |
| Loss/train      | 0.4102526903152466  |
| Loss/test       | 0.40167241960763933 |
| Metric/Accuracy | 0.855625            |
| Learning_rate   | 0.001               |

**Evaluation:**
The results support the hypothesis. Adding batch normalization improved the model's accuracy on the test set compared to the baseline model without batch normalization.

**Conclusion:**
Incorporating batch normalization layers enhanced the model's performance, likely due to improved training stability and convergence.

#### 21/10/2025, 18:58:02 Effect of increasing number convolutional layers to 8 without batch normalization

**Objective:**
Investigate the impact of increasing the number of convolutional layers to 8 without batch normalization on the model's performance.

**Hypothesis:**
Increasing the number of convolutional layers will allow the model to learn more complex features, leading to higher accuracy on the test set.

**Methodology:**
Modify the CNN architecture to include 8 convolutional layers without batch normalization.

```text
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
CNN                                      --
├─ModuleList: 1-1                        --
│    └─Conv2d: 2-1                       1,280
│    └─Identity: 2-2                     --
│    └─ReLU: 2-3                         --
│    └─MaxPool2d: 2-4                    --
│    └─Conv2d: 2-5                       147,584
│    └─Identity: 2-6                     --
│    └─ReLU: 2-7                         --
│    └─Conv2d: 2-8                       147,584
│    └─Identity: 2-9                     --
│    └─ReLU: 2-10                        --
│    └─Conv2d: 2-11                      147,584
│    └─Identity: 2-12                    --
│    └─ReLU: 2-13                        --
│    └─MaxPool2d: 2-14                   --
│    └─Conv2d: 2-15                      147,584
│    └─Identity: 2-16                    --
│    └─ReLU: 2-17                        --
│    └─Conv2d: 2-18                      147,584
│    └─Identity: 2-19                    --
│    └─ReLU: 2-20                        --
│    └─Conv2d: 2-21                      147,584
│    └─Identity: 2-22                    --
│    └─ReLU: 2-23                        --
│    └─MaxPool2d: 2-24                   --
│    └─Conv2d: 2-25                      147,584
│    └─Identity: 2-26                    --
│    └─ReLU: 2-27                        --
├─Sequential: 1-2                        --
│    └─Flatten: 2-28                     --
│    └─Linear: 2-29                      16,512
│    └─ReLU: 2-30                        --
│    └─Dropout: 2-31                     --
│    └─Linear: 2-32                      1,290
=================================================================
Total params: 1,052,170
Trainable params: 1,052,170
Non-trainable params: 0
=================================================================
```

This required to increase the number of max-pooling per convolutional layer to 3 to maintain sufficient spatial dimensions before the fully connected layers.

**Results:**

| Metric          | Value              |
| --------------- | ------------------ |
| Loss/train      | 0.7681955695152283 |
| Loss/test       | 0.7850289395451546 |
| Metric/Accuracy | 0.71015625         |
| Learning_rate   | 0.001              |

**Evaluation:**
The results do not support the hypothesis. Increasing the number of convolutional layers resulted in a decrease in accuracy on the test set compared to the baseline model with fewer layers.

**Conclusion:**
The increased complexity of the model may have led to overfitting or difficulties in training effectively without batch normalization.

#### 04/11/2025, 11:40:02 Effect of increasing number convolutional layers to 8 with batch normalization

**Objective:**
Investigate the impact of increasing the number of convolutional layers to 8 with batch normalization on the model's performance.

**Hypothesis:**
Increasing the number of convolutional layers with batch normalization will allow the model to learn more complex features, leading to higher accuracy on the test set.

**Methodology:**
Modify the CNN architecture to include 8 convolutional layers with batch normalization.

```text
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
CNN                                      --
├─ModuleList: 1-1                        --
│    └─Conv2d: 2-1                       1,280
│    └─Identity: 2-2                     --
│    └─ReLU: 2-3                         --
│    └─MaxPool2d: 2-4                    --
│    └─Conv2d: 2-5                       147,584
│    └─Identity: 2-6                     --
│    └─ReLU: 2-7                         --
│    └─Conv2d: 2-8                       147,584
│    └─Identity: 2-9                     --
│    └─ReLU: 2-10                        --
│    └─Conv2d: 2-11                      147,584
│    └─Identity: 2-12                    --
│    └─ReLU: 2-13                        --
│    └─MaxPool2d: 2-14                   --
│    └─Conv2d: 2-15                      147,584
│    └─Identity: 2-16                    --
│    └─ReLU: 2-17                        --
│    └─Conv2d: 2-18                      147,584
│    └─Identity: 2-19                    --
│    └─ReLU: 2-20                        --
│    └─Conv2d: 2-21                      147,584
│    └─Identity: 2-22                    --
│    └─ReLU: 2-23                        --
│    └─MaxPool2d: 2-24                   --
│    └─Conv2d: 2-25                      147,584
│    └─Identity: 2-26                    --
│    └─ReLU: 2-27                        --
├─Sequential: 1-2                        --
│    └─Flatten: 2-28                     --
│    └─Linear: 2-29                      16,512
│    └─ReLU: 2-30                        --
│    └─Dropout: 2-31                     --
│    └─Linear: 2-32                      1,290
=================================================================
Total params: 1,052,170
Trainable params: 1,052,170
Non-trainable params: 0
=================================================================
```

**Results:**

| Metric          | Value               |
| --------------- | ------------------- |
| Loss/train      | 0.409567654132843   |
| Loss/test       | 0.43624164432287216 |
| Metric/Accuracy | 0.8453125           |
| Learning_rate   | 0.001               |

**Evaluation:**
The results support the hypothesis. Increasing the number of convolutional layers with batch normalization improved the model's ability to learn complex features, as evidenced by the higher accuracy on the test set compared to the previous configuration without batch normalization.

**Conclusion:**
Incorporating batch normalization in a deeper architecture enhanced the model's performance, likely due to improved training stability and convergence.

#### 04/11/2025, 11:40:02 Effect of increasing number convolutional layers to 16 with batch normalization

**Objective:**
Investigate the impact of increasing the number of convolutional layers to 16 with batch normalization on the model's performance.

**Hypothesis:**
Increasing the number of convolutional layers with batch normalization will allow the model to learn even more complex features, leading to higher accuracy on the test set.

**Methodology:**
Modify the CNN architecture to include 16 convolutional layers with batch normalization.

```text
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
CNN                                      --
├─ModuleList: 1-1                        --
│    └─Conv2d: 2-1                       1,280
│    └─BatchNorm2d: 2-2                  256
│    └─ReLU: 2-3                         --
│    └─MaxPool2d: 2-4                    --
│    └─Conv2d: 2-5                       147,584
│    └─BatchNorm2d: 2-6                  256
│    └─ReLU: 2-7                         --
│    └─Conv2d: 2-8                       147,584
│    └─BatchNorm2d: 2-9                  256
│    └─ReLU: 2-10                        --
│    └─Conv2d: 2-11                      147,584
│    └─BatchNorm2d: 2-12                 256
│    └─ReLU: 2-13                        --
│    └─Conv2d: 2-14                      147,584
│    └─BatchNorm2d: 2-15                 256
│    └─ReLU: 2-16                        --
│    └─Conv2d: 2-17                      147,584
│    └─BatchNorm2d: 2-18                 256
│    └─ReLU: 2-19                        --
│    └─Conv2d: 2-20                      147,584
│    └─BatchNorm2d: 2-21                 256
│    └─ReLU: 2-22                        --
│    └─MaxPool2d: 2-23                   --
│    └─Conv2d: 2-24                      147,584
│    └─BatchNorm2d: 2-25                 256
│    └─ReLU: 2-26                        --
│    └─Conv2d: 2-27                      147,584
│    └─BatchNorm2d: 2-28                 256
│    └─ReLU: 2-29                        --
│    └─Conv2d: 2-30                      147,584
│    └─BatchNorm2d: 2-31                 256
│    └─ReLU: 2-32                        --
│    └─Conv2d: 2-33                      147,584
│    └─BatchNorm2d: 2-34                 256
│    └─ReLU: 2-35                        --
│    └─Conv2d: 2-36                      147,584
│    └─BatchNorm2d: 2-37                 256
│    └─ReLU: 2-38                        --
│    └─Conv2d: 2-39                      147,584
│    └─BatchNorm2d: 2-40                 256
│    └─ReLU: 2-41                        --
│    └─MaxPool2d: 2-42                   --
│    └─Conv2d: 2-43                      147,584
│    └─BatchNorm2d: 2-44                 256
│    └─ReLU: 2-45                        --
│    └─Conv2d: 2-46                      147,584
│    └─BatchNorm2d: 2-47                 256
│    └─ReLU: 2-48                        --
│    └─Conv2d: 2-49                      147,584
│    └─BatchNorm2d: 2-50                 256
│    └─ReLU: 2-51                        --
├─Sequential: 1-2                        --
│    └─Flatten: 2-52                     --
│    └─Linear: 2-53                      16,512
│    └─ReLU: 2-54                        --
│    └─Dropout: 2-55                     --
│    └─Linear: 2-56                      1,290
=================================================================
Total params: 2,236,938
Trainable params: 2,236,938
Non-trainable params: 0
=================================================================
```

**Results:**

| Metric          | Value               |
| --------------- | ------------------- |
| Loss/train      | 0.5733796954154968  |
| Loss/test       | 0.5665753045678139  |
| Metric/Accuracy | 0.78375            |
| Learning_rate   | 0.001               |

**Evaluation:**
The results do not support the hypothesis. Increasing the number of convolutional layers to 16 with batch normalization resulted in a decrease in accuracy on the test set compared to the previous configuration with 8 layers.

**Conclusion:**
The increased complexity of the model may have led to overfitting or difficulties in training effectively, even with batch normalization.
