---
layout: post
title: "Neural Networks Assignment Report"
tags: machine-learning report
---

[notebook]({{ "/notebooks/neural-networks/notebook.ipynb" | relative_url }})
[journal]({{ "/notebooks/neural-networks/journal.txt" | relative_url }})

## Introduction

For the lectures of Deep Learning and Deployment, we are tasked with exploring the intricacies of neural networks, their architectures, and their applications. This report aims to summarize the findings from the experiments conducted during the assignment.

You can find the the code for this assignment in the [notebook]({{ "/notebooks/neural-networks/notebook.ipynb" | relative_url }}). The [journal]({{ "/notebooks/neural-networks/journal.txt" | relative_url }}) contains a detailed log of the experiments, observations, and conclusions drawn during the assignment and is also included as appendix to this report.

All experiments are done on a provided neural network architecture and is based on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. The task of the neural network is to classify images of clothing into one of ten categories. The dataset consists of 60,000 training images and 10,000 test images, each being a 28x28 grayscale image.

The dataset is balanced, meaning that each class has an equal number of images. Therefore we are using accuracy as the primary metric to evaluate the performance of the neural network.

The provided neural network architecture is a feedforward neural network with two hidden layers. The number of units in each hidden layer can be adjusted. The activation function used is ReLU (Rectified Linear Unit), and the output layer produces raw logits for each class.

Multiple experiments were conducted, each focusing on different aspects of neural networks, such as hyperparameter tuning, architecture variations, etc. This report will detail the methodology, results, and conclusions drawn from these experiments.

Perhaps I overdid it a bit in terms of the number of experiments, but I wanted to explore as many aspects of neural networks as possible within the given timeframe. And even then I feel like I only scratched the surface of what is possible with neural networks.

## Method

The instructions for this assignment are as follows:

1. Define a hypothesis related to hyper tuning of a provided neural network.
2. Design an experiment based on the hypothesis.
3. Run the experiment.
4. Analyze the results and draw conclusions.
5. Repeat steps 2-4 to explore more aspects of neural networks.

## Results

As mentioned in the [journal]({{ "/notebooks/neural-networks/journal.txt" | relative_url }}), I conducted multiple experiments. The most interesting one was to test whether adjusting the width and depth of the neural network would lead to better performance.

The hypothesis was that increasing the depth of the network (i.e., adding more hidden layers) would lead to better performance, as deeper networks can learn more complex representations of the data. To test this hypothesis, I designed an experiment where I would train multiple versions of the neural network with varying depths and widths.

![Neural Networks Experiments]({{ "/assets/images/neural_networks_experiments.png" | relative_url }})

As shown in the graph above, the results were mixed. While some configurations with more layers did lead to better performance, others did not. In fact, the best performing configuration was a two-layer network with 384 and 512 units, achieving an accuracy of 0.8945.

This experiment highlighted the importance of careful hyperparameter tuning and the fact that more complex models do not always lead to better performance. It also underscored the need for systematic experimentation to identify optimal configurations.

## Discussion

It seems at every turn of setting up and running these experiments - I not only encountered unexpected results, but also gained a deeper understanding on how to conduct experiments in machine learning.

The outlined methodology was straightforward, but the execution revealed the complexity of neural networks, and also the importance of careful experimental design. For instance, I initially hypothesized that simply increasing the number of hidden units would lead to better accuracy. However, the results showed that there is an optimal range for the number of hidden units, and beyond that, performance can degrade. But I needed a second hypothesis in order to define the exact number of hidden units to compare results with. This was a valuable lesson in the importance of specificity in hypothesis formulation.

With these reflections in mind, I come to the conclusion that I need to conduct more experiments - not only to deepen my understanding of neural networks, but also to refine my experimental design skills.

## Conclusion

In conclusion, the experiments conducted during this assignment have provided valuable insights into the behavior of neural networks. The findings suggest that there are optimal configurations for various hyperparameters. However it is worth noting that multiple experiments yielded results that did not support the initial hypotheses. This highlights the complexity of neural networks and the relevance of the [no free lunch theorem](https://en.wikipedia.org/wiki/No_free_lunch_theorem) in machine learning.

## Appendix

### Journal Entries

#### 2025-10-07 15:00:51 Effect of Hidden Units on Accuracy

**Objective:**
Test whether increasing the number of hidden units in the neural network increases accuracy.

**Hypothesis:**
More hidden units will increase accuracy.

**Methodology:**
Ran a grid search with two layers, testing combinations of 16, 32, 64, 128, 256, 512, and 1024 units using the defined neural network and dataset.

**Results:**
Combination of 512 x 512 units achieved 0.8708 accuracy. The highest unit value did not yield the highest accuracy.

**Evaluation:**
The results do not support the hypothesis. Increasing hidden units beyond a certain point did not improve accuracy.

**Conclusion:**
There is an optimal range for the number of hidden units; simply increasing them does not guarantee better performance.

#### 2025-10-07 15:22:13 Effect of Hidden Units (384-640) on Accuracy

**Objective:**
Test whether the highest accuracy is achieved with hidden units between 384 and 512.

**Hypothesis:**
Highest accuracy is achieved between 384 and 512 units.

**Methodology:**
Ran a grid search with two layers, testing combinations of 384, 512, and 640 units using the defined neural network and dataset.

**Results:**
Combination of 384 x 512 units achieved 0.8761 accuracy. The increase between 512 x 384 and 512 x 512 was only 0.6 percent.

**Evaluation:**
The results support the hypothesis. The best accuracy was found in the range between 384 and 512 units, with only marginal gains beyond this range.

**Conclusion:**
There is an optimal range for the number of hidden units; increasing units beyond 512 does not significantly improve performance.

#### 2025-10-07 15:36:31 Effect of Number of Epochs on Accuracy

**Objective:**
Test whether training for more than 3 epochs increases the model's accuracy.

**Hypothesis:**
Training for more than 3 epochs will increase accuracy.

**Methodology:**
Trained the model for 5 epochs instead of 3, using the same network architecture and dataset. Also compared the effect of switching the layer composition from 512 x 384 to 384 x 512 units.

**Results:**
Training for 5 epochs did not increase accuracy; it remained at 0.8761. However, changing the layer composition from 512 x 384 to 384 x 512 increased accuracy from 0.8708 to 0.8795.

**Evaluation:**
The results do not support the hypothesis. Increasing the number of epochs from 3 to 5 did not improve accuracy, but changing the order of units in the layers did have a positive effect.

**Conclusion:**
Simply increasing the number of epochs does not guarantee better performance. The architecture and composition of layers can have a more significant impact on accuracy.

#### 2025-10-07 16:15:21 Effect of Increasing Epochs (5 vs 3) with 384 x 512 Units

**Objective:**
Test whether training for 5 epochs instead of 3, with a fixed combination of 384 x 512 units, increases accuracy.

**Hypothesis:**
Training for 5 epochs with 384 x 512 units will achieve higher accuracy than training for 3 epochs.

**Methodology:**
Trained the model with 384 x 512 units for 5 epochs and compared the accuracy to the result from 3 epochs.

**Results:**
The combination of 384 x 512 units with 5 epochs achieved 0.88 accuracy, which is higher than the accuracy achieved with 3 epochs.

**Evaluation:**
The results support the hypothesis. Increasing the number of epochs from 3 to 5 led to improved accuracy for this specific layer configuration.

**Conclusion:**
For the 384 x 512 unit configuration, training for more epochs can improve model performance, but the effect may depend on the chosen architecture and other hyperparameters.

#### 2025-10-07 16:16:32 Effect of Increasing Epochs (10 vs 5) with 384 x 512 Units

**Objective:**
Test whether training for 10 epochs instead of 5, with a fixed combination of 384 x 512 units, increases accuracy.

**Hypothesis:**
Training for 10 epochs with 384 x 512 units will achieve higher accuracy than training for 5 epochs.

**Methodology:**
Trained the model with 384 x 512 units for 10 epochs and compared the accuracy to the result from 5 epochs.

**Results:**
The combination of 384 x 512 units with 10 epochs achieved 0.8906 accuracy, which is higher than the accuracy achieved with 5 epochs.

**Evaluation:**
The results support the hypothesis. Increasing the number of epochs from 5 to 10 led to improved accuracy for this specific layer configuration.

**Conclusion:**
For the 384 x 512 unit configuration, training for more epochs can further improve model performance, but the effect may depend on the chosen architecture and other hyperparameters.

#### 2025-10-07 16:21:00 Effect of Increasing Epochs (20 vs 5) with Early Stopping

**Objective:**
Test whether training for 20 epochs instead of 5, with a set combination of layers, increases accuracy.

**Hypothesis:**
Training for 20 epochs with the same layer configuration will achieve higher accuracy than training for 5 epochs.

**Methodology:**
Trained the model for up to 20 epochs with early stopping enabled, using the same network architecture and dataset.

**Results:**
Early stopping halted training at 17 epochs, achieving 0.8945 accuracy. This was higher than the accuracy at 5 epochs.

**Evaluation:**
The results partially support the hypothesis. While accuracy increased, the improvement was due to early stopping rather than completing all 20 epochs.

**Conclusion:**
Allowing more epochs with early stopping can improve accuracy, but the model may not require all scheduled epochs to reach optimal performance.

#### 2025-10-07 16:30:14 Effect of Batch Size (32 vs 64) on Accuracy

**Objective:**
Test whether using a batch size of 32 instead of 64 increases accuracy.

**Hypothesis:**
Batch size of 32 will achieve higher accuracy than batch size of 64.

**Methodology:**
Trained the model with batch sizes of 32 and 64, comparing accuracy and training duration.

**Results:**
Batch size 32 achieved 0.8943 accuracy, while batch size 64 achieved 0.8945 accuracy. Training with batch size 32 stopped earlier (16 epochs) but took more time.

**Evaluation:**
The results do not support the hypothesis. Batch size 64 slightly outperformed batch size 32, and training time was longer for the smaller batch size.

**Conclusion:**
Reducing batch size does not necessarily improve accuracy and may increase training time.

#### 2025-10-07 16:37:06 Effect of Batch Size (128 vs 64) on Accuracy

**Objective:**
Test whether using a batch size of 128 instead of 64 increases accuracy.

**Hypothesis:**
Batch size of 128 will achieve higher accuracy than batch size of 64.

**Methodology:**
Trained the model with batch sizes of 128 and 64, comparing accuracy.

**Results:**
Batch size 128 achieved 0.8837 accuracy, while batch size 64 achieved 0.8945 accuracy.

**Evaluation:**
The results do not support the hypothesis. Batch size 64 outperformed batch size 128.

**Conclusion:**
Increasing batch size beyond a certain point can reduce model accuracy.

#### 2025-10-07 16:42:56 Effect of Learning Rate (1e-4 vs 1e-3) on Accuracy

**Objective:**
Test whether using a learning rate of 1e-4 instead of 1e-3 increases accuracy.

**Hypothesis:**
Learning rate of 1e-4 will achieve higher accuracy than 1e-3.

**Methodology:**
Trained the model with learning rates of 1e-4 and 1e-3, comparing accuracy.

**Results:**
Learning rate 1e-4 achieved 0.8903 accuracy, while 1e-3 achieved 0.8945 accuracy.

**Evaluation:**
The results do not support the hypothesis. Learning rate 1e-3 outperformed 1e-4.

**Conclusion:**
Lowering the learning rate does not always improve accuracy; optimal values depend on the model and data.

#### 2025-10-07 16:46:17 Effect of Learning Rate (1e-2 vs 1e-3) on Accuracy

**Objective:**
Test whether using a learning rate of 1e-2 instead of 1e-3 increases accuracy.

**Hypothesis:**
Learning rate of 1e-2 will achieve higher accuracy than 1e-3.

**Methodology:**
Trained the model with learning rates of 1e-2 and 1e-3, comparing accuracy.

**Results:**
Learning rate 1e-2 achieved 0.8495 accuracy, while 1e-3 achieved 0.8945 accuracy.

**Evaluation:**
The results do not support the hypothesis. Learning rate 1e-3 outperformed 1e-2.

**Conclusion:**
Increasing the learning rate too much can harm model performance.

#### 2025-10-07 19:02:10 Effect of Additional Layer Combinations on Accuracy

**Objective:**
Test whether adding an additional layer with different combinations of units increases accuracy.

**Hypothesis:**
Adding an additional layer with different unit combinations will increase accuracy.

**Methodology:**
Trained models with various three-layer configurations, comparing accuracy to the best previous result (0.8972).

**Results:**
All tested three-layer combinations achieved lower accuracy than 0.8972.

| Unit Combination | Accuracy |
| ---------------- | -------- |
| 512 x 348 x 512  | 0.8965   |
| 256 x 512 x 256  | 0.8961   |
| 384 x 256 x 512  | 0.8952   |
| 512 x 384 x 384  | 0.8949   |
| 256 x 384 x 512  | 0.8947   |
| 512 x 512 x 256  | 0.8946   |

**Evaluation:**
The results do not support the hypothesis. Not all three-layer configurations outperform the best previous result.

**Conclusion:**
Not every additional layer or unit combination leads to better performance; careful tuning is required.

#### 2025-10-07 19:25:05 Effect of Optimizer Choice (Adam vs SGD) on Accuracy

**Objective:**
Test whether switching the optimizer from Adam to SGD increases accuracy.

**Hypothesis:**
SGD will achieve higher accuracy than Adam.

**Methodology:**
Trained the model using both Adam and SGD optimizers, comparing accuracy.

**Results:**
SGD achieved 0.7596 accuracy, while Adam achieved 0.8965 accuracy.

**Evaluation:**
The results do not support the hypothesis. Adam outperformed SGD in this experiment.

**Conclusion:**
Optimizer choice can significantly impact model performance; Adam was superior to SGD for this task.
