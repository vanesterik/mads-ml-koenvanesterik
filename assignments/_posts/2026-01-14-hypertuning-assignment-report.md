---
layout: post
title: "Hypertuning Assignment Report"
tags: machine-learning report
---

![Sunflower](https://images.unsplash.com/photo-1540039906769-84cf3d448bc1?q=80&w=3135&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)

|               |                                       |
| ------------- | ------------------------------------- |
| **Date**      | 14th of January 2026                  |
| **Student**   | Koen van Esterik                      |
| **Institute** | HAN University of Applied Sciences    |
| **Course**    | Deep Learning & Deployment (2025 P1A) |
| **Subject**   | Hypertuning                           |

## Introduction

Training deep Convolutional Neural Networks (CNNs) often presents challenges related to slow convergence speeds and suboptimal final classification accuracy. These issues stem largely from the complexity of the optimization landscape and the phenomenon known as "Internal Covariate Shift", where the distribution of layer inputs changes during training {% cite BatchNormalization2025 %}. Addressing these inefficiencies is essential; achieving faster convergence and higher accuracy significantly reduces computational resources and enhances the practical applicability of the model in real-world scenarios.

This study aims to quantify the benefits of architectural normalization techniques. The primary research question is:

**"To what extent does the inclusion of Batch Normalization layers affect the convergence speed and final classification accuracy of a CNN on the flowers dataset?"**

The study is guided by the following hypothesis:

**Including Batch Normalization layers will increase final test accuracy by at least 0.1 (10%) and significantly reduce the number of epochs required to reach convergence.**

The scope of this research is limited to a quantitative ablation study {% cite AblationArtificialIntelligence2025a %} using a specific custom CNN architecture applied to the Flowers dataset, which is part of the MADS datasets repository {% cite groulsRaoulgMads_datasets2024 %}.

## Methodology

The experiments, conducted as part of the ablation study, compare two distinct variations of the same model architecture to isolate the specific impact of the variable in question:

1. **Experimental Group:** CNN architecture *with* Batch Normalization layers.
2. **Control Group:** Identical CNN architecture *without* Batch Normalization layers.

The study utilizes the **Flowers Dataset**. This dataset consists of approximately 3,000 large, complex images categorized into 5 distinct classes (labels). The complexity of the images serves as a robust benchmark for testing feature extraction capabilities in CNNs.

The base model is a custom Convolutional Neural Network (CNN) defined in the provided [notebook]({{ "/notebooks/hypertuning/notebook.ipynb" | relative_url }}). The training procedure was designed to ensure a fair "apples-to-apples" comparison:

1. **Hyperparameter Tuning:** The "Experimental" model (with Batch Norm) was first optimized using the `hyperopt` library to identify the best initial hyperparameters. This optimization phase consisted of 50 evaluations (`max_evals=50`), with each candidate model trained for a limited duration of **20 epochs**, as detailed in the configuration presented in Table 1.
2. **Controlled Testing:** The optimal hyperparameters derived from the tuning phase were then fixed and applied to both the "Experimental" and "Control" (without Batch Norm) architectures. To fully evaluate convergence behavior, both models were subsequently trained for an extended duration of **50 epochs**. This ensures that any difference in performance is attributable solely to the presence or absence of the normalization layers, rather than differences in learning rates, filter sizes, or training duration.

| Setting           | Value       |
| ----------------- | ----------- |
| Batch Size        | 64          |
| Epochs            | 50          |
| Train Steps       | 180         |
| Valid Steps       | 180         |
| Learning Rate     | 0.001       |
| Dropout           | 0.0 - 0.5   |
| Features          | 3           |
| Filters           | 32, 64, 128 |
| Kernel Size       | 3           |
| Number of Classes | 5           |

*Table 1: Settings used in the hyperparameter tuning experiments.*

Performance was evaluated using:

- **Accuracy:** To measure the percentage of correctly classified images.
- **CrossEntropyLoss:** To measure the divergence between the predicted probability distribution and the actual labels for both training and validation sets.

## Results

The inclusion of Batch Normalization resulted in a measurable improvement in model performance.

| Metric             | With Batch Norm | Without Batch Norm |
| ------------------ | --------------- | ------------------ |
| Test Accuracy      | 0.78            | 0.77               |
| Epochs to Converge | 25              | 49                 |

*Table 2: Performance comparison between models with and without Batch Normalization.*

![Control Experiment: Without Batch Normalization](/assets/images/without_batchnorm.png)
![Experimental Model: With Batch Normalization](/assets/images/with_batchnorm.png)

The hypothesis stated that accuracy would improve by at least 0.1 (10%).

- **Result:** The actual improvement was 0.01 (1%). Therefore, the accuracy threshold of the hypothesis was **not met**.
- **Observation:** While the second part of the hypothesis regarding convergence was strongly supported, showing a dramatic acceleration of training loss reduction with Batch Normalization (Figure 2, blue line), the results were not entirely stable. The validation loss for the normalized model (Figure 2, orange line) displayed significant volatility, fluctuating between 0.6 and 1.0 after the 5th step, while the control model's validation loss (Figure 1) maintained a smooth trajectory.

Research literature indicates that Batch Normalization requires specific learning rate configurations to mitigate this volatility {% cite bjorckUnderstandingBatchNormalization2018 %}. It appears that Batch Normalization's reliance on batch statistics and its weight scaling within the range of 0 to 1 makes the model more sensitive to the learning rate. The fixed learning rate of 0.001 may have been too high for the normalized model, resulting in instability in the validation loss. This suggests the need for at least one experiment where the learning rate is tailored specifically for the Batch Normalization model to explore potential improvements in stability.

![Additional Experiment: Batch Normalization with Learning Rate 0.1](/assets/images/with_batchnorm_lr_0_1.png)

In this experiment (Figure 3), the learning rate was adjusted to 0.1 for the Batch Normalization model. The results reveal that while the training loss still converges relatively quickly, the validation loss remains volatile. Although some stabilization is evident compared to the previous experiment (Figure 2), the fluctuations persist - indicating that further tuning of the learning rate may be essential to achieve a more smooth validation loss.

## Discussion

One significant challenge encountered during this research was the extreme duration of the hyperparameter tuning process (> 24 hours). The search for optimal parameters was computationally expensive and time-consuming, which limited the number of evaluations that could be performed within a reasonable timeframe. To enhance the efficiency of future experiments, the following strategies should be considered:

- Implementing the recommendation to use smaller input image dimensions, which will significantly decrease the computational load per epoch.
- Leveraging more advanced hardware to speed up the training process, enabling a broader search space or a higher number of `max_evals` within the same timeframe.

Furthermore, the experimental design relied on hyperparameter tuning solely on the "Experimental" model (with Batch Normalization) and applying those fixed parameters to the "Control" model. In the pursuit of isolating the specific effect of Batch Normalization, this was a methodologically sound decision, as it ensured that any performance difference was due to the architecture change rather than differing hyperparameters. However, this approach may have  disadvantaged the control model, as the optimal learning rate or filter sizes for a normalized network might differ from those of an unnormalized one. Future work could consider performing separate hyperparameter searches for both configurations to compare the *best possible version* of the control against the *best possible version* of the experiment, offering an additional perspective on performance capabilities.

Finally, the volatility observed in the Batch Normalization validation loss suggests that while the technique accelerates learning, it introduces instability if the learning rate is not adjusted accordingly. The model appears to converge rapidly to a low training loss, but the volatile validation loss indicates it may be overfitting to batch-specific statistics. This suggests experimentation with Batch Normalization in combination with learning rate schedules could be result in stabilizing validation performance.

## Conclusion

The results indicate that while Batch Normalization did not trigger a massive  jump in raw accuracy for this specific task, it is nevertheless an essential component of the architecture. The 0.01 (1%) gain in accuracy, combined with a drastic reduction in training time (convergence at epoch 25 vs. 49), confirms that Batch Normalization effectively mitigates optimization difficulties. It stabilizes the learning process, allowing the model to learn faster and generalize (somewhat) better.

This study was limited to a single dataset and architecture. The results suggest that normalization is essential for complex image data, but the magnitude of improvement may vary across different domains. Future work could also expand on investigating other datasets, determining if accuracy gain and speedup are consistent behaviors across different types of visual data.

## References

{% bibliography --cited %}