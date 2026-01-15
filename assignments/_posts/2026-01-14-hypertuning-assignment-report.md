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

Training deep Convolutional Neural Networks (CNNs) often presents challenges related to slow convergence speeds and suboptimal final classification accuracy. These issues stem largely from the complexity of the optimization landscape and the phenomenon known as "Internal Covariate Shift", where the distribution of layer inputs changes during training. {% cite BatchNormalization2025 %} Addressing these inefficiencies is essential; achieving faster convergence and higher accuracy significantly reduces computational resource consumption and enhances the practical applicability of the model in real-world scenarios.

This study aims to quantify the benefits of architectural normalization techniques. The primary research question is:

**"To what extent does the inclusion of Batch Normalization layers affect the convergence speed and final classification accuracy of a CNN on the flowers dataset?"**

The study is guided by the following hypothesis:

**Including Batch Normalization layers will increase final test accuracy by at least 0.1 (10%) and significantly reduce the number of epochs required to reach convergence.**

The scope of this research is limited to a quantitative ablation study using a specific custom CNN architecture applied to the Flowers dataset, which is part of the MADS datasets repository. {% cite groulsRaoulgMads_datasets2024 %} 

## Methodology

This research employs a quantitative **Ablation Study**. {% cite AblationArtificialIntelligence2025 %} The experiment compares two distinct variations of the same model architecture to isolate the specific impact of the variable in question:

1. **Experimental Group:** CNN architecture *with* Batch Normalization layers.
2. **Control Group:** Identical CNN architecture *without* Batch Normalization layers.

The study utilizes the **Flowers Dataset**. This dataset consists of approximately 3,000 large, complex images categorized into 5 distinct classes (labels). The complexity of the images serves as a robust benchmark for testing feature extraction capabilities in CNNs.

The base model is a custom Convolutional Neural Network (CNN) defined in the provided [notebook]({{ "/notebooks/hypertuning/notebook.ipynb" | relative_url }}). The training procedure was designed to ensure a fair "apples-to-apples" comparison:

1. **Hyperparameter Tuning:** The "Experimental" model (with Batch Norm) was first optimized using the `hyperopt` library. The optimization process was conducted over 50 evaluations (`max_evals=50`), as detailed in the configuration presented in Table 1.
2. **Controlled Testing:** The optimal hyperparameters derived from the experimental phase were then fixed and applied to the "Control" model (without Batch Norm). This ensures that any difference in performance is attributable solely to the presence or absence of the normalization layers, rather than differences in learning rates or filter sizes.

| Setting           | Value       |
| ----------------- | ----------- |
| Batch Size        | 64          |
| Epochs            | 20          |
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
| Test Accuracy      | 0.85            | 0.75               |
| Epochs to Converge | 10              | 18                 |

*Table 2: Performance comparison between models with and without Batch Normalization.*

!!! ADD LOSS vs EPOCHS PLOTS

The hypothesis stated that accuracy would improve by at least 0.1 (10%).

- **Result:** The actual improvement was 0.05 (5%). Therefore, the quantitative accuracy threshold of the hypothesis was **not met**.
- **Observation:** However, the second part of the hypothesis regarding convergence was strongly supported. The reduction in training epochs (approx. 8 epochs reduction) demonstrates a profound efficiency gain. The normalization of weights between layers proved to have a much larger impact on training stability than anticipated.

## Discussion

One significant challenge encountered during this research was the extreme duration of the hyperparameter tuning process (> 900 mins). The search for optimal parameters was computationally expensive and time-consuming, which limited the number of evaluations that could be performed within a reasonable timeframe. To improve the efficiency of future experiments, several strategies could be adopted.

- Applying the recommendation of using smaller input image dimensions, would drastically reduce the computational load per epoch.
- Utilizing more capable hardware would accelerate the training steps, allowing for a more extensive search space or a higher number of `max_evals` in the same amount of time.

Furthermore, the experimental design relied on hyperparameter tuning solely on the "Experimental" model (with Batch Normalization) and applying those fixed parameters to the "Control" model. In the pursuit of isolating the specific effect of Batch Normalization, this was a methodologically sound decision, as it ensured that any performance difference was due to the architecture change rather than differing hyperparameters. However, this approach may have  disadvantaged the control model, as the optimal learning rate or filter sizes for a normalized network might differ from those of an unnormalized one. Future work could consider performing separate hyperparameter searches for both configurations to compare the *best possible version* of the control against the *best possible version* of the experiment, offering an additional perspective on performance capabilities.

Finally, while Batch Normalization in this context successfully improved convergence speed, there is potential for further architectural enhancements. Future iterations of this research could investigate the inclusion of skip connections (residual connections) alongside normalization layers. The lectures on Deep Leaning suggest that such combinations can enhance both convergence speed and final accuracy, particularly in deeper networks. {% cite He2016DeepRL %}

## Conclusion

The results indicate that while Batch Normalization did not trigger a massive  jump in raw accuracy for this specific task, it is nevertheless an essential component of the architecture. The  gain in accuracy, combined with a drastic reduction in training time (convergence at epoch 10 vs. 18), confirms that Batch Normalization effectively mitigates optimization difficulties. It stabilizes the learning process, allowing the model to learn faster and generalize better.

This study was limited to a single dataset and architecture. The results suggest that normalization is critical for complex image data, but the magnitude of improvement may vary across different domains. Future work will expand this experimental setup to investigate other datasets, determining if the  accuracy gain and  speedup are consistent behaviors across different types of visual data.

## References

{% bibliography --cited %}