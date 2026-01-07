---
layout: post
title: "Deployment Assignment Report"
tags: machine-learning report
---

![INTERGALACTIC-NAMERATOR](https://raw.githubusercontent.com/vanesterik/intergalactic-namerator/refs/heads/main/references/rick-and-morty.png)

|               |                                       |
| ------------- | ------------------------------------- |
| **Date**      | 7th of January 2026                   |
| **Student**   | Koen van Esterik                      |
| **Institute** | HAN University of Applied Sciences    |
| **Course**    | Deep Learning & Deployment (2025 P1A) |
| **Subject**   | Deployment                            |

## Introduction

The primary focus of this research is the deployment of a containerized Artificial Intelligence (AI) application - **the Intergalactic Namerator** - on a remote Ubuntu server. This application utilizes a neural network to generate character names inspired by the "Rick & Morty" universe.

This research is significant as it serves as a foundational exercise in bridging the gap between local AI development and cloud-based production environments. While existing literature often focuses on model architecture and hyperparameter tuning, there is a recurring gap in practical, "lean" deployment strategies for developers working with limited server resources.

The central research question guiding this report is: *Can a Python-based AI application be successfully deployed and made publicly accessible on an Ubuntu server using Docker?* The scope is limited to the deployment phase, acknowledging that the underlying model is already optimized and the server is restricted to HTTP access.

## Method

This study utilized an exploratory case study design. The technical environment (the "sample") consisted of a remote Ubuntu 24.04 server equipped with 2 CPU cores and 16 GB of RAM.

The methodology involved the following steps:

- **Containerization**: Packaging the Python application and its dependencies using Docker.
- **Deployment**: Transferring and running the container on the target Ubuntu environment.
- **Data Collection & Analysis**: Success was measured through functional testing (accessibility via the public IP http://145.38.191.232/) and the implementation of a health check to monitor the container’s status.
- **Tools**: The primary software stack included Python, Docker, and the Ubuntu Command Line Interface (CLI).

## Results

The deployment was ultimately successful, though it required significant technical pivots. The initial deployment failed due to the storage footprint of the PyTorch framework, which exceeded the available disk space on the remote server.

### Key Findings

- **Optimization**: Success was achieved by explicitly excluding the standard PyTorch bundle and instead installing the CPU-only version. This drastically reduced the image size and allowed the application to compile and run within the server's constraints.
- **Accessibility**: The application is fully functional and accessible via its public IP, though it is currently limited to the HTTP protocol.
- **Validation**: The implemented health check confirmed that the "Intergalactic Namerator" can generate names successfully in the production environment.

## Discussion

This case study highlights the critical need for "intuition for production." The findings suggest that developers cannot assume remote environments will be as forgiving as local development machines.

A key takeaway is the importance of understanding **the theoretical intersection between software dependencies and hardware specifications**. The *wall* encountered with PyTorch's size serves as a practical example of why early deployment is necessary. In the future, a *placeholder* strategy—deploying a skeleton container first to test server limits, will be adopted to identify infrastructure bottlenecks before the final model is integrated.

Comparing these results to industry standards, the move toward CPU-only inference is a recognized best practice for cost-effective, non-latency-critical applications. Future research and iterations will focus on implementing **CI/CD pipelines** to automate these checks.

## Conclusion

The deployment of the Intergalactic Namerator proved that while Python-based AI apps can be deployed via Docker on Ubuntu, success is highly dependent on environment-specific optimizations.

### Main Takeaways

- Technical specifications of the host server must be audited against the dependency tree of the application before the launch date.
- Frameworks like PyTorch require careful management (such as selecting CPU-only builds) to fit within standard server constraints.
- A "fail-fast" approach—using placeholder code to test the deployment pipeline—is recommended for future practitioners to avoid last-minute hurdles.

## Appendix

- [Github repository](https://github.com/vanesterik/intergalactic-namerator)
- [PyPI package](https://pypi.org/project/intergalactic-namerator/)
- [Deployed application]([http://http://145.38.191.232/)