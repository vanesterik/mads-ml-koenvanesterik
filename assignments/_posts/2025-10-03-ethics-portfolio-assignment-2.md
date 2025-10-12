---
layout: post
title: "Ethics Portfolio Assignment #2"
tags: ethics
---

|               |                                       |
| ------------- | ------------------------------------- |
| **Date**      | 3rd of October 2025                   |
| **Student**   | Koen van Esterik                      |
| **Institute** | HAN University of Applied Sciences    |
| **Course**    | Deep Learning & Deployment (2025 P1A) |
| **Subject**   | Ethics                                |

![Dating](https://images.unsplash.com/photo-1604881991575-dfb1003d8811?q=80&w=2071&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)

## Introduction

For the lectures of ethics, we are asked to research the concept of directional acyclic graphs (DAGs). This to get a better understanding of how to indentify ethical dilemmas in the field of data science.

The terms used in the assignment description are defined as follows:

| Term                             | Definition                                                                                                                                                                    |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ethics                           | Values held by society as a whole. (e.g. societal norms)                                                                                                                      |
| Ethical Dilemma                  | A situation in which two or more conflicting moral imperatives, none of which overrides the other, confront an agent. {% cite EthicalDilemma2025 %}                           |
| Directional Acyclic Graphs (DAG) | A directed graph without cycles, meaning it's impossible to start at any node and return to it by following a consistently-directed path. {% cite DirectedAcyclicGraph2025 %} |
| Data Science                     | The extraction or extrapolation of knowledge from structured or unstructured data. {% cite DataScience2025 %}                                                                 |

These definitions will be used as the foundation for this assignment.

I will follow the instructions in the assignment description in order to comply with the assignment. These instructions are featured in the method section.

## Method

The instructions for this assignment are as follows:

1. Read the dating casus {% cite DatingappBreezeMag %}
2. Formulate your first impression on the ethical issues and how they came about
3. Draw a DAG (or multiple DAGs) for this dilemma
4. After this exercise (nr.3), are there aspects that you missed in your first impression?
5. What recommendation would you give to a data scientist who gets an assignment like this?

## Results

Initially when reading the dating casus, I got the impression from the text that people with dark skin or a non-Dutch background were getting less overall matches. This turns out to be an incorrect assumption. The actuall issue is that people with a dark skin or non-Dutch background are getting less matches with a similar ethinic background. This is a significant distinction, as it highlights the presence of bias in the matching algorithm rather than a general lack of matches.

The adjustment of my first impression was aided by the creation of a DAG. This helped me to visualize the relationships between the different factors involved in the ethical dilemma.

![Dating Casus DAG]({{ "/assets/images/dating-casus-dag.svg" | relative_url }})

It also enabled me to develop a potential solution to the ethical dilemma. By implementing an additional system upstream that measures the diversity of the matches and adjusts them as needed, we can enhance the candidate data used to train the algorithm, thereby reducing bias in the matching process.

Although I am not sure on what attributes this additional system should base its adjustments - since the casus states that physical attributes are not included in the used data. This is something that would need to be researched further.

## Discussion

The DAG helped me to visualize the relationships between the different factors involved in the ethical dilemma. It also helped me to identify potential solutions to the ethical dilemma.

But I can not help but feel that the DAG is a bit too simplistic. It does not capture the full complexity of the dating app. For example, how is the algorithm actually able to determine the similarity between users and candidates? What data points does it use, and how are they weighted? Especially considering that the used data does not include physical attributes according to the casus. These are important questions that the DAG does not address.

## Conclusion

After completing this assignment, I am convinced that DAGs are a useful tool for visualizing and analyzing ethical dilemmas. However, they should be used in conjunction with other tools and frameworks, such as the PLUS model, to ensure a comprehensive analysis of the ethical issues involved.

## References

{% bibliography --cited %}
