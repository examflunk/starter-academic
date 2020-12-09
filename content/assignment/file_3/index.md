---
title: Assignment 3
summary: The goal of this assignment is to learn about the Naive Bayes Classifier (NBC). It uses large movie review dataset , Builds the vocabulary list of the training data and removing the words that has frequency less than 5, Calculate the probability of P[“the”] and P[“the” | Positive]. It calculates  accuracy using five fold cross validation and Compares the effect of Smoothing on development dataset. Final accuracy is found using optimal parameter from smoothing on test dataset.

links:
  - icon_pack: fab
    icon: fa-download
    name: Download .ipynb
    url: 'https://drive.google.com/u/1/uc?id=1ZCj-0HP8OqDMbSwxeemEN8-dvOxhNi0Z&export=download'
    
tags:
- Algorithm 
date: "2020-04-27T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption: Photo by Toa Heftiba on Unsplash
  focal_point: Smart
---
This assignment is about implementation of KNN algorithm with the hyper parameters K values =1,3,5,7 and distance metrics = euclidean distance , euclidean distance and cosine distance.  Train with training dataset and then get the optimal hyperparameters using the predict of the model on Development dataset. Finally get accuracy of the model on the testing data set.

Analysis :- The performance of the model is good when the smoothing parameter value is 1 (laplace Smoothing) and its slowly decreases w.r.t  alpha values decreases. With the 5 fold cross validations on test data, Final accuracy scores are [0.8652, 0.8724, 0.8742, 0.8756, 0.8718] and a mean accuracy of 87.18%

{{% staticref "files/Manukonda_03.html" "newtab" %}}View my pynb{{% /staticref %}}
