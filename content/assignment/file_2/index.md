---
title: Assignement 2
summary: This assignment is about implementation of KNN algorithm with the hyper parameters K values =1,3,5,7 and distance metrics = euclidean distance , euclidean distance and cosine distance.  Train with training dataset and then get the optimal hyperparameters using the predict of the model on Development dataset. Finally get accuracy of the model on the testing data set.

links:
  - icon_pack: fab
    icon: fa-download
    name: Download .ipynb
    url: 'https://drive.google.com/file/d/1Fb0Hpwq2D4EaIGlKhfEe9fPgSlKu_ywi/view?usp=sharing/'
    
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

Analysis:-

- KMetrics :- optimal value of k would be = 5, From the above barchart, we can understand that my model more consistent  from k=3 and there shouldn't be any overfit with small k values and underfit with higher values of k, i will consider k=5 as the optimal parameter.

 - Distance Metric:- Cosine distance measure would be optimal, As we seen from the graph cosine metrics gave me better and   consistent values compared to the euclidean and normalized euclidean distance..

- Optimal Hyper parameters are k=5, and Cosine Distance

-  Final test accuracy for the test data is 96.66666666666667%, which says that our performence is consistent on test data as it   gave me same accuracy of developement data set for the optimal parameters of developementset

{{% staticref "files/Manukonda_02.html" "newtab" %}}View my Project{{% /staticref %}}
