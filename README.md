# "When Can I Trust It?" Contextualising Explainability Methods for Classifiers
Website accompanying the paper with the above title:

**Abstract** *The need for artificial intelligence systems to expose reasons for promoted decisions grows with the prevalence of these systems in society. In this work, we study, for carefully selected images, how an end user's trust is affected by visual explanations. Additionally, we complement our work by probing pretrained neural network's consistency for the selected images. Our research approach exposes the brittleness in these systems pointing toward a need to develop benchmarking methods connecting visual explanations to training data distribution and, additionally, move away from a flat output hierarchy toward including a concept ontology that matches the target domain in the system.*

###### This site adds code and additional material for reproducibility and additional material for part 3 in the paper.

**Part 3** This study focus on explaining, understanding and reasoning in relation to a singular classification. The flat hierarchy of 1000 classes in ImageNet-1k cannot, as expected, capture any deeper meaning in the images picturing the dog and the elephant. But that put aside, the brittleness, related to model choice, image transformation and weight selection for the pretrained ML-models poses important challenges for any non static target context. Our results  points in the same direction as other research that also discusses and lays bare the need to balance average case metrics on large datasets towards including a focus on performance in relation to a specific target context. The ontological mismatch become opaque when it is accompanied by a training datasets of incalculable size that makes it very challenging to identify if, and in what way, the input data is out of distribution.

Click the images for to get the ML-model comparisons.

[![](trust_1/images/dog.jpg)](https://k3larra.github.io/IKR/trust_1/version07.html?study_nbr=0)
[![](trust_1/images/elephant.jpg)](https://k3larra.github.io/IKR/trust_1/version07.html?study_nbr=1)
[![](trust_1/images/reptiles.jpg)](https://k3larra.github.io/IKR/trust_1/version07.html?study_nbr=2)

[Code to reproduce the experiments](ikr.ipynb)
