[//]: # (Image reference)
[slika1]: https://img.youtube.com/vi/2b2LPtljqdA/0.jpg
[xray]: documention/doc1.png
[results]: documention/doc2.png

# Bank churn prediction
This is a project for churn prediction written in **Python** and *Scikit-Learn* 


> Scikit-learn is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines
## Team Structure
1. Vladimir Vuksanovic
2. Goran Stojanovski
3. Goran Puntevski
4. Petar Trajchevski 
5. Dashmir Ibishi
---

| Video1 | Video2| Video3|
| ---- | ------ |-----------|
|[![Training][results]](https://www.youtube.com/watch?v=2b2LPtljqdA)|[![Training][slika1]](https://www.youtube.com/watch?v=2b2LPtljqdA)|[![Training][slika1]](https://www.youtube.com/watch?v=2b2LPtljqdA)|
## Project Specification
The following ML techniques wer used in this project
- Dimensionality Reduction
  - PCA
  - TSNE
- XG-Boost
- Random Forest

For training Random forest we used the following code snippet :

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,
...                            n_informative=2, n_redundant=0,
...                            random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
RandomForestClassifier(...)
print(clf.predict([[0, 0, 0, 0]]))
```


## Installation
For installing scikit-learn please do :
`
pip install scikit-learn
`


## Dataset

For download the dataset please click [HERE](http://google.com)

Screenshot of training performance : 
![Training](https://www.researchgate.net/profile/Atanu-Kumar-Paul/publication/321382145/figure/fig5/AS:579337981054986@1515136346983/Neural-network-training-validation-and-test-plot-for-the-output-response-Oxirane.png)

<p align = "center">
<img src="https://www.researchgate.net/profile/Tapash-Sarkar/publication/327720895/figure/fig3/AS:672135315996676@1537260955913/Neural-network-training-performance-plot-with-best-validation.png" alt = "Training" width="50%" height="50%"/>
</p>

![Training][xray]

<p align = "center">
<img src="documention/doc1.png" alt = "Training" width="50%" height="50%"/>
</p>



## Evalauation results

| Test | Traing | Validation|
| ---- | ------ |-----------|
| 99% | 92% |91%     |



## Fixed bugs :
- ~~Failure in installation because of ...~~
- ~~Improved model performance~~
- ~~Fixed bug in sort algorithm~~

## We tried the following Deep Learning Pretrained models:
- [x] Resnet50
- [ ] VGG16
- [ ] Densenet
