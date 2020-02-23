## K-Means Clustering with SGD
- k-means Clustering Algorithm(k-평균 알고리즘)은 주어진 데이터를 k개의 cluster(군집)으로 묶는 알고리즘이다. 
이 알고리즘의 기본 아이디어는 R2(2차원 평면)에서의 주어진 데이터  {pi} (1 <= i <= N, pi=(ai,bi))에 대해 주어진 함수를 최소가 되게 하는 중심(cluster point) x를 찾는 것이다.

본 문제는 Stochastic Gradient Descent Method를 이용해 k-means clustering을 구현해 본다.

아래의 주어진 평균 mu와 공분산 행렬 sigma에 대해 [다변수 정규 분포]를 통해
각각 1000개씩 총 4000개의 랜덤한 점을 만든다.      

mu1= [-5 -3], mu2= [5 -3], mu3= [0.0 5.0], mu4= [2.5 4.0]

sigma1 = [[0.8,0.1],[0.1,0.8]], sigma2 =[[1.2,0.6],[0.6,0.7]], sigma3 = [[0.5,0.05],[0.05,1.6]], sigma4 = [[1.5,0.05],[0.05,0.6]]  (2 by 2 matrix)

x1 = (0,0), x2 = (0,0), x3 = (0,0), x4 = (0,0) 에서 시작해서 k-means algorithm을 통해 cluster points를 찾는다.

---
```
import numpy as np
import matplotlib 
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
```
