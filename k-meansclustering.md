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
```
# Multivariate normal distribution
mu1 = [-5,-3]
mu2 = [5,-3]
mu3 = [0.0,5.0]
mu4 = [2.5,4.0]
sigma1 = [[0.8,0.1],[0.1,0.8]]
sigma2 = [[1.2,0.6],[0.6,0.7]]
sigma3 = [[0.5,0.05],[0.05,1.6]]
sigma4 = [[1.5,0.05],[0.05,0.6]]
```
```
x1, y1 = np.random.multivariate_normal(mu1,sigma1,1000).T
x2, y2 = np.random.multivariate_normal(mu2,sigma2,1000).T
x3, y3 = np.random.multivariate_normal(mu3,sigma3,1000).T
x4, y4 = np.random.multivariate_normal(mu4,sigma4,1000).T
```
```
data1 = np.column_stack((x1,y1))  #cluster1
data2 = np.column_stack((x2,y2))  #cluster2
data3 = np.column_stack((x3,y3))  #cluster3
data4 = np.column_stack((x4,y4))  #cluster4
```
```
x = [x1, x2, x3, x4]
y = [y1, y2, y3, y4]
```
```
#Initionalize parameter
gamma = 4
alpha = 0.75
beta = 0.99
omaxit = 50
imaxit = 10
mini_batch_size = 100
cluster_old = [[0,0],[0,0],[0,0],[0,0]]
```
```
# Main Loop
for k in range(omaxit):
    cluster_temp = cluster_old
    cluster_new = cluster_temp
    for l in range(imaxit):
        #Randomly Sample
        np.random.shuffle(data1)
        np.random.shuffle(data2)
        np.random.shuffle(data3)
        np.random.shuffle(data4)
        mini_batch1 = data1[0:mini_batch_size]
        mini_batch2 = data2[0:mini_batch_size]
        mini_batch3 = data3[0:mini_batch_size]
        mini_batch4 = data4[0:mini_batch_size]
        grad_temp = [np.sum(cluster_temp[0]-mini_batch1,axis=0),
                     np.sum(cluster_temp[1]-mini_batch2,axis=0),
                     np.sum(cluster_temp[2]-mini_batch3,axis=0),
                     np.sum(cluster_temp[3]-mini_batch4,axis=0),]
        grad_temp = np.divide(grad_temp, mini_batch_size)
        
        cluster_temp_new = cluster_old - gamma * grad_temp
        cluster_new = np.multiply(cluster_old,alpha) + np.multiply(cluster_temp_new, 1-alpha)
    gamma = beta * gamma
    cluster_old = cluster_new
```
```
print(cluster_new)
plt.plot(x,y,'x')
plt.plot(cluster_new[0][0], cluster_new[0][1], 'wo')
plt.plot(cluster_new[1][0], cluster_new[1][1], 'wo')
plt.plot(cluster_new[2][0], cluster_new[2][1], 'wo')
plt.plot(cluster_new[3][0], cluster_new[3][1], 'wo')
plt.axis('equal')
plt.show()
```

