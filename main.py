import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

K = 3
N = 200


data1 = np.random.randn(N//3,2) + np.array([5,6])
data2 = np.random.randn(N//3,2) + np.array([-5,-6])
data3 = np.random.randn(N//3,2) + np.array([-10,3])
data = np.concatenate((data1, data2, data3))

class point():
    def __init__(self, data):
        self.data = data
        self.k = np.random.randint(0,K)
    
    def __repr__(self):
        return str({"data":self.data, "k":self.k})


points = [point(d) for d in data]

def make_k_mapping(points):
    point_dict = defaultdict(list)
    for p in points:
        point_dict[p.k] = point_dict[p.k] + [p.data]
    return point_dict
def calc_k_means(point_dict):
    means = [np.mean(point_dict[k],axis=0) for k in range(K)]
    return means
def update_k(points,means):
    for p in points:   
        dists = [np.linalg.norm(means[k]-p.data) for k in range(K)]
        p.k = np.argmin(dists)

def fit(points, epochs=10):
    for e in range(epochs):
        point_dict = make_k_mapping(points)
        means = calc_k_means(point_dict)
        update_k(points, means)
    return means, points

  

if __name__ =="__main__":

  new_centroids, new_points = fit(points)
  x = [i.data[0] for i in new_points]
  y = [i.data[1] for i in new_points]


  x_centroids = [i.data[0] for i in new_centroids]
  y_centroids = [i.data[1] for i in new_centroids]
  print(new_points)
  plt.scatter(x,y)
  plt.scatter(x_centroids,y_centroids,c='red')

  plt.show()
