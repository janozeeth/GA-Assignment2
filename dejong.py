import numpy as np

def dejong1(x):
    return np.sum([x[i]**2 for i in range(len(x))])
 
def dejong2(x):
    return np.sum([100*(x[i+1]-x[i]**2)**2 + (x[i]-1)**2 for i in range(len(x)-1)])
 
def dejong3(x):
    return 6*len(x) + np.sum([np.floor(x[i]) for i in range(len(x))])
 
def dejong4(x):
    return np.sum([(i+1)*x[i]**4 for i in range(len(x))]) + np.random.normal(0,1)
 
def dejong5(x):
    a = np.zeros((2,25))
    a[0] = [-32,-16,0,16,32]*5
    a[1] = [-32]*5 + [-16]*5 + [0]*5 + [16]*5 + [32]*5
    return 1./(1./500 + np.sum([1./(i + (x[0]-a[0,i])**6 + (x[1]-a[1,i])**6) for i in range(25)]))
