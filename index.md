# Lowess Regression: Gradient Boosting
:  Anjali Goel

We use gradient boosting to help us improve models. It is performed by calculating the difference between the calculated and predicted output of a model. This is done over many trials and helps us to create a more accurate model.

To do gradient boosting, we must define the following kernels.

``` python
# Gaussian Kernel
def Gaussian(x):
	if len(x.shape)==1:
	    d = np.abs(x)
	else:
	    d = np.sqrt(np.sum(x**2,axis=1))
	return np.where(d>4,0,1/(np.sqrt(2*np.pi))*np.exp(-1/2*d**2))

# Tricubic Kernel
def Tricubic(x):
	if len(x.shape) == 1:
	    x = x.reshape(-1,1)
	d = np.sqrt(np.sum(x**2,axis=1))
	return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
	if len(x.shape) == 1:
	    x = x.reshape(-1,1)
	d = np.sqrt(np.sum(x**2,axis=1))
	return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
	if len(x.shape) == 1:
	    x = x.reshape(-1,1)
	d = np.sqrt(np.sum(x**2,axis=1))
	return np.where(d>1,0,3/4*(1-d**2))

#kernels are used to group together points for local regression
```
```python
#distance function to calculate distance between expected and calculated
def dist(u,v):
	if len(v.shape)==1:
	    v = v.reshape(1,-1)
	d = np.array([np.sqrt(np.sum((u-v[i])**2,axis=1)) for i in range(len(v))]) 
	return d
```

``` python
def lowess_with_xnew(x, y, xnew,f=2/3, iter=3, intercept=True, qwerty = 6, pcaComp = 2, alpha = 0.001)

#intialize
n = len(x)
r = int(ceil(f * n))
yest = np.zeros(n)

#reshape into matricies
if len(y.shape)==1: 
    y = y.reshape(-1,1)

if len(x.shape)==1: 
    x = x.reshape(-1,1)
  
if intercept: 
	#column of ones tacked onto matrix
    x1 = np.column_stack([np.ones((len(x),1)),x])
else:  
    x1 = x
```

```python
#bounds
h = [np.sort(np.sqrt(np.sum((x-x[i])**2,axis=1)))[r] for i in range(n)]
w = (1 - (np.clip(dist(x,x) / h, 0.0, 1.0)) ** 3) ** 3
```

```python
delta = np.ones(n) #square matrix filled with ones the same length as n
  for iteration in range(iter): #loops through based on how many times we specified it to cut the outliers
    for i in range(n): #loops trhough every observation and removes outliers
      W = np.diag(delta).dot(np.diag(w[i,:])) #this is the weights for removing values
      #because w is symmetric, switching the rows and columns wont matter
      #when we multiply two diag matrices we get a diag matrix
      b = np.transpose(x1).dot(W).dot(y)
      A = np.transpose(x1).dot(W).dot(x1)
      #prediction algorithms
      A = A + alpha*np.eye(x1.shape[1]) # if we want L2 regularization (ridge)
      beta = linalg.solve(A, b) #beta is the "solved" matrix for a and b between the independent and dependent variables
      yest[i] = np.dot(x1[i],beta) #set the y estimated values

    residuals = y.ravel() - yest #calculate residuals
    #you have to ravel y here because the shape of y is (330, 1) and y is (330, ), this is a subtle error that can happen within python
    #if you subtract a vector from a normal array python returns a square matrix, which is not what we want
    #print(y.shape) 
    #print(yest.shape) 
    s = np.median(np.abs(residuals)) #median of the residuals
    delta = np.clip(residuals / (qwerty * s), -1, 1) #calculate the new array with cut outliers 
    #print(delta.shape)
    delta = (1 - delta ** 3) ** 3 #assign more importance to observations that gave you less errors and vice versa
```
```python
#PCA
if x.shape[1]==1:
    f = interp1d(x.flatten(),yest,fill_value='extrapolate')
    output = f(xnew)
  else:
    output = np.zeros(len(xnew))
    for i in range(len(xnew)):
      ind = np.argsort(np.sqrt(np.sum((x-xnew[i])**2,axis=1)))[:r]
      #if you dont extract principle components, you would get an infinite loop
      #use delaunay triangulation 
      pca = PCA(n_components=pcaComp)
      x_pca = pca.fit_transform(x[ind])
      tri = Delaunay(x_pca,qhull_options='QJ')
      f = LinearNDInterpolator(tri,yest[ind])
      output[i] = f(pca.transform(xnew[i].reshape(1,-1))) # the output may have NaN's where the data points from xnew are outside the convex hull of X
  if sum(np.isnan(output))>0:
    g = NearestNDInterpolator(x,y.ravel()) 
    # output[np.isnan(output)] = g(X[np.isnan(output)])
    output[np.isnan(output)] = g(xnew[np.isnan(output)])
  return output
```
### Finally, here is our boosted function.
```python
def boosted_lowess(x, y, xnew, f=1/3,iter=2,intercept=True, qwerty = 6, pcaComp = 2, alpha = 0.001):
  # we need decision trees
  # for training the boosted method we use x and y
  model1 = Lowess_AG_MD(f=f,iter=iter,intercept=intercept, qwerty = qwerty, pcaComp = pcaComp, alpha = 0.001) # we need this for training the Decision Tree
  model1.fit(x,y)
  residuals1 = y - model1.predict(x)
  model2 = Lowess_AG_MD(f=f,iter=iter,intercept=intercept, qwerty = qwerty, pcaComp = pcaComp, alpha = 0.001)
  model2.fit(x,residuals1)
  output = model1.predict(xnew) + model2.predict(xnew)
  return output
```
