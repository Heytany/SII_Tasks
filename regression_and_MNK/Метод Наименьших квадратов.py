#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import math as m


def Kmt_method_test(X,Y):
    
    if (isinstance(X, np.ndarray)==True) | (isinstance(X, list)==True):
        FirstP=True
        
    if (isinstance(Y, np.ndarray)==True) | (isinstance(Y, list)==True):
        SecondP=True
    
    
    if  ((FirstP==False) & (SecondP==False)) | ((FirstP==False) | (SecondP==False)):
        print("Данная функция принимает как параметры только списки или массивы библиотеки Numpy")
        return [0,0]
    else:
      
        q=0; w=0;
        e=0;t=0;
        a=0;b=0;u=0;  
        
        if len(X)!=len(Y):
            print("Размеры передаваемых листов должны быть одинаковыми")
            return [0,0] 
        else:
            
            N=len(Y)
        
            try:
                for i in range(0,N):
                    q+=N*X[i]*Y[i]
                    w+=X[i]
                    e+=Y[i]
    
                r=q-w*e

                for i in range(0,N):
                    t=t+N*(X[i]**2)
                    u=u+X[i]
    
                u=(u**2)   

                for i in range(0,N):
                    a=r/(t-u)
                    b=1/N*(e-a*w)
    
                print("a  = ",a)
                print("b  = ",b)
    
                return [a,b]
            except:
                print("Значения хранимые в передаваемом листе должны быть числовыми и/или дробными")
                return [0,0]
        


# In[2]:



import pandas as pd
import numpy as np
data = pd.read_csv('HousePrices.csv')
vievs = np.array(data['X'])
downloads = np.array(data['Y'])




Z=Kmt_method_test(vievs,downloads)

#xx=np.linspace(0,8000,100)
xx = vievs
yy=Z[0]*xx+Z[1]

SomeSubPlot = plt.subplot(111)
SomeSubPlot.plot(xx,yy,'r', label=' Метод наименьших квадратов')
SomeSubPlot.plot(vievs,downloads,'*', label=' Рассеивание')
plt.ylabel(u'Просмотры')
plt.xlabel(u'Загрузки')
plt.title(u'МНК метод')

chartBox = SomeSubPlot.get_position()
SomeSubPlot.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])
SomeSubPlot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=1)
plt.grid(True)
plt.show()

print("____________________________________________________________________________________________________________________")
def colvec(rowvec):
    v = np.asarray(rowvec)
    return v.reshape(v.size,1)

#vievs = [5252,7620,941,1159,485,299,239,195,181,180]
#downloads = [21,46,9,8,3,6,4,2,2,2]

vievs = colvec(vievs)
downloads = colvec(downloads)



regr = LinearRegression()
regr.fit(vievs, downloads)
         
yy = regr.predict(vievs)

SomeSubPlot = plt.subplot(111)
SomeSubPlot.plot(vievs,yy,'r', label=' Метод наименьших квадратов')
SomeSubPlot.plot(vievs,downloads,'*', label=' Рассеивание')

plt.ylabel(u'Просмотры')
plt.xlabel(u'Загрузки')
plt.title(u'МНК метод - встроенные функции')


chartBox = SomeSubPlot.get_position()
SomeSubPlot.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])
SomeSubPlot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=1)
plt.grid(True)
plt.show()

print("____________________________________________________________________________________________________________________")
S=50
Cor=0.08
vievs=np.random.normal(500,4000,size=S)*Cor
downloads=np.random.normal(25,50,size=S)*Cor

for i in range(0,S):
    vievs[i]=m.fabs(vievs[i]) 
    downloads[i]=m.fabs(downloads[i]) 


Z=Kmt_method_test(vievs,downloads)

#xx=np.linspace(0,8000,100)
xx = vievs 
yy=Z[0]*xx+Z[1]

SomeSubPlot = plt.subplot(111)
SomeSubPlot.plot(xx,yy,'r', label=' Метод наименьших квадратов + Сглаженная случайная генерация по Гауссу')
SomeSubPlot.plot(vievs,downloads,'*', label=' Рассеивание')
plt.ylabel(u'Просмотры')
plt.xlabel(u'Загрузки')
plt.title(u'МНК метод - встроенные функции')

chartBox = SomeSubPlot.get_position()
SomeSubPlot.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])
SomeSubPlot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=1)
plt.grid(True)
plt.show()


# ## Далее идут просто примеры из справочников по регрессии и нампаю

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

N = 100     # число точек
sigma = 5   # стандартное отклонение 
k = 0.8    
b = 15      

x = np.array(range(N))
print(x)
f = np.array([k*z+b for z in range(N)])
y = f + np.random.normal(0, sigma, N)

# вычисляем коэффициенты
mx = x.sum()/N
my = y.sum()/N
a2 = np.dot(x.T, x)/N
a1 = np.dot(x.T, y)/N
print(a1)

kk = (a1 - mx*my)/(a2 - mx**2)
bb = my - kk*mx
ff = np.array([kk*z+bb for z in range(N)])

plt.plot(x, y,'*')
#plt.plot(f, c='green')
plt.plot(ff, c='red')
plt.grid(True)
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

N = 100     # число точек
sigma = ([3, 6, 2])   # стандартное отклонение 
k = ([0.5, 0.2, 0.1])  
b = ([2, 13, 9])      

x = np.array(range(N))
for i in range(len(k)):
  f = np.array([k[i]*z+b[i] for z in range(N)])
  y = f + np.random.normal(0, sigma[i], N)

# вычисляем коэффициенты
  mx = x.sum()/N
  my = y.sum()/N
  a2 = np.dot(x.T, x)/N
  a1 = np.dot(x.T, y)/N

  kk = (a1 - mx*my)/(a2 - mx**2)
  bb = my - kk*mx
  ff = np.array([kk*z+bb for z in range(N)])

  plt.plot(x, y,'*')
  #plt.plot(f, c='green')
  plt.plot(ff, c='red') 
  plt.grid(True)
  plt.show()


# In[ ]:




