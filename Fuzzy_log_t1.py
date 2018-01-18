import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import itertools 

# Membership functions for fuzzy logic (except Gaussian)

# Function to get the degree of a value x in a incremental ramp fuzzy interval
def up(a, b, x):
    a = float(a)
    b = float(b)
    x = float(x)
    if x < a:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    return 1.0

# Function to get the degree of a value x in a decremental ramp fuzzy interval
def down(a, b, x):
    return 1. - up(a, b, x)

# Function to get the degree of a value x in a triangular fuzzy interval
def tri(a, b, x):
    a = float(a)
    b = float(b)
    m = (a + b) / 2.
    first = (x - a) / (m - a)
    second = (b - x) / (b - m)
    return max(min(first, second), 0.)

# Function to get the degree of a value x in the trapezoidal fuzzy interval
def trap(A, x):
    
    try:
        if len(A)==4 :
            a = float(A[0])
            b = float(A[1])
            c = float(A[2])
            d = float(A[3])    
            
            m1 = 1.0/(b-a)
            b1 = -a/(b-a)
            m2 = -1.0/(d-c)
            b2 = d/(d-c)
            if a<= x <= b :
                val = float((m1*x + b1))
                #print 'First ramp'
            if b<= x <= c :
                val = float(1)
                #print 'Flat'
            if c<= x <= d :
                val = float((m2*x+b2))
                #print 'Second ramp'
            if (x<a or x>d):
                val = 0.0
                print 'Aqui caigo', A, x
            #print 'Trapezoidal degree', x
            return val
    except:
        
        print 'Vector must have 4 elements', x, A
    
    #first = (x - a) / (b - a)
    #second = (d - x) / (d - c)
        

# Function to find if the value belongs to a trapezoidal interval
def is_in(val, inter): 
    lim_i = inter[0]
    lim_s = inter[-1]
    if val >=lim_i and val <= lim_s:
        return 1
    else:
        return 0

# Function to find the mid-point of fuzzy interval
def midpoint(intva):
    t_v = len(intva)
    if t_v < 2:
        print 'The length of vector interval is wrong'
    else:
        midp = np.mean(intva)
        
    return int(midp)

def calc_favg(vetv):
    t_ve = len(vetv)
    prom = []
    for m in range(t_ve-1):
        if m == (t_ve-1):
            z = vetv[t_ve]-vetv[t_ve-1]
            prom.append(z)
        else:
            d = vetv[m]-np.mean(vetv[m+1:t_ve])
            prom.append(d)

    favg = np.mean(prom)
    
    return favg


# I decided to work with the data here, maybe is necessary to add a function
#    to read any document with the help of pandas

# Information about arrivals to New Zeland through the years 2000-2013 

dates = [1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992,\
1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000]
students = [3552, 4177, 3372, 3455, 3702, 3670, 3865, 3592, 3222, 3750, 3851,\
3231, 4170, 4554, 3872, 4439, 4266, 3219, 4305, 3928]

Df= pd.DataFrame({ 'Enrollment': students, 'Year' : dates }, columns = ['Year', 'Enrollment'] )    


# Sorting the data in ascending order
#new_stud = np.sort(Df.iloc[:,1])
new_stud = Df.iloc[:,1]
Dmax = max(Df.iloc[:, 1])
Dmin = min(Df.iloc[:, 1])
print 'Maximum data value', Dmax 
print 'Minimum data value', Dmin

l_tam = len(Df.iloc[:,0])
print 'Length of dates', l_tam

# Computing average distance between data sorted
dist = []
sum1 = 0
for i in range(l_tam-1):
    tmp = np.abs(Df.iloc[i+1,1]-Df.iloc[i,1])
    sum1 = sum1 + tmp
    dist.append(tmp)
AD = sum1/(l_tam - 1)
print 'Average Distance 2 =', AD

# Computing standard deviation
dist_mean = np.mean(dist)
print dist_mean

best_dist = []

for k in range(l_tam-1):
    if (dist[k]>= (AD-dist_mean) and dist[k]<= (AD+dist_mean)):
        best_dist.append(dist[k])
        
ADr = round(np.mean(best_dist),0)
print 'New average distance', ADr
# Limits of universe U = [Dmin-AD,Dmax+AD] normally
nl_inf = Dmin - ADr
nl_sup = Dmax + ADr
rang = np.arange(nl_inf,nl_sup)

print 'New inferior limit', nl_inf
print 'New superior limit', nl_sup

#Create the intervals of U
n_p = 7 # Number of intervals
int_sz = round ((nl_sup - nl_inf)/n_p) # Size of interval
print 'Size of interval {}'.format(int_sz)
A_names = [str(i) for i in np.arange(n_p)]
U_l = [nl_inf]

for i in range(1,int(n_p)):
    U_l.append(nl_inf+(i)*int_sz)  
    if len(U_l) == (int(n_p)-1):
        U_l.append(nl_inf+i*int_sz)  
    
U_l.append(nl_sup)
print U_l

# Create a list of lists with the intervals 
n_elem = 2
A_trap = []
x = []
for i in range(len(U_l)):
    x.append(U_l[i])
    if len(x)==n_elem:
        tmp = x[:]
        A_trap.append(tmp)
        del x[0]
print A_trap
 
# Fuzzyfy the data set using the trapezoid fuzzyfication approach
Fzytion_num = []    
Fzytion_set = []
Fzytion_val = []

for val in Df.iloc[:,1]:
    for p in range(len(A_trap)):
        xset = is_in(val,A_trap[p])
        if xset == 1:
            Fzytion_set.append(int(float(A_names[p]))) # Assigment of fuzzy set Ai
            Fzytion_val.append(val) # Values of the fuzzy set Ai
            Fzytion_num.append(A_trap[p])
        
vc = pd.Series(Fzytion_val) # To find some duplicates
df =vc[vc.duplicated(keep=False)]
dfp  = df.tolist() # List of duplicate values
inx_d = df.index.tolist() # List of indexes of duplicate values
print (Fzytion_val)                    
print (Fzytion_set)
print ''

# Create a list of Fvag values 
favg_e = 4
f_trap = []
xt = []
for i in range(len(Df.iloc[:,1])):
    xt.append(Df.iloc[i,1])
    if len(xt)==favg_e:
        tmp = xt[:]
        f_trap.append(tmp)
        del xt[0]

favg_val = [0, 0, 0, 0]
n_st = 3
dat = [0, 0, 0]
for k in range(n_st,len(Df.iloc[:,1])+1):
    x =  Df.iloc[:k,1].tolist()
    tpl = calc_favg(x)
    dat.append(tpl)


# Defuzzification part 