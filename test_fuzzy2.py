# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:40:29 2017

@author: albertnava
"""
# %%

import copy
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import itertools 
import pso_tofuzzy
from sklearn.metrics import mean_squared_error


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

# Function to find the fuzzy group of a fuzzy set before prediction
def find_grp(seq,x):
    for sublist in seq:
        if sublist[0] == x:
            return seq.index(sublist)
    #        yind = seq.index(sublist)
    #    if sublist[0] != x:
    #        yind = -1
    #return yind

# Function to find the mid-point of the group of trapezoidal fuzzy intervals
def defuz_midpoint(fgrl):
    t_v = len(fgrl)
    if t_v > 1:
        midp = 0
        for item in fgrl:
            tmp = np.mean([item[1],item[2]])
            midp += tmp
        midp = midp/t_v
    else:
        midp = np.mean([fgrl[0][1],fgrl[0][2]])
        
    return int(midp)

# Function to get a list with the intervals for any element of FLGRn
def get_trp_funs(intvs,grp):
    bsgr = []
    for n in range(1,len(grp)):
        bsgr.append(intvs[grp[n]])
    
    return bsgr

# Function to get the minimum degree of duplicate fuzzyfied elements (element, Ai)
def get_max_mf(gropued):
    tamDF = len(gropued)
    degmf = []
    #print 'Gruops of duplicates', gropued
    for i in range(tamDF):
        temtr = trap(gropued.iloc[i][2],gropued.iloc[i][1])
        degmf.append(temtr)
    #valmax = np.max(degmf)
    valmax = np.min(degmf)    
    indxt = degmf.index(valmax)
    max_ind = gropued.iloc[indxt][0]

    return max_ind

# Function to find duplicate values 
# It returns the all values that are the same and their indexes
def check_dupl(vectr): 
    Not_rep = []
    ind_r = []

    for i, item in enumerate(vectr):
        if item not in Not_rep:
            Not_rep.append(item)
        else:
            ind_r.append(i)

    for i in ind_r:
        a = vectr[i]
        if a in vectr:
            k = vectr.index(a)
            if k not in ind_r:
                ind_r.append(k)
    ind_r = sorted(ind_r)

    return Not_rep, ind_r

# Function to eliminate duplicate values in the list
def red_grp (lsts, inds):
    mod = []
    copa = copy.deepcopy(lsts)
    for item in inds:
        now = copa[item]
        bef = lsts[item-1][0]
        now.insert(0,bef)
        mod.append(now)

    for k in range(len(inds)):
        lsts[inds[k]] = mod[k]

    return lsts

# Function to compute the Mean absolute percentage error 
def MAPE(y_true, y_pred): # If y_true has a zero value then will be an error

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


    
# Information about arrivals to New Zeland through the years 2000-2013 
dates = [1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985,\
1986, 1987, 1988, 1989, 1990, 1991, 1992]
students = [13055, 13563, 13867, 14696, 15460, 15311, 15603, 15861, 16807, 16919, 16388, 15433, \
15497, 15145, 15163, 15984, 16859, 18150, 18970, 19328, 19337, 18876]

Df= pd.DataFrame({ 'Enrollment': students, 'Year' : dates }, columns = ['Year', 'Enrollment'] )    

# Sorting the data in ascending order
new_stud = np.sort(students)
Dmax = max(students)
Dmin = min(students)
print 'Maximum data value', Dmax 
print 'Minimum data value', Dmin

l_tam = len(dates)
print 'Length of dates', l_tam

# Computing average distance between data sorted
dist = []
dist2 = []
sum1 = 0
sum1a = 0
for i in range(l_tam-1):
    tmp = np.abs(new_stud[i]-new_stud[i+1])
    tmp1 = np.abs(students[i+1]-students[i])
    sum1a = sum1a + tmp1
    sum1 = sum1 + tmp
    dist2.append(tmp1)
    dist.append(tmp)
AD = sum1/(l_tam - 1)
Range = sum1a/(2*(l_tam-1))
print 'Average Distance 1', Range
print 'Average Distance 2', AD

# Computing standard deviation
sum2 = 0
for j in range (l_tam-1):
    tm = (dist[j]-AD)**2
    sum2 = sum2 + tm
sAD = np.sqrt(sum2 / l_tam)
print sAD

best_dist = []
#sum3 = 0
for k in range(l_tam-1):
    if (dist[k]>= (AD-sAD) and dist[k]<= (AD+sAD)):
        #sum3 = sum3 + dist[k]
        best_dist.append(dist[k])
        #print dist[k]
ADr = round(np.mean(best_dist),0)
print 'New average distance', ADr
# Limits of universe U = [Dmin-AD,Dmax+AD]
l_inf = Dmin - ADr
l_sup = Dmax + ADr
print 'New inferior limit', l_inf
print 'New superior limit', l_sup
rang = np.arange(l_inf,l_sup)
# Number of sets of U, n = (R-S)/2S 
R = l_sup - l_inf
S = 190  
n_i = round((R-S)/(2*S),0)
A_names = [str(i) for i in np.arange(n_i)] 
n_p = (2*n_i)

print 'Number of intervals  =', n_i # The paper uses this size of intervals
#m = R/Range
#print 'Number of intervals 2 = ', m

nl_inf = l_inf + ADr
nl_sup = l_sup - ADr

#Create the intervals of U
U_s = []
U_l = []
U_t = []
for i in range(1,int(n_p)):
    U_l.append(nl_inf+(i-1)*S)  
    if len(U_l) == (int(n_p)-1):
        U_l.append(nl_inf+i*S)  
    U_s.append(nl_inf+i*S)
    
U_l.append(l_sup)
U_l.insert(0,l_inf)

# Create a list of lists with parameters to form trapezoidal MF
Anew = []
x = []
for i in range(len(U_l)):
    x.append(U_l[i])
    if len(x)==4:
        tmp = x[:]
        Anew.append(tmp)
        del x[0]
        del x[0]
# The list A is the intervals used in the paper. TODO How to compute these intervals? 
A = [[12861,13055,13245,13436], [13245,13436,13626,13816],[13626,13816,14007,14197], \
[14007,14197,14388,14578], [14388,14578,14768,14959], [14768,14959,15149,15339], \
[15149,15339,15530,15720], [15530,15720,15910,16101], [15910,16101,16291,16482], \
[16291,16482,16672,16862], [16672,16862,17053,17243], [17053,17243,17433,17624], \
[17433,17624,17814,18004], [17814,18004,18195,18385], [18195,18385,18576,18766], \
[18576,18766,18956,19147], [18956,19147,19337,19531]]

# Fuzzyfy the data set using the trapezoid fuzzyfication approach
Fzytion_num = []    
Fzytion_set = []
Fzytion_val = []
for val in students:
    for p in range(len(Anew)):
        xset = is_in(val,A[p])
        #xset = is_in(val,Anew[p])
        if xset == 1:
            Fzytion_set.append(int(float(A_names[p]))) # Assigment of fuzzy set Ai
            Fzytion_val.append(val) # Values of the fuzzy set Ai
            #Fzytion_num.append(Anew[p])
            Fzytion_num.append(A[p])
        
print (Fzytion_val)                    
print (Fzytion_set)
print (len(Fzytion_set))
print ''

# Reduce the list of values by checking their membership function value
itx = list(range(len(Fzytion_val))) # Indexes just to check wich rows will be keep
col_names = ['indc', 'estd', 'fuzznum', 'Aip']
# Just a DataFrame to make it more visible
setg = pd.DataFrame({'estd':Fzytion_val,'indc':itx, 'fuzznum':Fzytion_num, 'Aip':Fzytion_set}, columns = col_names)
g = setg.groupby(['estd']).filter(lambda x : len(x)>1)
g = g.groupby('estd') # Just to get the groups with duplicate values


idx_d = [] # Indexes wich have higher degree of membership function
for name,group in g:
    fizt = get_max_mf(group)
    #print  group
    idx_d.append(fizt)

nidx = [item for item in itx if item not in idx_d] 
Nsetg = setg[setg.index.isin(nidx)].copy(deep=True) # Final Data frame with unique elements of students (is not the same
# as the values of the paper because of the partition is a little bit different)
#Nsetg = Nsetg.reset_index(drop=True) # After remove some rows the index list changed so it is necessary to "reindex"
Nsetg.index = pd.RangeIndex(len(Nsetg.index))
nlen = len(Nsetg['Aip'])
print 'New index'
print (Nsetg['Aip'].values.tolist())


# Create a list of first order relationships First_or = [[Ai,Aj], [Aj,Ak], ....]
n = 2 # Number of order 
cs = 1 # For 1st order
st_or = list (Nsetg['Aip'][k:k+n].values.tolist() for k in range (0, len(Nsetg['Aip'])-1, cs))
print 'Fisrt FSGs'
print st_or

First_or, indx = check_dupl(st_or)
band = True
cont = 0
while ( band and cont < 5):
    st_or = red_grp(st_or, indx)
    First_or, indx = check_dupl(st_or)
    if (indx == []):
        band = False
        print 'Indexes is empy'
    cont += 1

print 'Fuzzy set groups'
print st_or

# The paper says the if rules statement are in reverse order of the fuzzy set groups
ifstg = copy.deepcopy(st_or)
for i,item in enumerate(ifstg):
    item = item[::-1]
    ifstg[i] = item
print ifstg
    
# The process of "Deffuzyfication" by calculating the weights using PSO and the "if statements"
pop_size = 5 # Number of particles
search_s = [0, 1] # Limits of position of a square region
vel_s = [-0.01,0.01] # Limits of velocity of a square region
max_gens = 500 # Number of iterations
max_vel = 0.1 # Maximum velocity of a particle
c1, c2 = 2.0, 2.0 # Constants to update the velocity
w_mega = 1.4

weights = []
datpso = []
for k in range(nlen):
    tmo = len(ifstg[k])
    m = k+2
    if m == nlen:
        print 'Terminado'
        break
    vepso = []
    nelem = Nsetg['estd'][m]
    for i in range(tmo):
        temel = Nsetg['estd'][m-(i+1)]
        vepso.append(temel)
    datpso.append(vepso)
    best = pso_tofuzzy.search(max_gens, search_s, vel_s, pop_size, max_vel, c1, c2, w_mega, nelem, vepso)
    weights.append(best['position'])

# Once the weights are ready just multiply these with the past values of the current value
Fors_val = []
for x,y in zip(weights, datpso) :
    C = sum(np.multiply(x, y))
    Fors_val.append(C)
print Fors_val
print len(Fors_val)

# Computing the MSE (Mean squared error) and MAPE (Mean absolute percentage error)

mse = mean_squared_error(Nsetg['estd'][2:nlen], Fors_val)
mape = MAPE(Nsetg['estd'][2:nlen], Fors_val)


props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.figure()
plt.plot(Nsetg['estd'][2:nlen], marker = '^', linestyle = '-', color = 'r', label = 'Original')
plt.plot(Fors_val, marker = '*', linestyle = '--', color = 'b' , label = 'Forecasted')
plt.figtext(0.75, 0.3, 'MAPE = {} \n MSE = {}'.format(mape,mse), fontdict =  None,
            horizontalalignment = 'center', fontsize = 12, bbox = props)
plt.legend(loc='upper left')
plt.xlabel('Dates')
plt.ylabel('Enrollment')
plt.title('Fuzzy logic for enrollment students')
plt.xticks(range(len(dates)), dates)
plt.xticks(range(len(dates)), dates, rotation=45) #writes strings with 45 degree angle
plt.grid()
plt.show()