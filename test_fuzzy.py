# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:40:29 2017

@author: albertnava
"""
# %%

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import itertools 

# %%
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
    
# %% I decided to work with the data here, maybe is necessary to add a function
#    to read any document with the help of pandas

# Information about arrivals to New Zeland through the years 2000-2013 

dates = [1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985,\
1986, 1987, 1988, 1989, 1990, 1991, 1992]
students = [13055, 13563, 13867, 14696, 15460, 15311, 15603, 15861, 16807, 16919, 16388, 15433, \
15497, 15145, 15163, 15984, 16859, 18150, 18970, 19328, 19337, 18876]

test_d = pd.Series(students)


Df= pd.DataFrame({ 'Enrollment': students, 'Year' : dates }, columns = ['Year', 'Enrollment'] )    

# Sorting the data in ascending order
new_stud = np.sort(students)
Dmax = max(students)
Dmin = min(students)
print 'Maximum data value', Dmax 
print 'Minimum data value', Dmin

l_tam = len(dates)
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
print 'Average Distance', AD

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
A_names = [str(i) for i in np.arange(n_i)] # There is no error of np.arange
n_p = (2*n_i)


print 'Number of intervals 1 =', n_i 
m = R/Range
print 'Number of intervals 2= ', m

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
        #print "appended"
    U_s.append(nl_inf+i*S)
    #U_t.append((nl_inf+i*S)-(nl_inf+(i-1)*S))

#print U_l
#print len(U_l)

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
#print Anew    
#print len(Anew)
A = [[12861,13055,13245,13436], [13245,13436,13626,13816],[13626,13816,14007,14197], \
[14007,14197,14388,14578], [14388,14578,14768,14959], [14768,14959,15149,15339], \
[15149,15339,15530,15720], [15530,15720,15910,16101], [15910,16101,16291,16482], \
[16291,16482,16672,16862], [16672,16862,17053,17243], [17053,17243,17433,17624], \
[17433,17624,17814,18004], [17814,18004,18195,18385], [18195,18385,18576,18766], \
[18576,18766,18956,19147], [18956,19147,19337,19531]]

# Create all trapezoidal membershipfunctions
#fuzz_set_A = ctrl.Consequent(rang,'universe')
#for k in range(int(n)) :
#    fuzz_set_A[A_names[k]] = fuzz.trapmf(fuzz_set_A.universe,Anew[k])
#fuzz_set_A.view()
    
# Fuzzyfy the data set using the trapezoid fuzzyfication approach
Fzytion_num = []    
Fzytion_set = []
Fzytion_val = []
for val in students:
    for p in range(len(Anew)):
        #xset = is_in(val,A[p])
        xset = is_in(val,Anew[p])
        if xset == 1:
            Fzytion_set.append(int(float(A_names[p]))) # Assigment of fuzzy set Ai
            Fzytion_val.append(val) # Values of the fuzzy set Ai
            Fzytion_num.append(Anew[p])
            #Fzytion_num.append(A[p])
        
vc = pd.Series(Fzytion_val)
df =vc[vc.duplicated(keep=False)]
dfp  = df.tolist() # List of duplicate values
inx_d = df.index.tolist() # List of indexes of duplicate values
#print (Fzytion_val)                    
#print (Fzytion_set)
#print (dfp)
#print (inx_d)
print ''

# Elimiar uno de los valores que se repitan, al considerar el que tiene el mayor grado en
# su funcion mu

itx = list(range(len(Fzytion_val))) # Indexes just to check wich rows will be keep
col_names = ['indc', 'estd', 'fuzznum', 'Aip']
setg = pd.DataFrame({'estd':Fzytion_val,'indc':itx, 'fuzznum':Fzytion_num, 'Aip':Fzytion_set}, columns = col_names)
#print setg.head(7)
g = setg.groupby(['estd']).filter(lambda x : len(x)>1)
g = g.groupby('estd') # Just to get the groups with duplicate values


idx_d = [] # Indexes wich have higher degree of membership function
for name,group in g:
    fizt = get_max_mf(group)
    #print  group
    idx_d.append(fizt)
#print idx_d

nidx = [item for item in itx if item not in idx_d] 
Nsetg = setg[setg.index.isin(nidx)].copy(deep=True) # Final Data frame with unique elements of students (is not the same
# as the values of the paper because of the partition is a little bit different)
#print len(Nsetg)
#print 'New index'
#print nidx

# %%
# Create a list of first order relationships First_or = [[Ai,Aj], [Aj,Ak], ....]
n = 2 # Number of order 
cs = 1 # For 1st order
st_or = list (Nsetg['Aip'][k:k+n].values.tolist() for k in range (0, len(Nsetg['Aip'])-1, cs))

#for j,a in enumerate(st_or):
#    print j,a 
st_or.sort()
First_or = list(k for k,_ in itertools.groupby(st_or))
#print First_or
#print len(First_or)
# Create the list of the First Group Relationships FLGRn
FLGRs = [] # List of groups (indexes) with the same initial set
ukeys = [] # List of initial set (index)
for k,g in itertools.groupby(First_or, lambda x : x[0]):
    #print list(g)    
    FLGRs.append(list(g))
    ukeys.append(k)

FLGRn = [] 
for i,e in zip(ukeys,FLGRs):
    tmp = []
    tmp.append(i)
    #print (i,e)
    for it in e:
        tmp.append(it[1])
    FLGRn.append(tmp)

#print FLGRn



# %%
# The process of "Deffuzyfication" by calculating the centroid 

ai_g = [] # Just a vector to keep the indexes of FLGRn for each Ai
aine = [0] # Vector for values predicted
for item in Nsetg['Aip']:
    gii = find_grp(FLGRn,item)
    if gii != None:
        gind = FLGRn[gii]
        ai_g.append(gii)
        intervs = get_trp_funs(Anew,gind)
    else:
        gind = [Anew[item]]
        ai_g.append(item)
        print 'Value whitout group',item, gind
        intervs = gind
    #print 'Calculate midpoint for value', item, gind
    #intervs = get_trp_funs(Anew,gind)
    midpoint = defuz_midpoint(intervs)
    aine.append(midpoint)
print ''
print aine

