import numpy as np 
import math 
layer_outputs=[4.7,2.5,1.9]
E=math.e
exp_values=[]
for output in layer_outputs:
    exp_values.append(E**output)
    
print(exp_values)
norm_base=sum(exp_values)
norm_values=[]

for exp_value in exp_values:
    norm_values.append(exp_value/norm_base)

print(norm_values)
print(sum(norm_values))
print(norm_base)