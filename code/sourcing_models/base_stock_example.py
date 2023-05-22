import numpy as np
import matplotlib.pyplot as plt
from lib.single_sourcing import *

np.random.seed(10)

l = 2
h = 5
b = 495
r = 10
T = 1000

S = SingleSourcingModel(l=l, 
                        h=h, 
                        b=b,
                        T=T, 
                        I0=r,
                        r=r,
                        optimal_base_stock=True)

S.simulate()  

print("optimal cost:", S.optimal_cost)
print("average cost (base stock):", S.total_cost/T)

plt.figure()
plt.plot(S.inventory, '-o', label = r"inventory")
plt.plot(S.inventory_position, '-o', label = r"inventory position")
plt.plot(S.demand, '-o', label = r"demand")
plt.xlabel(r"time")
plt.ylabel(r"value")
plt.legend(loc = 4, ncol = 3)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(S.q, '-o', label = r"order")
plt.xlabel(r"time")
plt.ylabel(r"value")
plt.legend(loc = 4, ncol = 3)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(S.cost, '-o', label = r"cost")
plt.xlabel(r"time")
plt.ylabel(r"cost")
plt.legend(loc = 4, ncol = 3)
plt.tight_layout()
plt.show()
