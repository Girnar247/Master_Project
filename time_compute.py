from tqdm import tqdm
import time


t1 = time.time()
print(t1)

# for i in range(10000000):
#     a = i**2

for i in tqdm(range(10000000)):
    a = i**2
    
t2 = time.time()
print(t2)

print('Time diff: ', (t2-t1))

#
