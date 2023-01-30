import collections

a = collections.deque(maxlen=5)
for i in range(0,10):
    a.append(i)
    print(i)

print(list(a)[1:])