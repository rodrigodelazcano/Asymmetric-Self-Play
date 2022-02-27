import time

x = {"env_id": [3]}

start = time.perf_counter()
y = x.get("env_id").pop() if x.get("env_id") else 1
end = time.perf_counter()
print(y)
duration = end - start
print('TIME: ', duration)