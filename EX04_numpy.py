import numpy as np
person_type = np.dtype({
	'names':['name', 'Chinese', 'English', 'Math'],
	'formats':['S32', 'i', 'i', 'i']})
people = np.array([("Zhangfei", 66, 65, 30), ("GuanYu", 95, 85, 98), ("ZhaoYun", 93, 92, 96), ("HuangZhong", 90, 88, 77), ("DianWei", 80, 90, 90)], dtype=person_type)

chinese = people[:]['Chinese']
math = people[:]['Math']
english = people[:]['English']

print("Chinese Average:", np.mean(chinese))
print("Math Average:", np.mean(math))
print("English Average:", np.mean(english))

print("Chinese Min:", np.min(chinese))
print("Math Min:", np.min(math))
print("English Min:", np.min(english))

print("Chinese Max:", np.max(chinese))
print("Math Max:", np.max(math))
print("English Max:", np.max(english))

print("Chinese STD:", np.std(chinese))
print("Math STD:", np.std(math))
print("English STD:", np.std(english))

print("Chinese VAR:", np.var(chinese))
print("Math VAR:", np.var(math))
print("English VAR:", np.var(english))
