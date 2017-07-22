def distance(x_start, x_end, n=1):
	""" x_start, x_end - 2 vectors between which distance is to be measured
		n - order of the measured distance according to Minkowski distance's definition
	"""
	if len(x_start) != len(x_end):
		print("[ERROR] - Inconsistent dimensions of input vectors!")
		result = -1
	elif n < 1:
		print("[ERROR] - Order 'n' has to be >= 1!")
		result = -1
	else:
		tmp = [abs(x_end[i] - x_start[i]) for i in range(len(x_start)) ]
		tmpPower = [value**n for value in tmp]
		tmpSum = sum(tmpPower)
		result = tmpSum**(1/n)
	return(result)
	
def euclideanDistance(x_start, x_end):
	"""
	Function created to increase readability in external files.
	"""
	return(distance(x_start, x_end, n=2))