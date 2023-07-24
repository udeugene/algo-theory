## 1. Sliding window
### Maximum Sum Subarray of Size K
$O(N*K)$ solution
Calculate each k-size subarrays
```python
def max_subarray(k, arr):
	max_sum = 0
	window_sum = 0

	for i in range(len(arr) - k + 1):
		window_sum = 0
		for j in range(i, i + k):
			window_sum += arr[j]
		max_sum = max(max_sum, window_sum)
	return max_sum
```

$O(N)$ solution
Slide window of size k ahead by 1 element and recalculate sum
```python
def max_subarray(k, arr):
	max_sum = 0

	for i in range(k):
		window_sum += arr[i]
	max_sum = window_sum

	l = 0
	r = k
	while r < len(arr):
		window_sum -= arr[l]
		window_sum += arr[r]
		if window_sum > max_sum:
			max_sum = window_sum
		l += 1
		r += 1
	return max_sum
	
```
### Smallest Subarray with a given sum
**Description** 
	Given an array of positive numbers and a positive number ‘S’, find the length of the **smallest contiguous subarray whose sum is greater than or equal to ‘S’**. Return 0, if no such subarray exists.

Move right window index and check if it is possible to move left window index.
$O(n)$ or $O(n+n)$ complexity
```python
def min_subarr_len(target, nums):
	l = 0
	curr_sum = 0
	min_len = len(nums) + 1
	
	for r in range(len(nums)):
		curr_sum += nums[r]
		
		while curr_sum >= target:
			curr_sum -= nums[l]
			min_len = min(min_len, r - l + 1)
			l += 1
			
	if min_len > len(nums):
		return 0
	
	return min_len
```