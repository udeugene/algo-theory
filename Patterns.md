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
### Longest Substring with K Distinct Characters
Given a string, find the length of the **longest substring** in it **with no more than K distinct characters**.
```python
def longestKSubstr(self, s, k):
	curr = {}
	l = 0
	max_size = -1
	
	for i, r in enumerate(s):
		if r not in curr:
			curr[r] = 1
		else:
			curr[r] += 1
		
		while len(curr) > k: # if unique char more than k delete leftmost unique
			curr[s[l]] -= 1
			if curr[s[l]] == 0:
				del curr[s[l]]
			l += 1
			
		max_size = max(max_size, i - l + 1)                
	
	if len(curr) < k: # unique chars in s less than k
		return -1
		
	return max_size
```
### Fruits into Baskets
Pretty the same idea as the previous one
```python
def totalFruit(fruits):
    """
    :type fruits: List[int]
    :rtype: int
    """
    harvest = {}
    max_harvest = 1
    l = 0

    for i, f in enumerate(fruits):
        if f not in harvest:
            harvest[f] = 1
        else:
            harvest[f] += 1

        while len(harvest) > 2:
            harvest[fruits[l]] -= 1
            if harvest[fruits[l]] == 0:
                del harvest[fruits[l]]
            l += 1

        max_harvest = max(max_harvest, i - l + 1)

    return max_harvest
```