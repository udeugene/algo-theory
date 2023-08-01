## Sliding windows
#### Maximum Sum Subarray of Size K
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
#### [Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)
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
#### Longest Substring with K Distinct Characters
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
#### [Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/)
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
#### [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
Given a string, find the **length of the longest substring** which has **no repeating characters**.
```python
def lengthOfLongestSubstring(self, s: str) -> int:
	max_len = 0
	l = 0
	seen = {}

	for char in s:
		if char not in seen:
			seen[char] = 1
		else:
			seen[char] += 1

		while seen[char] > 1:
			seen[s[l]] -= 1
			if seen[s[l]] <= 0:
				del seen[s[l]]
			l += 1
		max_len = max(max_len, len(seen))

	return max_len
```

#### [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)
You are given a string `s` and an integer `k`. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most `k` times.
```python
def characterReplacement(self, s: str, k: int) -> int:
	curr = {}
	l = 0
	max_len = 0
	max_cnt = 0

	for i in range(len(s)):
		if s[i] not in curr:
			curr[s[i]] = 1
		else:
			curr[s[i]] += 1

		if curr[s[i]] > max_cnt:
			max_cnt = curr[s[i]]
		

		while i - l - max_cnt + 1 > k:
			curr[s[l]] -= 1
			l += 1

		max_len = max(max_len, i - l + 1)
	
	return max_len
```
#### [Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/)
```python
def longestOnes(self, nums: List[int], k: int) -> int:
	l=r=0    
	for r in range(len(nums)):
		if nums[r] == 0:
			k-=1
		if k<0:
			if nums[l] == 0:
				k+=1
			l+=1
	return r-l+1
```
#### [Permutation in String](https://leetcode.com/problems/permutation-in-string/)
Given two strings `s1` and `s2`, return `true` _if_ `s2` _contains a permutation of_ `s1`_, or_ `false` _otherwise_.

In other words, return `true` if one of `s1`'s permutations is the substring of `s2`.
```python
def checkInclusion(self, s1: str, s2: str) -> bool:
	l = 0
	match_cnt = 0
	s1_dict = {}

	for s in s1:
		if s not in s1_dict:
			s1_dict[s] = 1
		else:
			s1_dict[s] += 1

	for i, char in enumerate(s2):

		if char in s1_dict:
			s1_dict[char] -= 1
			if s1_dict[char] == 0:
				match_cnt += 1
		
		if i - l + 1 > len(s1):
			if s2[l] in s1_dict:
				if s1_dict[s2[l]] == 0:
					match_cnt -= 1
				s1_dict[s2[l]] += 1
			l += 1

		if match_cnt == len(s1_dict):
			return True
		
	return False
```
#### [Subarray Product Less Than K](https://leetcode.com/problems/subarray-product-less-than-k/)
Given an array of integers `nums` and an integer `k`, return _the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than_ `k`.
```python
def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
	l = 0
	cnt = 0
	curr_prod = 1

	if k <= 1:
		return 0

	for r in range(len(nums)):
		curr_prod *= nums[r]
		
		while curr_prod >= k:
			curr_prod /= nums[l]
			l += 1

		cnt += r - l + 1

	return cnt
```
## Two pointers
#### [Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)
Given a **1-indexed** array of integers `numbers` that is already **_sorted in non-decreasing order_**, find two numbers such that they add up to a specific `target` number. Let these two numbers be `numbers[index1]` and `numbers[index2]` where `1 <= index1 < index2 < numbers.length`.

Return _the indices of the two numbers,_ `index1` _and_ `index2`_, **added by one** as an integer array_ `[index1, index2]` _of length 2._

The tests are generated such that there is **exactly one solution**. You **may not** use the same element twice.
```python
def twoSum(numbers, target):
	"""
	:type numbers: List[int]
	:type target: int
	:rtype: List[int]
	"""

	l = 0
	r = len(numbers) - 1

	while l < r:
		if numbers[l] + numbers[r] == target:
			return [l + 1, r + 1]
		elif numbers[l] + numbers[r] > target:
			r -= 1
		else:
			l += 1
```
#### [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)
Given an integer array `nums` sorted in **non-decreasing order**, remove the duplicates [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm) such that each unique element appears only **once**. The **relative order** of the elements should be kept the **same**. Then return _the number of unique elements in_ `nums`.

Consider the number of unique elements of `nums` to be `k`, to get accepted, you need to do the following things:

- Change the array `nums` such that the first `k` elements of `nums` contain the unique elements in the order they were present in `nums` initially. The remaining elements of `nums` are not important as well as the size of `nums`.
- Return `k`.
```python
def removeDuplicates(nums: List[int]) -> int:
	l = 0
	r = 0

	while r < len(nums) - 1:
		r += 1
		if nums[r] > nums[l]:
			l += 1
		nums[l] = nums[r]
	
	return l + 1

```
#### [Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/)
```python
def sortedSquares(self, nums: List[int]) -> List[int]:
	n = len(nums)

	l = 0
	r = n - 1
	res = [0] * n

	while r >= l:
		r_sq = nums[r] ** 2
		l_sq = nums[l] ** 2

		if r_sq >= l_sq:
			res[r - l] = r_sq
			r -= 1
		else:
			res[r - l] = l_sq
			l += 1

	return res
```
#### [3Sum](https://leetcode.com/problems/3sum/)
Given an integer array nums, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

Notice that the solution set must not contain duplicate triplets.  
```python
def threeSum(self, nums: List[int]) -> List[List[int]]: 
	nums.sort() # sorting cause we need to avoid duplicates, with this duplicates will be near to each other
	l=[]
	for i in range(len(nums)):  #this loop will help to fix the one number i.e, i
		if i>0 and nums[i-1]==nums[i]:  #skipping if we found the duplicate of i
			continue 
		
		#NOW FOLLOWING THE RULE OF TWO POINTERS AFTER FIXING THE ONE VALUE (i)
		j=i+1 #taking j pointer larger than i (as said in ques)
		k=len(nums)-1 #taking k pointer from last 
		while j<k: 
			s=nums[i]+nums[j]+nums[k] 
			if s>0: #if sum s is greater than 0(target) means the larger value(from right as nums is sorted i.e, k at right) 
			#is taken and it is not able to sum up to the target
				k-=1  #so take value less than previous
			elif s<0: #if sum s is less than 0(target) means the shorter value(from left as nums is sorted i.e, j at left) 
			#is taken and it is not able to sum up to the target
				j+=1  #so take value greater than previous
			else:
				l.append([nums[i],nums[j],nums[k]]) #if sum s found equal to the target (0)
				j+=1 
				while nums[j-1]==nums[j] and j<k: #skipping if we found the duplicate of j and we dont need to check 
				#the duplicate of k cause it will automatically skip the duplicate by the adjustment of i and j
					j+=1   
	return l
```
#### [3Sum Closest](https://leetcode.com/problems/3sum-closest/)
Given an integer array `nums` of length `n` and an integer `target`, find three integers in `nums` such that the sum is closest to `target`.

Return _the sum of the three integers_.

You may assume that each input would have exactly one solution.
```python
def threeSumClosest(self, nums: List[int], target: int) -> int:
	import sys
	nums.sort()
	min_diff = sys.maxsize

	for i in range(len(nums)):
		j = i + 1
		k = len(nums) - 1

		while j < k:
			curr_sum = nums[i] + nums[j] + nums[k]
			curr_diff = abs(curr_sum - target)

			if curr_sum == target:
				return curr_sum

			if curr_diff < min_diff:
				min_diff = curr_diff
				min_sum = curr_sum
			
			if curr_sum < target:
				j += 1
			else:
				k -= 1
	return min_sum
```
#### [3Sum Smaller](https://practice.geeksforgeeks.org/problems/count-triplets-with-sum-smaller-than-x5549/1?utm_source=geeksforgeeks&utm_medium=article_practice_tab&utm_campaign=article_practice_tab)
Given an array **arr[]** of distinct integers of size **N** and a value **sum**, the task is to find the count of triplets **(i, j, k)**, having **(i<j<k)** with the sum of ****(arr[i] + arr[j] + arr[k])**** smaller than the given value sum.
```python
def countTriplets(self, arr, n, sumo):
	
	arr.sort()
	res = 0
	
	for i in range(n):
		
		if i > 0 and arr[i] == arr[i - 1]:
			continue
		
		j = i + 1
		k = n - 1
		
		while j < k:
			curr_sum = arr[i] + arr[j] + arr[k]
			
			if curr_sum < sumo:
				res += k - j # since arr[k] >= arr[j] each number between k and j is suitable for getting sum less then target
				j += 1
				while arr[j] == arr[j - 1] and j < k:
					j += 1
			else:
				k -= 1
	return res     
```
#### [Sort Colors](https://leetcode.com/problems/sort-colors/description/)
**Solution 1**: count each color and rewrite:
```python
def sortColors(nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    zeros = 0
    ones = 0
    twos = 0

    for num in nums:
        if num == 0:
            zeros += 1
        elif num == 1:
            ones += 1
        else:
            twos += 1

    nums[:zeros] = [0] * zeros
    nums[zeros: zeros + ones] = [1] * ones
    nums[zeros + ones:] = [2] * twos
```
**Solution 2**: Two pointers
```python
def sortColors(nums):
    l, m, r = 0, 0, len(nums) - 1

    while m <= r:
        if nums[m] == 0:
            nums[l], nums[m] = nums[m], nums[l]
            m += 1
            l += 1
        elif nums[m] == 1:
            m += 1
        else:
            nums[m], nums[r] = nums[r], nums[m]
            r -= 1

```
## Fast & Slow Pointers
#### [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/description/)
Return `true` _if there is a cycle in the linked list_. Otherwise, return `false`.
```python
def hasCycle(head: Optional[ListNode]) -> bool:
	slow, fast = head, head
	while fast is not None and fast.next is not None:
		slow = slow.next
		fast = fast.next.next

		if slow == fast:
			return True
			
	return False
```
#### [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)
Given the `head` of a linked list, return _the node where the cycle begins. If there is no cycle, return_ `null`. 
**Solution**
![[Pasted image 20230730220856.png]]
Fast pointer is 2 times faster then slow one. Slow and fast meet in point $Z$. By that time slow pointer travelled $a+b$, and the fast one $a + b + k \cdot (c + b)$ , where $k$ is positive integer, which is present when loop is small and fast pointer moved around it for many times.
We know that fast is 2 times faster, so:
$2 \cdot (a + b) = a + b + c + b + k \cdot (c + b)$
$k \cdot (c+b)$ can be ignored, because it equivalent to $k$ full cycles and do not affect point where slow and fast meet. So:
$a = c$
We need to push pointers until they meet in $Z$. And then push one pointer from head and other from $Z$ till they meet.
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

def detectCycle(head: Optional[ListNode]) -> Optional[ListNode]:
	fast, slow = head, head

	while fast and fast.next:
		slow = slow.next
		fast = fast.next.next

		if fast == slow:
			slow = head
			while slow != fast:
				slow = slow.next
				fast = fast.next
			return slow
	
	return None

```
#### [Happy Number](https://leetcode.com/problems/happy-number/)

Write an algorithm to determine if a number `n` is happy.
A **happy number** is a number defined by the following process:
- Starting with any positive integer, replace the number by the sum of the squares of its digits.
- Repeat the process until the number equals 1 (where it will stay), or it **loops endlessly in a cycle** which does not include 1.
- Those numbers for which this process **ends in 1** are happy.
- 
Return `true` _if_ `n` _is a happy number, and_ `false` _if not_.

**Solution** 
All numbers cycle, but ends up to 1 either to other number. Find cycle using fast & slow pointers.
```python
def find_square(num):
	s = 0
	while num > 0:
		digit = num % 10
		s += digit ** 2
		num = num // 10
	return s

def isHappy(n: int) -> bool:
	slow, fast = n, n

	while 1:
		slow = self.find_square(slow)
		fast = self.find_square(self.find_square(fast))
		if slow == fast:
			break
			
	if slow == 1:
		return True
	else:
		return False
```

#### [Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)
Given the `head` of a singly linked list, return _the middle node of the linked list_. 
If there are two middle nodes, return **the second middle** node.

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
def middleNode(head: Optional[ListNode]) -> Optional[ListNode]:
	slow, fast = head, head
	while fast and fast.next:
		slow = slow.next
		fast = fast.next.next

	return slow
```
#### [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)
Given the `head` of a singly linked list, return `true` _if it is a_ _palindrome_ _or_ `false` _otherwise_.

**Solution** 
Find mid then reverse half from mid to the end. Iterate from head and mid simultaneously.
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

def isPalindrome(head: Optional[ListNode]) -> bool:
	slow, fast = head, head

	while fast and fast.next: 
		slow = slow.next
		fast = fast.next.next

	curr = slow
	prev = None
	while curr:
		next = curr.next
		curr.next = prev
		prev = curr
		curr = next

	first_half = head
	while prev:
		if prev.val != first_half.val:
			return False

		prev = prev.next
		first_half = first_half.next

	return True
```
## Merge Intervals
#### [Merge Intervals](https://leetcode.com/problems/merge-intervals/)
Given an array of `intervals` where `intervals[i] = [starti, endi]`, merge all overlapping intervals, and return _an array of the non-overlapping intervals that cover all the intervals in the input_.

```python
def merge(intervals: List[List[int]]) -> List[List[int]]:
	intervals.sort(key=lambda x: x[0])
	merged = []

	i = 0
	while i < len(intervals):
		l = intervals[i][0]
		r = intervals[i][1]
		i += 1
		while i < len(intervals) and r >= intervals[i][0]:
			r = max(r, intervals[i][1])
			i += 1
		merged.append([l, r])

	return merged
```
#### [Insert Interval](https://leetcode.com/problems/insert-interval/)
You are given an array of non-overlapping intervals `intervals` where `intervals[i] = [starti, endi]` represent the start and the end of the `ith` interval and `intervals` is sorted in ascending order by `starti`. You are also given an interval `newInterval = [start, end]` that represents the start and end of another interval. 
Insert `newInterval` into `intervals` such that `intervals` is still sorted in ascending order by `starti` and `intervals` still does not have any overlapping intervals (merge overlapping intervals if necessary). 
Return `intervals` _after the insertion_.

**Solution**
1. Find position where new interval should be inserted to keep the whole intervals array sorted by the first element.
2. Merge overlapping intervals 
```python
def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
	l = newInterval[0]
	i = 0

	while i < len(intervals) and l > intervals[i][0]:
		i += 1

	intervals.insert(i, newInterval)

	merged = []
	i = 0
	while i < len(intervals):
		l = intervals[i][0]
		r = intervals[i][1]
		i += 1
		while i < len(intervals) and r >= intervals[i][0]:
			r = max(r, intervals[i][1])
			i += 1
		merged.append([l, r])

	return merged

```

## Cyclic Sort
#### [Missing Number](https://leetcode.com/problems/missing-number/)
Given an array `nums` containing `n` distinct numbers in the range `[0, n]`, return _the only number in the range that is missing from the array.

```python
def missingNumber(nums: List[int]) -> int:
	i = 0

	while i < len(nums):
		while nums[i] != i and nums[i] != len(nums):
			nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
		i += 1
	
	for i in range(len(nums)):
		if nums[i] != i:
			return i

	return len(nums)
```

#### [Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)
Given an array `nums` of `n` integers where `nums[i]` is in the range `[1, n]`, return _an array of all the integers in the range_ `[1, n]` _that do not appear in_ `nums`.

1) Classic cycle sort
```python
def findDisappearedNumbers(nums: List[int]) -> List[int]:
    i = 0

    while i < len(nums):
        while nums[i] != i + 1 and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]

        i += 1

    ans = []
    for i in range(len(nums)):
        if nums[i] != i + 1:
            ans.append(i + 1)

    return ans
```
2) Negate seen numbers
```python
def findDisappearedNumbers(nums: List[int]) -> List[int]:
	for num in nums:
		# Haven't seen before
		if nums[abs(num)-1]>0:
			# Store the fact that it has now been seen
			nums[abs(num)-1] *= -1

	# The numbers that weren't seen
	return [i+1 for i, num in enumerate(nums) if num>0]
```
