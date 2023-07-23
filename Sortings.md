## Insertion sort
```python
def insertion_sort(arr):
	for i in range(1, len(arr)):
		key = arr[i]
		# Insert A[i] into the sorted subarray A[1:(i-1)]
		j = i - 1
		while arr[j] > key and j >= 0:
			arr[j + 1] = arr[j]
			j -= 1
		arr[j + 1] = key
	return arr
```
## Bubblesort
```python
def bubblesort(arr):
	n = len(arr)
	for i in range(n):
		for j in range(n - 1, i, -1): # reverse index includes lower
			if arr[j] < arr[j - 1]:
				arr[j], arr[j - 1] = arr[j - 1], arr[j]
```
## Mergesort
```python
def merge(arr, p, q, r):
    n_l = q - p + 1
    n_r = r - q
    
    L = [0] * n_l
    R = [0] * n_r
    
    for i in range(n_l):
        L[i] = arr[p + i]
        
    for j in range(n_r):
        R[j] = arr[q + j + 1]
    
    i = 0
    j = 0
    k = p
    
    while i < n_l and j < n_r:
        if L[i] < R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
        
    while i < n_l:
        arr[k] = L[i]
        i += 1
        k += 1
        
    while j < n_r:
        arr[k] = R[j]
        j += 1
        k += 1
        
        
def merge_sort(arr, p, r):
    if p >= r:
        return
    
    q = (p + r) // 2
    
    merge_sort(arr, p, q)
    merge_sort(arr, q + 1, r)
    merge(arr, p, q, r)
    
    
merge_sort(arr, 0, len(arr) - 1)
arr
```
## Heapsort
```python
def parent(i):
    return ((i - 1) // 2)

def left(i): 
    return 2 * i + 1

def right(i):
    return 2 * i + 2

def max_heapify(arr, i, n):
    l = left(i)
    r = right(i)
    
    if l <= n and arr[l] > arr[i]:
        largest = l
    else:
        largest = i
        
    if r <= n and arr[r] > arr[largest]:
        largest = r
        
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        max_heapify(arr, largest, n)
        
def build_max_heap(arr, n):
    for i in range((n - 1) // 2 , -1, -1):
        max_heapify(arr, i, n)
        
def heapsort(arr, n):
    build_max_heap(arr, n)
    for i in range(n, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        n -= 1
        max_heapify(arr, 0, n)
```
## Quicksort
```python
def partition(arr, p, r):
    x = arr[r]
    i = p - 1
    for j in range(p, r):
        if arr[j] <= x:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            
    arr[i + 1], arr[r] = arr[r], arr[i + 1]
    
    return i + 1

def quicksort(arr, p, r):
    if p < r:
        q = partition(arr, p, r)
        quicksort(arr, p, q - 1)
        quicksort(arr, q + 1, r)
```
## Counting sort
Counting sort assumes that maximum element of array k is known and the array is in the range 0 to k
```python
def counting_sort(arr, n, k):
    arr_c = [0] * (k + 1)
    arr_b = [0] * n
    
    for i in range(n):
        arr_c[arr[i]] = arr_c[arr[i]] + 1
    # arr_c[i] contains numbers equal to i
    for i in range(1, k + 1):
        arr_c[i] = arr_c[i] + arr_c[i - 1]
    # arr_c[i] contains numbers greater or equal to i
        
    for i in range(n - 1, -1, -1):
        arr_b[arr_c[arr[i]] - 1] = arr[i] # arr_c contains 0-index, so is bigger then k by 1
        arr_c[arr[i]] -= 1 # handle duplicates
    return arr_b
```
## Radix sort
```python
```
## Bucket sort
Bucket sort assumes that the input is drawn from a uniform distribution 0 to 1
```python
def bucket_sort(arr):
	n = len(arr)
	arr_b =
```