## k-th order statistics
### Quickselect
```python
def randomized_select(arr, p, r, i):
	if p == r:
		return arr[p]
	
	q = randomized_partiton(arr, p, r)
	k = q - p + 1
	if i == k:
		return arr[q]
	elif i < k:
		return randomized_select(arr, p, q - 1, i)
	else:
		return randomized_select(arr, q + 1, r, i - k)
```
## Dynamic programming

Divide-and-conquer algorithm does more work than necessary, repeatedly solving the common subsubproblems. A dynamic-programming algorithm solves each subsubproblem just once and then saves its answer in a table, thereby avoiding the work of recomputing the answer every time it solves each subsubproblem.
To develop a dynamic-programming algorithm, follow a sequence of four steps:
1. Characterize the structure of an optimal solution. 
2. Recursively define the value of an optimal solution.
3. Compute the value of an optimal solution, typically in a bottom-up fashion.
4. Construct an optimal solution from computed information.

### Rod cutting
#### Recursive solution
Complexity: $O(2^{n-1})$
```python
def cut_rod(p, n):
    if n == 0:
        return 0
    
    q = float('-inf')
    
    for i in range(n):
        q = max([q, p[i] + cut_rod(p, n - i - 1)])
        
    return q
```
### Dynamic programming solution
Complexity: $O(n^2)$
**Top-down with memoization**
```python
def memoized_cut_rod(p, n):
	r = (n + 1) * [float('-inf')]
	return memoized_cut_rod_aux(p, n, r)

def memoized_cut_rod_aux(p, n, r):
	if r[n] >= 0:
		return r[n]

	if n == 0:
		q = 0
	else:
		q = float('-inf')
		for i in range(n): # i is the position of the first cut
			q = max([q, p[i] + memoized_cut_rod_aux(p, n - i - 1, r)])
	r[n] = q # remember the solution value for length n
	return q	

```
#### Bottom-up
```python
def bottom_up_cut_rod(p, n):
    r = (n + 1) * [float('-inf')]
    r[0] = 0

    for j in range(1, n + 1): # for increasing rod length j
        q = float('-inf')
        for i in range(j): # i is the position of the first cut
            q = max([q, p[i] + r[j - i - 1]])
        r[j] = q # remember the solution value for length j
    return r[n]
```