# LeetCode hot 100 刷题记录


## 哈希表
### 1. 两数之和

使用哈希表，空间换时间，时间复杂度O(N)、空间复杂度O(N)
```python 
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:

        hash = {}
        for i in range(len(nums)):
            if target - nums[i] not in hash:
                hash[nums[i]] = i
            else:
                return [i, hash[target - nums[i]]]
        return []
```

### 49. 字母异位词分组

时间复杂度：O(n * k * log k)，其中n是字符串数量，k是最长字符串的长度、空间复杂度：O(n * k)
```python 
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:

        hash = {}

        for str in strs:
            
            sorted_str = ''.join(sorted(str))

            if sorted_str in hash:
                hash[sorted_str].append(str)
            else:
                hash[sorted_str] = [str]
        return list(hash.values())
```

### 128. 最长连续序列
自己写的，总体时间复杂度：O(n²)，使用列表（hash = []）作为查找结构，列表的查找操作是 O(n)，应该使用集合（set），查找操作为 O(1)
```python 
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        max_count = 0
        if not nums:
            return 0

        hash = []
        for i in range(len(nums)):
            hash.append(nums[i])
            cur = nums[i]
            cur_count = 1
            while cur + 1 in hash:
                cur_count +=1
                cur = cur + 1
            cur = nums[i]
            while cur - 1 in hash:
                cur_count +=1
                cur = cur - 1
            max_count = max(max_count, cur_count)

        return max_count
```

```python 
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:
            return 0
        num_set = set(nums)
        max_count = 0

        for num in num_set:
            if num - 1 not in num_set:
                cur_count = 1
                cur_num = num

                while cur_num + 1 in num_set:
                    cur_count += 1
                    cur_num += 1 
                
                max_count = max(max_count, cur_count)
        return max_count
```

## 双指针

### 283. 移动零
自己做用快慢指针记录，但是选择移动0往后交换，这样会打乱非0的顺序，正确的应该是选择向前移动非0数字，慢指针用来记录要移动到哪，快指针遍历数组
```python 
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        slow = 0
        fast = 0
    

        while fast < len(nums):
            if nums[fast] != 0:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        
        while slow < len(nums):
            nums[slow] = 0
            slow +=1

        return nums

```
### 11. 盛最多水的容器

时间复杂度：O(N)，双指针总计最多遍历整个数组一次、空间复杂度：O(1)，只需要额外的常数级别的空间
```python 
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        max_area = 0

        while left < right:
            if height[left] < height[right]:
                cur_area = height[left] * (right - left)
                max_area = max(max_area, cur_area)
                left +=1
            else:
                cur_area = height[right] * (right - left)
                max_area = max(max_area, cur_area)
                right -=1
        return max_area
```

### 15. 三数之和

自己做想到了使用双指针，但是有很多没注意到:一是需要考虑重复，包括遍历的以及指针对应的，二是双指针对应一次遍历不仅仅只有一组。
时间复杂度：O(n^2)，空间复杂度：O(1)，返回值不计入，忽略排序的栈开销。
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        if not nums or nums[0] > 0:
            return []
        
        result = []
        
        for i in range(len(nums)-2):
            
            if i > 0 and nums[i] == nums[i-1]:
                continue
            if nums[i] > 0:
                break

            left = i + 1
            right = len(nums) - 1
            while left < right:
                if nums[i] + nums[left] + nums[right] > 0:
                    right -= 1
                elif nums[i] + nums[left] + nums[right] < 0:
                    left +=1
                else:
                    result.append([nums[i], nums[left], nums[right]])

                    while left < right and nums[left] == nums[left+1]:  
                        left += 1  
                    while left < right and nums[right] == nums[right-1]:  
                        right -= 1 

                    left += 1
                    right -= 1
        
        return result
```
      
### ⚠️ 42. 接雨水

接个锤子雨水啊，只记得用双指针。核心思路就是理解每个遍历位置的存水量=往左右看的最大值中的最小值减去当前位置的高度，只需要遍历一次，时间复杂度为O(n)，空间复杂度为O(1)

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0

        left = 0
        right = len(height) - 1
        left_max = 0
        right_max = 0
        res = 0

        while left < right:
            
            left_max = max(left_max, height[left])
            right_max = max(right_max, height[right])

            if left_max < right_max:
                res += left_max - height[left]
                left +=1
            else:
                res += right_max - height[right]
                right -=1
        return res
```


  ## 滑动窗口

### 3. 无重复字符的最长字串
  遍历的时候寻找窗口左边界，即重复出现的索引位置，时间复杂度：O(N)，其中N是字符串的长度。左指针和右指针分别会遍历整个字符串一次。空间复杂度：O(∣Σ∣)，其中Σ表示字符集（即字符串中可以出现的字符），∣Σ∣表示字符集的大小。在本题中没有明确说明字符集，因此可以默认为所有 ASCII 码在[0,128)内的字符，即∣Σ∣=128。我们需要用到哈希集合来存储出现过的字符，而字符最多有∣Σ∣个，因此空间复杂度为O(∣Σ∣)。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        hash = {}
        max_len = 0
        start = 0
        for i, char in enumerate(s):
            start = max(start, hash.get(char, -1) + 1)
            max_len = max(max_len, i - start + 1)
            hash[char] = i
        return max_len
  ```


### 438. 找到字符串中所有字母的异位词

哈希表计数+滑动窗口的方法脑子跟上了手没跟上，还得练啊！主循环：O(n)，n 是字符串 s 的长度，每次循环中的字典比较：O(k)，k 是字符集大小，总时间复杂度：O(n * k)，空间复杂度：O(Σ)，用于存储字符串p和滑动窗口中每种字母的数量。
因为题目给出字母都是小写的，因此可以直接用数组代替哈希表进行优化
 ```python           
  class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        hash = {}
        result = []
        # 创建一个哈希表记录p中各字符的出现次数
        for char in p:
           hash[char] =  hash.get(char, 0) + 1

        left = 0
        cur_hash = {}
        # 移动窗口
        for right, char in enumerate(s):
            cur_hash[char] = cur_hash.get(char, 0) + 1

            if right - left + 1 > len(p):
                cur_hash[s[left]] -= 1
                if cur_hash[s[left]] == 0:
                    del cur_hash[s[left]]
                left +=1
            if cur_hash == hash:
                result.append(left)
        return result
 ```
 ```python               
class Solution:  
    def findAnagrams(self, s: str, p: str) -> List[int]:  
        if len(s) < len(p):  
            return []  
            
        # O(1) 空间，因为是固定大小26  
        need = [0] * 26  
        window = [0] * 26  
        
        # O(m) 时间  
        for c in p:  
            need[ord(c) - ord('a')] += 1  
            
        res = []  
        
        # O(m) 时间  
        for i in range(len(p)):  
            window[ord(s[i]) - ord('a')] += 1  
            
        # O(1) 时间的数组比较  
        if window == need:  
            res.append(0)  
            
        # O(n-m) 时间  
        for i in range(len(p), len(s)):  
            window[ord(s[i - len(p)]) - ord('a')] -= 1  
            window[ord(s[i]) - ord('a')] += 1  
            
            # O(1) 时间的数组比较  
            if window == need:  
                res.append(i - len(p) + 1)  
                
        return res
 ```


## 字串

### 560. 和为K的子数组
前缀和的思路，但是写的时候非常傻逼没用哈希表记录前缀和，导致复杂度不行，时间复杂度：O(n)，空间复杂度：O(n)，记住！记住！记住！哈希表查找当前的sum减去目标值就行
 ```python  
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        
        hash = {}
        hash[0] = 1
        curr_sum = count = 0

        for num in nums:
            curr_sum += num
            count += hash.get(curr_sum - k, 0)
            hash[curr_sum] = hash.get(curr_sum, 0) + 1


        return count
 ```  

### 239. 滑动窗口最大值
哎，记起来了思路，使用双端单调队列，保持队列头是最大值，但是代码处理起来还是一堆问题，还是多练！记住：队列保存的是索引、先处理前k个即第一个窗口、先处理窗口边界（popleft）再考虑其它（pop）
 ```python  
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        due = deque()
        result = []

        for i in range(k):
            while due and nums[i] >= nums[due[-1]]:
                due.pop()
            due.append(i)
        
        result.append(nums[due[0]])


        for i in range(k, len(nums)):
            while due and due[0] <= i - k:
                due.popleft()
        
            while due and nums[i] >= nums[due[-1]]:
                due.pop()
            due.append(i)
            
            result.append(nums[due[0]])

        return result
 ```

### 76. 最小覆盖字串

垃圾题目，完全写不出来啊！！！！这道题的核心思路是使用滑动窗口配合哈希表来解决最小覆盖子串问题：用一个哈希表记录目标字符串t中每个字符需要的频率，另一个哈希表记录当前窗口中的字符频率，通过移动右指针扩大窗口来寻找可行解，当找到包含所有所需字符的窗口后（用valid变量记录已满足要求的字符数），再尝试通过移动左指针来缩小窗口，同时维护最小窗口的起始位置和长度，最终返回最小覆盖子串。
 ```python  
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        
        hash = {}
        for char in t:
            hash[char] = hash.get(char, 0) + 1

        min_length = float('inf')
        start = 0
        valid = 0
        left = 0
        cur_hash = {}

        for right in range(len(s)):
            if s[right] in hash:
                cur_hash[s[right]] = cur_hash.get(s[right], 0) + 1
                if cur_hash[s[right]] == hash[s[right]]:
                    valid +=1

            while valid == len(hash):
                if right - left +1 < min_length:
                    start = left
                    min_length = right - left + 1
                
                if s[left] in hash:
                    if cur_hash[s[left]] == hash[s[left]]:
                        valid -= 1
                    cur_hash[s[left]] -= 1
                left += 1
        return "" if min_length == float('inf') else s[start: start + min_length]
 ```
                 
## 普通数组

### 53. 最大子数组和
注意数组如果都是负数那么直接取最大值即可，搞不懂什么分治法，这不就逻辑很通顺
 ```python  
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:

        if max(nums) <= 0:
            return max(nums)

        max_sum = float('-inf')
        cur_sum = 0

        for i in range(len(nums)):
            if nums[i] + cur_sum > 0:
                cur_sum = cur_sum + nums[i]
                max_sum = max(max_sum, cur_sum)
            else:
                cur_sum = 0

        return  0 if max_sum == float('-inf') else max_sum 
 ```

### 56. 合并区间

先排序再慢慢合并就行，需要注意的是最后一个数组的处理，如果没法合并，两个都得加进去，如果可以合并，就直接加进去不用检测之后的
 ```python  
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) == 1:
            return intervals

        intervals.sort(key = lambda x:x[0])
        result = []

        for i in range(len(intervals) - 1):
            if intervals[i + 1][0] > intervals[i][1]:
                result.append(intervals[i])
                if i == len(intervals) -2:
                    result.append(intervals[i + 1])
            else:
                intervals[i+1] = [intervals[i][0], max(intervals[i][1], intervals[i + 1][1])]
                if i == len(intervals) -2:
                    result.append(intervals[i + 1])


        return result
 ```

### 189. 转轮数组

注意k可能大于数组长度，需要取余，然后翻转翻转再翻转即可，时间复杂度是 O(n)，空间复杂度是 O(1)
 ```python  
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        def reverse_arr(nums, start, end):
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start +=1
                end -=1
            
            return nums

        k = k % len(nums)
        reverse_arr(nums, 0, len(nums)-1)
        reverse_arr(nums, 0, k-1)
        reverse_arr(nums, k, len(nums)-1)
 ```  

### 238. 除自身以外数组的乘积

前缀积与后缀积再相乘即可，但是为了优化空间复杂度，可以在计算后缀积时直接与前缀积相乘
 ```python  
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        
        # 左边的积与右边的积相乘
        result = [1] * len(nums)

        for i in range(1, len(nums)):
            result[i] = result[i - 1] * nums[i - 1]

        right = 1
        for i in range(len(nums)-1, -1, -1):
            result[i] = result[i] * right
            right = right * nums[i]
        
        return result
 ```

### 41. 缺失的第一个正数

使用hash表记录符合直观，但是复杂度不符合，思路是将索引与值对应原地修改数组，但是原地修改一直有点问题,注意捋清楚是用While，平均下来，每个数只需要看一次就可以了，时间复杂度：O(N)，空间复杂度：O(1)

 ```python 
class Solution:  
    def firstMissingPositive(self, nums: List[int]) -> int:  
        n = len(nums)  
        
        # 第一次遍历：把数字放到正确的位置上  
        for i in range(n):  
            # nums[i] 应该放在 nums[i]-1 的位置上  
            # 比如 1 应该放在索引 0 的位置
            # 注意需要用while, 因为得一直交换,不然会漏掉交换
            while 1 <= nums[i] <= n and nums[nums[i]-1] != nums[i]:  
                # 交换 nums[i] 和 nums[nums[i]-1]  
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]  
        
        # 第二次遍历：检查每个位置的数字是否正确  
        for i in range(n):  
            if nums[i] != i + 1:  
                return i + 1  
        
        return n + 1
 ```
## 矩阵

### 73. 矩阵置零
空间复杂度 O(1)，使用矩阵的第一行和第一列作为标记，思路清晰，写代码也得速度

 ```python 
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        row, cols = len(matrix), len(matrix[0])
        row_bool = False
        cols_bool = False

        # 记录第一行是否存在0
        for j in range(cols):
            if matrix[0][j] == 0:
                row_bool = True
                break

        # 记录第一列是否存在0
        for i in range(row):
            if matrix[i][0] == 0:
                cols_bool = True
                break


        for i in range(1, row):
            for j in range(1, cols):
                if matrix[i][j] == 0:
                    matrix[0][j] = matrix[i][0] = 0
        
        for i in range(1, row):  
            for j in range(1, cols):  
                if matrix[i][0] == 0 or matrix[0][j] == 0:  
                    matrix[i][j] = 0 
        
        if row_bool:
             for j in range(cols):
                matrix[0][j] = 0
        
        if cols_bool:
             for i in range(row):
                matrix[i][0] = 0

 ```
        
### 54. 螺旋矩阵

从上下左右依次遍历，注意边界以及跳出循环的条件

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:

        row, cols = len(matrix), len(matrix[0])
        result = []
        # 上/下/左/右边界
        top = 0
        down = row
        left = 0
        right =cols

        while True:

            for i in range(left, right):
                result.append(matrix[top][i])
            top += 1

            if top >= down:
                break
                

         
            for i in range(top, down):
                result.append(matrix[i][right-1])
            right -= 1

            if left >= right:
                break

      
            for i in range(right-1, left-1, -1):
                result.append(matrix[down-1][i])
            down -= 1

            if top >= down:
                break

           
            for i in range(down - 1, top-1, -1):
                result.append(matrix[i][left])
            left += 1

            if left >= right:
                break

        return result
```
        
### 48. 旋转图像

注意对角线折叠的边界，不要折叠两次了
```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # 沿着对角线折叠,然后就是轴对称了

        for i in range(len(matrix)):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        for i in range(len(matrix)):
            for j in range(len(matrix[0])//2):
                matrix[i][j], matrix[i][len(matrix[0]) - j - 1] = matrix[i][len(matrix[0]) - j - 1],matrix[i][j]
```    

### 240. 搜索二维矩阵II
时间复杂度：O(m+n)。在搜索的过程中，如果我们没有找到target，那么我们要么将y减少1，要么将x增加1。由于(x,y)的初始值分别为(0,n−1)，因此y最多能被减少n次，x最多能被增加m次，总搜索次数为m+n。在这之后，x和y就会超出矩阵的边界

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:

        m, n = len(matrix), len(matrix[0])

        row = 0
        cols = n - 1

        while 0 <= row < m and 0 <= cols < n:
            if matrix[row][cols] == target:
                return True
            elif matrix[row][cols] > target:
                cols -= 1
            else:
                row += 1

        return False
```

## 链表

### 160. 相交链表

烦死了，知道思路但是不知道怎么循环终止条件，原理不相交最终两个都会是None

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:

        A = headA
        B = headB
        index = 0
        while A != B:
            A = A.next if A else headB
            B = B.next if B else headA

        return A
```


### 206. 反装链表
使用迭代法
```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        pre = None
        while cur:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        return pre
```
使用递归的方法，最烦的就是递归，就是学不会。因为递归会从头到尾，因为要链表翻转，所以会保留两个结点，每次将两个节点翻转同时使next的节点指向None

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        if not head or not head.next:
            return head

        newHead = self.reverseList(head.next)

        head.next.next = head
        head.next = None
        
        return newHead
```





### 141. 环形链表
快慢指针找环
```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head:
            return False
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
```

### 142. 环形链表II

头到环入口等于相遇点到环入口

```python
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        dummy = ListNode()
        dummy.next = head
        slow = fast = dummy
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                slow = dummy
                while True:
                    slow = slow.next
                    fast = fast.next
                    if slow == fast:
                        return slow


        return None
```

### 21. 合并有序链表 
```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:

        node1 = list1
        node2 = list2

        dummy = ListNode(0)
        cur_node = dummy

        while node1 and node2:
            if node1.val < node2.val:
                cur_node.next = node1
                node1 = node1.next
            else:
                cur_node.next = node2
                node2 = node2.next
            cur_node = cur_node.next

        if node1:
            cur_node.next = node1
        if node2:
            cur_node.next = node2
        
        return dummy.next
```

### 2. 两数相加
知道应该怎么做，但是写的时候逻辑不是很严谨，特别是进位判断写的乱七八糟，最好用数值存储结果计算，能够有效处理各种边界
```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:

        dummy = ListNode(0)
        cur_node = dummy
        carry = 0

        while l1 or l2 or carry:
            # 获取两个链表当前节点的值
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0

            # 计算和与进位
            total = x + y + carry
            carry = total // 10

            # 创建新节点
            cur_node.next = ListNode(total % 10)

            # 移动指针
            cur_node = cur_node.next
            l1 = l1.next if l1 else l1
            l2 = l2.next if l2 else l2
            
        return dummy.next
```    

           

## 图论
### 207 课程表
用DFS检测有向图中是否存在环。首先把课程依赖关系转换成图（用邻接表表示），然后用一个visited数组记录节点的访问状态（0未访问，1正在访问，2已完成访问）。在DFS遍历过程中，如果遇到状态为1的节点（正在访问），就说明存在环，返回False；如果遍历完所有节点都没有发现环，就返回True表示可以完成所有课程

### 208 实现Trie（前缀树）
用一个树形结构，其中每个节点存储一个字符和一个字典（children），字典用来指向子节点。从根节点开始，沿着路径走就能形成字符串。每个节点还有一个标记(is_end)表示是否是完整单词的结尾。插入时逐字符建立路径，查找时逐字符检查路径是否存在，区别在于查找完整单词时需要检查is_end标记，而查找前缀则不需要


## 堆
### 295 数据流中的中位数
用两个堆把数据分成两半：大顶堆存较小的一半，小顶堆存较大的一半；保持大顶堆的大小等于或比小顶堆多一个；添加数字时，通过在两个堆之间倒腾，保证大顶堆的最大值小于小顶堆的最小值；中位数就是：当总数为偶数时：两个堆顶的平均值；当总数为奇数时：大顶堆的堆顶（注意python默认是小顶堆）

## 栈
### 84 柱状图中的最大矩形
单调栈：当遇到一个小于栈顶的元素时，说明找到了栈顶元素的右边界，栈保持递增，所以栈顶下面的元素就是左边第一个小于当前高度的位置，这样就能确定一个矩形的左右边界，从而计算面积

## 回溯
### 51 N皇后
采用DFS回溯策略，通过逐行放置皇后的方式（保证行不冲突），使用set集合维护可用列（保证列不冲突），并利用坐标关系(行+列相等表示在同一主对角线，行-列相等表示在同一副对角线)来判断对角线冲突，当成功放置N个皇后时，将当前解加入结果集，最终返回所有可能的解

## 技巧

### 136. 只出现一次的数字
做过后就会了
```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result = 0
        for num in nums:
            result = result ^ num
            
        return result
```

### 169. 多数元素

有印象是类似投票，一人打全部，每次抵消最后剩下的就是最多的元素，写是写出来了，但是不够简洁
```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:

        max_sum = 1
        result = nums[0]

        for i in range(1, len(nums)):
            if nums[i] != result:
                max_sum -= 1
                if max_sum == 0:
                    result = nums[i+1]
            else:
                max_sum += 1

        return result
```

### 75. 颜色分类

有印象用三指针记录添加位置，但是代码写不对啊啊啊啊啊啊啊，遇到0：需要放置一个0、一个1、一个2，遇到1：需要放置一个1、一个2，遇到2：只需要放置一个2
```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        nums0 = nums1 = nums2 = 0
        for i in range(len(nums)):
            if nums[i] == 0:
                nums[nums2] = 2
                nums[nums1] = 1
                nums[nums0] = 0
                nums2+=1
                nums1+=1
                nums0+=1
            elif nums[i] == 1:
                nums[nums2] = 2
                nums[nums1] = 1
                nums1+=1
                nums2+=1
            else:
                nums[nums2] = 2
                nums2+=1
```

### 31. 下一个排列

有个地方思考错误了，交换的不是从后往前数第一个大于最后一个元素的数与最后一个元素
![image](https://github.com/user-attachments/assets/6bc19887-a3ef-440b-a9e6-500ac53b465e)
```python
class Solution:  
    def nextPermutation(self, nums: List[int]) -> None:  
        def reverse(nums, start, end):  
            while start < end:  
                nums[start], nums[end] = nums[end], nums[start]  
                start += 1  
                end -= 1  
        
        i = len(nums) - 2  
        # 1. 找到第一个升序对  
        while i >= 0 and nums[i] >= nums[i + 1]:  
            i -= 1  
            
        # 2. 如果找到了升序对  
        if i >= 0:  
            # 从后向前找第一个大于nums[i]的数  
            j = len(nums) - 1  
            while j > i and nums[j] <= nums[i]:  
                j -= 1  
            # 交换  
            nums[i], nums[j] = nums[j], nums[i]  
        
        # 3. 反转i之后的部分  
        reverse(nums, i + 1, len(nums) - 1)
```

### 287. 寻找重复数

不能修改数组完全没思路哎，还真是快慢指针，索引对应的i.next = nums[i]，然后就是找环的入口

```python
class Solution:  
    def findDuplicate(self, nums: List[int]) -> int:  
        # 1. 找到相遇点  
        slow = fast = nums[0]  
        while True:  
            slow = nums[slow]          # 慢指针走一步  
            fast = nums[nums[fast]]    # 快指针走两步  
            if slow == fast:           # 比较指针位置而不是值  
                break  
        
        # 2. 找到环的入口  
        slow = nums[0]                 # 慢指针回到起点  
        while slow != fast:            # 两个指针都走一步，直到相遇  
            slow = nums[slow]  
            fast = nums[fast]  
        
        return slow

```

