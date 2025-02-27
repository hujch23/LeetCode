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
时间复杂度：O(n+m)，其中n和m分别为两个链表的长度。因为每次循环迭代中，l1和l2只有一个元素会被放进合并链表中， 因此while循环的次数不会超过两个链表的长度之和
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
知道应该怎么做，但是写的时候逻辑不是很严谨，特别是进位判断写的乱七八糟，最好用数值存储结果计算，能够有效处理各种边界，时间复杂度：O(max(N,M))，其中 N 和 M 是两个链表的长度，空间复杂度：O(max(N,M))，需要创建一个新的链表存储结果
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


### 19. 删除链表的倒数第N个节点
傻逼一样想到的是翻转再翻转链表，还不如直接循环计算长度呢。使用双指针能够一遍扫描，快指针先走n步
```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:

        dummy = ListNode(0)
        dummy.next = head

        slow = fast = dummy

        for _ in range(n):
            fast = fast.next
        
        while fast.next:
            slow = slow.next
            fast = fast.next

        slow.next = slow.next.next

        return dummy.next
```

### 24. 两两交换链表中的节点
注意和链表翻转的交换是不一样的，两个题要快速捋清楚写出来
```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:

        if not head or not head.next:
            return head

        dummy = ListNode(0)
        dummy.next = head
        pre = dummy

        while pre.next and pre.next.next:
            left = pre.next
            right = pre.next.next

            left.next = right.next
            right.next = left
            pre.next = right

            pre = left

        return dummy.next
  ```

### 25. K个一维数组翻转
- 链表分区为已翻转部分+待翻转部分+未翻转部分
- 每次翻转前，要确定翻转链表的范围，这个必须通过k此循环来确定
- 需记录翻转链表前驱和后继，方便翻转完成后把已翻转部分和未翻转部分连接起来
- 初始需要两个变量pre和end，pre代表待翻转链表的前驱，end代表待翻转链表的末尾
- 经过k此循环，end到达末尾，记录待翻转链表的后继next = end.next
- 翻转链表，然后将三部分链表连接起来，然后重置pre和end指针，然后进入下一次循环
- 特殊情况，当翻转部分长度不足k时，在定位end完成后，end==null，已经到达末尾，说明题目已完成，直接返回即可
![image](https://github.com/user-attachments/assets/4ca0f2a2-cd68-4d1b-9dac-aaeec7960b46)

```python
class Solution:  
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:  
        
        dummy = ListNode(next  = head)
        pre = end = dummy

        def reverse(head):
            pre = None
            cur = start
            while cur:
                temp = cur.next
                cur.next = pre
                pre = cur
                cur = temp
            return pre

        while end.next:

            # 寻找翻转的起点和终点
            for _ in range(k):
                end = end.next
                if not end:
                    return dummy.next

            start = pre.next

            # 翻转前保留下一次翻转的起点
            next_begin = end.next

            # 翻转并重连
            pre.next = None  # 断开前面的连接  
            end.next = None  # 断开后面的连接
            pre.next = reverse(start)
            start.next = next_begin

            # 重置
            pre = end = start

        return dummy.next
```

### 138. 随机链表的复制

好歹看得懂题，思路是使用哈希映射，时间复杂度 O(N) ： 两轮遍历链表，使用 O(N) 时间。空间复杂度 O(N) ： 哈希表 dic 使用线性大小的额外空间
```python
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return None
        hash_map = {}

        cur = head
        while cur:
            hash_map[cur] = Node(cur.val)
            cur = cur.next

        cur = head
        while cur:
            if cur.next:
                hash_map[cur].next = hash_map[cur.next]
            if cur.random:
                hash_map[cur].random = hash_map[cur.random]

            cur = cur.next

        return hash_map[head]
```
还有一种方法是拼接 + 拆分 ABC--->AABBCC，然后第二个A的random不就是第一个A的random的next，主要是如何扩展以及去掉不需要的节点。时间复杂度 O(N) ： 三轮遍历链表，使用 O(N) 时间。
空间复杂度 O(1) ： 节点引用变量使用常数大小的额外空间
![image](https://github.com/user-attachments/assets/b2b782ce-cfea-489e-8f2b-401363ce5d4e)


```python
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return None
        
        # 链表扩展
        cur = head
        while cur:
            temp = Node(cur.val)
            temp.next = cur.next
            cur.next = temp
            cur = temp.next

        # 构建新节点的random指向
        cur = head
        while cur:
            if cur.random:
                cur.next.random = cur.random.next
            cur = cur.next.next

        # 链表拆分
        cur = res = head.next
        pre = head
        while cur.next:
            pre.next = pre.next.next
            cur.next = cur.next.next
            pre = pre.next
            cur = cur.next
        pre.next = None

        return res
```

### 148. 排序链表

逆天了这题：外层循环：size每次翻倍（1,2,4,8...）直到size大于等于链表长度，内层循环：每次取出两个长度为size的子链表，使用split函数分割链表，使用merge函数合并有序链表，维护已排序部分的尾节点tail

```python
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:

        if not head or not head.next:
            return head

        # 合并两个有序链表，返回合并后的头和尾  
        def merge(node1, node2):
            dummy = ListNode(0)
            tail = dummy

            while node1 and node2:
                if node1.val > node2.val:
                    tail.next = node2
                    node2 = node2.next
                else:
                    tail.next = node1
                    node1 = node1.next
                tail = tail.next

            tail.next = node1 if node1 else node2

            while tail.next:
                tail = tail.next

            return [dummy.next, tail]

        # 分割链表，返回第size个节点，并断开连接
        def split(head, size):
            if not head:
                return None
            for i in range(size-1):
                if not head.next:
                    break
                head = head.next

            next_head = head.next
            head.next = None

            return next_head

        dummy = ListNode(0)
        dummy.next = head

        

        # 计算链表长度
        cur = head
        length = 0
        while cur:
            length += 1
            cur = cur.next
        

        # size表示每次归并的子链表长度，从1开始，每次翻倍  
        size = 1
        while size < length:
            cur = dummy.next
            tail = dummy  # tail表示分割好的结尾

            while cur: # 因为需要遍历链表进行分割
                if not cur.next:
                    tail.next = cur
                    break
                left = cur
                right = split(left, size)  # 分割出第二个链表的头 
                cur = split(right, size)  # 分割出下一次归并的头

                merged = merge(left, right)
                tail.next = merged[0]   # 连接已排序部分  
                tail = merged[1]  # 更新tail为合并后的尾节点 

            size *= 2

        return dummy.next
```
       


### 23 合并K个升序链表

简单粗暴直接凉凉，时间复杂度：O(k^2n)，空间复杂度：没有用到与 k 和 n 规模相关的辅助空间，故渐进空间复杂度为 O(1)。
```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:

        def merge(node1, node2):
            dummy = ListNode(0)
            cur = dummy
            while node1 and node2:
                if node1.val > node2.val:
                    cur.next = node2
                    node2 = node2.next
                else:
                    cur.next = node1
                    node1 = node1.next
                cur = cur.next
            cur.next = node1 if node1 else node2

            return dummy.next
        
        if len(lists) == 0:
            return None
        
        if len(lists) == 1:
            return lists[0]
        
        # 初始化结果为第一个链表  
        result = lists[0]  
        
        # 两两合并  
        for i in range(1, len(lists)):  
            if lists[i]:  # 确保当前链表不为空  
                result = merge(result, lists[i])  
                
        return result 
 ```

![image](https://github.com/user-attachments/assets/0b582e7e-87ba-4565-82bf-89033efcd819)

 ```python     
class Solution:  
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:  
        heap = []  
        
        # 1. 初始化堆：将所有链表的第一个节点入堆  
        for i, node in enumerate(lists):  
            if node:  
                heappush(heap, (node.val, i, node))  
                # 三元组：(节点值，链表索引，节点引用)  
                # 节点值用于排序  
                # 链表索引用于在值相同时保持稳定性  
                # 节点引用用于构建结果链表  
        
        dummy = ListNode(0)  
        cur = dummy  
        
        # 2. 不断从堆中取出最小值，并添加下一个节点  
        while heap:  
            val, i, node = heappop(heap)  # 取出最小值  
            cur.next = node               # 连接到结果链表  
            cur = cur.next               # 移动指针  
            if node.next:                # 如果还有后续节点  
                heappush(heap, (node.next.val, i, node.next))  # 加入堆中  
        
        return dummy.next
```


### 146. LRU缓存
真要命的题，添加到头部：add_to_head()，哈希表存储key到节点的映射，删除节点：remove_node()，移动到头部：move_to_head()，删除尾部：remove_tail()
 ```python 
class ListNode:
    def __init__(self, key, value, pre = None, next= None):
        self.key = key
        self.value = value
        self.pre = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tail = ListNode(0, 0)
        self.head = ListNode(0, 0)
        self.tail.pre = self.head
        self.head.next = self.tail
        self.hash_map = {}

    def get(self, key: int) -> int:
        if key in self.hash_map:
            node =  self.hash_map[key]
            # 因为进行了查询，所以需要移动到最前边
            self.move_to_head(node)
            return node.value
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        # 如果key在hash_map中，替换值并移动到最前
        if key in self.hash_map:
            self.hash_map[key].value = value
            self.move_to_head(self.hash_map[key])
        else:
            node = ListNode(key, value)
            self.hash_map[key] = node
            self.add_to_head(node)
            # 如果超出容量，删除尾部节点  
            if len(self.hash_map) > self.capacity:  
                self.remove_tail()
    
    def move_to_head(self, node):

        # 先将node去掉再移到最前边
        self.remove_node(node)
        self.add_to_head(node)

    def add_to_head(self, node):
        node.next = self.head.next
        node.pre = self.head
        self.head.next.pre = node
        self.head.next = node

    def remove_node(self, node):
        node.pre.next = node.next
        node.next.pre = node.pre
        
    def remove_tail(self):
        last_node = self.tail.pre
        self.remove_node(last_node)
        self.hash_map.pop(last_node.key)
 ```

## 二叉树

### 94. 二叉树的中序遍历
二叉树的前、中以及后序遍历同统一递归模板
```python 
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:

        result = []
        

        def inorder(root):
            if not root:
                return 

            inorder(root.left)
            result.append(root.val)
            inorder(root.right)

        inorder(root)

        return result
 ```
也可使用遍历的方法，使用栈
```python 
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        
        stack = []
        result = []
        curr = root

        while stack or curr:
            while curr:
                stack.append(curr)
                curr = curr.left
            
            curr = stack.pop()
            result.append(curr.val)

            curr = curr.right

        return result
```
### 104. 二叉树的最大深度

递归实现：时间复杂度O(n)，空间复杂度O(h)，BFS实现：时间复杂度O(n)，空间复杂度O(w)，其中n是节点数，h是树高，w是最大宽度
```python 
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:

        if not root:
            return 0
        
        due = deque([root])
        max_depth = 0
        

        while due:
            max_depth += 1
            for _ in range(len(due)):
                node = due.popleft()

                if node.left:
                    due.append(node.left)

                if node.right:
                    due.append(node.right)

        return max_depth
```
```python 
class Solution:  
    def maxDepth(self, root: Optional[TreeNode]) -> int:  
        if not root:  
            return 0  
        
        # 返回左右子树的最大深度 + 1（当前层）  
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

### 226. 翻转二叉树
层次遍历即可
```python 
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:

        if not root:
            return None

        due = deque([root])
        while due:
            node = due.popleft()
            node.left, node.right = node.right, node.left 
            if node.left:
                due.append(node.left)
            if node.right:
                due.append(node.right)

        return root
```
### 101. 对称二叉树
有点印象但不多，稀里糊涂的逻辑不清晰
```python 
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:

        if not root:
            return True

        due = deque([(root.left, root.right)])

        while due:
            left, right = due.popleft()
            if not left and not right:
                continue
            if not left or not right or left.val != right.val:
                return False

            due.append((left.left, right.right))
            due.append((left.right, right.left))
            
        return True
```

### 543. 二叉树的直径
得多多联系递归啊！！！！！！！
```python 
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:  
        self.max_diameter = 0  
    
        def depth(node):  
            if not node:  
                return 0  
        
            left_depth = depth(node.left)   # 左子树深度  
            right_depth = depth(node.right) # 右子树深度  
            
            # 计算经过当前节点的最长路径（边的数量）  
            # left_depth + right_depth 就是边的数量  
            self.max_diameter = max(self.max_diameter, left_depth + right_depth)  
            
            # 返回当前节点到叶子节点的最大深度  
            return max(left_depth, right_depth) + 1  # 这里要加1，因为是计算深度  
    
        depth(root)  
        return self.max_diameter
```

### 102. 二叉树的层序遍历
```python 
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        result = []
        if not root:
            return []

        due = deque([root])
        while due:
            cur_level = []
            level_size = len(due)

            for i in range(level_size):
                node = due.popleft()
                cur_level.append(node.val)

                if node.left:
                    due.append(node.left)
                if node.right:
                    due.append(node.right)
            result.append(cur_level)
        return result
```

### 108. 将有序数组转换为二叉搜索树

知道对应的就是中序数组，但是好像不会写数组转二叉树，看到二叉树就要想到递归！！！！！！！！！
```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        
        def traversal(nums, left, right):
            if left > right:
                return None

            mid = (right + left) // 2

            root = TreeNode(nums[mid])
            root.left = traversal(nums, left, mid-1)
            root.right = traversal(nums, mid+1, right)

            return root

        return traversal(nums, 0, len(nums)-1)
```

### 98. 验证二叉搜索树

中序遍历验证是否是递增数组就行
```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:

        result = []

        def inorder(root):
            if not root:
                return 
            inorder(root.left)
            result.append(root.val)
            inorder(root.right)

        inorder(root)

        for i in range(len(result)-1):
            if result[i] >= result[i+1]:
                return False

        return True
```

### 203. 二叉搜索树中的第K小的元素
继续中序遍历即可，但是进阶要求如果树频繁变咋整，用中序遍历数组每次都得变，既然是二叉搜索树，那么左子树小于根节点小于右子树，那就看看k与左子树的节点数的比较区搜索呗
```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:

        
        nums = []

        def inorder(root):
            if not root:
                return 

            inorder(root.left)
            nums.append(root.val)
            inorder(root.right)

        inorder(root)

        return nums[k-1]
```
```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        
        left_count = self.findChildren(root.left)

        if left_count + 1 == k:
            return root.val
        elif left_count + 1 < k:
            return self.kthSmallest(root.right, k - left_count - 1)
        else:
            return self.kthSmallest(root.left, k)
    
    def findChildren(self, root):
        if not root:
            return 0
        return self.findChildren(root.left) + self.findChildren(root.right) + 1
```

### 193. 二叉树的右视图
层序遍历得到每层的节点值即可，每次留下最右边的，但有一说一，递归的方法真的叼
```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:

        if not root:
            return []
        
        due = deque([root])
        res = []
        while due:
            cur_level = []
            level_size = len(due)
            for _ in range(len(due)):
                node = due.popleft()
                cur_level.append(node.val)
                if node.left:
                    due.append(node.left)
                if node.right:
                    due.append(node.right)
            if len(cur_level) > 0:
                res.append(cur_level[-1])
                
        return res
```
```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        def dfs(node: Optional[TreeNode], depth: int) -> None:
            if node is None:
                return
            if depth == len(ans):  # 这个深度首次遇到
                ans.append(node.val)
            dfs(node.right, depth + 1)  # 先递归右子树，保证首次遇到的一定是最右边的节点
            dfs(node.left, depth + 1)
        dfs(root, 0)
        return ans

```

### 114. 二叉树展开为链表
思路是有的，就是慢慢移，但是代码写的不三不四，既然是左边移到右边，那就检测左边，暂留右边。当然这个题还可以使用递归或者迭代实现

```python
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        while root:
            if root.left:
                right_subtree = root.right
                root.right = root.left
                root.left = None

                cur = root
                while cur.right:
                    cur = cur.right
                cur.right = right_subtree

            root = root.right
```

### 105. 从前序遍历与中序遍历构造二叉树

大概记得点点思路，在下标画横线就想起来了，但是不会用递归，尴尬死了，递归简洁版使用index，此方法空间复杂度提高
```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder:
            return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])

        root.left = self.buildTree(preorder[1:mid+1], inorder[0:mid])
        root.right = self.buildTree(preorder[mid+1:], inorder[mid+1:])

        return root
       
```

通过传递 preorder 和 inorder 的索引范围，而不是直接切片，避免创建新的列表。这样可以将切片操作的空间复杂度从 O(n^2)降低到O(mn)

```python
class Solution:  
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:  
        # 使用一个哈希表来快速定位 inorder 中的索引  
        inorder_index_map = {val: idx for idx, val in enumerate(inorder)}  

        def helper(pre_start, pre_end, in_start, in_end):  
            # 如果索引范围无效，返回 None  
            if pre_start > pre_end or in_start > in_end:  
                return None  

            # 根节点的值是 preorder 的第一个元素  
            root_val = preorder[pre_start]  
            root = TreeNode(root_val)  

            # 找到根节点在 inorder 中的位置  
            mid = inorder_index_map[root_val]  

            # 左子树的节点数量  
            left_size = mid - in_start  

            # 构建左子树  
            root.left = helper(pre_start + 1, pre_start + left_size, in_start, mid - 1)  

            # 构建右子树  
            root.right = helper(pre_start + left_size + 1, pre_end, mid + 1, in_end)  

            return root  

        # 调用辅助函数，初始范围是整个 preorder 和 inorder  
        return helper(0, len(preorder) - 1, 0, len(inorder) - 1)

```

### 437. 路径总和
印象中是递归加回溯，重点是搞清楚输入参数以及递归流程，记得之前做过的前缀和？
```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:

        self.hash_map = {0:1}

        return self.dfs(root, 0, targetSum)

    def dfs(self, root, curr_sum, targetSum):
        if not root:
            return 0
        curr_sum += root.val
        count = self.hash_map.get(curr_sum - targetSum, 0)
        self.hash_map[curr_sum] = self.hash_map.get(curr_sum, 0) + 1

        count += self.dfs(root.left, curr_sum, targetSum)
        count += self.dfs(root.right, curr_sum, targetSum)

        self.hash_map[curr_sum] -= 1

        return count
```

### 236. 二叉树的最近公共祖先
背下来吧，垃圾题目完全不理解
```python
class Solution:  
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':  
        # 如果当前节点为空，或者当前节点是 p 或 q，则直接返回当前节点  
        if not root or root == p or root == q:  
            return root  

        # 在左子树中递归查找最近公共祖先  
        left = self.lowestCommonAncestor(root.left, p, q)  
        # 在右子树中递归查找最近公共祖先  
        right = self.lowestCommonAncestor(root.right, p, q)  

        # 如果左右子树的递归结果都不为空，说明 p 和 q 分别在当前节点的左右子树中  
        if left and right:  
            return root  

        # 如果只有一边不为空，说明最近公共祖先在这一边  
        return left if left else right  
```


### 124. 二叉树中的最大路径和
记住，递归返回的只能取一边，但是计算最大值时都需要加上，同时负数去掉
```python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.maxPath = float('-inf')
        self.dfs(root)
        return self.maxPath

    def dfs(self, root):
        if not root:
            return 0

        left = max(self.dfs(root.left), 0)
        right = max(self.dfs(root.right), 0)

        self.maxPath = max(self.maxPath, left + right + root.val)

        return max(left, right) + root.val
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

### 46. 全排列

感觉回溯忘记的一干二净了，凭着记忆写出来了一点点.核心思路是：用一个数组记录当前路径，用一个数组标记已经使用过的数字，在每一步中，选择一个未使用的数字加入路径，递归处理下一个位置，回溯时撤销选择，继续尝试其他可能
```python
class Solution:  
    def permute(self, nums: List[int]) -> List[List[int]]:  
        res = []  
        path = []  # 添加路径数组  
        used = [False] * len(nums)  # 添加访问标记数组  
        self.dfs(nums, path, used, res)  
        return res  # 添加返回值  

    def dfs(self, nums, path, used, res):  
        # 当路径长度等于nums长度时，找到一个排列  
        if len(path) == len(nums):  
            res.append(path[:])  
            return  

        for i in range(len(nums)):  
            # 如果当前数字已经使用过，跳过  
            if used[i]:  
                continue  
            
            # 选择当前数字  
            path.append(nums[i])  
            used[i] = True  
            
            # 递归  
            self.dfs(nums, path, used, res)  
            
            # 回溯，撤销选择  
            path.pop()  
            used[i] = False
```

### 78. 子集

注意当前的循环会重复选择之前的元素，需要一个起始索引来避免重复；子集的长度确实应该 <= 原数组长度，但不应该在这里就返回，实际上每个路径都是一个有效的子集，应该无条件添加
```python
class Solution:  
    def subsets(self, nums: List[int]) -> List[List[int]]:  
        res = []  
        self.dfs(nums, [], 0, res)  # 添加起始索引参数  
        return res  

    def dfs(self, nums, path, start, res):  
        # 当前路径就是一个子集，直接添加  
        res.append(path[:])  
        
        # 从start开始，避免重复选择  
        for i in range(start, len(nums)):  
            path.append(nums[i])  
            self.dfs(nums, path, i + 1, res)  # 注意是i+1  
            path.pop()
```
### 17. 电话号码的组合

这道题的核心思路是：对于输入的每个数字（例如"23"），我们需要依次选择每个数字对应的字母表中的一个字母（2对应'abc'中选一个，3对应'def'中选一个），通过回溯的方式尝试所有可能的组合。具体来说，就是先固定第一个数字的一个字母，然后递归处理后面的数字，当选择的字母个数等于输入数字的长度时，就找到了一个有效组合。
```python
class Solution:  
    def letterCombinations(self, digits: str) -> List[str]:  
        # 处理空字符串情况  
        if not digits:  
            return []  
            
        hash_map = {  
            '2': ['a','b','c'],  
            '3': ['d','e','f'],  
            '4': ['g','h','i'],  
            '5': ['j','k','l'],  
            '6': ['m','n','o'],  
            '7': ['p','q','r','s'],  
            '8': ['t','u','v'],  
            '9': ['w','x','y','z']  
        }  
        
        res = []  
        self.dfs(digits, [], 0, res, hash_map)  
        return res  
    
    def dfs(self, digits, path, index, res, hash_map):  
        # 找到一个组合  
        if len(path) == len(digits):  
            res.append(''.join(path))  
            return  
            
        # 获取当前数字对应的字母列表  
        curr_digit = digits[index]  
        letters = hash_map[curr_digit]  
        
        # 遍历当前数字对应的所有字母  
        for letter in letters:  
            path.append(letter)  # 选择一个字母  
            self.dfs(digits, path, index + 1, res, hash_map)  
            path.pop()  # 回溯
```

### 39. 组合总和

![image](https://github.com/user-attachments/assets/effdcc9d-eeb5-4ba3-adcd-865696443741)
![image](https://github.com/user-attachments/assets/9026580d-2f07-45c6-a1fa-46d93463bb0b)


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

