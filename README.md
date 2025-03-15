# LeetCode hot 100 刷题记录

**链表超级高频**：反转链表、反转链表II、LRU缓存

**数组超级高频**：接雨水、三数之和

**二叉树超高频**：构造二叉树、二叉树的公共祖先

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

### 49. ⚠️ 字母异位词分组

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

### 128. ⚠️ 最长连续序列
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

### 15. ⚠️ 三数之和

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
      
###  42. 接雨水

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

### 3.  ⚠️ 无重复字符的最长字串
  遍历的时候寻找窗口左边界，即重复出现的索引位置，时间复杂度：O(N)，其中N是字符串的长度。左指针和右指针分别会遍历整个字符串一次。空间复杂度：O(∣Σ∣)，其中Σ表示字符集（即字符串中可以出现的字符），∣Σ∣表示字符集的大小。在本题中没有明确说明字符集，因此可以默认为所有 ASCII 码在[0,128)内的字符，即∣Σ∣=128。我们需要用到哈希集合来存储出现过的字符，而字符最多有∣Σ∣个，因此空间复杂度为O(∣Σ∣)。

```python
def lengthOfLongestSubstring(self, s: str) -> int:  
    hash = {}          # 存储字符最后出现的位置  
    max_len = 0        # 最长无重复子串的长度  
    start = 0          # 当前无重复子串的起始位置  
    
    for i, char in enumerate(s):  
        # 如果字符重复，更新起始位置  
        start = max(start, hash.get(char, -1) + 1)  
        # 更新最大长度  
        max_len = max(max_len, i - start + 1)  
        # 记录字符位置  
        hash[char] = i  
        
    return max_len
  ```


### 438. ⚠️ 找到字符串中所有字母的异位词

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

### 560. ⚠️ 和为K的子数组
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

### 239. ⚠️ 滑动窗口最大值
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

### 76. ⚠️  最小覆盖字串

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

### 41. ⚠️ 缺失的第一个正数

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


### 206. 反转链表
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

### 92. 反转链表II
![image](https://github.com/user-attachments/assets/018d5864-c3f9-494c-81db-fec24e7c96ce)

```python
class Solution:  
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:  
        # 如果链表为空或不需要反转  
        if not head or left == right:  
            return head  
        
        # 创建虚拟头节点，简化边界处理  
        dummy = ListNode(0)  
        dummy.next = head  
        prev = dummy  
        
        # 找到反转起始位置的前一个节点  
        for _ in range(left - 1):  
            prev = prev.next  
        
        # 开始反转  
        start = prev.next  
        then = start.next  
        
        # 反转 left 到 right 之间的节点  
        for _ in range(right - left):  
            start.next = then.next  
            then.next = prev.next  
            prev.next = then  
            then = start.next  
        
        return dummy.next  
```

### 234. ⚠️回文链表

```python
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        def reverse(head):
            pre = None
            cur = head
            while cur:
                temp = cur.next
                cur.next = pre
                pre = cur
                cur = temp
            return pre

        slow = fast = head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        
        right = reverse(slow.next)
        slow.next = None
        left = head

        while left and right:
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        return True  
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

### 25. ⚠️ K个一维数组翻转
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

### 138.  ⚠️ 随机链表的复制

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

### 148. ⚠️ 排序链表

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
       


### 23 ⚠️ 合并K个升序链表

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


### 146. ⚠️ LRU缓存
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

大概记得点点思路，在下标画横线就想起来了，但是不会用递归，尴尬死了，递归简洁版使用index，此方法时间复杂度提高
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

时间复杂度从 O(n^2)降低到O(n)

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
### 200. 岛屿数量
DFS：从1开始搜索，需要注意的是搜索完的地方可以进行标记，递归的终止条件就是出边界了或者四个方向都是0
```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:

        # 用DFS搜，找有多少个连接在一块的1
        counter = 0
        m, n = len(grid), len(grid[0])
        def dfs(i, j):
            if i < 0 or i >=m or j < 0 or j >=n:
                return
            if grid[i][j] != "1":
                return 
            grid[i][j] = "0"

            dfs(i + 1, j) 
            dfs(i - 1, j) 
            dfs(i, j + 1) 
            dfs(i, j - 1)
       
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    counter += 1
                    dfs(i, j)

        return counter
```

### 994. 腐烂的橘子

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:

        # BFS首先找到所有腐烂橘子的位置
        queue = deque()  # 记录腐橘子的位置以及时间
        fresh_count = 0
        min_time = 0
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.append((i, j, 0))
                elif grid[i][j] == 1:
                    fresh_count += 1

        if fresh_count == 0:
            return 0

        while queue:
            curr_i, curr_j, time = queue.popleft()
            directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
            for di, dj in directions:
                next_di = curr_i + di
                next_dj = curr_j + dj
                min_time = max(min_time, time)
                if 0<=next_di<m and 0<=next_dj<n and grid[next_di][next_dj] == 1:
                    grid[next_di][next_dj] = 2
                    queue.append((next_di, next_dj, time + 1))
                    fresh_count -= 1
        return min_time if fresh_count == 0 else -1
```

### 207 课程表
用DFS检测有向图中是否存在环。首先把课程依赖关系转换成图（用邻接表表示），然后用一个visited数组记录节点的访问状态（0未访问，1正在访问，2已完成访问）。在DFS遍历过程中，如果遇到状态为1的节点（正在访问），就说明存在环，返回False；如果遍历完所有节点都没有发现环，就返回True表示可以完成所有课程
![image](https://github.com/user-attachments/assets/b2573e09-119e-4c98-8af1-aec1e216f791)

```python
class Solution:  
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:  
        # 构建图  
        graph = [[] for _ in range(numCourses)]  
        for cur, pre in prerequisites:  
            graph[pre].append(cur)  
        
        # 访问状态数组：0 = 未访问，1 = 正在访问，2 = 已完成访问  
        visited = [0] * numCourses  

        # 深度优先搜索函数  
        def dfs(course):  
            if visited[course] == 1:  # 如果当前节点正在访问，说明存在环  
                return False  
            if visited[course] == 2:  # 如果当前节点已经访问完成，直接返回 True  
                return True  
            
            # 标记当前节点为正在访问  
            visited[course] = 1  

            # 递归访问所有邻居节点  
            for neighbor in graph[course]:  
                if not dfs(neighbor):  # 如果邻居节点检测到环，返回 False  
                    return False  
            
            # 标记当前节点为访问完成  
            visited[course] = 2  
            return True  

        # 遍历所有课程，检查是否存在环  
        for course in range(numCourses):  
            if not dfs(course):  # 如果某个课程检测到环，返回 False  
                return False  
        
        return True
```





### 208 实现Trie（前缀树）
用一个树形结构，其中每个节点存储一个字符和一个字典（children），字典用来指向子节点。从根节点开始，沿着路径走就能形成字符串。每个节点还有一个标记(is_end)表示是否是完整单词的结尾。插入时逐字符建立路径，查找时逐字符检查路径是否存在，区别在于查找完整单词时需要检查is_end标记，而查找前缀则不需要
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        if node.is_end:
            return True
        else:
            return False
            
    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```


## 二分查找

### 35. 搜索插入位置
```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:

        left = 0
        right = len(nums) - 1

        while left <= right:
            middle = (left + right) //2
            if nums[middle] == target:
                return middle
            elif nums[middle] > target:
                right = middle - 1
            else:
                left = middle + 1  
                
        return right + 1   
```
### 74. 搜索二维矩阵
直接将二维矩阵视为一维数组进行二分查找，时间复杂度为 O(log(m*n))，是最优的
```python
class Solution:  
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:  
        if not matrix or not matrix[0]:  
            return False  
        
        rows, cols = len(matrix), len(matrix[0])  
        left, right = 0, rows * cols - 1  
        
        while left <= right:  
            mid = (left + right) // 2  
            # 将一维索引转换为二维坐标  
            row, col = mid // cols, mid % cols  
            
            if matrix[row][col] == target:  
                return True  
            elif matrix[row][col] < target:  
                left = mid + 1  
            else:  
                right = mid - 1  
                
        return False
```

### 34. ⚠️ 查找元素的第一个和最后一个

最优解应该是使用两次二分查找：第一次二分查找找左边界，第二次二分查找找右边界，这样可以将时间复杂度优化到 O(log n)

```python
class Solution:  
    def searchRange(self, nums: List[int], target: int) -> List[int]:  
        def findBound(nums, target, isFirst):  
            left, right = 0, len(nums) - 1  
            
            while left <= right:  
                mid = (left + right) // 2  
                if nums[mid] == target:  
                    if isFirst:  
                        # 如果是查找左边界，即使找到了目标值也要继续向左找  
                        if mid == 0 or nums[mid-1] != target:  
                            return mid  
                        right = mid - 1  
                    else:  
                        # 如果是查找右边界，即使找到了目标值也要继续向右找  
                        if mid == len(nums)-1 or nums[mid+1] != target:  
                            return mid  
                        left = mid + 1  
                elif nums[mid] < target:  
                    left = mid + 1  
                else:  
                    right = mid - 1  
            return -1  
        
        if not nums:  
            return [-1, -1]  
            
        left = findBound(nums, target, True)  
        if left == -1:  
            return [-1, -1]  
        right = findBound(nums, target, False)  
        
        return [left, right]
```

### ⚠️ 搜索旋转排序数组

每次二分后，一定有一半是有序的，判断哪一半有序：通过比较 nums[left] 和 nums[mid]，判断目标值在哪一半：，如果在有序的那一半，用普通二分查找，如果不在有序的那一半，去另一半找
```python
class Solution:  
    def search(self, nums: List[int], target: int) -> int:  
        if not nums:  
            return -1  
            
        left, right = 0, len(nums) - 1  
        
        # 找到旋转点  
        while left <= right:  # 使用 <=   
            mid = (left + right) // 2  
            if nums[mid] == target:  
                return mid  
                
            # 判断哪一部分是有序的  
            if nums[left] <= nums[mid]:  # 左半部分有序  
                if nums[left] <= target < nums[mid]:  
                    right = mid - 1  
                else:  
                    left = mid + 1  
            else:  # 右半部分有序  
                if nums[mid] < target <= nums[right]:  
                    left = mid + 1  
                else:  
                    right = mid - 1  
                    
        return -1
```

### 153. 寻找旋转排序数组中的最小值
```python
class Solution:  
    def findMin(self, nums: List[int]) -> int:  
        left = 0  
        right = len(nums) - 1  
        
        while left < right:  
            mid = (left + right) // 2  
            
            if nums[mid] > nums[right]:  
                # 最小值在右半部分  
                left = mid + 1  
            else:  
                # 最小值在左半部分（包括mid）  
                right = mid  
                
        return nums[left]
```
### 4.寻找两个正序数组的中位数 
碰到这个题直接摆烂
```python
class Solution:  
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:  
        # 确保 nums1 的长度更短，这样二分查找的范围更小，更有效率  
        if len(nums1) > len(nums2):  
            nums1, nums2 = nums2, nums1  
        
        m, n = len(nums1), len(nums2)  
        
        # 计算合并后数组的左半部分应有的元素个数  
        # 对于总长度为奇数的情况：左半部分多一个元素  
        # 对于总长度为偶数的情况：左右部分元素个数相等  
        total_left = (m + n + 1) // 2  
        
        # 在 nums1 中二分查找分割点  
        # left和right表示nums1可能的分割位置范围  
        left, right = 0, m  
        
        while left <= right:  
            # i 是 nums1 的分割点：nums1[0..i-1] | nums1[i..m-1]  
            i = (left + right) // 2  
            # j 是 nums2 的分割点：nums2[0..j-1] | nums2[j..n-1]  
            # i + j = total_left，保证左半部分总数符合要求  
            j = total_left - i  
            
            # 获取分割点周围的四个数  
            # nums1_left：nums1分割点左边的数  
            # nums1_right：nums1分割点右边的数  
            # nums2_left：nums2分割点左边的数  
            # nums2_right：nums2分割点右边的数  
            
            # 处理边界情况：如果分割点在数组边缘  
            nums1_left = float('-inf') if i == 0 else nums1[i-1]  
            nums1_right = float('inf') if i == m else nums1[i]  
            nums2_left = float('-inf') if j == 0 else nums2[j-1]  
            nums2_right = float('inf') if j == n else nums2[j]  
            
            # 判断分割是否合适  
            # 合适的分割需要满足：左半部分的最大值 <= 右半部分的最小值  
            if nums1_left <= nums2_right and nums2_left <= nums1_right:  
                # 找到合适的分割点  
                if (m + n) % 2 == 0:  
                    # 如果总长度为偶数  
                    # 中位数 = (左半部分最大值 + 右半部分最小值) / 2  
                    return (max(nums1_left, nums2_left) +   
                           min(nums1_right, nums2_right)) / 2  
                else:  
                    # 如果总长度为奇数  
                    # 中位数就是左半部分的最大值  
                    return max(nums1_left, nums2_left)  
                    
            elif nums1_left > nums2_right:  
                # 如果nums1左边的值大于nums2右边的值  
                # 说明nums1的分割点太靠右了，需要向左移动  
                right = i - 1  
            else:  
                # 如果nums2左边的值大于nums1右边的值  
                # 说明nums1的分割点太靠左了，需要向右移动  
                left = i + 1  

# 举例说明：  
# nums1 = [1, 3, 5]  
# nums2 = [2, 4, 6]  
# 总长度 = 6（偶数）  
# total_left = (6 + 1) // 2 = 3  

# 假设 i = 1：  
# nums1: [1 | 3, 5]       i = 1  
# nums2: [2, 4 | 6]       j = 2  
# nums1_left = 1  
# nums1_right = 3  
# nums2_left = 4  
# nums2_right = 6  

# 检查是否满足条件：  
# nums1_left(1) <= nums2_right(6) √  
# nums2_left(4) <= nums1_right(3) ×  
# 不满足条件，需要增大i  

# 最终找到正确的分割：  
# nums1: [1, 3 | 5]  
# nums2: [2 | 4, 6]  
# 中位数 = (max(3,2) + min(5,4)) / 2 = (3 + 4) / 2 = 3.5
```

## 堆
需要学习：如何构造堆、快排、桶排
### 215. ⚠️ 数组中的第K个最大元素
无脑背堆的相关函数就行，思路是很简单的
```python
class Solution:  
    def findKthLargest(self, nums: List[int], k: int) -> int:  
        # 创建一个最小堆  
        # 我们维护一个大小为k的最小堆，这样堆顶就是第k大的元素  
        heap = []  

        # 遍历数组中的每个元素  
        for i in range(len(nums)):  
            # 情况1：堆的大小小于k，直接将元素加入堆中  
            if len(heap) < k:  
                heapq.heappush(heap, nums[i])  
            # 情况2：堆的大小等于k  
            else:  
                # 如果当前元素大于堆顶元素  
                # 说明找到了一个更大的元素，应该替换掉堆顶  
                if nums[i] > heap[0]:  
                    # heapreplace 等价于先 heappop 再 heappush  
                    # 但是效率更高，因为只需要一次向下调整  
                    heapq.heapreplace(heap, nums[i])  
        
        # 返回堆顶元素，即第k大的数  
        return heap[0]
```
如果面试让手写堆，那咱就手搓呗：
```python
class Solution:  
    def findKthLargest(self, nums: List[int], k: int) -> int:  
        # 初始化一个大小为k的最小堆  
        heap = []  
        
        # 向下调整堆，维护最小堆性质  
        def sift_down(arr, start, end):  
            root = start  
            while True:  
                # 找到左子节点  
                child = 2 * root + 1  
                # 如果左子节点超出范围，说明已经是叶子节点，结束调整  
                if child > end:  
                    break  
                # 如果右子节点存在且小于左子节点，选择右子节点  
                if child + 1 <= end and arr[child + 1] < arr[child]:  
                    child += 1  
                # 如果子节点小于根节点，交换位置  
                if arr[child] < arr[root]:  
                    arr[root], arr[child] = arr[child], arr[root]  
                    root = child  
                else:  
                    break  
        
        # 向上调整堆，维护最小堆性质  
        def sift_up(arr, child):  
            while child > 0:  
                parent = (child - 1) // 2  
                if arr[parent] > arr[child]:  
                    arr[parent], arr[child] = arr[child], arr[parent]  
                    child = parent  
                else:  
                    break  
        
        # 遍历数组  
        for num in nums:  
            if len(heap) < k:  
                # 堆未满，将元素加入堆底，然后向上调整  
                heap.append(num)  
                sift_up(heap, len(heap) - 1)  
            elif num > heap[0]:  
                # 堆已满，且当前元素大于堆顶  
                # 替换堆顶元素，然后向下调整  
                heap[0] = num  
                sift_down(heap, 0, len(heap) - 1)  
        
        return heap[0]
```
```python
class Solution:
    def findKthLargest(self, nums, k):
        def quick_select(nums, k):
            # 随机选择基准数
            pivot = random.choice(nums)
            big, equal, small = [], [], []
            # 将大于、小于、等于 pivot 的元素划分至 big, small, equal 中
            for num in nums:
                if num > pivot:
                    big.append(num)
                elif num < pivot:
                    small.append(num)
                else:
                    equal.append(num)
            if k <= len(big):
                # 第 k 大元素在 big 中，递归划分
                return quick_select(big, k)
            if len(nums) - len(small) < k:
                # 第 k 大元素在 small 中，递归划分
                return quick_select(small, k - len(nums) + len(small))
            # 第 k 大元素在 equal 中，直接返回 pivot
            return pivot
        
        return quick_select(nums, k)
```
### 347. ⚠️ 前K个高频元素
还是调包吧，用哈希表统计元素频率，然后对频率进行堆排序就行，相比上边的题就多个哈希表
```python
class Solution:  
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:  
        # 统计每个数字出现的频率  
        hash_map = {}  
        for num in nums:  
            hash_map[num] = hash_map.get(num, 0) + 1  
        
        # 创建最小堆，存储(频率,数字)元组  
        heap = []  
        
        # 遍历哈希表的所有键值对  
        for num, freq in hash_map.items():  
            if len(heap) < k:  
                # 堆未满，直接加入(频率,数字)  
                heapq.heappush(heap, (freq, num))  
            else:  
                # 堆已满，且当前频率大于堆顶频率  
                if freq > heap[0][0]:  
                    # 替换堆顶元素  
                    heapq.heapreplace(heap, (freq, num))  
        
        # 提取堆中的数字（第二个元素）  
        return [item[1] for item in heap]
```

### ⚠️ 295 数据流中的中位数
用两个堆把数据分成两半：大顶堆存较小的一半，小顶堆存较大的一半；保持大顶堆的大小等于或比小顶堆多一个；添加数字时，通过在两个堆之间倒腾，保证大顶堆的最大值小于小顶堆的最小值；中位数就是：当总数为偶数时：两个堆顶的平均值；当总数为奇数时：大顶堆的堆顶（注意python默认是小顶堆）
```python
class MedianFinder:

    def __init__(self):
        self.min_head = []
        self.max_head = []
        

    def addNum(self, num: int) -> None:

        if len(self.min_head) == len(self.max_head):
            # 先放进小顶堆，然后把最小的放大顶堆，保持大顶堆数量多一个
            heapq.heappush(self.min_head, num)
            small = heapq.heappop(self.min_head)
            heapq.heappush(self.max_head, -small)


        else:
            # 先放大顶堆，然后把最大的放小顶堆
            heapq.heappush(self.max_head, -num)
            large = - heapq.heappop(self.max_head)
            heapq.heappush(self.min_head, large)

        

    def findMedian(self) -> float:
        # 大堆等于小堆，相加取平均
        if len(self.min_head) == len(self.max_head):
            return (self.min_head[0] - self.max_head[0]) /2
        
        else:
            return -self.max_head[0]
```

## 栈
### 20. 有效的括号
这个题思路是很简单的，利用栈的特性即可，但是我在写代码中遇到了两个问题。一是没有定义对应关系直接判断，习惯默认左右括号是对应的了；二是忽略了有可能一开始就是右括号所以栈为空的情况
```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
         # 定义括号对应关系  
        pairs = {')': '(', ']': '[', '}': '{'}  
        for char in s:
            if char == '(' or  char == '[' or  char == '{':
                stack.append(char)
            else:
                if not stack or pairs[char] != stack.pop():
                    return False

        return True if len(stack) == 0 else False
```

### 71. 简化路径
首先跳过连续的路径分隔符，读取每个有效的路径部分，根据路径部分的不同类型进行不同的处理，使用栈来维护最终的简化路径，实现返回上一级目录和忽略无效路径的逻辑，最后将栈中的路径部分重新组装成标准路径格式
```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []
        n = len(path)
        i = 0
        while i < n:
            cur_path = ""
            while i < n and path[i] == '/':  
                i += 1
            while i < n and path[i] != '/':
                cur_path += path[i]
                i += 1
            if cur_path == '..':
                if stack:
                    stack.pop()
            elif cur_path == '.' or cur_path == '':
                continue
            else:
                stack.append(cur_path)
            
            i += 1
        
        res = "/" + "/".join(stack)
```
### 224. 基本计算器

关键在于将符号状态压入栈中，并在遇到括号时动态调整当前的操作符，从而能够处理任意深度的括号嵌套和正负号变化
```python
class Solution:  
    def calculate(self, s: str) -> int:  
        stack = [1]  # 符号栈，初始为正号  
        ans = 0      # 最终结果  
        num = 0      # 当前数字  
        op = 1       # 当前操作符（1表示正号，-1表示负号）  

        for c in s:  
            if c == ' ':  
                continue  # 跳过空格  
            elif c.isdigit():  
                num = num * 10 + int(c)  # 构建数字  
            else:  
                ans += op * num  # 将当前数字加入结果  
                num = 0  # 重置数字  

                if c == '+':  
                    op = stack[-1]  # 使用栈顶的符号  
                elif c == '-':  
                    op = -stack[-1]  # 取栈顶符号的相反数  
                elif c == '(':  
                    stack.append(op)  # 将当前符号压入栈  
                elif c == ')':  
                    stack.pop()  # 括号结束，弹出最近的符号状态  

        return ans + op * num  # 处理最后一个数字 
```

### 155. 最小栈
设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈，这个题目的核心是如何在常数操作实现获取栈中最小值，思路就是利用一个辅助栈，说白了就是放进去时每次记录最小值，这样栈顶就是最小值了
```python
class MinStack:

    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
        else:
            self.stack.append((val, min(val, self.stack[-1][-1])))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    
    def getMin(self) -> int:
        return self.stack[-1][-1]
```

### 394. ⚠️ 字符串解码
这道题我的思路是正确的，但是在处理的时候没法正确的提取连续的字符串和数字，特别是数字陷入了误区，直接也按照字符的方式从栈中取出来就好，再用int转换即可
```python
class Solution:  
    def decodeString(self, s: str) -> str:  
        stack = []  
        
        for char in s:  
            if char != ']':  
                stack.append(char)  
            else:  
                # 提取字符串  
                cur_string = ""  
                while stack and stack[-1] != '[':  
                    cur_string = stack.pop() + cur_string  
                
                # 弹出 '['  
                stack.pop()  
                
                # 提取数字  
                num = ""  
                while stack and stack[-1].isdigit():  
                    num = stack.pop() + num  
                
                # 将重复后的字符串压回栈中  
                stack.append(int(num) * cur_string)  
        
        return "".join(stack)
```
### 739. 每日温度
啊啊啊啊啊啊啊啊，啊啊啊啊啊啊啊，栈放数组索引而不是值，对于这种需要计算范围的，用索引！！！！！！！！！！！！！！！！！！！！！！！
```python
class Solution:  
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:  
        n = len(temperatures)  
        result = [0] * n  # 初始化结果数组  
        stack = []        # 用于存储下标的栈  
        
        for i in range(n):  
            # 当栈不为空，且当前温度大于栈顶温度时  
            while stack and temperatures[i] > temperatures[stack[-1]]:  
                prev_idx = stack.pop()  # 获取栈顶温度的下标  
                result[prev_idx] = i - prev_idx  # 计算等待天数  
            stack.append(i)  # 将当前下标入栈  
        
        return result
```



### 84 ⚠️ 柱状图中的最大矩形
单调栈：当遇到一个小于栈顶的元素时，说明找到了栈顶元素的右边界，栈保持递增，所以栈顶下面的元素就是左边第一个小于当前高度的位置，这样就能确定一个矩形的左右边界，从而计算面积
```python
class Solution:  
    def largestRectangleArea(self, heights: List[int]) -> int:  
        # 在数组两端添加0，简化边界情况处理  
        # 左边的0确保第一个元素也能正确计算宽度  
        # 右边的0确保最后能弹出所有栈中元素  
        heights = [0] + heights + [0]  
        
        # 单调递增栈，存储下标  
        stack = []  
        # 记录最大矩形面积  
        max_area = 0  
        
        # 遍历每个位置  
        for i in range(len(heights)):  
            # 当前高度小于栈顶高度时，说明找到了右边界  
            # 需要计算栈顶元素为高度的矩形面积  
            while stack and heights[i] < heights[stack[-1]]:  
                # 获取当前高度  
                height = heights[stack.pop()]  
                
                # 计算宽度：右边界(i) - 左边界(新栈顶的下标) - 1  
                # 由于添加了哨兵0，不用担心栈为空的情况  
                width = i - stack[-1] - 1  
                
                # 更新最大面积  
                max_area = max(max_area, width * height)  
            
            # 将当前下标入栈  
            stack.append(i)  
            
        return max_area
```

## 贪心算法

### 121. 买股票的最佳时机
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        min_price = prices[0]
        max_price = 0
        for i in range(1, len(prices)):
            min_price = min(min_price, prices[i])
            max_price = max(max_price, prices[i] - min_price)
            

        return max_price
```

### 55. 跳跃游戏
每走一步后都想着能够走最远
```python
class Solution:  
    def canJump(self, nums: List[int]) -> bool:  
        n = len(nums)  
        
        # 处理特殊情况  
        if n == 1:  
            return True  
            
        # 初始化最远可以跳到的位置  
        max_steps = nums[0]  
        
        # 遍历数组（除了最后一个元素）  
        for i in range(n-1):  
            # 更新最远距离  
            # max_steps-1 表示之前的最远距离减去走到当前位置消耗的一步  
            # nums[i] 表示从当前位置可以跳的步数  
            max_steps = max(max_steps - 1, nums[i])  
            
            # 如果在某个位置无法继续前进  
            if max_steps == 0:  
                return False  
        
        # 判断最终是否能到达终点  
        return max_steps > 0
```

### 45. ⚠️ 跳跃游戏II
每次在上次能跳到的范围（end）内选择一个能跳的最远的位置（也就是能跳到max_far位置的点）作为下次的起跳点 ！
```python
class Solution:  
    def jump(self, nums: List[int]) -> int:  
        n = len(nums)  
        max_far = 0    # 目前能跳到的最远位置  
        end = 0        # 上次跳跃可达范围的右边界  
        steps = 0      # 跳跃次数  
        
        # 遍历数组，但不访问最后一个元素  
        for i in range(n - 1):  
            # 更新目前能跳到的最远位置  
            max_far = max(max_far, i + nums[i])  
            
            # 到达上次跳跃能到达的右边界  
            if i == end:  
                # 更新边界  
                end = max_far  
                steps += 1  
                
        return steps
```

### 763. ⚠️ 划分字母区间
你提到的思路很好：找到每个字母最后出现的位置，然后当遍历到某个位置时，如果这个位置是当前区间内所有字母的最后出现位置，就可以划分
```python
class Solution:  
    def partitionLabels(self, s: str) -> List[int]:  
        # 记录每个字母最后出现的位置  
        last = {}  
        for i, c in enumerate(s):  
            last[c] = i  
            
        result = []  
        start = 0   # 当前区间的起始位置  
        end = 0     # 当前区间的结束位置  
        
        # 遍历字符串  
        for i, c in enumerate(s):  
            # 更新当前区间的结束位置  
            end = max(end, last[c])  
            
            # 如果当前位置到达区间结束位置，说明可以划分  
            if i == end:  
                result.append(end - start + 1)  
                start = i + 1  
                
        return result
```

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
这道组合总和的核心思路是：从给定的数组中选择数字（每个数字可以重复使用），通过回溯的方式尝试不同的组合，记录当前组合的和，当和等于目标值时就找到一个解。为了避免重复组合（如[2,3]和[3,2]算同一个），我们规定每次只能选择当前数字或它后面的数字，这需要用一个start参数来控制选择的起始位置。当当前和超过目标值时就停止当前分支的尝试（剪枝）。

更形象的比喻：就像是在一个自助餐厅，每种食物（candidates中的数字）可以重复取，但要求总重量（和）正好等于某个值（target），并且为了避免重复记录，规定只能从左到右取食物。
![image](https://github.com/user-attachments/assets/effdcc9d-eeb5-4ba3-adcd-865696443741)
![image](https://github.com/user-attachments/assets/9026580d-2f07-45c6-a1fa-46d93463bb0b)




### 总结
全排列是一个考虑顺序的排列问题，就像是在排队，每个位置都可以安排任何一个还没用过的数字。比如[1,2,3]，第一个位置可以是1、2或3，选了1后第二个位置可以选2或3，以此类推。这需要用一个used数组来标记哪些数字已经使用过，避免重复使用。

子集问题则像是一排开关，每个数字只面临"选"或"不选"的决定，不考虑顺序。为了避免生成重复的子集（如[1,2]和[2,1]算作相同），我们规定只能往后选择，这需要一个start参数来控制选择的起始位置。每个数字的选择都不会影响其他数字，形成了独立的决策。

电话号码字母组合像是在转动多层转盘，每个数字（转盘）对应特定的几个字母选项，必须在每一层都选择一个。这需要一个映射表来存储数字到字母的对应关系，且每个位置的选择都是从其对应的字母集合中进行，具有固定的选择范围。

组合总和则类似于凑零钱问题，特点是每个数字可以重复使用，且需要所选数字的和等于目标值。为了避免重复组合（如[2,3]和[3,2]算作相同），同样需要start参数来限制选择的方向。当当前和超过目标值时就可以停止尝试（剪枝），提高效率。

这些问题虽然都用到了回溯的思想，但各自的约束条件和处理方式不同：全排列需要考虑顺序和使用标记，子集关注选择与否，电话号码组合依赖固定映射，而组合总和允许重复选择并需要满足和的条件。理解这些差异对于选择正确的解题策略至关重要
![image](https://github.com/user-attachments/assets/3d8871e4-7e0f-4d21-abd0-d0857a1c3e6b)
![image](https://github.com/user-attachments/assets/7f546cd3-602b-4891-8940-3f4a4f96f138)


### 22.括号生成

![image](https://github.com/user-attachments/assets/23edb140-adf4-42a3-b63b-d515972ca5c4)
![image](https://github.com/user-attachments/assets/4ecd1660-cd65-46de-a0e6-b3ad7841a240)


### 79. 单词搜索

```python
def exist(self, board: List[List[str]], word: str) -> bool:  
    count = {}  
    m, n = len(board), len(board[0])  

    # 统计网格中的字符频率  
    for i in range(m):  
        for j in range(n):  
            count[board[i][j]] = count.get(board[i][j], 0) + 1  

    # 检查word中的字符是否超过网格中的字符  
    for c in word:  
        if c not in count or word.count(c) > count[c]:  
            return False  

    # 看看word的首尾字符那个在网格中频率低，从低的开始搜  
    if count.get(word[0], 0) > count.get(word[-1], 0):  
        word = word[::-1]  

    def dfs(i, j, k):  
        if k == len(word):  # k为匹配上的字符数  
            return True  
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[k]:  
            return False  
        temp = board[i][j]  
        board[i][j] = '#'  
        result = dfs(i-1, j, k+1) or dfs(i+1, j, k+1) or dfs(i, j+1, k+1) or dfs(i, j-1, k+1)  
        board[i][j] = temp  
        return result  

    # 递归搜索  
    for i in range(m):  
        for j in range(n):  
            if board[i][j] == word[0] and dfs(i, j, 0):  
                return True  
    
    return False
```

### 131. 分割回文串
回溯算法框架下的"选择"是分割点的位置，用回文判断作为剪枝条件，每个位置都可能是分割点，但只有形成回文才继续递归，需要维护当前的分割方案并及时回溯
```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:

        def is_reverse(start, end):
            while start < end:
                if s[start] != s[end]:
                    return False
                start += 1
                end -= 1
            return True

        def backtrack(start):
            if start == len(s):
                res.append(path[:])
                return 
            
            for end in range(start, len(s)):
                if is_reverse(start, end):
                    path.append(s[start:end+1])
                    backtrack(end + 1)
                    path.pop()
        res = []
        path = []

        backtrack(0)

        return res
```
### 51 N皇后
采用DFS回溯策略，通过逐行放置皇后的方式（保证行不冲突），使用set集合维护可用列（保证列不冲突），记录皇后每行所在位置，并利用坐标关系(行+列相等表示在同一主对角线，行-列相等表示在同一副对角线)来判断对角线冲突，当成功放置N个皇后时，将当前解加入结果集，最终返回所有可能的解
```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:

         # 记录可用列
        s = set(range(n))
        # 记录每一行皇后所在位置
        col = [0] * n
        res = []

        def dfs(i, s):
            if i == n:
                res.append(['.'*col[i] + 'Q' + '.'*(n - col[i] - 1)  for i in range(n) ])
                return

            for j in s:
                if all(x + col[x] != i + j and x - col[x] != i - j for x in range(i)):
                    col[i] = j
                    dfs(i+1, s - {j})

        dfs(0, s)
        return res
```

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

## 动态规划
### 70. 爬楼梯
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        dp = [1] * n 
        dp[1] = 2

        for i in range(2, n):
            dp[i] = dp[i- 1] + dp[i-2]

        return dp[n-1]
```

### ⚠️ 118. 杨辉三角
```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:

        if numRows == 1:
            return [[1]]
        
        res = [[1], [1,1]]
        pre_level = [1, 1]
        
        for i in range(2, numRows):
            # 每一行都是从1开始，每一行数量等于行数，每一行最后一个都是1
            cur_level = []
            cur_level.append(1)
            for j in range(1, i):
                cur_level.append(pre_level[j-1] + pre_level[j])

            cur_level.append(1)
            pre_level = cur_level
            res.append(cur_level)

        return res
```
### 198. 打家劫舍
```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]

        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])

        return max(dp)
```
### ⚠️ 279. 完全平方数
![image](https://github.com/user-attachments/assets/00e8cda0-8eba-4217-b176-ea44c263d5b4)
```python
class Solution:  
    def numSquares(self, n: int) -> int:  
        # 初始化 dp 数组  
        dp = [float('inf')] * (n + 1)  
        dp[0] = 0  # 数字 0 不需要任何完全平方数  
        
        # 遍历每个数字 i  
        for i in range(1, n + 1):  
            # 遍历所有可能的完全平方数 j^2  
            j = 1  
            while j * j <= i:  
                dp[i] = min(dp[i], dp[i - j * j] + 1)  
                j += 1  
        
        return dp[n]
```
### 322. 零钱兑换
和上边的题目一样，属于同一种动态规划问题
```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for coin in coins:
                if i - coin >=0:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1
   ```

### ⚠️ 139. 单词拆分
![image](https://github.com/user-attachments/assets/2c30bc9b-361a-4638-9144-a5be52fee86d)

```python
class Solution:  
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:  
        # 将 wordDict 转换为集合，方便快速查找  
        wordSet = set(wordDict)  
        # 初始化 dp 数组，dp[i] 表示 s[:i] 是否可以被拼接  
        dp = [False] * (len(s) + 1)  
        dp[0] = True  # 空字符串可以被拼接  
        
        # 遍历字符串 s  
        for i in range(1, len(s) + 1):  
            # 遍历 wordDict 中的每个单词  
            for word in wordSet:  
                # 如果当前单词可以匹配 s[i-len(word):i]  
                if i >= len(word) and s[i-len(word):i] == word:  
                    dp[i] = dp[i] or dp[i-len(word)]  
        
        return dp[len(s)]
```
### 300. 最长递增子序列
```python
class Solution:  
    def lengthOfLIS(self, nums: List[int]) -> int:  
        dp = [1] * len(nums)  # dp[i] 表示以 nums[i] 结尾的最长递增子序列长度  
        for i in range(1, len(nums)):  
            for j in range(i):  
                if nums[j] < nums[i]:  
                    dp[i] = max(dp[i], dp[j] + 1)  
        return max(dp)  # 返回全局最长递增子序列的长度
```
![image](https://github.com/user-attachments/assets/37a83107-2274-4db5-a4a5-d358873a310d)

```python
class Solution:  
    def lengthOfLIS(self, nums: List[int]) -> int:  
        def binarySearch(sub, x):  
            # 手动实现二分查找，找到第一个大于等于 x 的位置  
            left, right = 0, len(sub) - 1  
            while left < right:  
                mid = (left + right) // 2  
                if sub[mid] >= x:  
                    right = mid  
                else:  
                    left = mid + 1  
            return left  

        sub = []  # 用于存储当前的递增子序列  
        for x in nums:  
            if not sub or x > sub[-1]:  
                sub.append(x)  # 如果 x 大于 sub 的最后一个元素，直接添加  
            else:  
                # 使用二分查找找到第一个大于等于 x 的位置  
                pos = binarySearch(sub, x)  
                sub[pos] = x  # 替换该位置的值为 x  
        return len(sub)
```

### ⚠️152. 乘积最大子数组
由于存在负数，那么会导致最大的变最小的，最小的变最大的。因此还需要维护当前最小值imin
```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:

        max_prod = nums[0]
        min_prod = nums[0]
        result_max = nums[0]

        for i in range(1, len(nums)):
            if nums[i] < 0:
                max_prod, min_prod = min_prod, max_prod
            max_prod = max(nums[i],max_prod*nums[i])
            min_prod = min(nums[i],min_prod*nums[i])
            result_max = max(result_max,max_prod)
        return result_max
```
### ⚠️ 416. 分割等和子集
```python
def canPartition(self, nums: List[int]) -> bool:  
    # 1. 判断是否可能存在解  
    total = sum(nums)  
    if total % 2 != 0:  # 总和为奇数，必不可能平分  
        return False  
    target = total // 2  # 目标和（每个子集的和）  

    # 2. 初始化dp数组  
    dp = [False] * (target + 1)  # dp[i]表示是否可以凑出和为i  
    dp[0] = True  # 空集的和为0，总是可以达到  

    # 3. 动态规划过程  
    for num in nums:  # 遍历每个数字  
        for j in range(target, num-1, -1):  # 从后往前遍历  
            dp[j] = dp[j] or dp[j-num]  # 状态转移  

    return dp[target]  # 返回是否能凑出目标和
  ```
### ⚠️ 最长有效括号
![image](https://github.com/user-attachments/assets/ed9d4e26-189e-4201-a409-2b706fb87da5)
```python
def longestValidParentheses(self, s: str) -> int:  
    if not s:  
        return 0  
        
    dp = [0] * len(s)  

    for i in range(1, len(s)):  # 从索引1开始  
        if s[i] == ')':  
            # 情况1：()型  
            if s[i-1] == '(':  
                dp[i] = (dp[i-2] if i >= 2 else 0) + 2  
            # 情况2：))型  
            elif i - dp[i-1] > 0 and s[i-dp[i-1]-1] == '(':  
                dp[i] = dp[i-1] + 2  
                # 加上之前的有效括号长度  
                if i - dp[i-1] >= 2:  
                    dp[i] += dp[i-dp[i-1]-2]  
                    
    return max(dp) if dp else 0
```

![image](https://github.com/user-attachments/assets/e6b184ea-83ef-4d13-b0be-53391b7f9ded)

```python
class Solution:  
    def longestValidParentheses(self, s: str) -> int:  
        # 处理边界情况  
        if not s or len(s) == 1:  
            return 0  
        
        # 栈初始化，放入-1作为哨兵  
        stack = [-1]  
        max_length = 0  
        
        # 遍历字符串  
        for i in range(len(s)):  
            if s[i] == '(':  # 遇到左括号  
                stack.append(i)  # 索引入栈  
            else:  # 遇到右括号  
                stack.pop()  # 弹出栈顶  
                if not stack:  # 栈空，说明没有匹配的左括号  
                    stack.append(i)  # 当前右括号索引入栈作为新的参考点  
                else:  # 栈不空，说明找到了匹配  
                    # 计算当前有效括号长度  
                    curr_length = i - stack[-1]  
                    max_length = max(curr_length, max_length)  
        
        return max_length
```

### 62. 不同路径
```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                dp[0][j] = dp[i][0] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]

        return dp[m-1][n-1]
```

### 64. 最小路径和
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:  
        m, n = len(grid), len(grid[0])  
        dp = [[0]* n for _ in range(m)]  
        
        # 初始化第一列  
        dp[0][0] = grid[0][0]  # 起点  
        for i in range(1, m):  
            dp[i][0] = dp[i-1][0] + grid[i][0]  
        
        # 初始化第一行  
        for j in range(1, n):  
            dp[0][j] = dp[0][j-1] + grid[0][j]  

        # 正确的状态转移  
        for i in range(1, m):  
            for j in range(1, n):  
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]  

        return dp[m-1][n-1]
```
### 5. 最长回文子串
```python
def longestPalindrome(self, s: str) -> str:  
    # 中心扩展函数  
    def extend_center(left, right):  
        # 向两边扩展，直到不满足回文条件  
        while left >= 0 and right < len(s) and s[left] == s[right]:  
            left -= 1  
            right += 1  
        # 返回有效的回文子串  
        return s[left + 1:right]  

    result = ""  # 存储最长回文子串  

    # 遍历每个可能的中心点  
    for i in range(len(s)):  
        odd = extend_center(i, i)      # 奇数长度回文  
        even = extend_center(i, i+1)   # 偶数长度回文  

        # 取最长的子串  
        result = max(result, odd, even, key=len)  
    return result
```

### 1143. 最长公共子序列
![image](https://github.com/user-attachments/assets/785113ff-4125-48cf-b71f-24da2d280878)
```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m = len(text1)+1
        n = len(text2)+1

        dp = [[0] * n for _ in range(m)]


        for i in range(1, m):
            for j in range(1, n):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m-1][n-1]
```
### 72. 编辑距离
```python
def minDistance(self, word1: str, word2: str) -> int:  
    m = len(word1) + 1  # 包含空字符串情况  
    n = len(word2) + 1  
    
    # dp[i][j] 表示 word1前i个字符 转换到 word2前j个字符 需要的最少操作数  
    dp = [[0] * n for _ in range(m)]  
    
    # 初始化：空字符串转换到另一个字符串需要的操作数  
    for i in range(m):  
        dp[i][0] = i    # 删除操作  
    for j in range(n):  
        dp[0][j] = j    # 插入操作  
    
    # 状态转移  
    for i in range(1, m):  
        for j in range(1, n):  
            if word1[i-1] == word2[j-1]:  
                dp[i][j] = dp[i-1][j-1]  # 字符相同，不需要操作  
            else:  
                dp[i][j] = min(  
                    dp[i-1][j-1],  # 替换操作  
                    dp[i-1][j],    # 删除操作  
                    dp[i][j-1]     # 插入操作  
                ) + 1  
                
    return dp[m-1][n-1]
```


## ACM模式
```python
# 不同的split()用法  
line = "A,B,C"  
print(line.split(','))  # ['A', 'B', 'C']  

line = "A B C"  
print(line.split())     # ['A', 'B', 'C']  默认按空格分割  
print(line.split(' '))  # ['A', 'B', 'C']  显式按空格分割  

# 处理多个空格  
line = "A  B   C"  
print(line.split())     # ['A', 'B', 'C']  处理多个空格  
print(line.split(' '))  # ['A', '', 'B', '', '', 'C']  保留空字符串
```
![image](https://github.com/user-attachments/assets/8c163a79-b080-4067-9906-48499d60c752)


## 递归实现快排
```python
def quick_sort(nums):
    if len(nums) <= 1:
        return nums
    pivot = nums[0]

    num_left = [x for x in nums[1:] if x < pivot]
    nums_right = [x for x in nums[1:] if x > pivot]


    return quick_sort(num_left) + [pivot] + quick_sort(nums_right)

nums = [3, 4, 2, 1, 5, 6, 7]
print(quick_sort(nums))
```

## 迭代实现快排

```python
def quick_sort(nums):
    if len(nums) <= 1:
        return nums
    stack = [(0, len(nums)-1)]

    while stack:
        left, right = stack.pop()
        if left >= right:
            continue
        pivot = nums[left]
        low, high = left, right
        while low < high:
            while low < high and nums[high] >= pivot:
                high -= 1
            while low < high and nums[low] <= pivot:
                low += 1
            nums[low], nums[high] = nums[high], nums[low]
        nums[low], nums[left] = nums[left], nums[low]
        stack.append((left, low - 1))
        stack.append((low + 1, right))
    return nums


nums = [1, 5, 7, 4, 3, 1, 8, 2]
print(quick_sort(nums))
```
## 归并排序
```python
def merge_sort(nums):
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])

    return merge(left, right)

def merge(left, right):
    res = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            res.append(left[i])
            i += 1
        else:
            res.append((right[j]))
            j += 1
    res.extend(left[i:])
    res.extend(right[j:])
    return res

nums = [1, 3, 5, 7, 2, 4, 6, 8]
print(merge_sort(nums))
```


## 堆排序
```python
def heap_sort(nums):
    n = len(nums)

    # 构建最大堆
    for i in range(n // 2 - 1, -1, -1):
        heapify(nums, n, i)

    # 逐步将堆顶元素（最大值）移到数组末尾，并调整堆
    for i in range(n - 1, 0, -1):
        nums[0], nums[i] = nums[i], nums[0]  # 将堆顶元素与末尾元素交换
        heapify(nums, i, 0)  # 调整剩余的堆

    return nums

def heapify(nums, n, i):
    """
    调整以索引 i 为根的子树，使其满足最大堆的性质。
    nums: 数组
    n: 堆的大小
    i: 当前需要调整的节点索引
    """
    largest = i  # 假设当前节点是最大值
    left = 2 * i + 1  # 左子节点索引
    right = 2 * i + 2  # 右子节点索引

    # 如果左子节点存在且大于当前最大值  
    if left < n and nums[left] > nums[largest]:
        largest = left

    # 如果右子节点存在且大于当前最大值
    if right < n and nums[right] > nums[largest]:
        largest = right

    # 如果最大值不是当前节点，则交换，并递归调整
    if largest != i:
        nums[i], nums[largest] = nums[largest], nums[i]
        heapify(nums, n, largest)

nums = [3, 4, 2, 1, 5, 6, 7]
sorted_nums = heap_sort(nums)
print("排序结果:", sorted_nums)
```
