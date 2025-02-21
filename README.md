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

