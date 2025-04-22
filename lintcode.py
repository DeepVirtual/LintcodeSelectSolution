# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:56:19 2018

@author: cz
"""

#1. A + B Problem
class Solution:
    """
    @param a: An integer
    @param b: An integer
    @return: The sum of a and b 
    """
    #https://stackoverflow.com/questions/30696484/a-b-without-arithmetic-operators-python-vs-c
    #https://www.jiuzhang.com/solution/a-b-problem/#tag-other-lang-python
    
    def aplusb(self, a, b):
        # write your code here
        int_max=0x7FFFFFFF
        
        while b!=0:
            a,b=(a^b) & int_max,(a&b)<<1
        return a
a=100
b=-100           
if __name__ == "__main__":
    print(Solution().aplusb(a, b))            


#2. Trailing Zeros 
class Solution:
    """
    @param: n: An integer
    @return: An integer, denote the number of trailing zeros in n!
    """
    def trailingZeros(self, n):
        # write your code here, try to do it without arithmetic operators.
        
        
        def cal0(n):
            f=5
            ans=0
            while f<n:
              ans+=n//f
              f*=5
            return ans
        
        return  cal0(n) if n>4 else 0
        
#3. Digit Counts   
class Solution:
    """
    @param: : An integer
    @param: : An integer
    @return: An integer denote the count of digit k in 1..n
    """
    def digitCounts(self, k, n):
        # write your code here
        assert(n>=0 and  0<=k<=n )
        ans=0
        
        for x in range(n+1):
            while True:
                
                if x%10==k:
                    ans+=1
                x=x//10
                if x==0:
                    break
        return ans
n = 12
k = 0
if __name__ == "__main__":
    print(Solution().digitCounts( k, n))     
                
#4. Ugly Number II              
class Solution:
    """
    @param n: An integer
    @return: the nth prime number as description.
    """
    def nthUglyNumber(self, n):
        # write your code here

        import heapq
        heap=[1]
        
        for i in range(1,n+1):
            value= heapq.heappop(heap)
            heapq.heappush(heap,value*2)
            heapq.heappush(heap,value*3)
            heapq.heappush(heap,value*5)
            while  value==heap[0]:
                   heapq.heappop(heap)
        return value
        
n=9
if __name__ == "__main__":
    print(Solution().nthUglyNumber(n)) 
        
        
#5. Kth Largest Element         
class Solution:
    # @param k & A a integer and an array
    # @return ans a integer
    
    def kthLargestElement(self, k, A):
        n=len(A)
        k-=1
        def partition(s,e):
            p,q=s+1,e
            
            while p<=q:
                if A[p]>A[s]:
                    p+=1
                else:
                    A[p],A[q]=A[q],A[p]
                    q-=1
            A[s],A[q]=A[q], A[s]
            m=q
            
            if m==k:
                return A[m]
            elif m<k:
                return partition(m+1,e)
            else:
                return partition(s,m-1)
        return partition(0,n-1)
A=[9,3,2,4,8]
k=3                    
if __name__ == "__main__":
    print(Solution().kthLargestElement( k, A))                    
                
#6. Merge Two Sorted Arrays 
class Solution:
    """
    @param A: sorted integer array A
    @param B: sorted integer array B
    @return: A new sorted integer array
    """
    def mergeSortedArray(self, A, B):
        # write your code here
        
        i=0
        j=0
        l=[]

        while i<len(A)  and j < len(B):
           
            if A[i]<B[j]:
               l.append(A[i])
               i+=1
            else:
               l.append(B[j])
               j+=1
        if i<len(A):
          l+=A[i:]
        if j<len(B):
          l+=B[j:]
        return l

A=[1,2,3,4]

B=[2,4,5,6]
if __name__ == "__main__":
    print(Solution().mergeSortedArray( A, B))               
          
#7. Serialize and Deserialize Binary Tree               
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

 
 

class Solution:
    """
    @param root: An object of TreeNode, denote the root of the binary tree.
    This method will be invoked first, you should design your own algorithm 
    to serialize a binary tree which denote by a root node to a string which
    can be easily deserialized by your own "deserialize" method later.
    """
    def serialize(self, root):
        # write your code here
        from collections import deque
        string=''
        queue=deque([root])
        while queue:
              cur=queue.popleft()
              if cur:
                 string+=','+str(cur.val)
                 queue.append(cur.left)
                 queue.append(cur.right)
              else:
                 string+=',None'
        return string
            
   

    """
    @param data: A string serialized by your serialize method.
    This method will be invoked second, the argument data is what exactly
    you serialized at method "serialize", that means the data is not given by
    system, it's given by your own serialize method. So the format of data is
    designed by yourself, and deserialize it here as you serialize it in 
    "serialize" method.
    """
    def deserialize(self, data):
        # write your code here
        from collections import deque
        data=data.split(',')
        data=deque(data)
        _,val=data.popleft(),data.popleft()
        root=TreeNode(val) if val!='None' else None
        queue=deque([root])
        
        while queue:
              node=queue.popleft()
              if node:
                 left,right=data.popleft(),data.popleft()
                 leftnode=None  if left=='None' else TreeNode(int(left))
                 rightnode=None  if right=='None' else TreeNode(int(right)) 
                 node.left=leftnode
                 node.right=rightnode
                 queue.append(node.left)
                 queue.append(node.right)
        return root

#root=TreeNode(3)
#root.left=TreeNode(9)
#root.right=TreeNode(20)
#root.right.left=TreeNode(15)
#root.right.right=TreeNode(7)

#
#An example of testdata: Binary tree {3,9,20,#,#,15,7}, denote the following structure:
#
#  3
# / \
#9  20
#  /  \
# 15   7        
if __name__ == "__main__":
    print(Solution().serialize( root))  


    
#8. Rotate String        
class Solution:
    """
    @param str: An array of char
    @param offset: An integer
    @return: nothing
    """
    def rotateString(self, str, offset):
        # write your code here 
        #work in python 2
        n=len(str)
        if n>0:
            offset=offset%n
        
        M=offset
        str[:-M]=str[:-M][::-1]
        str[-M:]=str[-M:][::-1]
        str[:]=str[:][::-1]
        #print(str)
            
            
str="abcdefg"
offset=2
if __name__ == "__main__":
    print(Solution().rotateString(str, offset))        
        
#9. Fizz Buzz        
class Solution:
    """
    @param n: An integer
    @return: A list of strings.
    """
    def fizzBuzz(self, n):
        # write your code here

        ans=[]
        for i in range(1,n+1):
            if i%3==0 and i%15!=0:
                ans.append("fizz")
            elif i%5==0 and i%15!=0:
                ans.append("buzz")
            elif i%15==0 :
                ans.append("fizz buzz")
            else:
                ans.append(str(i))
        return ans
n=15
if __name__ == "__main__":
    print(Solution().fizzBuzz( n)) 

                  
#11. Search Range in Binary Search Tree                 
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: param root: The root of the binary search tree
    @param k1: An integer
    @param k2: An integer
    @return: return: Return all keys that k1<=key<=k2 in ascending order
    """
    def searchRange(self, root, k1, k2):
        # write your code here
        res=[]
        
        def searchRange(node,k1,k2,res):
            if not node:
                return 
            if node.val>k2:
               searchRange(node.left,k1,k2,res)
            elif node.val<k1:
               searchRange(node.right,k1,k2,res)
            else:
                res.append(node.val)
                searchRange(node.left,k1,k2,res)
                searchRange(node.right,k1,k2,res)
        searchRange(root,k1,k2,res)
        return sorted(res)
                
#12. Min Stack            
class MinStack:
    
    def __init__(self):
        # do intialization if necessary
        self.stack=[]
        self.minstack=[]
    """
    @param: number: An integer
    @return: nothing
    """
    def push(self, number):
        # write your code here
        self.stack.append(number)
        if not self.minstack or number <= self.minstack[-1]:
             self.minstack.append(number)

    """
    @return: An integer
    """
    def pop(self):
        # write your code here
        
        if self.stack[-1]==self.minstack[-1]:
            self.minstack.pop()
        return self.stack.pop()

    """
    @return: An integer
    """
    def min(self):
        # write your code here
        return self.minstack[-1]
        
#13. Implement strStr()        
class Solution:
    """
    @param: source: source string to be scanned.
    @param: target: target string containing the sequence of characters to match
    @return: a index to the first occurrence of target in source, or -1  if target is not part of source.
    """
    def strStr(self, source, target):
        # write your code here 
        if target is None:
            return -1
        if not target:
            return 0
        if not source or not target:
            return -1
        
        ns=len(source)
        nt=len(target)
        if ns< nt :
            return -1
        for i in range(ns-nt+1):
            
            j=0
            tempi=i
            while j<nt and target[j]==source[tempi]:
            
                j+=1
                tempi+=1
                print(j,tempi)
            if j==nt:
                return i
        return -1
source = "source" 
target = "target" 
source = "abcdabcdefg" 
target = "bcd"               
if __name__ == "__main__":
    print(Solution().strStr( source, target))                
            
#14. First Position of Target        
class Solution:
    """
    @param nums: The integer array.
    @param target: Target to find.
    @return: The first position of target. Position starts from 0.
    """
    def binarySearch(self, nums, target):
        # write your code here
        n=len(nums)
        
        left=0
        right=n-1
        
        while left +1<right:
            mid=(left+right)//2
            if nums[mid]>=target:
                right=mid
            else:
                left=mid
        if nums[left]==target:
            return left
        elif nums[right]==target:
            return right
        else:
            return -1
            
nums=[1, 2, 3, 3, 4, 5, 10] 
target=3               
if __name__ == "__main__":
    print(Solution(). binarySearch( nums, target))            
                
                
#15. Permutations            
class Solution:
    """
    @param: nums: A list of integers.
    @return: A list of permutations.
    """
    def permute(self, nums):
        # write your code here
        res=[]
        
        def helper(res,path,nums):
            if not nums:
                res.append(path)
                return 
            for i in range(len(nums)):
                helper(res,path+[nums[i]],nums[:i]+nums[i+1:])
        
        if nums==[]:
            return [[]]
        if nums=='':
            return []
        helper(res,[],nums)
        return res
nums = [1,2,3]
if __name__ == "__main__":
    print(Solution().permute( nums))   

#16. Permutations II
class Solution:
    """
    @param: :  A list of integers
    @return: A list of unique permutations
    """

    def permuteUnique(self, nums):
        # write your code here
        res=[]
        
        def helper(res,path,nums):
            if not nums:
                res.append(path)
                return 
            for i in range(len(nums)):
                if i>0 and nums[i]==nums[i-1]:
                    continue
                helper(res,path+[nums[i]],nums[:i]+nums[i+1:])
        
        if nums==[]:
            return [[]]
        if nums is None:
            return []
        helper(res,[],sorted(nums))
        return res
nums = [1,2,2]
if __name__ == "__main__":
    print(Solution().permuteUnique( nums))   
        
#17. Subsets        
class Solution:
    """
    @param nums: A set of numbers
    @return: A list of lists
    """
    def subsets(self, nums):
        # write your code here
        
        def dfs(res,path,nums):
            res.append(path)
            if not nums:
                return 
            for i in range(len(nums)):
                dfs(res,path+[nums[i]],nums[i+1:])
        res=[]
        dfs(res,[],sorted(nums))
        return res
nums = [1,2,3]
if __name__ == "__main__":
    print(Solution().subsets( nums))
            
#18. Subsets II        
class Solution:
    """
    @param nums: A set of numbers.
    @return: A list of lists. All valid subsets.
    """
    def subsetsWithDup(self, nums):
        # write your code here
        def dfs(res,path,nums):
            res.append(path)
            if not nums:
                return 
            for i in range(len(nums)):
                if i>0 and nums[i-1]==nums[i]:
                    continue
                dfs(res,path+[nums[i]],nums[i+1:])
        res=[]
        dfs(res,[],sorted(nums))
        return res
nums = [1,2,2]
if __name__ == "__main__":
    print(Solution().subsetsWithDup( nums)) 

#20. Dices Sum       
class Solution:
    # @param {int} n an integer
    # @return {tuple[]} a list of tuple(sum, probability)
    def dicesSum(self, n):
        # Write your code here
#https://zhengyang2015.gitbooks.io/lintcode/dices_sum_20.html
        
        dp=[[0 for y in range(6*n+1)]  for x in range(n+1)]
        
        for i in range(1,7):
            dp[1][i]=1/6
        
        for i in range(2,n+1):
            for j in range(i,6*n+1):
                for k in range(1,7):
                    if j>k:
                        dp[i][j]+=dp[i-1][j-k]
                
                dp[i][j]/=6.0
        res=[]
        
        for i in range(n,6*n+1):
            res.append([i, dp[n][i]])
        return res
n=2
if __name__ == "__main__":
    print(Solution().dicesSum( n)) 
        
#Flatten List
class Solution(object):

    # @param nestedList a list, each element in the list 
    # can be a list or integer, for example [1,2,[1,2]]
    # @return {int[]} a list of integer
    def flatten(self, nestedList):
        # Write your code here
        
        def flat(ele,res):
            if isinstance(ele,int):
                res.append(ele)
                
            else:
                for e in ele:
                    res=flat(e,res)
            return res
        res=[]
        
        return flat(nestedList,res)
nestedList=[4,[3,[2,[1]]]]
if __name__ == "__main__":
    print(Solution().flatten(nestedList)) 

#24. LFU Cache     
class LFUCache:
    """
    @param: capacity: An integer
    """
    def __init__(self, capacity):
        # do intialization if necessary
        self.capacity=capacity
        self.map={}   #  key value dictionary
        self.freq_time={} # key to freq time dictionary
        self.prior_queue=[] #  (freq,time,key)
        self.time=0
        self.update=set()
        

    """
    @param: key: An integer
    @param: value: An integer
    @return: nothing
    """
    def set(self, key, value):
        # write your code here

        import heapq
        self.time+=1
        
        if not key in self.map:
            
           if self.capacity <=len(self.map):
        
                 while self.prior_queue and self.prior_queue[0][2] in self.update:
                       _,_,k=heapq.heappop(self.prior_queue)
                       f,t=self.freq_time[k]
                       heapq.heappush(self.prior_queue,(f,t,k))
                       self.update.remove(k)
                 _,_,k=heapq.heappop(self.prior_queue)
                 self.freq_time.pop(k)
                 self.map.pop(k)
                
           self.freq_time[key] =(0,self.time)
           heapq.heappush(self.prior_queue,(0,self.time,key))
           
        else:
            f,_=self.freq_time[key]
            self.freq_time[key]=(f+1,self.time)
            self.update.add(key)
        
        self.map[key]=value
            
 
    
    def get(self, key):
        # write your code here 
        """
         @param: key: An integer
         @return: An integer
        """
        self.time+=1
        if self.capacity<=0:
            return 
        
        if key in self.map:
            f,_=self.freq_time[key]
            self.freq_time[key]=(f+1,self.time)
            self.update.add(key)
            return self.map[key]
        return -1
#28. Search a 2D Matrix            
class Solution:
    """
    @param matrix: matrix, a list of lists of integers
    @param target: An integer
    @return: a boolean, indicate whether matrix contains target
    """
    def searchMatrix(self, matrix, target):
        # write your code here
        if not matrix:
            return False
        array=[]
        for row in matrix:
            array+=row
        l=0
        r=len(array)-1
        
        while l<r:
            
            mid=(l+r)//2
            
            if array[mid]==target:
                return True
            elif array[mid]>target:
                r=mid-1
            else:
                l=mid+1
        return array[l]==target
matrix=[
    [1, 3, 5, 7],
    [10, 11, 16, 20],
    [23, 30, 34, 50]
]   

target=3
if __name__ == "__main__":
    print(Solution().searchMatrix(matrix, target))     
        
#29. Interleaving String        
class Solution:
    """
    @param s1: A string
    @param s2: A string
    @param s3: A string
    @return: Determine whether s3 is formed by interleaving of s1 and s2
    """
    def isInterleave(self, s1, s2, s3):
        # write your code here
        if not s1 and not s2 and not s3:
            return True
        
        
        
        n1=len(s1)
        n2=len(s2)
        n3=len(s3)
        
        if n1+n2!=n3:
            return False
        
        dp=[[False] * (n2+1)   for _ in range(n1+1)]
        
        dp[0][0]=True
        
        for i in range(1,n1+1):
            dp[i][0]= ( dp[i-1][0] and s1[i-1]==s3[i-1])
        for j in range(1,n2+1):
            dp[0][j]= ( dp[0][j-1] and s2[j-1]==s3[j-1])
            
            
        for i in range(1,n1+1):
           for j in range(1,n2+1):
               
               if (dp[i-1][j] and s1[i-1]==s3[i+j-1])  or (dp[i][j-1] and s2[j-1]==s3[i+j-1]):
                   dp[i][j]=True
        return dp[n1][n2]
s1 = "aabcc"
s2 = "dbbca"
s3 = "aadbbcbcac"
s3 = "aadbbbaccc"
if __name__ == "__main__":
    print(Solution().isInterleave( s1, s2, s3))          
        
#30. Insert Interval        
"""
Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    """
    @param intervals: Sorted interval list.
    @param newInterval: new interval.
    @return: A new interval list.
    """
    def insert(self, intervals, newInterval):
        # write your code here
        
        insert_pos=0
        res=[]
        for interval in intervals:
            if newInterval.start>interval.end:
                res.append(interval)
                insert_pos+=1
            elif newInterval.end<interval.start:
                res.append(interval)
            else:
                newInterval.end=max(newInterval.end,interval.end)
                newInterval.start=min(newInterval.start,interval.start)
        res.insert(insert_pos,newInterval)
        return res
intervals=[]
newInterval=Interval(5,7)
if __name__ == "__main__":
    print(Solution().insert( intervals, newInterval))
                
#31. Partition Array  
class Solution:
    """
    @param nums: The integer array you should partition
    @param k: An integer
    @return: The index after partition
    """
    def partitionArray(self, nums, k):
        # write your code here


       l=0
       r=len(nums)-1
       
       while l<=r:
           
             while l<=r and nums[l] < k:
                 l+=1
             
             while l<=r and nums[r] >= k:
                 r-=1
             
             if l<r:
                 nums[l], nums[r]=nums[r],nums[l]
       return [nums,l,r]
nums=[7,7,9,8,6,6,8,7,9,8,6,6]
k=7
if __name__ == "__main__":
    print(Solution().partitionArray( nums, k))

#32. Minimum Window Substring 
class Solution:
    """
    @param source : A string
    @param target: A string
    @return: A string denote the minimum window, return "" if there is no such a string
    """
    def minWindow(self, source , target):
        # write your code here
        
        from collections import Counter
        need=Counter(target)
        missing=len(target)
        
        
        I=0
        i=0
        J=0
        for j,c in enumerate(source,1):
            if c in need and need[c]>0:
                
                missing-=1
            if c in need:
                need[c]-=1
            
            if not missing:
                while i<j and (source[i] not in need or need[source[i]]<0):
                    if source[i] in need:
                         need[source[i]]+=1
                    i+=1
                if not J or    J-I > j-i:
                    J=j
                    I=i
        return source[I:J]
 
source = "ADOBECODEBANC"
target = "ABC"     
if __name__ == "__main__":
    print(Solution().minWindow( source , target))   
            
#33. N-Queens            
class Solution:
    """
    @param: n: The number of queens
    @return: All distinct solutions
    """
    def solveNQueens(self, n):
        # write your code here
        
        import copy
        
        board=[['.' for _ in range(n)]  for _ in range(n)] 
        emptycol=[True for _ in range(n)]
        empty45=[True for _ in range(2*n-1)]
        empty135=[True for _ in range(2*n-1)]
        
        res=[]
        def backtrack(row):
            if row==n:
                res.append([''.join(row) for row in board])
                
            for col in range(n):
                    if emptycol[col]  and empty45[row-col+n-1]  and  empty135[row+col]:
                       board[row][col]='Q'
                       emptycol[col]=False
                       empty45[row-col+n-1]=False
                       empty135[row+col]=False
                       backtrack(row+1)
                       emptycol[col]=True
                       empty45[row-col+n-1]=True
                       empty135[row+col]=True
                       board[row][col]='.'
        backtrack(0)
        return res
n=4
if __name__ == "__main__":
    print(Solution().solveNQueens( n)) 
                       
#34. N-Queens II                
class Solution:
    """
    @param n: The number of queens.
    @return: The total number of distinct solutions.
    """
    def totalNQueens(self, n):
        # write your code here
        self.col={}
        self.n=n
        self.res=0
        self.search(0)
        return self.res
    def attack(self,row,col):
        for c,r in self.col.items():
            if c-r==col-row or c+r==col+row:
                return True
        return False
    
    def search(self,row):
        if row==self.n:
            self.res+=1
            return 
        
        for col in range(self.n):
            if col in self.col:
                continue
            if self.attack(row,col):
                continue
            self.col[col]=row
            self.search(row+1)
            del self.col[col]
    
            
            
n=4
if __name__ == "__main__":
    print(Solution().totalNQueens( n))         
        
#35. Reverse Linked List        
"""
Definition of ListNode

class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: n
    @return: The new head of reversed linked list.
    """
    def reverse(self, head):
        # write your code here
          
        cur=head
        prev=None
        while cur:
              temp=cur.next
              cur.next=prev
              prev=cur
              cur=temp
        return prev
        
        
#36. Reverse Linked List II   
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: ListNode head is the head of the linked list 
    @param m: An integer
    @param n: An integer
    @return: The head of the reversed ListNode
    """
    def reverseBetween(self, head, m, n):
        # write your code here
        def reverse(head):
            if not head:
                return head
            prev=None
            cur=head
            while cur:
                temp=cur.next
                cur.next=prev
                prev=cur
                cur=temp
            return prev
        
        def findkth(head,k):
            cur=head
            for _ in range(k):
                if not cur:
                    return cur
                cur=cur.next
            return cur
        dummy= ListNode(-1,head)
        mth_prev=findkth(dummy,m-1)
        mth=mth_prev.next
        nth=findkth(dummy,n)
        nth_next=nth.next
        nth.next=None
        
        
        mth_prev.next=reverse(mth)
        mth.next=nth_next
        
        return dummy.next
        
#https://leetcode.com/problems/reverse-linked-list-ii/discuss/30666/Simple-Java-solution-with-clear-explanation?page=2        
#37. Reverse 3-digit Integer
class Solution:
    """
    @param number: A 3-digit number.
    @return: Reversed number.
    """
    def reverseInteger(self, number):
        # write your code here
        return int(str(number)[::-1])
#38. Search a 2D Matrix II
class Solution:
    """
    @param matrix: A list of lists of integers
    @param target: An integer you want to search in matrix
    @return: An integer indicate the total occurrence of target in the given matrix
    """
    def searchMatrix(self, matrix, target):
        # write your code here
        if not matrix:
            return 0
        def search(row,col,target):
            if matrix[row][col]==target:
                self.res+=1
                return 
            if matrix[row][col]<target:
                for (r,c) in ((row+1,col),(row,col+1)):
                    if 0<=r<len(matrix)  and 0<=c<len(matrix[0])  and (r,c) not in self.visited:
                       self.visited.add((r,c)) 
                       search(r,c,target)
            
        self.res=0
        self.visited=set((0,0))
        search(0,0,target)  
        return self.res
target=3        
matrix=[
  [1, 3, 5, 7],
  [2, 4, 7, 8],
  [3, 5, 9, 10]
]   
matrix=[[62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80],
        [63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81],
        [64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82],
        [65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83],
        [66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84],
        [67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85]]
target=81     
if __name__ == "__main__":
    print(Solution().searchMatrix(matrix, target))         
        
#39. Recover Rotated Sorted Array        
class Solution:
    """
    @param nums: An integer array
    @return: nothing
    """
    def recoverRotatedSortedArray(self, nums):
        # write your code here
        i=1
        minI=0
        while i<len(nums) :
            
            if nums[i]<nums[i-1]:
                minI=i
                break
            i+=1
        nums[:minI]=nums[:minI][::-1]
        nums[minI:]=nums[minI:][::-1]
        nums[:]=nums[::-1]
        return nums
nums=    [4, 5, 1, 2, 3]
if __name__ == "__main__":
    print(Solution().recoverRotatedSortedArray( nums))   

#40. Implement Queue by Two Stacks
class MyQueue:
    
    def __init__(self):
        # do intialization if necessary
        self.stack1=[]
        self.stack2=[]

    """
    @param: element: An integer
    @return: nothing
    """
    def push(self, element):
        # write your code here
        self.stack1.append(element)

    """
    @return: An integer
    """
    def pop(self):
        # write your code here
        self.top()
        return self.stack2.pop()

    """
    @return: An integer
    """
    def top(self):
        # write your code here
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2[-1]
#41. Maximum Subarray 
class Solution:
    """
    @param nums: A list of integers
    @return: A integer indicate the sum of max subarray
    """
    def maxSubArray(self, nums):
        # write your code here
        
        dp=[0]*len(nums)
        
        res=nums[0]
        dp[0]=nums[0]
        
        for i in range(1,len(nums)):
            if dp[i-1]<=0:
                dp[i]=nums[i]
            else:
                dp[i]=dp[i-1]+nums[i]
            res=max(dp[i],res)
        return res
nums= [-2,2,-3,4,-1,2,1,-5,3]
if __name__ == "__main__":
    print(Solution().maxSubArray(nums))   
               
#42. Maximum Subarray II        
class Solution:
    """
    @param: nums: A list of integers
    @return: An integer denotes the sum of max two non-overlapping subarrays
    """
    def maxTwoSubArrays(self, nums):
        # write your code here
        
        n=len(nums)
        
        left=[0]*n
        right=[0]*n
        
        left[0]=nums[0]
        max_end_here=nums[0]
        max_sofar=nums[0]
        
        for i in range(1,n):
            max_end_here=max(nums[i],max_end_here+nums[i])
            max_sofar=max(max_sofar,max_end_here)
            left[i]=max_sofar
        
        right[n-1]=nums[n-1]
        max_end_here=nums[n-1]
        max_sofar=nums[n-1]
        
        for i in range(n-2,-1,-1):
            max_end_here=max(nums[i],max_end_here+nums[i])
            max_sofar=max(max_sofar,max_end_here)
            right[i]=max_sofar
            
        res=float('-inf')
        
        for i in range(n-1):
            res=max(res,left[i]+right[i+1])
        return res
nums= [1, 3, -1, 2, -1, 2]
if __name__ == "__main__":
    print(Solution().maxTwoSubArrays(nums))   
        
#43. Maximum Subarray III
class Solution:
    """
    @param nums: A list of integers
    @param k: An integer denote to find k non-overlapping subarrays
    @return: An integer denote the sum of max k non-overlapping subarrays
    """
    def maxSubArray(self, nums, k):
        # write your code here
        n=len(nums)
        if not nums or k<=0 or k>n:
            return -1
        
        localMax=[[0 for _ in range(k+1)] for _ in range(n+1)]
        globalMax=[[0 for _ in range(k+1)] for _ in range(n+1)]
        
        for j in range(1,k+1):
            localMax[j-1][j]=float('-inf')
            for i in range(j,n+1):
                localMax[i][j]=max(globalMax[i-1][j-1],localMax[i-1][j])+nums[i-1]
                if i==j:
                    globalMax[i][j]=localMax[i][j]
                else:
                    globalMax[i][j]=max(globalMax[i-1][j],localMax[i][j])
        return globalMax[n][k]
nums= [-1,4,-2,3,-2,3]
k=2
if __name__ == "__main__":
    print(Solution().maxSubArray(nums, k))                    
                    
#44. Minimum Subarray    
class Solution:
    """
    @param: nums: a list of integers
    @return: A integer indicate the sum of minimum subarray
    """
    def minSubArray(self, nums):
        # write your code here        
        
        n=len(nums)
        dp=[0]*n
        res=nums[0]
        dp[0]=nums[0]
        
        for i in range(1,n):
            dp[i]=min(nums[i],nums[i]+dp[i-1])
            res=min(res,dp[i])
        return res
nums= [1, -1, -2, 1]

if __name__ == "__main__":
    print(Solution().minSubArray( nums))   

       
#45. Maximum Subarray Difference
class Solution:
    """
    @param nums: A list of integers
    @return: An integer indicate the value of maximum difference between two substrings
    """
    def maxDiffSubArrays(self, nums):
        # write your code here
        n=len(nums)
        
        leftmin_so_far=[0]*n
        leftmax_so_far=[0]*n
        rightmin_so_far=[0]*n
        rightmax_so_far=[0]*n
        
                     
        leftmin_so_far[0]=nums[0]
        leftmax_so_far[0]=nums[0]
        
        min_end_here=nums[0]
        max_end_here=nums[0]
        
        for i in range(1,n):
            min_end_here=min(nums[i],min_end_here+nums[i])
            leftmin_so_far[i]=min(min_end_here,leftmin_so_far[i-1])
            
            max_end_here=max(nums[i],max_end_here+nums[i])
            leftmax_so_far[i]=max(max_end_here,leftmax_so_far[i-1])
            
      
        rightmin_so_far[n-1]=nums[n-1]
        rightmax_so_far[n-1]=nums[n-1]
        min_end_here=nums[n-1]
        max_end_here=nums[n-1]
        
        for i in range(n-2,-1,-1):
            min_end_here=min(nums[i],min_end_here+nums[i])
            rightmin_so_far[i]=min(min_end_here,rightmin_so_far[i+1])
            
            max_end_here=max(nums[i],max_end_here+nums[i])
            rightmax_so_far[i]=max(max_end_here,rightmax_so_far[i+1])
        
        res=float('-inf')
        
        for i in range(n-1):
            res=max(abs(leftmax_so_far[i]-rightmin_so_far[i+1]),res)
            res=max(abs(leftmin_so_far[i]-rightmax_so_far[i+1]),res)
            
        print('leftmax_so_far',leftmax_so_far)    
        print('rightmin_so_far',rightmin_so_far)
        print('leftmin_so_far',leftmin_so_far)    
        print('rightmax_so_far',rightmax_so_far)
        
        return res
nums=    [-5,-4]
if __name__ == "__main__":
    print(Solution().maxDiffSubArrays(nums))      


#46. Majority Element
class Solution:
    """
    @param: nums: a list of integers
    @return: find a  majority number
    """
    def majorityNumber(self, nums):
        # write your code here
        
        from collections import defaultdict
        count=defaultdict(int)
        for n in nums:
            count[n]+=1
            if count[n]==len(nums)//2+1:
                return n

nums=[1, 1, 1, 1, 2, 2, 2]
if __name__ == "__main__":
    print(Solution().majorityNumber( nums))        
        
#47. Majority Element II  
class Solution:
    """
    @param: nums: a list of integers
    @return: The majority number that occurs more than 1/3
    """
    def majorityNumber(self, nums):
        # write your code here
        
        from collections import defaultdict
        count=defaultdict(int)
        for n in nums:
            count[n]+=1
            if count[n]==len(nums)//3+1:
                return n
nums=[1, 2, 1, 2, 1, 3, 3]
if __name__ == "__main__":
    print(Solution().majorityNumber(nums))     
        
        
#48. Majority Number III         
class Solution:
    """
    @param nums: A list of integers
    @param k: An integer
    @return: The majority number
    """
    def majorityNumber(self, nums, k):
        # write your code here        
        from collections import defaultdict
        count=defaultdict(int)
        for n in nums:
            count[n]+=1
            if count[n]==len(nums)//k+1:
                return n
                
nums=[3,1,2,3,2,3,3,4,4,4]
k=3        
if __name__ == "__main__":
    print(Solution().majorityNumber(nums, k))        
        
#49. Sort Letters by Case        
class Solution:
    """
    @param: chars: The letter array you should sort by Case
    @return: nothing
    """
    def sortLetters(self, chars):
        # write your code here
        i=0
        j=len(chars)-1
        chars=list(chars)
        while i<=j:
            
            while i<=j and chars[i].islower():
                i+=1
            while i<=j and chars[j].isupper():
                j-=1
            if i<j:
                chars[i], chars[j]=chars[j],chars[i]
        print(chars)
                
chars="abAcD"
if __name__ == "__main__":
    print(Solution().sortLetters( chars)) 

#Product of Array Exclude Itself
class Solution:
    """
    @param: nums: Given an integers array A
    @return: A long long array B and B[i]= A[0] * ... * A[i-1] * A[i+1] * ... * A[n-1]
    """
    def productExcludeItself(self, nums):
        # write your code here
        
        n=len(nums)
        
        res=[1]*n
        
        for i in range(1,n):
            res[i]=res[i-1]*nums[i-1]
        print(res)
        
        right=1
        
        for i in range(n-1,-1,-1):
            res[i]=res[i]*right
            right=nums[i]*right
        return res
        
nums=[1, 2, 3]        
if __name__ == "__main__":
    print(Solution().productExcludeItself( nums))        

#51. Previous Permutation
class Solution:
    """
    @param: nums: A list of integers
    @return: A list of integers that's previous permuation
    """
    def previousPermuation(self, nums):
        # write your code here    
        n=len(nums)
        
        if n<=1:
            return nums
        
        for i in range(n-2,-1,-1):
            if nums[i]>nums[i+1]:
                for j in range(n-1,i,-1):
                    if nums[j]<nums[i]:
                       nums[j],nums[i]=nums[i],nums[j]
                       nums[i+1:]=reversed (sorted(nums[i+1:]))
                       break
                break
            else:
                if i==0:
                   nums.reverse()
        return nums
nums=    [1,3,2,3]
nums=   [1,2,3,4]
if __name__ == "__main__":
    print(Solution().previousPermuation(nums))                
        
        
#52. Next Permutation         
class Solution:
    """
    @param nums: A list of integers
    @return: A list of integers
    """
    def nextPermutation(self, nums):
        # write your code here
        n=len(nums)
        
        if n<=1:
            return nums
        
        for i in range(n-2,-1,-1):
            if nums[i]<nums[i+1]:
                for j in range(n-1,i,-1):
                    if nums[j]>nums[i]:
                       nums[j],nums[i]=nums[i],nums[j]
                       nums[i+1:]=sorted(nums[i+1:])
                       break
                break
            else:
                if i==0:
                   nums.reverse()
        return nums
    
nums=[1,3,2,3]  [1,3,3,2] 
nums=[4,3,2,1]  [1,2,3,4]
if __name__ == "__main__":
    print(Solution().nextPermutation( nums)) 

#53. Reverse Words in a String 
class Solution:
    """
    @param: s: A string
    @return: A string
    """
    def reverseWords(self, s):
        # write your code here
        if not s:
            return s
        slist=s.split()
        if len(slist)==1:
            return s
        else:
            return ' '.join(reversed(slist))
s="How are you?"        
if __name__ == "__main__":
    print(Solution().reverseWords( s))       
        
#54. String to Integer (atoi)         
class Solution:
    """
    @param str: A string
    @return: An integer
    """
    def atoi(self, str):
        # write your code here
        if not str:
            return 0
      
        intmax=(1<<31)-1
        res=0
        i=0
        sign=1
        str=str.strip()
        n=len(str)
        if str[i]=='-':
            sign=-1
            i+=1
        elif str[i]=='+':
            i+=1
        
       
        for j in range(i,n):
            if str[j]<'0' or str[j]>'9':
                break
            
            res=res*10+int(str[j])
            
            
        res*=sign
        if res>intmax:
            return intmax
        elif res< intmax*-1:
            return intmax*-1-1
        else:
            return res

str= "    -5211314"       
if __name__ == "__main__":
    print(Solution().atoi( str))       

#55. Compare Strings            
class Solution:
    """
    @param A: A string
    @param B: A string
    @return: if string A contains all of the characters in B return true else return false
    """
    def compareStrings(self, A, B):
        # write your code here
        from collections import Counter
        AC=Counter(A)
        BC=Counter(B)
        dif = BC-AC
        return not dif
#56. Two Sum
class Solution:
    """
    @param numbers: An array of Integer
    @param target: target = numbers[index1] + numbers[index2]
    @return: [index1 + 1, index2 + 1] (index1 < index2)
    """
    def twoSum(self, numbers, target):
        # write your code here
        
        for i,n in enumerate(numbers):
            if target-n in numbers:
                if target-n !=n:
                    return [i,numbers.index(target-n)]
                else:
                    return [i,len(numbers)-1-(numbers[::-1].index(target-n))]
            
numbers=[2, 7, 11, 15]
target=9
numbers=[0,4,3,0]
target=0        
if __name__ == "__main__":
    print(Solution().twoSum(numbers, target))        
        

#57. 3Sum
class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @return: Find all unique triplets in the array which gives the sum of zero.
    """
    def threeSum(self, numbers):
        # write your code here
        if not numbers:
            return []
        n=len(numbers)
        res=[]
        numbers.sort()
        for i in range(0,n-2):
            if i>0 and numbers[i]==numbers[i-1]:
                continue
            l=i+1
            r=n-1
            
            while l<r:
               s=numbers[i]+numbers[l]+numbers[r]
               if s>0:
                  r-=1
               elif s<0:
                  l+=1
               else:
                  res.append([numbers[i],numbers[l],numbers[r]])
                  while l<r  and numbers[l]==numbers[l+1]:
                      l+=1
                  while l<r  and numbers[r]==numbers[r-1]:
                      r-=1
                  l+=1
                  r-=1
        return res
    
numbers=[-1, 0, 1 ,2 ,-1, -4]
if __name__ == "__main__":
    print(Solution().threeSum(numbers))                   
        
        
#58. 4Sum        
class Solution:
    """
    @param numbers: Give an array
    @param target: An integer
    @return: Find all unique quadruplets in the array which gives the sum of zero
    """
    def fourSum(self, numbers, target):
        # write your code here
        def findNsum(nums,target,N,result,results):
            if N<2 or len(nums)<N or nums[0]*N>target or nums[-1]*N<target:
                return 
            if N==2:
                l=0
                r=len(nums)-1
                while l<r:
                    s=nums[l]+nums[r]
                    if s>target:
                        r-=1
                    elif s<target:
                        l+=1
                    else:
                        results.append(result+[nums[l],nums[r]])
                        while l<r  and nums[l]==nums[l+1]:
                            l+=1
                        while l<r and nums[r]==nums[r-1]:
                            r-=1
                        l+=1
                        r-=1
                        
            else:
                for i in range(len(nums)-N+1):
                   if i==0 or nums[i-1]!=nums[i]:
                      findNsum(nums[i+1:],target-nums[i],N-1,result+[nums[i]],results)
                      
                      
        results=[]
        findNsum(sorted( numbers),target,4,[],results)
        return results
numbers = [1 ,0, -1, 0, -2, 2]  
target = 0     
if __name__ == "__main__":
    print(Solution().fourSum( numbers, target))        
                
#59. 3Sum Closest 
class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @param target: An integer
    @return: return the sum of the three integers, the sum closest target.
    """
    def threeSumClosest(self, numbers, target):
        # write your code here

        if not numbers:
               return 0
        numbers.sort()
        
        res=numbers[0]+numbers[1]+numbers[2]
        
        for i in range(len(numbers)-2):
            l=i+1
            r=len(numbers)-1
            
            while l <r:
                sumN=numbers[i]+numbers[l]+numbers[r]
                
                if sumN==target:
                    return sumN
                if abs(sumN-target )< abs(res-target ):
                    res=sumN
                
                if sumN<target:
                    l+=1
                elif sumN>target:
                    r-=1
        return res
numbers = [-1, 2, 1, -4]
target = 1
if __name__ == "__main__":
    print(Solution(). threeSumClosest(numbers, target))

#60. Search Insert Position                
class Solution:
    """
    @param A: an integer sorted array
    @param target: an integer to be inserted
    @return: An integer
    """
    def searchInsert(self, A, target):
        # write your code here
        if not A:
            return 0
        
        n=len(A)
        
        l=0
        r=n-1
        
        while l<=r:
            mid=(l+r)//2
            
            if A[mid]==target:
                return mid
            elif  A[mid]>target:
                r=mid-1
            else:
                l=mid+1
                
        return l
    def searchInsert(self, A, target):
        # write your code here
        if not A:
            return 0
        
        n=len(A)
        
        l=0
        r=n-1
        
        while l+1<r:
            mid=(l+r)//2
            
            if A[mid]==target:
                return mid
            elif  A[mid]>target:
                r=mid
            else:
                l=mid
        
        if A[l]>=target:
            return l
        if A[r]>=target:
            return r
        
                
        return n
    
    
    
    
      
A=[1,3,5,6]
target=5 → 2

A=[1,3,5,6]
target= 2 → 1

A=[1,3,5,6]
target=7 → 4

A=[1,3,5,6]
target= 0 → 0
                    
if __name__ == "__main__":
    print(Solution().searchInsert(A, target))


#61. Search for a Range   
class Solution:
    """
    @param A: an integer sorted array
    @param target: an integer to be inserted
    @return: a list of length 2, [index1, index2]
    """
    def searchRange(self, A, target):
        # write your code here
        if not A:
           return [-1,-1]     
        n=len(A)
        #get the leftmost 
        l=0
        r=n-1
        while l<=r:
            mid=(l+r)//2
            if A[mid]<target:
                l=mid+1
            else:
                r=mid-1
        if l<0 or l>=n or A[l]!=target:
            return [-1,-1]
        left=l
        
        l=0
        r=n-1
        while l<=r:
            mid=(l+r)//2
            if A[mid]>target:
                r=mid-1
            else:
                l=mid+1
        right=r
        return [left,right]
A=[5, 7, 7, 8, 8, 10]
target= 8        
if __name__ == "__main__":
    print(Solution().searchRange(A, target))                          
                        
#62. Search in Rotated Sorted Array 
class Solution:
    """
    @param A: an integer rotated sorted array
    @param target: an integer to be searched
    @return: an integer
    """
    def search(self, A, target):
        # write your code here
        if not A:
            return -1
        n=len(A)
        l=0
        r=n-1
        
        while l<=r:
            mid=(l+r)//2
            if A[mid]==target:
                return mid
            if  A[mid]<A[r]: # right part is sorted 
                 if A[mid]< target and target <= A[r]:
                     l=mid+1
                 else:
                     r=mid-1
            else:#left part is sorted
                if A[l]<=target and target < A[mid]:
                    r=mid-1
                else:
                    l=mid+1
        return -1
A=[4, 5, 1, 2, 3] 
target=1
A=[4, 5, 1, 2, 3]
target=0
A=[6,8,9,1,3,5]
target=5
if __name__ == "__main__":
    print(Solution().search(A, target))                    
            
                
#63. Search in Rotated Sorted Array II  
class Solution:
    """
    @param A: an integer ratated sorted array and duplicates are allowed
    @param target: An integer
    @return: a boolean 
    """
    def search(self, A, target):
        # write your code here
        if not A:
            return False
        l=0
        
        n=len(A)
        r=n-1
       
        
        while l<=r:
            mid=(l+r)>>1
            if A[mid]==target:
                return True
            if  A[mid]==A[l]  and A[mid]==A[r]:
                l+=1
                r-=1
            elif A[mid] < A[r]:# right part is sorted
                 if A[mid]<target and target<=A[r]:
                     l=mid+1
                 else:
                     r=mid-1
            else:# left part is sorted
                if A[l]<=target and target < A[mid]:
                    r=mid-1
                else:
                    l=mid+1
        return False
A=[1, 1, 0, 1, 1, 1] 
target=0
A=[4, 5, 1, 2, 3]
target=0
A=[6,8,9,1,3,5]
target=5
A=[1, 1, 1, 1, 1, 1]
target=0
if __name__ == "__main__":
    print(Solution().search(A, target))                    
                
#64. Merge Sorted Array                 
class Solution:
    """
    @param: A: sorted integer array A which has m elements, but size of A is m+n
    @param: m: An integer
    @param: B: sorted integer array B which has n elements
    @param: n: An integer
    @return: nothing
    """
    def mergeSortedArray(self, A, m, B, n):
        # write your code here
        i=m-1
        j=n-1
        index=m+n-1
        while    i>-1 and j>-1:
             if A[i]<B[j]:
                 A[index]=B[j]
                 j-=1
             else:
                 A[index]=A[i]
                 i-=1
             index-=1
        while j>-1:
            A[index]=B[j]
            index-=1
            j-=1
      
A = [1, 2, 3, empty, empty]
B = [4, 5]
if __name__ == "__main__":
    print(Solution().search(A, target))            
                 
#65. Median of two Sorted Arrays  
class Solution:
    """
    @param: A: An integer array
    @param: B: An integer array
    @return: a double whose format is *.5 or *.0
    """
    def findMedianSortedArrays(self, A, B):
    
        def findKth(A,B,k):
            a=len(A)
            b=len(B)
            if a>b:
               A,B=B,A
            print(A,B)
            if not A:
               return B[k]
            if k==a+b-1:
               return max(A[-1],B[-1])
        
            i=min(a-1,k//2)
            j=min(b-1,k-i)
        
        
            if A[i]>B[j]:
               return findKth(A[:i],B[j:],i)
            else:
               return findKth(A[i:],B[:j],j)
        
        l=len(A)+len(B)
        
        if l%2==1:
            return findKth(A,B,l//2)
        else:
            return (findKth(A,B,l//2-1)+findKth(A,B,l//2))/2
        
A=[1,2,3,4,5,6]
B=[2,3,4,5]       

[1, 2, 2, 3, 3, 4, 4, 5, 5, 6]     
A=[2]
B=[]
if __name__ == "__main__":
    print(Solution().findMedianSortedArrays( A, B))           
        
#66. Binary Tree Preorder Traversal 
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: A Tree
    @return: Preorder in ArrayList which contains node values.
    """
    def preorderTraversal(self, root):
        # write your code here
        
        def tranverse(node,res):
            
            if node:
                res.append(node.val)
                tranverse(node.left,res)
                tranverse(node.right,res)
            return res
        res=[]
        return tranverse(root,res)
    
    
    
        #Non_recursive
        if not root:
            return []
        stack=[root]
        preorder=[]
        
        while stack:
            node=stack.pop()
            preorder.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return preorder
                
#67. Binary Tree Inorder Traversal         
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: A Tree
    @return: Inorder in ArrayList which contains node values.
    """
    def inorderTraversal(self, root):
        # write your code here
        if not root:
            return []
        
        stack=[]
        inorder=[]
        
        cur=root
        while cur:
            stack.append(cur)
            cur=cur.left
        
        
        while stack:
            node=stack.pop()
            inorder.append(node.val)
            if node.right:
                cur=node.right
                while cur:
                    stack.append(cur)
                    cur=cur.left
        return inorder
#68. Binary Tree Postorder Traversal 
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: A Tree
    @return: Postorder in ArrayList which contains node values.
    """
    def postorderTraversal(self, root):
        # write your code here    
        if not root:
            return []
        
        stack=[root]
        postorder=[]
        while stack:
            node=stack.pop()
            postorder.insert(0,node.val)
            
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return postorder
            
#69. Binary Tree Level Order Traversal 
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
class Solution:
    """
    @param root: A Tree
    @return: Level order a list of lists of integer
    """
    def levelOrder(self, root):
        # write your code here
        if not root:
            return []
        
        q=[root]
        result=[]
        
        while q:
            new_q=[]
            result.append([n.val for n in q])
            
            for n in q:
                if n.left:
                    
                  new_q.append(n.left)
                if n.right:
                  new_q.append(n.right)
            q=new_q
                    
        return result
    
root=TreeNode(1)  
root.left=    TreeNode(2) 
root.right=    TreeNode(3) 
root.left.left=TreeNode(4)
root.left.right=TreeNode(5) 
root.left.right.left=TreeNode(7)
root.left.right.right=TreeNode(8)
root.right.right=TreeNode(6)



    1
   / \
  2   3
 / \   \
4   5   6
   / \
  7   8 
if __name__ == "__main__":
    print(Solution().levelOrder(root)) 

#70. Binary Tree Level Order Traversal II
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: A tree
    @return: buttom-up level order a list of lists of integer
    """
    def levelOrderBottom(self, root):
        # write your code here
        if not root:
            return []
        
        q=[root]
        result=[]
        
        while q:
            new_q=[]
            result.append([n.val for n in q])
            
            for n in q:
                if n.left:
                    
                  new_q.append(n.left)
                if n.right:
                  new_q.append(n.right)
            q=new_q
                    
        return result[::-1]


#71. Binary Tree Zigzag Level Order Traversal
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: A Tree
    @return: A list of lists of integer include the zigzag level order traversal of its nodes' values.
    """
    def zigzagLevelOrder(self, root):
        # write your code here
        if not root:
            return []
        
        q=[root]
        result=[]
        direction=True
        
        while q:
            new_q=[]
            if direction:
               result.append([n.val for n in q])
            else:
               result.append([n.val for n in q][::-1])
            direction= not direction
                
            
            for n in q:
                if n.left:
                    
                  new_q.append(n.left)
                if n.right:
                  new_q.append(n.right)
            q=new_q
                    
        return result
    
    
root=TreeNode(1)  
root.left=    TreeNode(2) 
root.right=    TreeNode(3) 
root.left.left=TreeNode(4)
root.left.right=TreeNode(5) 
root.left.right.left=TreeNode(7)
root.left.right.right=TreeNode(8)
root.right.right=TreeNode(6)
    1
   / \
  2   3
 / \   \
4   5   6
   / \
  7   8 
if __name__ == "__main__":
    print(Solution().zigzagLevelOrder( root))         
        
#72. Construct Binary Tree from Inorder and Postorder Traversal        
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param inorder: A list of integers that inorder traversal of a tree
    @param postorder: A list of integers that postorder traversal of a tree
    @return: Root of a tree
    """
    def buildTree(self, inorder, postorder):
        # write your code here
        if not inorder or not postorder:
            return None
        
        root=TreeNode(postorder[-1])
        
        rootPOS=inorder.index(postorder[-1])
        
        root.left=self.buildTree(inorder[:rootPOS],postorder[:rootPOS])
        root.right=self.buildTree(inorder[rootPOS+1:],postorder[rootPOS:-1])
        return root

#73. Construct Binary Tree from Preorder and Inorder Traversal 
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param inorder: A list of integers that inorder traversal of a tree
    @param postorder: A list of integers that postorder traversal of a tree
    @return: Root of a tree
    """
    def buildTree(self,  preorder,inorder):
        # write your code here
#(a) Inorder (Left, Root, Right) : 4 2 5 1 3
#(b) Preorder (Root, Left, Right) : 1 2 4 5 3
#(c) Postorder (Left, Right, Root) : 4 5 2 3 1 
        if not inorder :
            return None
        
        root=TreeNode(preorder[0])
        rootPOS=inorder.index(preorder[0])
        root.left=self.buildTree(preorder[1:rootPOS+1],inorder[:rootPOS])
        root.right=self.buildTree(preorder[rootPOS+1:],inorder[rootPOS+1:])
        return root
        
#74. First Bad Version 
"""
class SVNRepo:
    @classmethod
    def isBadVersion(cls, id)
        # Run unit tests to check whether verison `id` is a bad version
        # return true if unit tests passed else false.
You can use SVNRepo.isBadVersion(10) to check whether version 10 is a 
bad version.
"""


class Solution:
    """
    @param: n: An integer
    @return: An integer which is the first bad version.
    """
    def findFirstBadVersion(self, n):
        # write your code here
        if n<=1:
            return n
        
        
        l=1
        r=n
        
        while l+1<r:
            mid=(l+r)>>1
            if SVNRepo.isBadVersion(mid):
                r=mid
            else:
                l=mid
        
        
        for i in range(l,r+1):
            if  not SVNRepo.isBadVersion(i)  and SVNRepo.isBadVersion(i+1):
                return i+1

#75. Find Peak Element
class Solution:
    """
    @param: A: An integers array.
    @return: return any of peek positions.
    """
    def findPeak(self, A):
        # write your code here
        n=len(A)
        
        if n==1:
            return A[0]
        
        start=0
        end=n-1
        
        while start+1<end:
            mid=(start+end)>>1
            if A[mid]<A[mid+1]:
                start=mid
            elif A[mid]<A[mid-1]:
                end=mid
            else:
                end=mid
        
        if A[start]>A[end]:
            return start
        else:
            return end
A=[1, 2, 1, 3, 4, 5, 7, 6]
if __name__ == "__main__":
    print(Solution().findPeak( A))            
            
#76. Longest Increasing Subsequence 
class Solution:
    """
    @param nums: An integer array
    @return: The length of LIS (longest increasing subsequence)
    """
    def longestIncreasingSubsequence(self, nums):
        # write your code here
        #n**2
        if not nums:
            return 0
        
        n=len(nums)
        
        dp=[1]*n
        
        for i , val in enumerate(nums):
            for j in range(i):
                if nums[j]<val:
                   dp[i]=max(dp[i],dp[j]+1)
        return max(dp)
        #n*lgn
        if not nums:
            return 0
        
        n=len(nums)
        
        tail=[0]*n
        size=0
        
        for x in nums:
            i=0
            j=size
            while i!=j:
                mid=(i+j)>>1
                if tail[mid]<x:
                    i=mid+1
                else:
                    j=mid
            tail[i]=x
            size=max(size,i+1)
        return size
                
#77. Longest Common Subsequence
class Solution:
    """
    @param A: A string
    @param B: A string
    @return: The length of longest common subsequence of A and B
    """
    def longestCommonSubsequence(self, A, B):
        # write your code here
        m=len(A)
        n=len(B)
        
        if not m or not n:
            return 0
        
        dp=[[0 for _ in range(n+1)]  for _ in range(m+1)]
        
        for i in range(1,n+1):
            for j in range(1,m+1):
                if A[i-1]==B[j-1]:
                    dp[i][j]=dp[i-1][j-1]+1
                else:
                    dp[i][j]=max(dp[i][j-1],dp[i-1][j])
        return dp[m][n]
#78. Longest Common Prefix
class Solution:
    """
    @param strs: A list of strings
    @return: The longest common prefix
    """
    def longestCommonPrefix(self, strs):
        # write your code here
        if not strs:
            return ''
        
        if len(strs)==1:
            return strs[0]
        
        
        for i in range(len(strs[0])):
            for string in strs[1:]:
                if i+1>len(string)  or strs[0][i]!=string[i]:
                    return strs[0][:i]
        return strs[0]
strs=["ABCD", "ABEF"]
strs=["abc","abcd","","ab","ac"]
if __name__ == "__main__":
    print(Solution().longestCommonPrefix( strs))           
        
#79. Longest Common Substring        
class Solution:
    """
    @param A: A string
    @param B: A string
    @return: the length of the longest common substring.
    """
    def longestCommonSubstring(self, A, B):
        # write your code here
        m=len(A)
        n=len(B) 
        
        if not m or not n:
            return 0
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        ans=0
        for i in range(1,m+1):
            for j in range(1,n+1):
                if A[i-1]==B[j-1]:
                    dp[i][j]=dp[i-1][j-1]+1
                ans=max(ans,dp[i][j])
        print(dp)
        return ans
                    
A="www.lintcode.com code"
B="www.ninechapter.com code"        
if __name__ == "__main__":
    print(Solution().longestCommonSubstring( A, B))                 
#80. Median        
class Solution:
    """
    @param nums: A list of integers
    @return: An integer denotes the middle number of the array
    """
    def median(self, nums):
        # write your code here        
        def partition(s,e,k):
            
            p=s+1
            q=len(nums)-1
            
            while p<=q:
                if nums[p]>nums[s]:
                    p+=1
                else:
                    nums[p],nums[q]=nums[q],nums[p]
                    q-=1
            nums[s],nums[q]=nums[q],nums[s]
            
            m=q
            
            if m==k:
                return nums[m]
            elif m>k:
                return partition(s,m-1,k)
            else:
                return partition(m+1,e,k)
        
        if len(nums)%2==1:
            return  partition(0,len(nums)-1,len(nums)//2)
        else:
            #print(partition(0,len(nums)-1,len(nums)//2))
            #print(partition(0,len(nums)-1,len(nums)//2-1))
            return partition(0,len(nums)-1,len(nums)//2)
nums=[4, 5, 1, 2, 3]   
nums=[7, 9, 4, 5]             
if __name__ == "__main__":
    print(Solution().median( nums))         
        
#81. Find Median from Data Stream
import heapq        
class Solution:
    """
    @param nums: A list of integers
    @return: the median of numbers
    """
    
    def medianII(self, nums):
        # write your code here 
        minheap=[]
        maxheap=[]
        ans=[]
        
            
        def add_to_heap(minheap,maxheap,num):
            if not maxheap or num < -maxheap[0]:
                heapq.heappush(maxheap,-num)
            else:
                heapq.heappush(minheap,num)
                
        
        def balance(minheap,maxheap):
            while len(maxheap) < len(minheap):
                heapq.heappush(maxheap,-heapq.heappop(minheap))
            while len(maxheap) > len(minheap)+1:
                heapq.heappush(minheap,-heapq.heappop(maxheap))
            
            
        for num in nums:
            add_to_heap(minheap,maxheap,num)
            balance(minheap,maxheap)
            median=-maxheap[0]
            ans.append(median)
            print(minheap,maxheap)
        return ans
nums=[4, 5, 1, 3, 2, 6, 0,200,100]            
if __name__ == "__main__":
    print(Solution(). medianII( nums))         
        
#82. Single Number
class Solution:
    """
    @param A: An integer array
    @return: An integer
    """
    def singleNumber(self, A):
        # write your code here
        
        res=A[0]
        
        for i in range(1,len(A)):
            res^=A[i]
        return res
A=[1,2,2,1,3,4,3]
if __name__ == "__main__":
    print(Solution().singleNumber( A))       


#83. Single Number II 
class Solution:
    """
    @param A: An integer array
    @return: An integer
    """
    def singleNumberII(self, A):
        # write your code here
        # 创建一个长度为32的数组countsPerBit，  
        # countsPerBit[i]表示A中所有数字在i位出现的次数  
        countsPerBit=[0]*32
        res=0
        
        for i in range(32):
            for j in A:
                if (j>>i)&1 ==1:
                    countsPerBit[i]=(countsPerBit[i]+1)%3
            res|=countsPerBit[i]<<i
        return res
A=[1,1,2,3,3,3,2,2,4,1]        
if __name__ == "__main__":
    print(Solution().singleNumberII( A))        
        
#84. Single Number III         
class Solution:
    """
    @param: A: An integer array
    @return: An integer array
    """
    def singleNumberIII(self, A):
        # write your code here        
#http://fisherlei.blogspot.com/2015/10/leetcode-single-number-iii-solution.html
#Why diff &= ~(diff - 1)
#First, this the original formula to get the last set bit. The diff &= -diff is just an abbreviation with the knowledge of ~(diff - 1) = - (diff - 1) - 1 = -diff.
#
#If diff is set on the least significant bit, then this is trivially provable: the least significant bit is the last set bit. After the -1 operation, this least significant bit became 0, and is the only change to all bits of diff. Then we ~ the result, which means the least significant bit gets reverted to 1, while all the other bits are guaranteed to have been reverted. Thus the least significant bit is the only bit that is left unchanged and that could survive the & operation.
#If diff is unset on the least significant bit: let's focus on the rightmost occurrence of 10 in diff. The 1 bit in this 10 is the last set bit of diff. After -1 operation, this 10 becomes 01. All the 0 bits to the right of this rightmost 10 are all change to 1 bits, and all the other whatever bits to the left of this rightmost 10 would be remain unchanged:
#**..**10_00..00
#after -1:
#**..**01_11..11
#Then we do ~ operation. The **..** part would all be reverted, and none of them would survive the upcoming & operation. 01 would become back 10, and would both survive the & operation, although the bit 1 is the only one we care about. All those 11..11 part gets reverted back to 00..00 after the ~ operation, and does not matter to the & operation. Thus the only thing surviving the & operation would be the rightmost 10, or the last set bit which is within it.
#
#Incidentally, it does not really matter which bit we choose for the second pass to succeed, but since there is an elegant way to find the rightmost set bit, then let's use that.
#        
        dif=0
        for num in A:
            dif^=num
        dif=dif & (~(dif-1))
        
        c1=0
        c2=0
        for num in A:
            if dif&num==0:
                c1^=num
                
            else:
                c2^=num
        return [c1,c2]
        
#85. Insert Node in a Binary Search Tree 
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: node: insert this node into the binary search tree
    @return: The root of the new binary search tree.
    """
    def insertNode(self, root, node):
        # write your code here
        if not root:
            return node
        
        cur=root
        
        while cur !=node:
            if node.val<cur.val:
                if not cur.left:
                    cur.left=node
                cur=cur.left
            else:
                if not cur.right:
                    cur.right=node
                cur=cur.right
        return root
                
        
#86. Binary Search Tree Iterator         
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

Example of iterate a tree:
iterator = BSTIterator(root)
while iterator.hasNext():
    node = iterator.next()
    do something for node 
"""


class BSTIterator:
    """
    @param: root: The root of binary tree.
    """
    def __init__(self, root):
        # do intialization if necessary
        self.stack=[]
        self.root=root

    """
    @return: True if there has next node, or false
    """
    def hasNext(self ):
        # write your code here
        if self.stack or self.root:
            return True

    """
    @return: return next node
    """
    def next(self ):
        # write your code here
        
        while self.root:
            self.stack.append(self.root)
            
            self.root=self.root.left
        node=self.stack.pop()
        nxt=node
        self.root=node.right
        return nxt
                
#87. Remove Node in Binary Search Tree 
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: value: Remove the node with given value.
    @return: The root of the binary search tree after removal.
    """
    def removeNode(self, root, value):
        # write your code here 
        if root.val > value:
            root.left=self.removeNode(root.left,value)
        elif root.val < value:
            root.right=self.removeNode(root.right,value)
        else:
            if not root.left:
                return root.right
            elif not root.right:
                return root.left
            else:
                smallestright=root.right
                while smallestright.left:
                    smallestright=smallestright.left
                smallestright.left=root.left
                return root.right
        return root
                      
#88. Lowest Common Ancestor of a Binary Tree 
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: A: A TreeNode in a Binary.
    @param: B: A TreeNode in a Binary.
    @return: Return the least common ancestor(LCA) of the two nodes.
    """
    def lowestCommonAncestor(self, root, A, B):
        # write your code here

        if not root:
          return root
        if root==A or root==B:
            return root
        
        left=self.lowestCommonAncestor(root.left,A,B)
        right=self.lowestCommonAncestor(root.right,A,B)
        
        if left  and right:
            return root
        
        if left:
            return left 
        if right:
            return right
        return None
            
#89. k Sum 
class Solution:
    """
    @param A: An integer array
    @param k: A positive integer (k <= length(A))
    @param target: An integer
    @return: An integer
    """
    
    def kSum(self, A, k, target):
        # write your code here
        
#        self.result=0
#        def findkSum(A,k,target):
#            
#            if not A or len(A)<k or A[0]*k >target or     A[-1]*k <target  :
#               return 
#            
#            if k==1:
#                for a in A:
#                    if a==target:
#                       self.result+=1
#                
#                
#        
#            elif k==2:
#               r=len(A)-1
#               l=0
#               while l<r:
#                   tempsum=A[r]+A[l]
#                   if tempsum>target:
#                       r-=1
#                   elif tempsum<target:
#                       l+=1
#                   else:
#                       self.result+=1
#                       while l<r and A[l]==A[l+1]:
#                           l+=1
#                       while l<r and A[r]==A[r-1]:
#                           r-=1
#                       r-=1
#                       l+=1
#            else:         
#              for i in range(len(A)-k+1):
#                if i==0 or A[i]!=A[i-1]:
#                   findkSum(A[i+1:],k-1,target-A[i])
#            
#        
#        A.sort()
#        
#        findkSum(A,k,target)
#        return self.result
            
            n=len(A)
            ksum=[[[0 for _ in range(k+1)]  for _ in range(n+1)]  for _ in range(target+1)]
            for i in range(n+1):
               ksum[0][i][0]=1
        
            for i in range(1,target+1):
              for j in range(1,n+1):
                for l in range(1,min(j,k)+1):
                   ksum[i][j][l] =ksum[i][j-1][l]
                   if i>=A[j-1]  :
                       ksum[i][j][l]+=ksum[i-A[j-1]][j-1][l-1]
        
            return ksum[target][n][k]
                       
A=[1,3,4,5,8,10,11,12,14,17,20,22,24,25,28,30,31,34,35,37,38,40,42,44,45,48,51,54,56,59,60,61,63,66]
k = 24
target = 842 
A=[1,2,3,4]
k = 2
target = 5 
A=[1,3,5,7,10,13,14,17,19,22,24,27,30,33,34,36,38,41] 
k=5
target=176                    
if __name__ == "__main__":
    print(Solution().kSum( A, k, target))        
                               
#90. k Sum II                          
class Solution:
    """
    @param: A: an integer array
    @param: k: a postive integer <= length(A)
    @param: targer: an integer
    @return: A list of lists of integer
    """
    def kSumII(self, A, k, target):
        # write your code here
        def findkSum(A,k,target,result,results):
            
            
            if not A or len(A)<k or A[0]*k >target or     A[-1]*k <target  :
               return 
            
            if k==1:
                for a in A:
                    if a==target:
                       results.append([a])
                
                
        
            elif k==2:
               r=len(A)-1
               l=0
               while l<r:
                   tempsum=A[r]+A[l]
                   if tempsum>target:
                       r-=1
                   elif tempsum<target:
                       l+=1
                   else:
                       results.append(result+[A[r],A[l] ])
                       while l<r and A[l]==A[l+1]:
                           l+=1
                       while l<r and A[r]==A[r-1]:
                           r-=1
                       r-=1
                       l+=1
            else:         
              for i in range(len(A)-k+1):
                if i==0 or A[i]!=A[i-1]:
                   findkSum(A[i+1:],k-1,target-A[i],result+[A[i]],results)
            
        results=[]
        A.sort()
        
        findkSum(A,k,target,[],results)
        
        return results 
A=[1,4,5,6,8]
k=1
target=4                          
if __name__ == "__main__":
    print(Solution().kSumII( A, k, target)) 
       
#91. Minimum Adjustment Cost
class Solution:
    """
    @param: A: An integer array
    @param: target: An integer
    @return: An integer
    """
    def MinAdjustmentCost(self, A, target):
        # write your code here                                                          
#http://ryanleetcode.blogspot.com/2015/05/minimum-adjustment-cost.html
#https://www.geeksforgeeks.org/find-minimum-adjustment-cost-of-an-array/                       
#https://zhengyang2015.gitbooks.io/lintcode/minimum_adjustment_cost_91.html
        if not A:
          return 0
      
        #you can assume each number in the array is a positive integer and not greater than 100.
        dp=[[0 for _ in range(101)] for _ in range(len(A)+1) ]
        
        for i in range(1,101):
            dp[1][i]=abs(A[0]-i)
            
        for i in range(2,len(A)+1):
            for j in range(1,101):
                dp[i][j]=float('inf')
                for k in range(1,101):
                    if abs(k-j)<=target:
                        dp[i][j]=min(dp[i][j],dp[i-1][k]+abs(j-A[i-1]))
        
        res=float('inf')
        for j in range(1,101):
        
            res=min(res,dp[len(A)][j])
        return res
A=[1,4,2,3] 
target = 1    
if __name__ == "__main__":
    print(Solution().MinAdjustmentCost( A, target) )
        
#92. Backpack
class Solution:
    """
    @param m: An integer m denotes the size of a backpack
    @param A: Given n items with size A[i]
    @return: The maximum size
    """
    def backPack(self, m, A):
        # write your code here
#https://aaronice.gitbooks.io/lintcode/content/dynamic_programming/backpack.html
#f[i][S] “前i”个物品，取出一些能否组成和为S            
#        if not A:
#            return 0
#        n=len(A)
#        dp =[ [0 for _ in range(m+1)] for _ in range(n+1)  ]  
#        dp[0][0]=True
#        
##        for j in range(1,m+1):
##            dp[0][j]=False
##            
##        for j in range(1,n+1) :   
##            dp[j][0]=True
#            
#            
#        for i in range(1,n+1):
#            for j in range(m+1):
#                if dp[i-1][j] or ( j-A[i-1] >=0 and dp[i-1][j-A[i-1]] ):
#                    dp[i][j]=True
#                else:
#                    dp[i][j]=False
#        
#        
#        for k in range(m,0,-1):
#            if dp[n][k]:
#                return k
#        return 0
        if not A:
            return 0
        n=len(A)
        
        dp=[0 for _ in range(m+1)]
        dp[0]=True
        ans=0
        for item in A:
            for j in range(m,0,-1):
                if j-item>=0 and dp[j-item]:
                    dp[j]=True
                    ans=max(ans,j)
        return ans
                    
A=[81,112,609,341,164,601,97,709,944,828,627,730,460,523,643,901,602,508,401,442,738,443,555,471,97,644,184,964,418,492,920,897,99,711,916,178,189,202,72,692,86,716,588,297,512,605,209,100,107,938,246,251,921,767,825,133,465,224,807,455,179,436,201,842,325,694,132,891,973,107,284,203,272,538,137,248,329,234,175,108,745,708,453,101,823,937,639,485,524,660,873,367,153,191,756,162,50,267,166,996,552,675,383,615,985,339,868,393,178,932] 
m = 80000  
if __name__ == "__main__":
    print(Solution().backPack( m, A) )                    

#93. Balanced Binary Tree
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """
    def isBalanced(self, root):
        # write your code here
        def getHeight(node):
            if not node:
                return 0
            lh=getHeight(node.left)
            rh=getHeight(node.right)
            return max(lh,rh)+1
        
        if not root:
            return True
        
        lh=getHeight(root.left)
        rh=getHeight(root.right)
        return abs(lh-rh) <=1  and self.isBalanced(root.left)  and self.isBalanced(root.right)
                    
#94. Binary Tree Maximum Path Sum 
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """
    def maxPathSum(self, root):
        # write your code here
        self.maxseen=float('-inf')
        def pathSum(node):
            if not node:
                return 0
            
            left=max(0,pathSum(node.left))
            right=max(0,pathSum(node.right))
            
            self.maxseen=max(self.maxseen,left+right+node.val)
            return node.val+max(left,right)
            
        pathSum(root)   
        return self.maxseen        
  1
 / \
2   3                      
root= TreeNode(1)
root.left=TreeNode(2)
root.right=TreeNode(3) 
if __name__ == "__main__":
    print(Solution().maxPathSum( root) )   

#95. Validate Binary Search Tree 
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of binary tree.
    @return: True if the binary tree is BST, or false
    """
    def isValidBST(self, root):
        # write your code here
        
        def inorder(root):
            if root.left:
                inorder(root.left)
            ans.append(root.val)
            if root.right:
                inorder(root.right)
        
        if not root:
            return True
        ans=[]
        inorder(root)
        
        print(ans)
        last=None
        for node in  sorted(ans):
            print(last,node)
            if not last:
                last=node
                continue
            
            if last==node:
                return False
            last=node
            
        if ans[:]==sorted(ans):
            return True
        else:
            return False

                
#{2,1,2}                
            
root= TreeNode(2)
root.left=TreeNode(1)
root.right=TreeNode(2) 
if __name__ == "__main__":
    print(Solution().isValidBST( root) )
                   
                    
#96. Partition List            
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: The first node of linked list
    @param x: An integer
    @return: A ListNode
    """
    def partition(self, head, x):
        # write your code here            
        
        
        
#Given 1->4->3->2->5->2->null and x = 3,
#return 1->2->2->4->3->5->null.  
        smallhead=ListNode(None)
        largehead=ListNode(None)
        
        
            
        cur=head
        smallheadcur    =smallhead
        largeheadcur    =largehead
        while cur:
            if cur.val<x:
                smallheadcur.next=cur
                if smallheadcur.next:
                   smallheadcur=smallheadcur.next
            else:
                largeheadcur.next=cur
                largeheadcur=largeheadcur.next
            #print(smallheadcur.val,largeheadcur.val)
            cur=cur.next
        if largeheadcur.next:
            largeheadcur.next=None
            
        smallheadcur.next=largehead.next
        
        temp=smallhead.next
        while temp:
            print(temp.val,sep=',', end='')
            temp=temp.next
        
        return smallhead.next
#1->4->3->2->5->2
head=ListNode(1)
head.next=ListNode(4)
head.next.next=ListNode(3)
head.next.next.next=ListNode(2)
head.next.next.next.next=ListNode(5)
head.next.next.next.next.next=ListNode(2)


head=ListNode(3)
head.next=ListNode(3)
head.next.next=ListNode(1)
head.next.next.next=ListNode(2)
head.next.next.next.next=ListNode(4)

x=3
if __name__ == "__main__":
    print(Solution().partition( head, x) )        
        
#97. Maximum Depth of Binary Tree                
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """
    def maxDepth(self, root):
        # write your code here
#  1
# / \ 
#2   3
#   / \
#  4   5            
#        
        def height(node):
           if not node:
               return 0
           return max(height(node.left),height(node.right))+1
       
        return height(root)
        
#98. Sort List      
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: The head of linked list.
    @return: You should return the head of the sorted linked list, using constant space complexity.
    """
    def sortList(self, head):
        # write your code here
        def merge(list1,list2) :
            if not list1:
                return list2
            if not list2:
                return list1
            
            head=None
            
            if list1.val<list2.val:
                head=list1
                list1=list1.next
            else:
                head=list2
                list2=list2.next
                
            temp=head
            
            while list1 and list2:
                  if list1.val<list2.val:
                      temp.next=list1
                      list1=list1.next
                      temp=temp.next
                  else:
                      temp.next=list2
                      list2=list2.next
                      temp=temp.next
            if list1:
                temp.next=list1
            if list2:
                temp.next=list2
                
            return head
        
        
        if not head:
            return head
        
        if not head.next:
            return head
        
        slow=head
        fast=head
        
        while fast.next and fast.next.next:
              fast=fast.next.next
              slow=slow.next
        mid=slow.next
        slow.next=None
        
        
        
        list1=self.sortList(head)
        list2=self.sortList(mid)
        
        newhead=merge(list1,list2)
        return newhead
        
#99. Reorder List        
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: The head of linked list.
    @return: nothing
    """
    def reorderList(self, head):
        # write your code here
        if not head or not head.next or not head.next.next:
            return head
        
        pfast=head
        pslow=head
        
        while pfast.next and pfast.next.next:
            pfast=pfast.next.next
            pslow=pslow.next
        
        part2_head=pslow.next
        pslow.next=None
        
        cur=part2_head.next
        prev=part2_head
        part2_head.next=None
        
        while cur:
            temp=cur.next
            cur.next=prev
            prev=cur
            cur=temp
        
        
        part1_node=head
        part2_node=prev
        
        while part2_node:
               temp=part2_node.next
               part2_node.next=part1_node.next
               part1_node.next=part2_node
               part2_node=temp
               part1_node=part1_node.next.next
        cur=head
        while cur:
           print(cur.val,sep=',',end='')
           cur=cur.next
        return head
               
#1->2->3->4        
            
head=ListNode(1)
head.next=ListNode(2)
head.next.next=ListNode(3)
head.next.next.next=ListNode(4)
if __name__ == "__main__":
    print(Solution().reorderList( head) )               
#100. Remove Duplicates from Sorted Array        
class Solution:
    """
    @param: nums: An ineger array
    @return: An integer
    """
    def removeDuplicates(self, nums):
        # write your code here
        n=len(nums)
        if n==0 :
            return 0
        index=0
        for i in range(1,n):
            if nums[index]!=nums[i]:
                index+=1
                nums[index]=nums[i]
        return index+1
nums=   [-10,0,1,2,3]
nums=   [-10,0,1,1,1,2,3]
nums=[]
if __name__ == "__main__":
    print(Solution().removeDuplicates( nums) )                 
            
#101. Remove Duplicates from Sorted Array II    
class Solution:
    """
    @param: nums: An ineger array
    @return: An integer
    """
    def removeDuplicates(self, nums):
        # write your code here  
        n=len(nums)
        if n==0 :
            return 0
        if n==1:
            return 1
        
        if n==2:
            return 2
        
        index=1
       
        for i in range(2,n):
            if  nums[index-1]!=nums[i]:
                index+=1
                nums[index]=nums[i]
        print(nums[:index+1])
        return index+1
        
nums=   [-10,0,1,2,3]
nums=   [-10,0,1,1,1,2,3]
nums=[]        
if __name__ == "__main__":
    print(Solution().removeDuplicates( nums) )           
                
#102. Linked List Cycle                       
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""


class Solution:
    """
    @param: head: The first node of linked list.
    @return: True if it has a cycle, or false
    """
    def hasCycle(self, head):
        # write your code here
        if not head:
           return   False
        fast=head
        slow=head
        
        while fast.next and fast.next.next:
            fast=fast.next.next
            slow=slow.next
            if fast==slow:
                return True
        return False
            
#103. Linked List Cycle II        
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""


class Solution:
    """
    @param: head: The first node of linked list.
    @return: The node where the cycle begins. if there is no cycle, return null
    """
    def detectCycle(self, head):
        # write your code here
        if not head or not head.next or not head.next.next:
            return None
#        fast=head.next.next
#        slow=head.next
        
        fast=head
        slow=head
        
        while fast and fast.next:
              fast=fast.next.next
              slow=slow.next
              if not fast or not fast.next:
                  return None
              if fast==slow:
                  break
        fast=head
        
        while True:
            fast=fast.next
            slow=slow.next
            if fast==slow:
                break
        return slow
            
        
       
    
-21->10->17->8->4->26->5    
head= ListNode(21)
head.next=ListNode(10)
head.next.next=ListNode(17) 
head.next.next.next=ListNode(8) 
head.next.next.next.next=ListNode(4) 
head.next.next.next.next.next=ListNode(26) 
head.next.next.next.next.next.next =ListNode(5)  

head.next.next.next.next.next.next.next=head2

->35>33->-7 -16->27->-12->6
head2= ListNode(35)
head2.next=ListNode(33)
head2.next.next=ListNode(7) 
head2.next.next.next=ListNode(16) 
head2.next.next.next.next=ListNode(27) 
head2.next.next.next.next.next=ListNode(12) 
head2.next.next.next.next.next.next =ListNode(6)
  
head2.next.next.next.next.next.next.next=head3

29- 12->5->9->20->14->14-
head3= ListNode(29)
head3.next=ListNode(12)
head3.next.next=ListNode(5) 
head3.next.next.next=ListNode(9) 
head3.next.next.next.next=ListNode(20) 
head3.next.next.next.next.next=ListNode(14) 
head3.next.next.next.next.next.next =ListNode(14)

head3.next.next.next.next.next.next.next=head4

2->13->-24->21->23->-21->5 
head4= ListNode(2)
head4.next=ListNode(13)
head4.next.next=ListNode(24) 
head4.next.next.next=ListNode(21) 
head4.next.next.next.next=ListNode(23) 
head4.next.next.next.next.next=ListNode(21) 
head4.next.next.next.next.next.next =ListNode(5)

head4.next.next.next.next.next.next.next= head4.next.next

-21->10->17->8->4->26->5
->35>33->-7 -16->27->-12->6
->29- 12->5->9->20->14->14-
>2->13->-24->21->23->-21->5 

if __name__ == "__main__":
    print(Solution().detectCycle(  head) )              
        
#104. Merge K Sorted Lists  
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""
import heapq
class Solution:
    """
    @param lists: a list of ListNode
    @return: The head of one sorted list.
    """
    def mergeKLists(self, lists):
        # write your code here   
        heap=[]
        for i,ll in  enumerate(lists):
            if ll:
                heapq.heappush(heap,(ll.val,i,ll))
        
        dummy=ListNode(-1)
        
        cur=dummy
        while heap:
            _,idx,node=heapq.heappop(heap)
            cur.next=node
            cur=cur.next
            if cur.next:
                heapq.heappush(heap,(cur.next.val,idx,cur.next))
        return dummy.next
                
#105. Copy List with Random Pointer             
"""
Definition for singly-linked list with a random pointer.
class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None
"""


class Solution:
    # @param head: A RandomListNode
    # @return: A RandomListNode
    def copyRandomList(self, head):
        # write your code here
        if not head :
           return None


        root=   RandomListNode(head.label) 
        cur=head
        cur_copy=root
        while cur:
              if cur.next:
                  nextnode=RandomListNode(cur.next.label)
              else:
                  nextnode=None
              if cur.random:
                  nextrandom=RandomListNode(cur.random.label)
              else:
                  nextrandom=None
              
              cur_copy.next=  nextnode 
              cur_copy.random=  nextrandom
              cur=cur.next
              cur_copy=cur_copy.next
        return root
              
#106. Convert Sorted List to Binary Search Tree            
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next

Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    """
    @param: head: The first node of linked list.
    @return: a tree node
    """
    def sortedListToBST(self, head):
        # write your code here 
        
        
        if not head:
            return None
        if not head.next:
            return TreeNode(head.val)
        
        fast=head.next.next
        slow=head
        
        while fast and fast.next:
            fast=fast.next.next
            slow=slow.next
        
        temp=slow.next
        slow.next=None
        
        root=TreeNode(temp.val)
        root.left=self.sortedListToBST(head)
        root.right=self.sortedListToBST(temp.next)
        return root
        
#107. Word Break            
class Solution:
    """
    @param: s: A string
    @param: dict: A dictionary of words dict
    @return: A boolean
    """
    def wordBreak(self, s, dict):
        # write your code here 
           
        if not s:
            return len(s)==0
        if not dict:
            return False
        n=len(s)
        dp=[False for _ in range(n+1)]
        dp[0]=True
        
        
        maxlen=max([len(w)  for w in dict])
        
        
        for i in range(n+1):
            for j in range(1,min(i,maxlen)+1):
                if not dp[i-j]:
                    continue
                if s[i-j:i]  in dict:
                    dp[i]=True
                    break
        return dp[n]
            

            
s = "lintcode"   
dict = ["lint", "code"]     
if __name__ == "__main__":
    print(Solution().wordBreak( s, dict))            
        
#108. Palindrome Partitioning II         
class Solution:
    """
    @param s: A string
    @return: An integer
    """
    def minCut(self, s):
        # write your code here
        n=len(s)
        
        dp=[i-1 for i in range(n+1)]
        
        for i in range(n+1):
            j=0
            k=1
            while i+j<n and i-j>=0 and s[i+j]==s[i-j]:
                dp[i+j+1]=min(dp[i+j+1],dp[i-j]+1)
                j+=1
            while i+k<n and i-k+1>=0 and s[i+k]==s[i-k+1]:
                dp[i+k+1]=min(dp[i+k+1],dp[i-k+1]+1)
                k+=1
        return dp[n]
s="aab"
s='bb'
if __name__ == "__main__":
    print(Solution().minCut(s))         
        
#109. Triangle         
class Solution:
    """
    @param triangle: a list of lists of integers
    @return: An integer, minimum path sum
    """
    def minimumTotal(self, triangle):
        # write your code here  
        n=len(triangle)
        dp=[[float('inf') for _ in range(n)] for _ in range(n)]
        dp[0][0]=triangle[0][0]
        
        for i in range(1,n):
            for j in range(i+1):
                
               if j==0:
                  dp[i][j] =dp[i-1][j]+ triangle[i][j]
               elif j==i:
                  dp[i][j]=dp[i-1][j-1]+ triangle[i][j]
               else:
                  dp[i][j]=min(dp[i-1][j-1],dp[i-1][j])+ triangle[i][j]
               
        #print(dp)
        return min(dp[n-1][i] for i in range(n))
  
triangle=[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
] 
triangle=[[-7],
 [-2,1],
 [-5,-5,9],
 [-4,-5,4,4],
 [-6,-6,2,-1,-5],
 [3,7,8,-3,7,-9],
 [-9,-1,-9,6,9,0,7],
 [-7,0,-6,-8,7,1,-4,9],
 [-3,2,-6,-9,-7,-6,-9,4,0],
 [-8,-6,-3,-9,-2,-6,7,-5,0,7],
 [-9,-1,-2,4,-2,4,4,-1,2,-5,5],
 [1,1,-6,1,-2,-4,4,-2,6,-6,0,6],
 [-3,-3,-6,-2,-6,-2,7,-9,-5,-7,-5,5,1]]       

if __name__ == "__main__":
    print(Solution().minimumTotal(triangle))        
        
#110. Minimum Path Sum        
class Solution:
    """
    @param grid: a list of lists of integers
    @return: An integer, minimizes the sum of all numbers along its path
    """
    def minPathSum(self, grid):
        # write your code here
        m=len(grid)
        n=len(grid[0])
        if not grid:
            return 0
        dp=[[float('inf') for _ in range(n)] for _ in range(m)]
        dp[0][0]=grid[0][0]
        for i in range(1,m):
            dp[i][0]=grid[i][0]+dp[i-1][0]
        for j in range(1,n):
            dp[0][j]=grid[0][j]+dp[0][j-1]
        #print(dp)    
        for i in range(1,m):
            for j in range(1,n):
                
                dp[i][j]=min(dp[i-1][j],dp[i][j-1])+grid[i][j]
        #print(dp)
        return dp[m-1][n-1]
        
grid=[[1,2],
      [1,1]]  
if __name__ == "__main__":
    print(Solution().minPathSum(grid))      
        
#111. Climbing Stairs        
class Solution:
    """
    @param n: An integer
    @return: An integer
    """
    def climbStairs(self, n):
        # write your code here   
        
        dp=[0 for _ in range(n+1)]
        if n==0:
            return 0
        dp[1]=1
        if n>1:
           dp[2]=2
        if n>2:
           dp[3]=3
        
        for i in range(4,n+1):
            print(dp[n])
            dp[i]+=dp[i-1]
            dp[i]+=dp[i-2]
        print(dp)
        return dp[n]
n=5    
if __name__ == "__main__":
    print(Solution().climbStairs( n))
        
#112. Remove Duplicates from Sorted List        
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: head is the head of the linked list
    @return: head of linked list
    """
    def deleteDuplicates(self, head):
        # write your code here 
        
        if not head :
            return None
        head=cur
        
        while cur:
            while cur.next and  cur.val==cur.next.val:
                cur.next=cur.next.next
            cur=cur.next
        return head
#113. Remove Duplicates from Sorted List II                 
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: head is the head of the linked list
    @return: head of the linked list
    """
    def deleteDuplicates(self, head):
        # write your code here            
        if not head:
            return None
      
        dummy=ListNode(float('inf'))
        dummy.next=head
        cur=dummy
        while cur:
            if cur and cur.next and cur.next.next and cur.next.next.val==cur.next.val  and cur.next.val !=cur.val:
               temp=cur.next
               while temp and temp.next and  temp.next.val==temp.val :
                     temp=temp.next
               cur.next=temp.next
           
               
            if cur.next and cur.next.next and cur.next.next.val==cur.next.val:
                 continue
            else:    
                 cur=cur.next
#        cur=dummy
#        while cur:
#            print(cur.val,sep=',',end='')
#            cur=cur.next
               
        return dummy.next
        
head=ListNode(1)
head.next=   ListNode(2) 
head.next.next=   ListNode(3) 
head.next.next.next=   ListNode(3) 
head.next.next.next.next=   ListNode(4)
head.next.next.next.next.next=   ListNode(4)   
head.next.next.next.next.next.next=   ListNode(5)  

0->1->1->2->3



head=ListNode(1)
head.next=   ListNode(1) 
head.next.next=   ListNode(1) 
head.next.next.next=   ListNode(2) 
head.next.next.next.next=   ListNode(3)


head=ListNode(1)
head.next=   ListNode(1) 
head.next.next=   ListNode(1) 
head.next.next.next=   ListNode(1) 
head.next.next.next.next=   ListNode(1)
head.next.next.next.next.next=   ListNode(2)   
head.next.next.next.next.next.next=   ListNode(2)  
head.next.next.next.next.next.next.next=   ListNode(2) 
head.next.next.next.next.next.next.next.next=   ListNode(2) 
head.next.next.next.next.next.next.next.next.next=   ListNode(2) 
      
        
Given 1->2->3->3->4->4->5, return 1->2->5.
Given 1->1->1->2->3, return 2->3.        
if __name__ == "__main__":
    print(Solution().deleteDuplicates( head))        
        
#114. Unique Paths  
class Solution:
    """
    @param m: positive integer (1 <= m <= 100)
    @param n: positive integer (1 <= n <= 100)
    @return: An integer
    """
    def uniquePaths(self, m, n):
        # write your code here
        
        dp=[[0 for _ in range(n)]  for _ in range(m)]
        
        for i in range(m):
            dp[i][0]=1
        for j in range(n):
            dp[0][j]=1
        
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j]+=dp[i-1][j]
                dp[i][j]+=dp[i][j-1]
        return dp[m-1][n-1]
m = 4 
n = 5
m = 3 
n = 3                
if __name__ == "__main__":
    print(Solution().uniquePaths( m, n))              
            
#115. Unique Paths II         
class Solution:
    """
    @param obstacleGrid: A list of lists of integers
    @return: An integer
    """
    def uniquePathsWithObstacles(self, obstacleGrid):
        # write your code here  
        if not obstacleGrid:
            return 0
        
        m=len(obstacleGrid)
        n=len(obstacleGrid[0])
        dp=[[0 for _ in range(n)]  for _ in range(m)]
        for i in range(m):
            if obstacleGrid[i][0]==0:
               dp[i][0]=1
            else:
                break
        for j in range(n):
            if obstacleGrid[0][j]==0:
               dp[0][j]=1
            else:
                break
        
        for i in range(1,m):
            for j in range(1,n):
                if obstacleGrid[i][j]!=1:
                    dp[i][j]+=dp[i-1][j]
                    dp[i][j]+=dp[i][j-1]
                
        return dp[m-1][n-1]
obstacleGrid=[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]        
if __name__ == "__main__":
    print(Solution().uniquePathsWithObstacles( obstacleGrid))             
        
#116. Jump Game   
class Solution:
    """
    @param A: A list of integers
    @return: A boolean
    """
    def canJump(self, A):
        # write your code here
        if not A:
            return False
        
       
        m=len(A)
        if m==1:
            return True
       
        
        dp=[False for _ in range(m)]
        
        if A[0]>0:
            dp[0]=True
        
        for i in range(m):
            for j in range(1,A[i]+1):
                if dp[i] and i+j < m:
                    dp[i+j]=True
        return dp[m-1]
        
 
        
A = [2,3,1,1,4], return true.

A = [3,2,1,0,4], return false.     
if __name__ == "__main__":
    print(Solution().canJump( A))          
        
#117. Jump Game II        
class Solution:
    """
    @param A: A list of integers
    @return: An integer
    """
    def jump(self, A):
        # write your code here
#        if not A:
#            return False
#        m=len(A)
#        if m==1:
#            return True
#        dp=[float('inf') for _ in range(m)]
#        dp[0]=0
#        for i in range(m):
#            for j in range(1,A[i]+1):
#                if i+j < m:
#                    dp[i+j]=min(dp[i]+1,dp[i+j])
#        return dp[m-1]
        
        m=len(A)
        if m==1:
            return 0
        maxreach=A[0]+0
        step=1
        i=0
        
        while  maxreach <m-1:
               step+=1
               for j in range(i+1,A[i]+i+1):
                   if A[j]+j>maxreach:
                       maxreach= A[j]+j
                       temp=j
                i=temp
        return step
 
A = [2,3,1,1,4]

if __name__ == "__main__":
    print(Solution().jump( A))          
        
The minimum number of jumps to reach the last index is 2. 
(Jump 1 step from index 0 to 1, then 3 steps to the last index        
        
#118. Distinct Subsequences         
class Solution:
    """
    @param: : A string
    @param: : A string
    @return: Count the number of distinct subsequences
    """

    def numDistinct(self, S, T):
        # write your code here
        #看leetcode的描述更清晰
        #给出字符串S和字符串T，计算S的不同的子序列中T出现的个数。
        m=len(T)
        n=len(S)
        
        dp=[[0 for _ in range(n+1)] for _ in range(m+1)]
        
        for j in range(n+1):
            dp[0][j]=1
        
        for i in range(m):
            for j in range(n):
                if T[i]==S[j]:
                    dp[i+1][j+1]=dp[i+1][j]+dp[i][j]
                else:
                    dp[i+1][j+1]=dp[i+1][j]
        return dp[m][n]
               *  * ]
#      S = [acdabefbc]
#mem[1] = [0111222222]
#mem[2] = [0000022244]        

S = "rabbbit"
T = "rabbit"  
if __name__ == "__main__":
    print(Solution().numDistinct( S, T))          
        
#119. Edit Distance             
class Solution:
    """
    @param word1: A string
    @param word2: A string
    @return: The minimum number of steps.
    """
    def minDistance(self, word1, word2):
        # write your code here
        #看leetcode的描述更清晰
        m=len(word1)
        n=len(word2)
        
        dp=[[0 for _ in range(n+1)]for _ in range(m+1)]
        
        for i in range(m+1):
            dp[i][0]=i
        for j in range(n+1):
            dp[0][j]=j
            
        for i in range(1,m+1):
            for j in range(1,n+1):
                if word1[i-1]==word2[j-1]:
                    dp[i][j]=dp[i-1][j-1]
                else:
                    dp[i][j]=min(dp[i-1][j-1]+1,dp[i-1][j]+1,dp[i][j-1]+1)
        return dp[m][n]
word1 = "horse"
word2 = "ros"  
word1 = "intention"
word2 = "execution"          
if __name__ == "__main__":
    print(Solution().minDistance( word1, word2))        
        
        
#120. Word Ladder        
class Solution:
    """
    @param: start: a string
    @param: end: a string
    @param: dict: a set of string
    @return: An integer
    """
    def ladderLength(self, start, end, dict):
        # write your code here
        #看leetcode的描述更清晰
        
       
        import string 
        dic=set(dict)
        dic.add(end)
        dic.add(start)
        
        from collections import deque
      
        q=deque([(start,1)])
       
    
        while q:
            word,step=q.popleft()
           
            if word==end:
                return step
            
            for j in range(len(start)):
                    for k in string.ascii_lowercase:
                        if word[j] !=k:
                            nextword=word[:j]+k+word[j+1:]
                            if nextword in dic:
                                q.append((nextword,step+1))
                                dic.remove(nextword)
                                
                
                    
        return 0
                
                
            
             
        
start = "hit"
end = "cog"
dict = ["hot","dot","dog","lot","log"]   

start ="sand"     
end =       "acne" 

dict =["slit","bunk","wars","ping","viva","wynn","wows","irks","gang","pool","mock","fort","heel","send","ship","cols","alec","foal","nabs","gaze","giza","mays","dogs","karo","cums","jedi","webb","lend","mire","jose","catt","grow","toss","magi","leis","bead","kara","hoof","than","ires","baas","vein","kari","riga","oars","gags","thug","yawn","wive","view","germ","flab","july","tuck","rory","bean","feed","rhee","jeez","gobs","lath","desk","yoko","cute","zeus","thus","dims","link","dirt","mara","disc","limy","lewd","maud","duly","elsa","hart","rays","rues","camp","lack","okra","tome","math","plug","monk","orly","friz","hogs","yoda","poop","tick","plod","cloy","pees","imps","lead","pope","mall","frey","been","plea","poll","male","teak","soho","glob","bell","mary","hail","scan","yips","like","mull","kory","odor","byte","kaye","word","honk","asks","slid","hopi","toke","gore","flew","tins","mown","oise","hall","vega","sing","fool","boat","bobs","lain","soft","hard","rots","sees","apex","chan","told","woos","unit","scow","gilt","beef","jars","tyre","imus","neon","soap","dabs","rein","ovid","hose","husk","loll","asia","cope","tail","hazy","clad","lash","sags","moll","eddy","fuel","lift","flog","land","sigh","saks","sail","hook","visa","tier","maws","roeg","gila","eyes","noah","hypo","tore","eggs","rove","chap","room","wait","lurk","race","host","dada","lola","gabs","sobs","joel","keck","axed","mead","gust","laid","ends","oort","nose","peer","kept","abet","iran","mick","dead","hags","tens","gown","sick","odis","miro","bill","fawn","sumo","kilt","huge","ores","oran","flag","tost","seth","sift","poet","reds","pips","cape","togo","wale","limn","toll","ploy","inns","snag","hoes","jerk","flux","fido","zane","arab","gamy","raze","lank","hurt","rail","hind","hoot","dogy","away","pest","hoed","pose","lose","pole","alva","dino","kind","clan","dips","soup","veto","edna","damp","gush","amen","wits","pubs","fuzz","cash","pine","trod","gunk","nude","lost","rite","cory","walt","mica","cart","avow","wind","book","leon","life","bang","draw","leek","skis","dram","ripe","mine","urea","tiff","over","gale","weir","defy","norm","tull","whiz","gill","ward","crag","when","mill","firs","sans","flue","reid","ekes","jain","mutt","hems","laps","piss","pall","rowe","prey","cull","knew","size","wets","hurl","wont","suva","girt","prys","prow","warn","naps","gong","thru","livy","boar","sade","amok","vice","slat","emir","jade","karl","loyd","cerf","bess","loss","rums","lats","bode","subs","muss","maim","kits","thin","york","punt","gays","alpo","aids","drag","eras","mats","pyre","clot","step","oath","lout","wary","carp","hums","tang","pout","whip","fled","omar","such","kano","jake","stan","loop","fuss","mini","byrd","exit","fizz","lire","emil","prop","noes","awed","gift","soli","sale","gage","orin","slur","limp","saar","arks","mast","gnat","port","into","geed","pave","awls","cent","cunt","full","dint","hank","mate","coin","tars","scud","veer","coax","bops","uris","loom","shod","crib","lids","drys","fish","edit","dick","erna","else","hahs","alga","moho","wire","fora","tums","ruth","bets","duns","mold","mush","swop","ruby","bolt","nave","kite","ahem","brad","tern","nips","whew","bait","ooze","gino","yuck","drum","shoe","lobe","dusk","cult","paws","anew","dado","nook","half","lams","rich","cato","java","kemp","vain","fees","sham","auks","gish","fire","elam","salt","sour","loth","whit","yogi","shes","scam","yous","lucy","inez","geld","whig","thee","kelp","loaf","harm","tomb","ever","airs","page","laud","stun","paid","goop","cobs","judy","grab","doha","crew","item","fogs","tong","blip","vest","bran","wend","bawl","feel","jets","mixt","tell","dire","devi","milo","deng","yews","weak","mark","doug","fare","rigs","poke","hies","sian","suez","quip","kens","lass","zips","elva","brat","cosy","teri","hull","spun","russ","pupa","weed","pulp","main","grim","hone","cord","barf","olav","gaps","rote","wilt","lars","roll","balm","jana","give","eire","faun","suck","kegs","nita","weer","tush","spry","loge","nays","heir","dope","roar","peep","nags","ates","bane","seas","sign","fred","they","lien","kiev","fops","said","lawn","lind","miff","mass","trig","sins","furl","ruin","sent","cray","maya","clog","puns","silk","axis","grog","jots","dyer","mope","rand","vend","keen","chou","dose","rain","eats","sped","maui","evan","time","todd","skit","lief","sops","outs","moot","faze","biro","gook","fill","oval","skew","veil","born","slob","hyde","twin","eloy","beat","ergs","sure","kobe","eggo","hens","jive","flax","mons","dunk","yest","begs","dial","lodz","burp","pile","much","dock","rene","sago","racy","have","yalu","glow","move","peps","hods","kins","salk","hand","cons","dare","myra","sega","type","mari","pelt","hula","gulf","jugs","flay","fest","spat","toms","zeno","taps","deny","swag","afro","baud","jabs","smut","egos","lara","toes","song","fray","luis","brut","olen","mere","ruff","slum","glad","buds","silt","rued","gelt","hive","teem","ides","sink","ands","wisp","omen","lyre","yuks","curb","loam","darn","liar","pugs","pane","carl","sang","scar","zeds","claw","berg","hits","mile","lite","khan","erik","slug","loon","dena","ruse","talk","tusk","gaol","tads","beds","sock","howe","gave","snob","ahab","part","meir","jell","stir","tels","spit","hash","omit","jinx","lyra","puck","laue","beep","eros","owed","cede","brew","slue","mitt","jest","lynx","wads","gena","dank","volt","gray","pony","veld","bask","fens","argo","work","taxi","afar","boon","lube","pass","lazy","mist","blot","mach","poky","rams","sits","rend","dome","pray","duck","hers","lure","keep","gory","chat","runt","jams","lays","posy","bats","hoff","rock","keri","raul","yves","lama","ramp","vote","jody","pock","gist","sass","iago","coos","rank","lowe","vows","koch","taco","jinn","juno","rape","band","aces","goal","huck","lila","tuft","swan","blab","leda","gems","hide","tack","porn","scum","frat","plum","duds","shad","arms","pare","chin","gain","knee","foot","line","dove","vera","jays","fund","reno","skid","boys","corn","gwyn","sash","weld","ruiz","dior","jess","leaf","pars","cote","zing","scat","nice","dart","only","owls","hike","trey","whys","ding","klan","ross","barb","ants","lean","dopy","hock","tour","grip","aldo","whim","prom","rear","dins","duff","dell","loch","lava","sung","yank","thar","curl","venn","blow","pomp","heat","trap","dali","nets","seen","gash","twig","dads","emmy","rhea","navy","haws","mite","bows","alas","ives","play","soon","doll","chum","ajar","foam","call","puke","kris","wily","came","ales","reef","raid","diet","prod","prut","loot","soar","coed","celt","seam","dray","lump","jags","nods","sole","kink","peso","howl","cost","tsar","uric","sore","woes","sewn","sake","cask","caps","burl","tame","bulk","neva","from","meet","webs","spar","fuck","buoy","wept","west","dual","pica","sold","seed","gads","riff","neck","deed","rudy","drop","vale","flit","romp","peak","jape","jews","fain","dens","hugo","elba","mink","town","clam","feud","fern","dung","newt","mime","deem","inti","gigs","sosa","lope","lard","cara","smug","lego","flex","doth","paar","moon","wren","tale","kant","eels","muck","toga","zens","lops","duet","coil","gall","teal","glib","muir","ails","boer","them","rake","conn","neat","frog","trip","coma","must","mono","lira","craw","sled","wear","toby","reel","hips","nate","pump","mont","died","moss","lair","jibe","oils","pied","hobs","cads","haze","muse","cogs","figs","cues","roes","whet","boru","cozy","amos","tans","news","hake","cots","boas","tutu","wavy","pipe","typo","albs","boom","dyke","wail","woke","ware","rita","fail","slab","owes","jane","rack","hell","lags","mend","mask","hume","wane","acne","team","holy","runs","exes","dole","trim","zola","trek","puma","wacs","veep","yaps","sums","lush","tubs","most","witt","bong","rule","hear","awry","sots","nils","bash","gasp","inch","pens","fies","juts","pate","vine","zulu","this","bare","veal","josh","reek","ours","cowl","club","farm","teat","coat","dish","fore","weft","exam","vlad","floe","beak","lane","ella","warp","goth","ming","pits","rent","tito","wish","amps","says","hawk","ways","punk","nark","cagy","east","paul","bose","solo","teed","text","hews","snip","lips","emit","orgy","icon","tuna","soul","kurd","clod","calk","aunt","bake","copy","acid","duse","kiln","spec","fans","bani","irma","pads","batu","logo","pack","oder","atop","funk","gide","bede","bibs","taut","guns","dana","puff","lyme","flat","lake","june","sets","gull","hops","earn","clip","fell","kama","seal","diaz","cite","chew","cuba","bury","yard","bank","byes","apia","cree","nosh","judo","walk","tape","taro","boot","cods","lade","cong","deft","slim","jeri","rile","park","aeon","fact","slow","goff","cane","earp","tart","does","acts","hope","cant","buts","shin","dude","ergo","mode","gene","lept","chen","beta","eden","pang","saab","fang","whir","cove","perk","fads","rugs","herb","putt","nous","vane","corm","stay","bids","vela","roof","isms","sics","gone","swum","wiry","cram","rink","pert","heap","sikh","dais","cell","peel","nuke","buss","rasp","none","slut","bent","dams","serb","dork","bays","kale","cora","wake","welt","rind","trot","sloe","pity","rout","eves","fats","furs","pogo","beth","hued","edam","iamb","glee","lute","keel","airy","easy","tire","rube","bogy","sine","chop","rood","elbe","mike","garb","jill","gaul","chit","dons","bars","ride","beck","toad","make","head","suds","pike","snot","swat","peed","same","gaza","lent","gait","gael","elks","hang","nerf","rosy","shut","glop","pain","dion","deaf","hero","doer","wost","wage","wash","pats","narc","ions","dice","quay","vied","eons","case","pour","urns","reva","rags","aden","bone","rang","aura","iraq","toot","rome","hals","megs","pond","john","yeps","pawl","warm","bird","tint","jowl","gibe","come","hold","pail","wipe","bike","rips","eery","kent","hims","inks","fink","mott","ices","macy","serf","keys","tarp","cops","sods","feet","tear","benz","buys","colo","boil","sews","enos","watt","pull","brag","cork","save","mint","feat","jamb","rubs","roxy","toys","nosy","yowl","tamp","lobs","foul","doom","sown","pigs","hemp","fame","boor","cube","tops","loco","lads","eyre","alta","aged","flop","pram","lesa","sawn","plow","aral","load","lied","pled","boob","bert","rows","zits","rick","hint","dido","fist","marc","wuss","node","smog","nora","shim","glut","bale","perl","what","tort","meek","brie","bind","cake","psst","dour","jove","tree","chip","stud","thou","mobs","sows","opts","diva","perm","wise","cuds","sols","alan","mild","pure","gail","wins","offs","nile","yelp","minn","tors","tran","homy","sadr","erse","nero","scab","finn","mich","turd","then","poem","noun","oxus","brow","door","saws","eben","wart","wand","rosa","left","lina","cabs","rapt","olin","suet","kalb","mans","dawn","riel","temp","chug","peal","drew","null","hath","many","took","fond","gate","sate","leak","zany","vans","mart","hess","home","long","dirk","bile","lace","moog","axes","zone","fork","duct","rico","rife","deep","tiny","hugh","bilk","waft","swig","pans","with","kern","busy","film","lulu","king","lord","veda","tray","legs","soot","ells","wasp","hunt","earl","ouch","diem","yell","pegs","blvd","polk","soda","zorn","liza","slop","week","kill","rusk","eric","sump","haul","rims","crop","blob","face","bins","read","care","pele","ritz","beau","golf","drip","dike","stab","jibs","hove","junk","hoax","tats","fief","quad","peat","ream","hats","root","flak","grit","clap","pugh","bosh","lock","mute","crow","iced","lisa","bela","fems","oxes","vies","gybe","huff","bull","cuss","sunk","pups","fobs","turf","sect","atom","debt","sane","writ","anon","mayo","aria","seer","thor","brim","gawk","jack","jazz","menu","yolk","surf","libs","lets","bans","toil","open","aced","poor","mess","wham","fran","gina","dote","love","mood","pale","reps","ines","shot","alar","twit","site","dill","yoga","sear","vamp","abel","lieu","cuff","orbs","rose","tank","gape","guam","adar","vole","your","dean","dear","hebe","crab","hump","mole","vase","rode","dash","sera","balk","lela","inca","gaea","bush","loud","pies","aide","blew","mien","side","kerr","ring","tess","prep","rant","lugs","hobo","joke","odds","yule","aida","true","pone","lode","nona","weep","coda","elmo","skim","wink","bras","pier","bung","pets","tabs","ryan","jock","body","sofa","joey","zion","mace","kick","vile","leno","bali","fart","that","redo","ills","jogs","pent","drub","slaw","tide","lena","seep","gyps","wave","amid","fear","ties","flan","wimp","kali","shun","crap","sage","rune","logs","cain","digs","abut","obit","paps","rids","fair","hack","huns","road","caws","curt","jute","fisk","fowl","duty","holt","miss","rude","vito","baal","ural","mann","mind","belt","clem","last","musk","roam","abed","days","bore","fuze","fall","pict","dump","dies","fiat","vent","pork","eyed","docs","rive","spas","rope","ariz","tout","game","jump","blur","anti","lisp","turn","sand","food","moos","hoop","saul","arch","fury","rise","diss","hubs","burs","grid","ilks","suns","flea","soil","lung","want","nola","fins","thud","kidd","juan","heps","nape","rash","burt","bump","tots","brit","mums","bole","shah","tees","skip","limb","umps","ache","arcs","raft","halo","luce","bahs","leta","conk","duos","siva","went","peek","sulk","reap","free","dubs","lang","toto","hasp","ball","rats","nair","myst","wang","snug","nash","laos","ante","opal","tina","pore","bite","haas","myth","yugo","foci","dent","bade","pear","mods","auto","shop","etch","lyly","curs","aron","slew","tyro","sack","wade","clio","gyro","butt","icky","char","itch","halt","gals","yang","tend","pact","bees","suit","puny","hows","nina","brno","oops","lick","sons","kilo","bust","nome","mona","dull","join","hour","papa","stag","bern","wove","lull","slip","laze","roil","alto","bath","buck","alma","anus","evil","dumb","oreo","rare","near","cure","isis","hill","kyle","pace","comb","nits","flip","clop","mort","thea","wall","kiel","judd","coop","dave","very","amie","blah","flub","talc","bold","fogy","idea","prof","horn","shoo","aped","pins","helm","wees","beer","womb","clue","alba","aloe","fine","bard","limo","shaw","pint","swim","dust","indy","hale","cats","troy","wens","luke","vern","deli","both","brig","daub","sara","sued","bier","noel","olga","dupe","look","pisa","knox","murk","dame","matt","gold","jame","toge","luck","peck","tass","calf","pill","wore","wadi","thur","parr","maul","tzar","ones","lees","dark","fake","bast","zoom","here","moro","wine","bums","cows","jean","palm","fume","plop","help","tuba","leap","cans","back","avid","lice","lust","polo","dory","stew","kate","rama","coke","bled","mugs","ajax","arts","drug","pena","cody","hole","sean","deck","guts","kong","bate","pitt","como","lyle","siam","rook","baby","jigs","bret","bark","lori","reba","sups","made","buzz","gnaw","alps","clay","post","viol","dina","card","lana","doff","yups","tons","live","kids","pair","yawl","name","oven","sirs","gyms","prig","down","leos","noon","nibs","cook","safe","cobb","raja","awes","sari","nerd","fold","lots","pete","deal","bias","zeal","girl","rage","cool","gout","whey","soak","thaw","bear","wing","nagy","well","oink","sven","kurt","etna","held","wood","high","feta","twee","ford","cave","knot","tory","ibis","yaks","vets","foxy","sank","cone","pius","tall","seem","wool","flap","gird","lore","coot","mewl","sere","real","puts","sell","nuts","foil","lilt","saga","heft","dyed","goat","spew","daze","frye","adds","glen","tojo","pixy","gobi","stop","tile","hiss","shed","hahn","baku","ahas","sill","swap","also","carr","manx","lime","debs","moat","eked","bola","pods","coon","lacy","tube","minx","buff","pres","clew","gaff","flee","burn","whom","cola","fret","purl","wick","wigs","donn","guys","toni","oxen","wite","vial","spam","huts","vats","lima","core","eula","thad","peon","erie","oats","boyd","cued","olaf","tams","secs","urey","wile","penn","bred","rill","vary","sues","mail","feds","aves","code","beam","reed","neil","hark","pols","gris","gods","mesa","test","coup","heed","dora","hied","tune","doze","pews","oaks","bloc","tips","maid","goof","four","woof","silo","bray","zest","kiss","yong","file","hilt","iris","tuns","lily","ears","pant","jury","taft","data","gild","pick","kook","colt","bohr","anal","asps","babe","bach","mash","biko","bowl","huey","jilt","goes","guff","bend","nike","tami","gosh","tike","gees","urge","path","bony","jude","lynn","lois","teas","dunn","elul","bonn","moms","bugs","slay","yeah","loan","hulk","lows","damn","nell","jung","avis","mane","waco","loin","knob","tyke","anna","hire","luau","tidy","nuns","pots","quid","exec","hans","hera","hush","shag","scot","moan","wald","ursa","lorn","hunk","loft","yore","alum","mows","slog","emma","spud","rice","worn","erma","need","bags","lark","kirk","pooh","dyes","area","dime","luvs","foch","refs","cast","alit","tugs","even","role","toed","caph","nigh","sony","bide","robs","folk","daft","past","blue","flaw","sana","fits","barr","riot","dots","lamp","cock","fibs","harp","tent","hate","mali","togs","gear","tues","bass","pros","numb","emus","hare","fate","wife","mean","pink","dune","ares","dine","oily","tony","czar","spay","push","glum","till","moth","glue","dive","scad","pops","woks","andy","leah","cusp","hair","alex","vibe","bulb","boll","firm","joys","tara","cole","levy","owen","chow","rump","jail","lapp","beet","slap","kith","more","maps","bond","hick","opus","rust","wist","shat","phil","snow","lott","lora","cary","mote","rift","oust","klee","goad","pith","heep","lupe","ivan","mimi","bald","fuse","cuts","lens","leer","eyry","know","razz","tare","pals","geek","greg","teen","clef","wags","weal","each","haft","nova","waif","rate","katy","yale","dale","leas","axum","quiz","pawn","fend","capt","laws","city","chad","coal","nail","zaps","sort","loci","less","spur","note","foes","fags","gulp","snap","bogs","wrap","dane","melt","ease","felt","shea","calm","star","swam","aery","year","plan","odin","curd","mira","mops","shit","davy","apes","inky","hues","lome","bits","vila","show","best","mice","gins","next","roan","ymir","mars","oman","wild","heal","plus","erin","rave","robe","fast","hutu","aver","jodi","alms","yams","zero","revs","wean","chic","self","jeep","jobs","waxy","duel","seek","spot","raps","pimp","adan","slam","tool","morn","futz","ewes","errs","knit","rung","kans","muff","huhs","tows","lest","meal","azov","gnus","agar","sips","sway","otis","tone","tate","epic","trio","tics","fade","lear","owns","robt","weds","five","lyon","terr","arno","mama","grey","disk","sept","sire","bart","saps","whoa","turk","stow","pyle","joni","zinc","negs","task","leif","ribs","malt","nine","bunt","grin","dona","nope","hams","some","molt","smit","sacs","joan","slav","lady","base","heck","list","take","herd","will","nubs","burg","hugs","peru","coif","zoos","nick","idol","levi","grub","roth","adam","elma","tags","tote","yaws","cali","mete","lula","cubs","prim","luna","jolt","span","pita","dodo","puss","deer","term","dolt","goon","gary","yarn","aims","just","rena","tine","cyst","meld","loki","wong","were","hung","maze","arid","cars","wolf","marx","faye","eave","raga","flow","neal","lone","anne","cage","tied","tilt","soto","opel","date","buns","dorm","kane","akin","ewer","drab","thai","jeer","grad","berm","rods","saki","grus","vast","late","lint","mule","risk","labs","snit","gala","find","spin","ired","slot","oafs","lies","mews","wino","milk","bout","onus","tram","jaws","peas","cleo","seat","gums","cold","vang","dewy","hood","rush","mack","yuan","odes","boos","jami","mare","plot","swab","borg","hays","form","mesh","mani","fife","good","gram","lion","myna","moor","skin","posh","burr","rime","done","ruts","pays","stem","ting","arty","slag","iron","ayes","stub","oral","gets","chid","yens","snub","ages","wide","bail","verb","lamb","bomb","army","yoke","gels","tits","bork","mils","nary","barn","hype","odom","avon","hewn","rios","cams","tact","boss","oleo","duke","eris","gwen","elms","deon","sims","quit","nest","font","dues","yeas","zeta","bevy","gent","torn","cups","worm","baum","axon","purr","vise","grew","govs","meat","chef","rest","lame"]  
if __name__ == "__main__":
    print(Solution().ladderLength(start, end, dict))                
      
#121. Word Ladder II 
from  collections import defaultdict, deque
import string 
 
class Solution:
    """
    @param: start: a string
    @param: end: a string
    @param: dict: a set of string
    @return: a list of lists of string
    """
    def findLadders(self, start, end, dict):
        # write your code here
#        使用BFS去計算每個字變成終點的字的距離(step of transformation)，並且用Hash Map去記錄著
#同時也用Hash Map記錄著每個字跟他的鄰居們(next word)
#
#在用DFS去遍歷，從起點到終點，同時只選擇那些下一個字的距離是目前字的距離的少1
        
        dict.add(start)
        dict.add(end)
        distance=  defaultdict(int) 
        graph=defaultdict(list) 
        
        for word in dict:
            for j in range(len(word)):
                for k in string.ascii_lowercase:
                    nextword=word[:j]+k+word[j+1:]
                    if word[j]!=k  and nextword in dict:
                        graph[word]+=[nextword]
        
                        
        
        
        
        
        def bfs(start, end, dict, distance,graph):
            q=deque([end])
            visited=set([end])
            step=0
            
            while q:
               n=len(q)
                
               for _ in range(n):
                    word=q.popleft()
                    distance[word]=step
                    for nextword in  graph[word]:
                        if nextword not in visited:
                            visited.add(nextword)
                            q.append(nextword)
                            
               step+=1
            
                
                    
        def dfs(cur_word, end, path, distance,graph,res):
            if cur_word==end:
                path.append(cur_word)
                res.append(path+[])
                path.pop()
                return 
                
                 
            
            for next_word in graph[cur_word]:
                    if distance[next_word] +1==distance[cur_word]:
                        path+=[cur_word]
                        dfs(next_word, end, path, distance,graph,res)
                        path.pop()
        
        res=[] 
        bfs(start, end, dict, distance,graph)
        #print(distance)
        #print(graph)              
        dfs(start, end, [], distance,graph,res)  
        return res
 

start = "hit"
end = "cog"
dict = set(["hot","dot","dog","lot","log"])
Return
  [
    ["hit","hot","dot","dog","cog"],
    ["hit","hot","lot","log","cog"]
  ]
start = "qa"
end = "sq"
dict =["si","go","se","cm","so","ph","mt","db","mb","sb","kr","ln","tm","le","av","sm","ar","ci","ca","br","ti","ba","to","ra","fa","yo","ow","sn","ya","cr","po","fe","ho","ma","re","or","rn","au","ur","rh","sr","tc","lt","lo","as","fr","nb","yb","if","pb","ge","th","pm","rb","sh","co","ga","li","ha","hz","no","bi","di","hi","qa","pi","os","uh","wm","an","me","mo","na","la","st","er","sc","ne","mn","mi","am","ex","pt","io","be","fm","ta","tb","ni","mr","pa","he","lr","sq","ye"]  
  
if __name__ == "__main__":
    print(Solution().findLadders( start, end, dict))


#122. Largest Rectangle in Histogram
class Solution:
    """
    @param height: A list of integer
    @return: The area of largest rectangle in the histogram
    """
    def largestRectangleArea(self, height):
        # write your code here
        stack=[-1]
        height.append(0)
        
        ans=0
        for i in range(len(height)):
            while height[i]<height[stack[-1]]:
                h=height[stack.pop()]
                w=i-stack[-1]-1
                ans=max(ans,h*w)
            stack.append(i)
        return ans
height = [2,1,5,6,2,3]
if __name__ == "__main__":
    print(Solution().largestRectangleArea( height))


#123. Word Search
class Solution:
    """
    @param board: A list of lists of character
    @param word: A string
    @return: A boolean
    """
    def exist(self, board, word):
        # write your code here
        
        def find(board,i,x,y):
            if i==len(word):
                return True
            if x<0 or y < 0 or x>=len(board) or y>=len(board[0]):
                return False
            if board[x][y]!=word[i]:
                return False
            else:
                temp=board[x][y]
                board[x][y]='#'
                if find(board,i+1,x-1,y) or find(board,i+1,x+1,y) or find(board,i+1,x,y-1) or find(board,i+1,x,y+1):
                    return True
                else:
                    board[x][y]=temp 
                    return False
               
                
                
        
        for x in range(len(board)):
                for y in range(len(board[0])):
                  if find(board,0,x,y):
                      return True
        return False
                      
   
                            

board =[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

word = "ABCCED", return true.
word = "SEE", return true.
word = "ABCB", return false.

board2=[]
for row in board:
    board2.append(list(row))
    
    
board =["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaab"]
word ="baaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
board2=[]
for row in board:
    board2.append(list(row))

board=board2

if __name__ == "__main__":
    print(Solution().exist( board, word))

#124. Longest Consecutive Sequence
class Solution:
    """
    @param num: A list of integers
    @return: An integer
    """
    def longestConsecutive(self, num):
        # write your code here
        dic={}
        
        for x in num:
            dic[x]=1
        ans=0
        for x in num:
            if x in dic:
                length=1
                left=x-1
                right=x+1
                del dic[x]
                while left in dic:
                    del dic[left]
                    #print(left)
                    left-=1
                    length+=1
                    #print(left,length)
                while right in dic:
                    del dic[right]
                    right+=1
                    length+=1
            if ans < length:
                    ans=length
        return ans
        
num=    [100, 4, 200, 1, 3, 2]
if __name__ == "__main__":
    print(Solution().longestConsecutive( num))
    
    
    
    
#125. Backpack II
class Solution:
    """
    @param m: An integer m denotes the size of a backpack
    @param A: Given n items with size A[i]
    @param V: Given n items with value V[i]
    @return: The maximum value
    """
    def backPackII(self, m, A, V):
        # write your code here
        #用子问题定义状态：即f[i][v]表示前 i 件物品恰放入一个容量为 j 的背包可以获得的最大价值。
#        item=[]
#        for size,value in zip(A,V):
#            item.append((size,value))
#        item.sort(key= lambda x: (-x[1],x[0])  )
#        sofar=0
#        self.res=0
#        def search(m,item,sofar):
#            if self.res<sofar:
#                self.res=sofar
#            if not item:
#                return 
#            for   i,itm in enumerate(item):
#                  s,v=itm
#                  if m-s>=0:
#                      search(m-s,item[:i]+item[i+1:],sofar+v)
#        search(m,item,sofar)
#        return  self.res    
        n=len(A)
        dp=[[0 for _ in range(m+1)] for _ in range(n+1)]
        
        for i in range(1,n+1):
            for j in range(1,m+1):
                if j<A[i-1]:
                    dp[i][j]=dp[i-1][j]
                else:
                    dp[i][j]=max(dp[i-1][j],dp[i-1][j-A[i-1]]+V[i-1])
        return dp[n][m]
    
A=[2, 3, 5, 7]
V= [1, 5, 2, 4]    
m=10 
m=1000
A=[71,34,82,23,1,88,12,57,10,68,5,33,37,69,98,24,26,83,16,26,18,43,52,71,22,65,68,8,40,40,24,72,16,34,10,19,28,13,34,98,29,31,79,33,60,74,44,56,54,17,63,83,100,54,10,5,79,42,65,93,52,64,85,68,54,62,29,40,35,90,47,77,87,75,39,18,38,25,61,13,36,53,46,28,44,34,39,69,42,97,34,83,8,74,38,74,22,40,7,94]
V=[26,59,30,19,66,85,94,8,3,44,5,1,41,82,76,1,12,81,73,32,74,54,62,41,19,10,65,53,56,53,70,66,58,22,72,33,96,88,68,45,44,61,78,78,6,66,11,59,83,48,52,7,51,37,89,72,23,52,55,44,57,45,11,90,31,38,48,75,56,64,73,66,35,50,16,51,33,58,85,77,71,87,69,52,10,13,39,75,38,13,90,35,83,93,61,62,95,73,26,85]                
if __name__ == "__main__":
    print(Solution().backPackII( m, A, V))            
    
    
#127. Topological Sorting 
"""
Definition for a Directed graph node
class DirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
"""


class Solution:
    """
    @param: graph: A list of Directed graph node
    @return: Any topological order for the given graph.
    """
    def topSort(self, graph):
        # write your code here
        def topSort(self, graph):
        # write your code here
        def dfs(i,counter,ans):
            ans.append(i)
            counter[i]-=1
            for j in i.neighbors:
                counter[j]-=1
                if counter[j]==0 :
                   dfs(j,counter,ans)
        counter={}
        ans=[]
        for i in graph:
            counter[i]=0
        for i in graph:
            for j in i.neighbors:
                counter[j]=counter[j]+1
        
        
        for i in graph:
            if counter[i]==0 :
                dfs(i,counter,ans)
        return ans
                
if __name__ == "__main__":
    print(Solution().topSort(graph))            
                    
#128. Hash Function    
class Solution:
    """
    @param key: A string you should hash
    @param HASH_SIZE: An integer
    @return: An integer
    """
    def hashCode(self, key, HASH_SIZE):
        # write your code here
        ans=0
        for x in key:
            ans=(ans*33+ ord(x)) % HASH_SIZE
        return ans
    
#129. Rehashing    
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""
class Solution:
    """
    @param hashTable: A list of The first node of linked list
    @return: A list of The first node of linked list which have twice size
    """
    def rehashing(self, hashTable):
        # write your code here
       n=len(hashTable)
        newhash=[None for _ in range(2*n)]
        for i,x in enumerate(hashTable):
            while x:
                pos=x.val%(2*n)
                if not newhash[pos]:
                       newhash[pos]=ListNode(x.val)
                else:
                       cur= newhash[pos]
                       while  cur.next:
                              cur=cur.next
                       cur.next=ListNode(x.val)
                x=x.next
        return newhash
Given [null, 21->9->null, 14->null, null],

return [null, 9->null, null, null, null, 21->null, 14->null, null]    
                                 
#130. Heapify                    
class Solution:
    """
    @param: A: Given an integer array
    @return: nothing
    """
    def heapify(self, A):
        # write your code here
        #從尾到頭掃一遍，如果遇到本身的值比parent element的值小的
#就一直跟parent element的值交換，直到交換到最頂層或是本身的值大於parent element的值
        def  move_up(i,A):
            while (i-1)//2 >=0 and A[(i-1)//2]>A[i]:
                  A[(i-1)//2],A[i]=A[i],A[(i-1)//2]
                  i=(i-1)//2
        for i in range(1,len(A)):
            move_up(i,A)
            print(A)
A=[3,2,1,4,5]            
if __name__ == "__main__":
    print(Solution().heapify(A))             
                
#131. The Skyline Problem                
class Solution:
    """
    @param buildings: A list of lists of integers
    @return: Find the outline of those buildings
    """
    def buildingOutline(self, buildings):
        # write your code here   
#        import heapq
#        events=sorted([ ( L,-H,R) for L,R ,H in buildings ]  + list({(R,0,None) for _ ,R,_ in buildings}))
#        res=[[0,0]]
#        hq=[(0,float('inf'))]
#        for x,negH,R in events:
#            while x >= hq[0][1]:
#                heapq.heappop(hq)
#            if negH:
#                heapq.heappush(hq,(negH,R))
#            if res[-1][1]+hq[0][0]:
#                res+=[x,-hq[0][0]],
#        return res[1:]
    
    
        import heapq
        events=sorted([ ( L,-H,R) for L,R ,H in buildings ]  + list({(R,0,None) for _ ,R,_ in buildings}))
        res=[[0,0]]
        hq=[(0,float('inf'))]
        for x,negH,R in events:
            while x >= hq[0][1]:
                heapq.heappop(hq)
            if negH:
                heapq.heappush(hq,(negH,R))
            if res[-1][1]+hq[0][0]:
                
                res+=[x,-hq[0][0]],
        res2=[]
        for i in range(1,len(res)-1):
            if res[i][1]==0:
                continue
            else:
                res2.append(list((res[i][0],res[i+1][0],res[i][1]))) 
        return res2
    
    
    
buildings=[[1,3,3],[2,4,4],[5,6,1]]    
[[1,3],[2,4],[4,0],[5,1],[6,0]]   
[[1,2,3],[2,4,4],[5,6,1]]    
buildings=[ [2 ,9, 10], [3, 7, 15], [5 ,12, 12], [15, 20, 10], [19, 24 ,8] ]
if __name__ == "__main__":
    print(Solution().buildingOutline( buildings))    
    
#132. Word Search II    
class Solution:
    """
    @param board: A list of lists of character
    @param words: A list of string
    @return: A list of string
    """
    def wordSearchII(self, board, words):
        # write your code here
        tries={}
        m=len(board)
        n=len(board[0])
        for word in words:
            curDict=tries
            for c in word:
                curDict=curDict.setdefault(c,{})
            curDict['#']='#'
                    
        def find(board,i,j,path,res,tries):
            if '#' in tries:
               res.add(path)
               
               
            if  not (0<=i<m  and  0<=j<n)  or board[i][j] not in tries:
                return 
            
            temp=board[i][j]
            board[i][j]='@'
            find(board,i+1,j,path+temp,res,tries[temp])
            find(board,i-1,j,path+temp,res,tries[temp])
            find(board,i,j+1,path+temp,res,tries[temp])
            find(board,i,j-1,path+temp,res,tries[temp])
            board[i][j]=temp
        res=set()
        
        
        for i in range(m):
            for j in range(n):
                find(board,i,j,'',res,tries)
        return list(res)
                 
            
words = ["oath","pea","eat","rain"]            
board =[
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]
board =[['d','o','a','f'],
['a','g','a','i'],
['d','c','a','n' ]]  
words =["dog", "dad", "dgdg", "can", "again"]      
if __name__ == "__main__":
    print(Solution().wordSearchII( board, words))        
        
#133. Longest Word
class Solution:
    """
    @param: dictionary: an array of strings
    @return: an arraylist of strings
    """
    def longestWords(self, dictionary):
        # write your code here
        curlen=0
        res=[]
        for word in dictionary:
            if len(word)>curlen:
                res=[word ]
                curlen=len(word)
            elif len(word)==curlen:
                res.append(word)
        return res
dictionary=[
  "dog",
  "google",
  "facebook",
  "internationalization",
  "blabla"
]
dictionary=[
  "like",
  "love",
  "hate",
  "yes"
]
if __name__ == "__main__":
    print(Solution().longestWords(dictionary))                 
                
#134. LRU Cache        
class LRUCache:
    """
    @param: capacity: An integer
    """
    def __init__(self, capacity):
        # do intialization if necessary
        from collections import  OrderedDict
        self.cap=capacity
        self.array=OrderedDict()

    """
    @param: key: An integer
    @return: An integer
    """
    def get(self, key):
        # write your code here
        if key in self.array:
            temp=self.array[key]
            del self.array[key]
            self.array[key]=temp
            return temp
        else:
            return -1
        
        

    """
    @param: key: An integer
    @param: value: An integer
    @return: nothing
    """
    def set(self, key, value):
        # write your code here 
        if key in self.array:
            del self.array[key]
        elif self.cap==len(self.array):
             self.array.popitem(last=False)
        self.array[key]=value
        
#135. Combination Sum
class Solution:
    """
    @param candidates: A list of integers
    @param target: An integer
    @return: A list of lists of integers
    """
    def combinationSum(self, candidates, target):
        # write your code here\
        
        def search(target,candidates,path,res,start):
            if target==0:
                
                   res.append(path[:])
                   return 
            else:
                for i in range(start,len(candidates)):
                    if (i==0 or candidates[i]!=candidates[i-1]) and target-candidates[i]>=0:
                        search(target-candidates[i],candidates,path+[candidates[i]],res,i)
                
            
        
        candidates.sort()
        res=[]
        search(target,candidates,[],res,0)
        return res
candidates=    [2,3,6,7]    
target =7        
[7]
[2, 2, 3]        
if __name__ == "__main__":
    print(Solution().combinationSum( candidates, target))               
        
#136. Palindrome Partitioning        
class Solution:
    """
    @param: s: A string
    @return: A list of lists of string
    """
    def partition(self, s):
        # write your code here
        def isPalindrome(string):
            n=len(string)
            if len(string)==1:
                return True
            for i in range(n//2):
                if string[i]!=string[n-1-i]:
                    return False
            return True
            
            
        def cut(s,res,path):
            if not s:
                res.append(path[:])
            else:
                for i in range(1,len(s)+1):
                    if isPalindrome(s[:i]):
                        cut(s[i:],res,path+[s[:i]])
        res=[]
        cut(s,res,[])
        return res
s='aab'
if __name__ == "__main__":
    print(Solution().partition( s))     
                
                        
#137. Clone Graph                
"""
Definition for a undirected graph node
class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
"""


class Solution:
    """
    @param: node: A undirected graph node
    @return: A undirected graph node
    """
    def cloneGraph(self, node):
        # write your code here
        if not node:
            return node
        
        #get all nodes
        from collections import deque
        mapping={}
        
        q=deque([node])
        
        while q:
            onenode=q.popleft()
            mapping[onenode]=UndirectedGraphNode(onenode.label)
            for nei in onenode.neighbors:
                if nei not in mapping:
                    q.append(nei)
        
        for anode in mapping:
            for nei in anode.neighbors:
                mapping[anode].neighbors.append(mapping[nei])
        return mapping[node]
        
            
if __name__ == "__main__":
    print(Solution().partition( s))                 
#138. Subarray Sum    
class Solution:
    """
    @param nums: A list of integers
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def subarraySum(self, nums):
        # write your code here
      
#        def search(nums,path,res,target,start):
#            if target==0 and path:
#                res.append(path[:])
#                return 
#            for i in range(start,len(nums)):
#                search(nums,path+[i],res,target+nums[i],i+1)
#        res=[]
#        search(nums,[],res,0,0)
#        print(res)
#        summ=0
#        for i in res[0]:
#            summ+=nums[i]
#        print(summ)
#        return [res[0][0],res[0][-1]]
        sumdict={0:-1}
        if nums[0]==0:
            return [0,0]
       
        summ=0
        for i in range(len(nums)):
            summ+=nums[i]
            if summ in sumdict:
                return [sumdict[summ]+1,i]
            sumdict[summ]=i
            
      
nums=[-3, 1, 2, -3, 4]
nums=[-5,10,5,-3,1,1,1,-2,3,-4]
#return [0, 2] or [1, 3]
if __name__ == "__main__":
    print(Solution().subarraySum(nums))       
    
#139. Subarray Sum Closest
class Node:
    def __init__(self,_value,_pos):
        self.value=_value
        self.pos=_pos
#    def __cmp__(self,other):
#        if self.value==other.value:
#            return self.pos-other.pos
#        else:
#            return self.value-other.value
    def __lt__(self, other):
        if self.value == other.value:
            return self.pos < other.pos
        else:
            return self.value < other.value
        
        
class Solution:
    """
    @param: nums: A list of integers
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def subarraySumClosest(self, nums):
        # write your code here   
        summ=0
        s=[]
        s.append(Node(0,-1))
        
        
        for i in range(len(nums)):
            summ+=nums[i]
            s.append(Node(summ,i))
        
        s.sort()
        
        res=[0,0]
        ans=float('inf')
        for i in range(len(nums)-1):
            if s[i+1].value-s[i].value<ans or s[i+1].value-s[i].value==ans and min(s[i+1].pos,s[i].pos)+1<res[0]:
                ans=s[i+1].value-s[i].value
                res[0]=min(s[i+1].pos,s[i].pos)+1
                res[1]=max(s[i+1].pos,s[i].pos)
        return res        
nums=[6,-4,-8,3,1,7]    
if __name__ == "__main__":
    print(Solution().subarraySumClosest( nums))                     
        
#140. Fast Power        
class Solution:
    """
    @param a: A 32bit integer
    @param b: A 32bit integer
    @param n: A 32bit integer
    @return: An integer
    """
    def fastPower(self, a, b, n):
        # write your code here
        
        if n==0:
            return 1%b
        if n==1:
            return a%b
        if n%2==0:
            return (self.fastPower(a,b,n//2) **2)%b
        else:
            return (self.fastPower(a,b,n//2) **2*a)%b

a=2
b=3
n=31

if __name__ == "__main__":
    print(Solution().fastPower(a, b, n)) 


#141. Sqrt(x)     
class Solution:
    """
    @param x: An integer
    @return: The sqrt of x
    """
    def sqrt(self, x):
        # write your code here
        
        if x==1 or x==0:
            return x

        for i in range(x):
           if i*i>x:
               return i-1
x=10
if __name__ == "__main__":
    print(Solution().sqrt( x)) 
    
#142. O(1) Check Power of 2
class Solution:
    """
    @param n: An integer
    @return: True or false
    """
    def checkPowerOf2(self, n):
        # write your code here
        if n<=0:
            return False
        return n&(n-1)==0
if __name__ == "__main__":
    print(Solution(). checkPowerOf2( n))


#143. Sort Colors II
class Solution:
    """
    @param colors: A list of integer
    @param k: An integer
    @return: nothing
    """
    def sortColors2(self, colors, k):
        # write your code here
        n=len(colors)
        def rainbowsort(colors,left,right,lowNum,highNum):
            if lowNum==highNum:
                return 
            if left>=right:
                return 
            l=left
            r=right
            midNum=(lowNum+highNum)/2
            while l <=r:
                while l<=r and colors[l]<= midNum:
                    l+=1
                while l<=r and colors[r]> midNum:
                    r-=1
                if l<=r:
                    colors[l],colors[r]=colors[r],colors[l]
                
            rainbowsort(colors,left,r,lowNum,midNum)
            rainbowsort(colors,l,right,midNum+1,highNum)
        rainbowsort(colors,0,n-1,1,k)
        print(colors)
colors=[3, 2, 2, 1, 4]
k=4
if __name__ == "__main__":
    print(Solution().sortColors2(colors, k))

#144. Interleaving Positive and Negative Numbers
class Solution:
    """
    @param: A: An integer array.
    @return: nothing
    """
    def rerange(self, A):
        # write your code here
        
        n=len(A)
        
        hi=n-1
        lo=0
        
        while lo<=hi:
            while lo<=hi and A[lo]<0:
                lo+=1
            while lo<=hi and A[hi]>0:
                hi-=1
            if lo<=hi:
                A[lo],A[hi] = A[hi] , A[lo]
                lo+=1
                hi-=1
        negCount=lo
        posCount=n-hi-1
        print(A)
        
        lo=1        if negCount>=posCount  else 0
        
        hi=n-2        if   posCount >=negCount else n-1
        
        
        while lo<hi:
            A[lo],A[hi] = A[hi] , A[lo]
            lo+=2
            hi-=2
        print(A)
A=[-1, -2, -3, 4, 5, 6,-9]             
if __name__ == "__main__":
    print(Solution().rerange( A))        
        
#145. Lowercase to Uppercase    
class Solution:
    """
    @param character: a character
    @return: a character
    """
    def lowercaseToUppercase(self, character):
        # write your code here
        x=ord('A')-ord('a')
        return chr(x+ord(character))

#148. Sort Colors
class Solution:
    """
    @param nums: A list of integer which is 0, 1 or 2 
    @return: nothing
    """
    def sortColors(self, nums):
        # write your code here
        n=len(nums)
        i=0
        left=0
        right=n-1
        print(nums) 
        while i<=right:
            while i <=right and nums[i]==0:
                nums[left],nums[i]=nums[i],nums[left]
                left+=1
                i+=1
            print(nums,i,left,right)    
            while i <=right and nums[i]==1:
                i+=1
            print(nums,i,left,right)
            
            while i <=right and nums[i]==2:
                nums[right],nums[i]=nums[i],nums[right]
                right-=1
            print(nums,i,left,right) 
            
                
        
nums=[2,1,0,1,2,0,2]
if __name__ == "__main__":
    print(Solution().sortColors( nums))        
            
#149. Best Time to Buy and Sell Stock                
class Solution:
    """
    @param prices: Given an integer array
    @return: Maximum profit
    """
    def maxProfit(self, prices):
        # write your code here
        low=float('inf')
        
        total=0
        
        for x in prices:
            if x-low>total:
                total=x-low
            if x<low:
                low=x
        return total
prices=[3,2,3,1,2]
if __name__ == "__main__":
    print(Solution().maxProfit( prices))    
            
#150. Best Time to Buy and Sell Stock II
class Solution:
    """
    @param prices: Given an integer array
    @return: Maximum profit
    """
    def maxProfit(self, prices):
        # write your code here
        
        low=float('inf')
        profit=0
        for x in prices:
            if x>low:
                
               profit+=x-low
               low=x
            else:
                 
               low=x
        return profit
            
prices= [2,1,2,0,1]
prices= [1,2,4]
if __name__ == "__main__":
    print(Solution().maxProfit( prices)) 
          
[2,1,2,0,1], return 2        
        
#151. Best Time to Buy and Sell Stock III        
class Solution:
    """
    @param prices: Given an integer array
    @return: Maximum profit
    """
    def maxProfit(self, prices):
        # write your code here
        
#https://blog.csdn.net/fightforyourdream/article/details/14503469 
#这里我们先解释最多可以进行k次交易的算法，然后最多进行两次我们只需要把k取成2即可。我们还是使用“局部最优和全局最优解法”。我们维护两种量，一个是当前到达第i天可以最多进行j次交易，最好的利润是多少（global[i][j]），另一个是当前到达第i天，最多可进行j次交易，并且最后一次交易在当天卖出的最好的利润是多少（local[i][j]）。下面我们来看递推式，全局的比较简单，

#global[i][j]=max(local[i][j],global[i-1][j])，
#也就是去当前局部最好的，和过往全局最好的中大的那个（因为最后一次交易如果包含当前天一定在局部最好的里面，否则一定在过往全局最优的里面）。

#全局（到达第i天进行j次交易的最大收益） = max{局部（在第i天交易后，恰好满足j次交易），全局（到达第i-1天时已经满足j次交易）}
        n=len(prices)
        if n==0:
            return 0
        
        def profit(prices,k):
            
            globalmax=[[0 for _ in range(k+1)]for _ in range(n)]
            
            localmax=[[0 for _ in range(k+1)]for _ in range(n)]
            
            for i in range(1,n):
                for j in range(1,k+1):
                    dif=prices[i]-prices[i-1]
                    localmax[i][j]=max(globalmax[i-1][j-1]+max(0,dif),localmax[i-1][j]+dif)
                    globalmax[i][j]=max(localmax[i][j], globalmax[i-1][j])
            return globalmax[n-1][k]
        return profit(prices,2)
prices=[4,4,6,1,1,4,2,5]# return 6.        
    
if __name__ == "__main__":
    print(Solution().maxProfit( prices)) 
                    
#152. Combinations
class Solution:
    """
    @param n: Given the range of numbers
    @param k: Given the numbers of combinations
    @return: All the combinations of k numbers out of 1..n
    """
    def combine(self, n, k):
        # write your code here
        def add(k,path,res,index):
            if len(path)==k:
                res.append(path[:])
                return 
            for i in range(index,n+1):
                path.append(i)
                add(k,path,res,i+1)
                path.pop()
        res=[]
        add(k,[],res,1)
        return res
n = 4
k=2    
if __name__ == "__main__":
    print(Solution(). combine( n, k))  
        
Given n = 4 and k = 2, a solution is:

[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4]
]        
        
        
#153. Combination Sum II
class Solution:
    """
    @param num: Given the candidate numbers
    @param target: Given the target number
    @return: All the combinations that sum to target
    """
    def combinationSum2(self, num, target):
        # write your code here
        def search(num,target,path,res,index):
            if target==0:
                res.append(path[:])
            for i in range(index,len(num)):
                if (i==index or num[i]!=num[i-1] )  and target-num[i]>=0:
                     
                
                    search(num,target-num[i],path+[num[i]],res,i+1)
        res=[]
        num.sort()
        search(num,target,[],res,0)
        return res
num=        [10,1,6,7,2,1,5]
target=8
if __name__ == "__main__":
    print(Solution().combinationSum2( num, target))        
        
[10,1,6,7,2,1,5] and target 8        
        
 [
  [1,7],
  [1,2,5],
  [2,6],
  [1,1,6]
]       
        
#154. Regular Expression Matching        
class Solution:
    """
    @param s: A string 
    @param p: A string includes "." and "*"
    @return: A boolean
    """
    def isMatch(self, s, p):
        # write your code here
#https://leetcode.com/problems/regular-expression-matching/discuss/5723/My-DP-approach-in-Python-with-comments-and-unittest        
        m=len(p)
        n=len(s)
        dp=[[False for _ in range(n+1)] for _ in range(m+1)]
        
        dp[0][0]=True
     
        for i in range(2,m+1):
            if p[i-1]=='*':
               dp[i][0]=dp[i-2][0]
        
        
        for i in range(1,m+1):
            for j in range(1,n+1):
                if p[i-1]!='*':
                    if  p[i-1]=='.'  or  p[i-1]==s[j-1]:
                        dp[i][j]=dp[i-1][j-1]
                else:
                    dp[i][j]=dp[i-2][j]  or dp[i-1][j]
                    if p[i-2]==s[j-1] or p[i-2]=='.':
                        dp[i][j]=dp[i][j] or dp[i][j-1]
        return dp[-1][-1]

s="aa"
p="a" 
s="aab"
p="c*a*b"                       
if __name__ == "__main__":
    print(Solution().isMatch( s, p))        
                                
#155. Minimum Depth of Binary Tree                    
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of binary tree
    @return: An integer
    """
    def minDepth(self, root):
        # write your code here
        
        if not root:
            return 0
        
        from collections import deque
        q=deque([root])
        
        step=1
        while q:
            tempq=deque()
            for _ in range(len(q)):
                
                node=q.popleft()
                if not node.left and not node.right:
                    return step
                else:
                    if node.left:
                        tempq.append(node.left)
                    if node.right:
                        tempq.append(node.right)
            q=tempq
            step+=1
        return step
                
            
  1
 / \ 
2   3
   / \
  4   5              
                        
root=TreeNode(1)                   
root.left=TreeNode(2)   
root.right=TreeNode(3)   
root.right.left=TreeNode(4)   
root.right.right=TreeNode(5)   
if __name__ == "__main__":
    print(Solution().minDepth( root))   

#156. Merge Intervals
"""
Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    """
    @param intervals: interval list.
    @return: A new interval list.
    """
    def merge(self, intervals):
        # write your code here
        res=[]
        intervals.sort(key=lambda x:x.start)
        a='@'
        
#        for x in intervals:
#            print(x.start,x.end)
        for i in range(len(intervals)-1):
            if a=='@':
                a=intervals[i].start
                b=intervals[i].end
            b=max(b,intervals[i].end)
            if intervals[i+1].start>b:
                res.append(Interval(a,b))
                a='@'
            
        if a=='@':
            res.append(intervals[-1])
        else:
            res.append(Interval(a,max(b,intervals[-1].end)))
        for x in res:
            print(x.start,x.end)
        return res
            
intervals=[Interval(1,4),Interval(0,2),Interval(3,5)]
intervals=[                     
  Interval(1, 3),              
  Interval(2, 6),       Interval(8, 10),           
  Interval(15, 18)            ]
intervals=[                     
  (1, 3),              
  (2, 6),       (8, 10),           
  (10, 18)]

intervals=[Interval(2,3),Interval(4,5),Interval(6,7),Interval(8,9),Interval(1,10)]

intervals=[(0,2),(1,4),(3,5)]
intervals=[(2,3),(4,5),(6,7),(8,9),(1,10)]
       
if __name__ == "__main__":
    print(Solution().merge(intervals))
    
#157. Unique Characters
class Solution:
    """
    @param: str: A string
    @return: a boolean
    """
    def isUnique(self, str):
        # write your code here
        
        dic={}
        for s in str:
            if s in dic:
                return False
            dic[s]=1
        return True
    
    
    
    
#158. Valid Anagram
class Solution:
    """
    @param s: The first string
    @param t: The second string
    @return: true or false
    """
    def anagram(self, s, t):
        # write your code here
        if not s and not t:
            return True
        if not s or not t:
            return False
        n=len(s)
        m=len(t)
        if n!=m:
            return False
        
        from collections import Counter
        return Counter(s)==Counter(t)
            
s="abcd"
t="dcab"           
if __name__ == "__main__":
    print(Solution().anagram(s, t))             
            
            
#159. Find Minimum in Rotated Sorted Array
class Solution:
    """
    @param nums: a rotated sorted array
    @return: the minimum number in the array
    """
    def findMin(self, nums):
        # write your code here
        
        n=len(nums)
        if n==1:
            return nums[0]
        if n==2:
            return min(nums)
        
        l=0
        r=n-1
        
        while l+1<r:
            mid=(l+r)//2
            #print(l,r,mid)
            if nums[mid]>nums[l] and nums[r]<nums[l]:#left part is sorted min is in the right part
               l=mid+1
            else:
                r=mid
        print(nums[mid],nums[l],nums[r])
        return min(nums[mid],nums[l],nums[r])
 
nums=[2, 4, 5, 6, 7, 0, 1]
nums=[4, 5, 6, 7, 0, 1, 2] 
nums=[5, 6, 7, 0, 1, 2, 4]
nums=[6, 7, 0, 1, 2, 4, 5]
if __name__ == "__main__":
    print(Solution().findMin( nums))  

#160. Find Minimum in Rotated Sorted Array II
class Solution:
    """
    @param nums: a rotated sorted array
    @return: the minimum number in the array
    """
    def findMin(self, nums):
        # write your code here
        n=len(nums)
        if n==1:
            return nums[0]
        if n==2:
            return min(nums)
        
        l=0
        r=n-1
        while l+1<r:
            
            mid=(l+r)//2
            #print(l,r,mid)
            if nums[mid]>=nums[l] and nums[r]<=nums[l]:#left part is sorted min is in the right part
               l=mid+1
            else:
                r=mid
        print(nums[mid],nums[l],nums[r])
        return min(nums[mid],nums[l],nums[r])
nums=[4,4,4,4,4,0,1,3]
nums=[4,4,4,4,0,0,0,0]

nums=[4,4,0,1,3,3,3,3]
if __name__ == "__main__":
    print(Solution().findMin( nums)) 

#161. Rotate Image
class Solution:
    """
    @param matrix: a lists of integers
    @return: nothing
    """
    def rotate(self, matrix):
        # write your code here
        n=len(matrix)
        
        
        for i in range(n):
            for j in range(i+1,n):
              matrix[i][j] ,  matrix[j][i] = matrix[j][i]m, matrix[i][j]
              
        for i in range(n):
            matrix[i].reverse()
            
#162. Set Matrix Zeroes        
class Solution:
    """
    @param matrix: A lsit of lists of integersa
    @return: nothing
    """
    def setZeroes(self, matrix):
        # write your code here
        m=len(matrix)
        n=len(matrix[0])
        
        
        row=set()
        col=set()
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j]==0:
                    row.add(i)
                    col.add(j)
                    continue
            if i in row:
                continue
            
        for  i in range(m):
            for j in range(n):
                if i in row or j in col:
                    matrix[i][j]=0
                
#163. Unique Binary Search Trees                    
class Solution:
    """
    @param n: An integer
    @return: An integer
    """
    def numTrees(self, n):
        # write your code here
#一棵树由根节点，左子树和右子树构成。
#对于目标n，根节点可以是1, 2, ..., n中的任意一个，假设根节点为k，那么左子树的可能性就是numTrees(k-1)种，
#右子树的可能性就是numTrees(n-k)种，他们的乘积就根节点为k时整个树的可能性。把所有k的可能性累加就是
#最终结果        
        hashtable={0:1,1:1,2:2}
        
        def dfs(hashtable,n):
            if n in hashtable:
                return hashtable[n]
            res=0
            for i in range(1,n+1):
                res+=dfs(hashtable,i-1)*dfs(hashtable,n-i)
            hashtable[n]=res
            return res
        return dfs(hashtable,n)
n=3
if __name__ == "__main__":
    print(Solution().numTrees( n)) 
                
#164. Unique Binary Search Trees II
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
class Solution:
    # @paramn n: An integer
    # @return: A list of root
    def generateTrees(self, n):
        # write your code here     
        def dfs(start,end):
            if start>end:
                return [None]
            res=[]
            
            for rootval in range(start,end+1):
                leftTree=dfs(start,rootval-1)
                rightTree=dfs(rootval+1,end)
                for i in leftTree:
                   for j in rightTree:
                      
                     root=TreeNode(rootval)
                     root.left=i
                     root.right=j
                     res.append(root)
            return res
        return dfs(1,n)
n=3
if __name__ == "__main__":
    print(Solution().generateTrees( n))         
        
#165. Merge Two Sorted Lists   
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param l1: ListNode l1 is the head of the linked list
    @param l2: ListNode l2 is the head of the linked list
    @return: ListNode head of linked list
    """
    def mergeTwoLists(self, l1, l2):
        # write your code here
        if not l1 :
            return l2
        if not l2:
            return l1
        
        dummy=ListNode(-1)
        cur=dummy
        while l1 and l2:
              if l1.val<l2.val:
                  cur.next=l1
                  l1=l1.next
              else:
                  cur.next=l2
                  l2=l2.next
              cur=cur.next
        if l1:
            cur.next=l1
        if l2:
            cur.next=l2
        return dummy.next
            
#166. Nth to Last Node in List
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""
class Solution:
    """
    @param: head: The first node of linked list.
    @param: n: An integer
    @return: Nth to last node of a singly linked list. 
    """
    def nthToLast(self, head, n):
        # write your code here
        if not head:
            return None
        
        length=0
        cur=head
        while cur:
            cur=cur.next
            length+=1
        cur=head
        for _ in range(length-n):
            cur=cur.next
            
        return cur
            
        
        
            
root=      ListNode(3)
root.next=      ListNode(2)
root.next.next=      ListNode(1) 
root.next.next.next=      ListNode(5)      
head=root
n=2        
if __name__ == "__main__":
    print(Solution().nthToLast( head, n))           
#Given a List  3->2->1->5->null and n = 2, return node  whose value is 1        
        
#167. Add Two Numbers        
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param l1: the first list
    @param l2: the second list
    @return: the sum list of l1 and l2 
    """
    def addLists(self, l1, l2):
        # write your code here
        if not l1 :
            return l2
        if not l2:
            return l1
        
        dummy=ListNode(-1)
        carry=0
        cur=dummy
        while l1 and l2:
            cur.next=ListNode((l1.val+l2.val+carry)%10)
            carry=(l1.val+l2.val+carry)//10
            
            l1=l1.next
            l2=l2.next
            cur=cur.next
        while l1:
            cur.next=ListNode((l1.val+carry)%10)
            carry=(l1.val+carry)//10
            
            l1=l1.next
            
            cur=cur.next
        while l2:
            cur.next=ListNode((l2.val+carry)%10)
            carry=(l2.val+carry)//10
            
           
            l2=l2.next
            cur=cur.next
        if carry:
            
           cur.next=ListNode(carry)
        cur=dummy.next
        while cur:
            print(cur.val)
            cur=cur.next
        return dummy.next
            
l1=      ListNode(7)
l1.next=      ListNode(1)
l1.next.next=      ListNode(6) 
l2=      ListNode(5)
l2.next=      ListNode(9)
l2.next.next=      ListNode(2) 

l1=      ListNode(3)
l1.next=      ListNode(1)
l1.next.next=      ListNode(5) 
l2=      ListNode(5)
l2.next=      ListNode(9)
l2.next.next=      ListNode(2) 

l1=      ListNode(1)
l1.next=      ListNode(1)
l1.next.next=      ListNode(1) 
l1.next.next.next=      ListNode(1)
l1.next.next.next.next=      ListNode(1)

l2=      ListNode(9)
l2.next=      ListNode(8)
l2.next.next=      ListNode(8) 
l2.next.next.next=      ListNode(8)
l2.next.next.next.next=      ListNode(8)

if __name__ == "__main__":
    print(Solution().addLists(l1, l2))           
             
            
  
        
#Given 7->1->6 + 5->9->2. That is, 617 + 295.
#
#Return 2->1->9. That is 912.
#1->1->1->1->1->null
#9->8->8->8->8->null
#Given 3->1->5 and 5->9->2, return 8->0->8.        
        
#168. Burst Balloons        
class Solution:
    """
    @param nums: A list of integer
    @return: An integer, maximum coins
    """
    def maxCoins(self, nums):
        # write your code here
        
#        def burst(nums,memo):
#            if tuple(nums) in memo:
#                return memo[tuple(nums)]
#            n=len(nums)
#            if len(nums)==3:
#                memo[tuple(nums)]=nums[1]
#                return nums[1]
#            
#            maxcoins=0
#            for i in range(1,n-1):
#                maxcoins=max(maxcoins,nums[i-1]*nums[i]*nums[i+1]+burst(nums[:i]+nums[i+1:],memo))
#            memo[tuple(nums)]=maxcoins
#            return maxcoins
#                
#        memo={}
#        return burst([1]+nums+[1],memo)
        
#        @dharmendra2 This dp works in this way: we scan the array from len 2 to len n with 
#        all possible start points and end points. For each combination, we will find the 
#        best way to burst balloons. dp[i][j] means we are looking at a combination with
#        start point at index i and end point at index j with len of j - i. In this combination,
#        we use the third loop to find the best way to burst.
#        “nums[left] * nums[i] * nums[right]” means we burst all balloons from left to i
#        and all ballons from i to right. So only balloons left, i and right exits in current 
#        combination therefore we can do this operation. “+ dp[left][i] + dp[i][right]” means 
#        add the value from best burst in range(left, i) and range(i, right).
        n=len(nums)
        
        dp=[[0 for _ in range(n)]  for _ in range(n)]
        
        for length in range(n):
            for i in range(0,n-length):
                j=i+length
                for k in range(i,j+1):
                    if i==k:
                        leftdp=0
                    else:
                        leftdp=dp[i][k-1]
                    if k==j:
                        rightdp=0
                    else:
                        rightdp=dp[k+1][j]
                    
                    if i==0:
                        left=1
                    else:
                        left=nums[i-1]
                    if j==n-1:
                        right=1
                        
                    else:
                        right=nums[j+1]
                    dp[i][j]=max(dp[i][j],leftdp+rightdp+left*nums[k]*right)
        return dp[0][n-1]
        
        
        
        
        
nums=[4, 1, 5, 10]
nums=[35,16,83,87,84,59,48,41,20,54]
nums=[8,2,6,8,9,8,1,4,1,5,3,0,7,7,0,4,2,2,5,5]    
if __name__ == "__main__":
    print(Solution().maxCoins(nums))    
        
#Given [4, 1, 5, 10]
#Return 270
#
#nums = [4, 1, 5, 10] burst 1, get coins 4 * 1 * 5 = 20
#nums = [4, 5, 10]    burst 5, get coins 4 * 5 * 10 = 200 
#nums = [4, 10]       burst 4, get coins 1 * 4 * 10 = 40
#nums = [10]          burst 10, get coins 1 * 10 * 1 = 10
#
#Total coins 20 + 200 + 40 + 10 = 270

        
#169. Tower of Hanoi
class Solution:
    """
    @param n: the number of disks
    @return: the order of moves
    """
    def towerOfHanoi(self, n):
        # write your code here
        
#        res=[]
#        
#        def move(n,fromDisk,toDisk,auxiliary):
#            if n==1:
#                res.append('Disk 1 move from '+ fromDisk+ ' to ' + toDisk)
#                return 
#            move(n-1,fromDisk,auxiliary,toDisk)
#            res.append('disk '+str(n)+' move from '+ fromDisk+ ' to ' + toDisk)
#            move(n-1,auxiliary,toDisk,fromDisk)
#        move(n,'A','C','B')
#        return res
#    
        res=[]
        
        def move(n,fromDisk,toDisk,auxiliary):
            if n==1:
                res.append('from '+ fromDisk+ ' to ' + toDisk)
                return 
            move(n-1,fromDisk,auxiliary,toDisk)
            res.append('from '+ fromDisk+ ' to ' + toDisk)
            move(n-1,auxiliary,toDisk,fromDisk)
        move(n,'A','C','B')
        return res
n=3
if __name__ == "__main__":
    print(Solution().towerOfHanoi( n))                    
        
#170. Rotate List
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: the List
    @param k: rotate to the right k places
    @return: the list after rotation
    """
    def rotateRight(self, head, k):
        # write your code here
        
        if not head:
            return None
        
        if k==0:
            return head
        
        if not head.next:
            return head
        
        cur=head
        
        length=0
        
        while cur:
            length+=1
            cur=cur.next
            
        k=k%length 
        
        if k==0:
            return head
        
        
        cur=head
        
        for _ in range(length-k-1):
            cur=cur.next
        
        newhead=cur.next
        cur.next=None
        
        cur=newhead
        while cur.next:
            cur=cur.next
        cur.next=head
        
        
        cur=newhead
        
        while cur:
            print(cur.val)
            cur=cur.next
        return newhead
k = 2
head=ListNode(1)
head.next=ListNode(2)
head.next.next=ListNode(3)
head.next.next.next=ListNode(4)
head.next.next.next.next=ListNode(5)
            

k = 0
head=ListNode(1)


k=100
head=ListNode(0)
head.next=ListNode(1)

if __name__ == "__main__":
    print(Solution().rotateRight( head, k))                    
                
#171. Anagrams        
class Solution:
    """
    @param strs: A list of strings
    @return: A list of strings
    """
    def anagrams(self, strs):
        # write your code here
        
        from collections import defaultdict
        dic=defaultdict(list)
        
        
        for x in strs:
            sortedword=''.join(sorted(x))
            dic[sortedword]+=[x]
        
        res=[]
        for key,val in dic.items():
            if len(val)>=2:
                res+=val
        return res
                
            
            
strs=["lint", "intl", "inlt", "code"]  
strs=["ab", "ba", "cd", "dc", "e"]  
if __name__ == "__main__":
    print(Solution().anagrams( strs))                 
    
#172. Remove Element
class Solution:
    """
    @param: A: A list of integers
    @param: elem: An integer
    @return: The new length after remove
    """
    def removeElement(self, A, elem):
        # write your code here 
        if not A:
            return 0
        i=0
        j=0
        
        n=len(A)
        while j<n:
            if A[j]!=elem:
                A[i]=A[j]
                i+=1
                j+=1
                
            else:
                j+=1
        return i
            
A= [0,4,4,0,0,2,4,4]     
elem=4          
if __name__ == "__main__":
    print(Solution().removeElement( A, elem))                 
            
    
#173. Insertion Sort List
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: The first node of linked list.
    @return: The head of linked list.
    """
    def insertionSortList(self, head):
        # write your code here
#https://leetcode.com/problems/insertion-sort-list/discuss/46420/An-easy-and-clear-way-to-sort-(-O(1)-space-)
        if not head:
            return None
        dummy=ListNode(0)
        pre=dummy#insert node between pre and pre.next
        cur=head#the node will be inserted
        
        while cur:
            next=cur.next
            #find the right place to insert
            while pre.next  and pre.next.val < cur.val:
                pre=pre.next
            #insert between pre and pre.next
            cur.next=pre.next
            pre.next=cur
            cur=next
            pre=dummy
        return dummy.next
        
        
#174. Remove Nth Node From End of List        
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: The first node of linked list.
    @param n: An integer
    @return: The head of linked list.
    """
    def removeNthFromEnd(self, head, n):
        # write your code here
        
        if not head:
            return None
        if n==0:
            return head
        
        fast=head
        slow=head
        
        for _ in range(n):
            fast=fast.next
        if not fast:
            return head.next
        while fast.next:
            fast=fast.next
            slow=slow.next
        slow.next=slow.next.next
        cur=head
        while cur:
            print(cur.val)
            cur=cur.next
        return head
       
#Given linked list: 1->2->3->4->5->null, and n = 2.
#
#After removing the second node from the end, the linked list becomes 1->2->3->5->null.        
        
head=ListNode(1)
head.next=ListNode(2)
head.next.next=ListNode(3)
head.next.next.next=ListNode(4)
head.next.next.next.next=ListNode(5)            
n=1       
if __name__ == "__main__":
    print(Solution().removeNthFromEnd(head, n))         
        
        
#175. Invert Binary Tree
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def invertBinaryTree(self, root):
        # write your code here
        
        if not root:
            return None
        
        def invert(node):
            
            if not node:
                return None
            right=invert(node.left)
            left=invert(node.right)
            node.left=left
            node.right=right
            return node
        return invert(root)
        
        
#176. Route Between Two Nodes in Graph
"""
Definition for a Directed graph node
class DirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
"""


class Solution:
    """
    @param: graph: A list of Directed graph node
    @param: s: the starting Directed graph node
    @param: t: the terminal Directed graph node
    @return: a boolean value
    """

    def hasRoute(self, graph, s, t):
        # write your code here
        visited={}
        for x in graph:
            visited[x]=0
        
        def dfs(visited,cur,t):
            if visited[cur]==1:
                return False
            if cur==t:
                return True
            visited[cur]=1
            for nextcur in cur.neighbors:
                if visited[nextcur]==0 and dfs(visited,nextcur,t):
                    return True
            return False
        
        return dfs(visited,s,t)
            
#177. Convert Sorted Array to Binary Search Tree With Minimal Height
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    """
    @param: A: an integer array
    @return: A tree node
    """
    def sortedArrayToBST(self, A):
        # write your code here
        def convert(l):
            n=len(l)
            if n==0:
                return None
            if n==1:
                return TreeNode(l[0])
            mid=(0+n-1)//2
            root=TreeNode(l[mid])
            
            root.left=convert(l[:mid])
            root.right=convert(l[mid+1:])
            return root
        return convert(A)
            
#178. Graph Valid Tree
class Solution:
    """
    @param n: An integer
    @param edges: a list of undirected edges
    @return: true if it's a valid tree, or false
    """
    def validTree(self, n, edges):
        # write your code here
#思路：判断一个图是不是一棵树①首先应该有n-1条边　②边没有形成环        
#vertex和edge的validation，|E| = |V| - 1，
#也就是要验证 edges.length == n-1，如果该条件不满足，则Graph一定不是valid tree。  
        if len(edges)!=n-1:
           return False    
        
        from    collections import defaultdict 
        neighbors  =    defaultdict (list) 
        for k,v in edges:
            neighbors[k]+=[v]
            neighbors[v]+=[k]
        self.visited=set()
        def dfs(i,parent):
            self.visited.add(i)
            
            for nei in neighbors[i]:
                if nei not in self.visited :
                    if not dfs(nei,i):
                        return False
                elif nei!=parent:
                    return False
            return True
        
        res=dfs(0,-1)
        print(self.visited)
        print(res)
        return res  and len(self.visited)==n
n = 5 
edges = [[0, 1], [0, 2], [0, 3], [1, 4]]  
edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]  ]
if __name__ == "__main__":
    print(Solution().validTree( n, edges))                
            
#179. Update Bits
class Solution:
    """
    @param n: An integer
    @param m: An integer
    @param i: A bit position
    @param j: A bit position
    @return: An integer
    """
    def updateBits(self, n, m, i, j):
        # write your code here
#http://www.chenguanghe.com/lintcode-update-bits/ 
#python deal with 32th bit differs from Java
        
#        for k in range(i,j+1):
#            n=n & ~(1<<k) #set kth bit in n to 0
#            n = n|((m& (1<<(k-i)))<<i)#set kth bit in n to m's kth bit
#        
#        if n>(1<<31):
#            return n-(1<<32)
        a=[]
        for _ in range(32):
            a.append(n%2)
            n=n//2
        
        for k in range(i,j+1):
            a[k]=m%2
            m=m//2
        
        n=0
        for i in range(31):
            if a[i]>0:
                n|=(a[i]<<i)
#If the sign bit is “1”, then the number is negative in value  

        if a[31]==1:
            n-=1<<31 
            n=~n+1
            # ~n+1
        return n
            
#Two's Complement binary for Negative Integers:
#
#Negative numbers are written with a leading one instead of a leading zero.
# So if you are using only 8 bits for your twos-complement numbers, 
# then you treat patterns from "00000000" to "01111111" as the whole numbers from 0 to 127, 
# and reserve "1xxxxxxx" for writing negative numbers. A negative number, -x, 
# is written using the bit pattern for (x-1) with all of the bits complemented 
# (switched from 1 to 0 or 0 to 1). So -1 is
# complement(1 - 1) = complement(0) = "11111111", and -10 is complement(10 - 1) =
# complement(9) = complement("00001001") = "11110110". This means that negative numbers
# go all the way down to -128 ("10000000").
#
#Of course, Python doesn't use 8-bit numbers. It USED to use however many bits were 
#native to your machine, but since that was non-portable, it has recently switched to
# using an INFINITE number of bits. Thus the number -5 is treated by bitwise operators
# as if it were written "...1111111111111111111011".                        
############################################################            
# Negative numbers to binary system with Python            #
#                                                          #
#                                                          #
#Just need to add 2**32 (or 1 << 32) to the negative value # 
#bin(-1+(1<<32))                                           #
# or Bitwise AND (&) with 0xffffffff (2**32 - 1) first:    #
#    0xffffffff used as mask                               #
#'0b11111111111111111111111111111111'                      #         
            
        
            
#m & (1<<(k-i)) 是从m的第一个bit开始扫描.因为已知j-i = size of (m)2
#((m & (1<<(k-i)))<<i) 扫描后, 往左shift i位对准n上的i位.
#n = n | ((m & (1<<(k-i)))<<i) 把n的第i位到j位设为m的0~(j-i)位 
        
n=int('10000000000',2)        
m=int('10101',2)  
i=2
j=6 

n=1
m=-1
i=0
j=31  
n= -521
m=0
i=31
j=31       
if __name__ == "__main__":
    print(Solution().updateBits(n, m, i, j))  
        
#180. Binary Representation
class Solution:
    """
    @param n: Given a decimal number that is passed in as a string
    @return: A string
    """
    def binaryRepresentation(self, n):
        # write your code here
        
        [a,b]=n.split('.')
        a='{:b}'.format(int(a))
        #print(a,b)
        
        
        def frac_to_binary( num):
            if num==''  or int(num)==0:
                return ''
            if int(num)%10!=5:
                return None
            from decimal import Decimal 
            num=Decimal('0.'+num)
            res=''
            while num:
                num*=2
                if num>=1:
                    res+='1'
                    num-=1
                else:
                    res+='0'
                num=num.normalize()
                if num and str(num)[-1]!='5':
                    return None
            #print(res)
            return res
        b=frac_to_binary( b)
        print(b)
        if b  is None:
            return 'ERROR'
        elif b=='':
            return a
        else:
            return a+'.'+b
n="3.72" 
n="3.5" 
n="3.0" 
n='3.'                 
if __name__ == "__main__":
    print(Solution().binaryRepresentation( n))              
        
        
#181. Flip Bits
class Solution:
    """
    @param a: An integer
    @param b: An integer
    @return: An integer
    """
    def bitSwapRequired(self, a, b):
        # write your code here
        c=a^b
        cnt=0
        print('{:b}'.format(c))
        for i in range(32):
          if ( c & (1<<i) )!=0:
              cnt+=1
        return cnt
a=14
b=31
if __name__ == "__main__":
    print(Solution().bitSwapRequired(a, b) )  
            
class Solution:
    """
    @param a, b: Two integer
    return: An integer
    """
    def bitSwapRequired(self, a, b):
        # write your code here
        c = a ^ b
        cnt = 0   
        for i in range(32):
            if c & (1 << i) != 0:
                cnt += 1
        return cnt

#182. Delete Digits
class Solution:
    """
    @param A: A positive integer which has N digits, A is a string
    @param k: Remove k digits
    @return: A string
    """
    def DeleteDigits(self, A, k):
        # write your code here
        n=len(A)
        if k==n:
            return ''
        if k>n:
            return ''
        remain=n-k
        
        A_list=list(A)
        res=[]
        index=-1
        for i in range(k,n):
            minmin=float('inf')
            for j in range(index+1,i+1):
                if int(A[j]) < minmin:
                    minmin=int(A[j])
                    index=j
            res.append(str(minmin))
        return str(int(''.join(res)))

 
A = "178542"
k = 4 
A = "254193"
k=1 
A="123454321"
k=1     
        
if __name__ == "__main__":
    print(Solution().DeleteDigits( A, k))            
                
#183. Wood Cut
class Solution:
    """
    @param L: Given n pieces of wood with length L[i]
    @param k: An integer
    @return: The maximum length of the small pieces
    """
    def woodCut(self, L, k):
        # write your code here
        
        largest=max(L)
        
        if k>sum(L):
            return 0
        
        if k==sum(L):
            return 1
        
        
        l=0
        r=largest
        #print(largest)
        #print(len(L))
        
        while l+1<r:
            mid=(r+l)//2
            
            res=0
            for i in range(len(L)):
                res+=L[i]//mid
            if res==k:
                l=mid
            elif res>k:
                l=mid
            else:
                r=mid
        res_m=0
        res_l=0
        res_r=0
        for i in range(len(L)):
                res_m+=L[i]//mid
                res_l+=L[i]//l
                res_r+=L[i]//r
        dic={res_m:mid,res_l:l,res_r:r}
        print(dic)
        
        
        array=[(key-k,v) for (key,v) in dic.items() if key-k >=0]
        
        
        array.sort(key=lambda x: ( x[0] ,-x[1]))
       
        return array[0][1]
        
           
L=[232, 124, 456]
k=7# return 114.       
if __name__ == "__main__":
    print(Solution().woodCut(L, k))              
                
#184. Largest Number
class Solution:
    """
    @param nums: A list of non negative integers
    @return: A string
    """
    def largestNumber(self, nums):
        # write your code here
        
#        # works in Python 2 only 
#        # pythno remove cmp
#        
#        nums=sorted(nums,cmp=lambda x ,y: 1 if str(x)+str(y) < str(y)+str(x)  else -1)
#        #print nums
#       
#        largest=''.join([str(x) for x in nums])
        class comparekey(str):
            def __lt__(x,y):
                return x+y>y+x
        s=[str(x) for x in nums]
        
        s.sort(key=comparekey)
        
        return ''.join(s) if s[0]!='0'  else '0' 


nums=[1, 20, 23, 4, 8]        
nums= [  11, 10, 1 ] 
    
if __name__ == "__main__":
    print(Solution().largestNumber(nums))

#185. Matrix Zigzag Traversal         
class Solution:
    """
    @param matrix: An array of integers
    @return: An array of integers
    """
    def printZMatrix(self, matrix):
        # write your code here
        
        m=len(matrix)
        if m==0:
            return []
        n=len(matrix[0])
        
        from collections import defaultdict
        dd=defaultdict(list)
        
        for i in range(m):
            for j in range(n):
                dd[i+j+1].append(matrix[i][j])
        
        res=[]
        for key in dd:
            if key%2==1:
                res+=reversed(dd[key])
            else:
                res+=dd[key]
        return  res        
 
matrix=       [
  [1, 2,  3,  4],
  [5, 6,  7,  8],
  [9,10, 11, 12]
]
if __name__ == "__main__":
    print(Solution().printZMatrix( matrix))
        
        
#186. Max Points on a Line
"""
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
"""

class Solution:
    """
    @param points: an array of point
    @return: An integer
    """
    def maxPoints(self, points):
        # write your code here
        n=len(points)
        from collections import defaultdict
        d=defaultdict(int)
        if n<=2:
            return n
        
        def gcd(a,b):
            if b==0:
                return a
            return gcd(b,a%b)
        
        res=0
        for i in range(n):
            d.clear()
            
            curmax=0
            overlap=0
            for j in range(i+1,n):
                dx=points[i].x-points[j].x
                dy=points[i].y-points[j].y
                if dx==0 and dy==0:
                    overlap+=1
                    continue
                
                com=gcd(dx,dy)
                dx=dx//com
                dy=dy//com
                
                d[(dx,dy)]+=1
                curmax=max(curmax, d[(dx,dy)])
            res=max(res,curmax+overlap+1)
        return res

            
if __name__ == "__main__":
    print(Solution().maxPoints( points))        
        
#187. Gas Station
class Solution:
    """
    @param gas: An array of integers
    @param cost: An array of integers
    @return: An integer
    """
    def canCompleteCircuit(self, gas, cost):
        # write your code here
        n=len(gas)
        total=0
        summ=0
        start=0
        if n==0:
            return -1
        if sum(gas)-sum(cost)<0:
            return -1
        for i in range(n):
            summ+=gas[i]-cost[i]
            
            if summ<0:
                summ=0
                start=i+1
        
        return start
gas=[5]
cost=[4] 
gas=[1,1,3,1] 
cost=[2,2,1,1]   
gas=[0,4,1,1]
cost=[1,1,4,1]           
if __name__ == "__main__":
    print(Solution().canCompleteCircuit( gas, cost))              
        
#189. First Missing Positive        
class Solution:
    """
    @param A: An array of integers
    @return: An integer
    """
    def firstMissingPositive(self, A):
        # write your code here
        n=len(A)
        if n==0:
            return 1
        i=0
        while i<n:
            while A[i]>0 and A[i]<=n and A[i]!=i+1 and A[i]!=A[A[i]-1]:
                t=A[i]
                A[i]=A[t-1]
                A[t-1]=t
            i+=1
        print(A)
        for i in range(n):
            if A[i]!=i+1:
                return i+1
        return n+1
A=[1,2,0]
A=[3,4,-1,1]    
A=[-1]   
A=[99,94,96,11,92,5,91,89,57,85,66,63,84,81,79,61,74,78,77,30,64,13,58,18,70,69,51,12,32,34,9,43,39,8,1,38,49,27,21,45,47,44,53,52,48,19,50,59,3,40,31,82,23,56,37,41,16,28,22,33,65,42,54,20,29,25,10,26,4,60,67,83,62,71,24,35,72,55,75,0,2,46,15,80,6,36,14,73,76,86,88,7,17,87,68,90,95,93,97,98]  
if __name__ == "__main__":
    print(Solution().firstMissingPositive( A)) 

#190. Next Permutation II    
class Solution:
    """
    @param nums: An array of integers
    @return: nothing
    """
    def nextPermutation(self, nums):
        # write your code here
        n=len(nums)
        temp=[0]*n
        i=n-1
        while i>0 and nums[i]<=nums[i-1]:
            i-=1
        if i==0:
            for j in range(n):
                temp[j]=nums[j]
            for j in range(n):
                nums[j]=temp[n-1-j]
        #print(i)
        
        
        for k in range(i-1):
            temp[k]=nums[k]
            
        
        p=n-1
        
        while p>i and nums[p]<=nums[i-1]:
            p-=1
        nums[i-1],nums[p]=nums[p],nums[i-1]
        
        temp[i-1]=nums[i-1]
        
        for h in range(n-1,i-1,-1):
            temp[n-1+i-h]=nums[h]
        for h in range(n):
            nums[h]=temp[h]
        print(nums)
        return 
nums=[1,2,3]  #1,3,2
nums=[3,2,1] # 1,2,3
nums=[1,1,5] # 1,5,1
nums=[1,3,2]
if __name__ == "__main__":
    print(Solution().nextPermutation( nums))    
         
#191. Maximum Product Subarray
class Solution:
    """
    @param nums: An array of integers
    @return: An integer
    """
    def maxProduct(self, nums):
        # write your code here   
        n=len(nums)
        if n==0:
           return 0
#确定状态:
#最后一步: 考虑最后一个数A[n], 那么乘积最大的就是前面以n - 1为结尾的最大乘积, 再乘上这个数
#子问题: 如果要求以n为结尾的子数组的最大乘积, 先要求以n - 1为结尾的最大乘积
#在这里要注意, 如果A[n]是个负数, 因为负负得正, 所以我们需要以n - 1为结尾的最小乘积
#转移方程:
#维护两个数组, f[i] 和 g[i], f[i]用于记录最大值, g[i]用于记录最小值.
#A[i]代表数组中的第i个数
#转移方程: f[i] = max(i -> 1...n | f[i - 1] * A[i], g[i - 1] * A[i], A[i])
#g[i] = min(i -> 1...n | f[i - 1] * A[i], g[i - 1] * A[i], A[i])
#初始条件与边界情况:
#f[0] = A[0], g[0] = A[0]
#计算顺序:
#从左往右
#最终结果max(i -> 0...n | f[i])
        
        
        
        f=[0 for _ in range(n)]
        g=[0 for _ in range(n)]
        f[0]=nums[0]
        g[0]=nums[0]
        
        
        for i in range(1,n):
            f[i]=max(nums[i],max(nums[i]*f[i-1],nums[i]*g[i-1]))
            g[i]=min(nums[i],min(nums[i]*f[i-1],nums[i]*g[i-1]))
        return max(f)
nums=[2,3,-2,4]
nums=[-2,0,-1]
if __name__ == "__main__":
    print(Solution().maxProduct( nums))        
        
#192. Wildcard Matching
class Solution:
    """
    @param s: A string 
    @param p: A string includes "?" and "*"
    @return: is Match?
    """
    def isMatch(self, s, p):
        # write your code here
        m=len(s)
        n=len(p)
        dp=[[0 for _ in range(n+1)] for _ in range(m+1)]
        
        dp[0][0]=True
        for i in range(1,m+1):
            dp[i][0]=False
            
        print(dp)
        for j in range(1,n+1):
            if p[j-1]=='*':
               dp[0][j]=dp[0][j-1]
            else:
                dp[0][j]=False
                
        for i in range(1,m+1):
            for j in range(1,n+1):
                if p[j-1]!='*':
                    dp[i][j]=dp[i-1][j-1]  and ( p[j-1]==s[i-1] or p[j-1]=='?')
                else:
                    dp[i][j]=dp[i-1][j]  or dp[i][j-1]
        print(dp)
        return dp[m][n]
s="aa"
p='a'  

s="aa"
p='aa' 
s="aaa"
p='aa'
s="aa"
p='*'
s="aa"
p='a*' 
s="ab"
p='?*'  
s="aab"
p='c*a*b'          
if __name__ == "__main__":
    print(Solution().isMatch( s, p))                
        
#196. Missing Number        
class Solution:
    """
    @param nums: An array of integers
    @return: An integer
    """
    def findMissing(self, nums):
        # write your code here
        
#Xor    
        n=len(nums)
        ans=0
        for i in range(n+1):
            ans^=i
        for i in range(n):
            ans^=nums[i]
        
        return ans 
nums=[0, 1, 3]        
if __name__ == "__main__":
    print(Solution().findMissing(nums))        
        
            
#197. Permutation Index             
class Solution:
    """
    @param A: An array of integers
    @return: A long integer
    """
    def permutationIndex(self, A):
        # write your code here
        #先得了解 康托展开
        #https://zh.wikipedia.org/wiki/%E5%BA%B7%E6%89%98%E5%B1%95%E5%BC%80
        n=len(A)
        
        index=1
        for i in range(n):
            count=0
            factor=1
            
            for j in range(i+1,n):
                if A[j]<A[i]:
                    count+=1
            for k in range(1,n-i):
                factor*=k
            index+=count*factor
        return index
        
        
A=     [1,2,4]   
A=[3 ,5, 7 ,4 ,1, 2 ,9, 6 ,8]
if __name__ == "__main__":
    print(Solution().permutationIndex( A))         
        
        
#198. Permutation Index II
class Solution:
    """
    @param A: An array of integers
    @return: A long integer
    """
    def permutationIndexII(self, A):
        # write your code here
        n=len(A)
        if n==0:
            return 0
        
        
        
        count={}
        factor=1
        index=1
        multi_factor=1
        
        for i in range(n-1,-1,-1):
            rank=0
            if A[i] not in count:
                count[A[i]]=0
            count[A[i]]+=1
            multi_factor*=count[A[i]]
            for j in range(i+1,n):
                if A[j]<A[i]:
                    rank+=1
            index+=factor*rank//multi_factor
            factor*=(n-i)
        return index
            
                
A=     [1,2,4]   
A=[3 ,5, 7 ,4 ,1, 2 ,9, 6 ,8]       
if __name__ == "__main__":
    print(Solution().permutationIndexII( A))        
        
#200. Longest Palindromic Substring
class Solution:
    """
    @param s: input string
    @return: the longest palindromic substring
    """
    def longestPalindrome(self, s):
        # write your code here        
        
#https://www.felix021.com/blog/read.php?2040 
#https://leetcode.com/problems/longest-palindromic-substring/discuss/3337/Manacher-algorithm-in-Python-O(n)
 
        T='#'.join('^{}$'.format(s))
        
        C=0
        R=0
        n=len(T)
        p=[0 for _ in range(n)]
        for i in range(1,n-1):
            if R-i>0:
                p[i]=min(p[C-(i-C)],R - i)
            else:
                p[i]=1
#            p[i] = (R > i) and min(R - i, p[2*C - i]) 
            while T[i-p[i]]==T[i+p[i]]:
                p[i]+=1
            
            if p[i]+i>R:
                R=p[i]+i
                C=i
        maxlen,centerindex=max( (n,i)for i , n in enumerate(p))
        #return s[(centerindex- maxlen)//2:(centerindex+ maxlen)//2]
        return s[(centerindex-maxlen+1)//2:maxlen-1+(centerindex-maxlen+1)//2]
s="abcdzdcab"   
s="aaaabaaa"    
s="ccd" 
if __name__ == "__main__":
    print(Solution().longestPalindrome(s))            
            
#201. Segment Tree Build                
"""
Definition of SegmentTreeNode:
class SegmentTreeNode:
    def __init__(self, start, end):
        self.start, self.end = start, end
        self.left, self.right = None, None
"""
class Solution:
    """
    @param: start: start value.
    @param: end: end value.
    @return: The root of Segment Tree.
    """
    def build(self, start, end):
        # write your code here 
        if start>end:
            return None
        root=SegmentTreeNode(start, end)
        if start==end:
            return root
        
        root.left=self.build(root.start,(root.start + root.end) // 2)
        root.right=self.build((root.start + root.end) // 2+1,root.end)
        return root
            
        

#202. Segment Tree Query        
"""
Definition of SegmentTreeNode:
class SegmentTreeNode:
    def __init__(self, start, end, max):
        self.start, self.end, self.max = start, end, max
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of segment tree.
    @param start: start value.
    @param end: end value.
    @return: The maximum number in the interval [start, end]
    """
    def query(self, root, start, end):
        # write your code here        
        
#### Segment Tree, Divide and Conquer
#- 根据[start,end]跟 mid of (root.start, root.end) 做比较:
#- 简单的2个case: [start,end]全在mid左, 或者[start, end]全在mid右
#- 稍微复杂的3rd case: [start, end]包含了mid, 那么就break into 2 queries
#- [start, node.left.end], [node.right.start, end]        
        
        if start==root.start and end==root.end:
            return root.max
        
        mid=(root.start+root.end)//2
        if mid>=end:
            return self.query(root.left,start,end)
        if start>mid:
            return self.query(root.right,start,end)
        #start <= mid && end > mid
        leftmax=self.query(root.left,start,root.left.end)
        rightmax=self.query(root.right,root.right.start,end)
        return max(leftmax,rightmax)
            
        

#203. Segment Tree Modify
"""
Definition of SegmentTreeNode:
class SegmentTreeNode:
    def __init__(self, start, end, max):
        self.start, self.end, self.max = start, end, max
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of segment tree.
    @param index: index.
    @param value: value
    @return: nothing
    """
    def modify(self, root, index, value):
        # write your code here
        if root.start==index and root.end==index:
            root.max=value
            return 
        
        
        mid=(root.start+root.end)//2
        if mid >=index and index>=root.start:
            self.modify(root.left,index,value)
        if mid<index and root.end>=index:
            self.modify(root.right,index,value)
        root.max=max(root.left.max,root.right.max)

#204. Singleton
class Solution:
    # @return: The same instance of this class every time
    instance=None
    @classmethod
    def getInstance(cls):
        # write your code here
        if cls.instance is None:
            cls.instance=Solution()
        return cls.instance
        
        
        

#205. Interval Minimum Number
"""
Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""



class SegmentTree(object):
    def __init__(self,start,end,min=0):
        self.start=start
        self.end=end
        self.min=min
        self.left,self.right=None,None
#Always use self for the first argument to instance methods.

#Always use cls for the first argument to class methods.              
    @classmethod
    def build(cls,start,end,a):
        if start>end:
            return None
        if start==end:
            return SegmentTree(start,end,a[start])
        root=SegmentTree(start,end,a[start])
        mid=(start+end)//2
        root.left=cls.build(start,mid,a)
        root.right=cls.build(mid+1,end,a)
        root.min=min(root.left.min,root.right.min)
        return root
    @classmethod
    def query(self,root,start,end):
        if root.start>end or root.end<start:
            return float('inf')#make it big so that this one is not select
        if root.start>=start and root.end<=end:
            return root.min
        return min(self.query(root.left,start,end),self.query(root.right,start,end))

class Solution:
    """
    @param A: An integer array
    @param queries: An query list
    @return: The result list
    """
    def intervalMinNumber(self, A, queries):
        # write your code here 
        root=SegmentTree.build(0,len(A)-1,A)
        res=[]
        
        for x in queries:
            #res.append(SegmentTree.query(root,x.start,x.end))
            res.append(SegmentTree.query(root,x[0],x[1]))
        return res

A= [1,2,7,8,5]
queries=[(1,2),(0,4),(2,4)]
if __name__ == "__main__":
    print(Solution().intervalMinNumber(A, queries))            
                   
#206. Interval Sum
"""
Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class SegmentTree(object):
    def __init__(self,start,end,sum=0):
        self.start=start
        self.end=end
        self.sum=sum
        self.left,self.right=None,None
#Always use self for the first argument to instance methods.

#Always use cls for the first argument to class methods.              
    @classmethod
    def build(cls,start,end,a):
        if start>end:
            return None
        if start==end:
            return SegmentTree(start,end,a[start])
        root=SegmentTree(start,end,a[start])
        mid=(start+end)//2
        root.left=cls.build(start,mid,a)
        root.right=cls.build(mid+1,end,a)
        root.sum=root.left.sum+root.right.sum
        return root
    @classmethod
    def query(self,root,start,end):
        if root.start>end or root.end<start:
            return 0#make it big so that this one is not select
        if root.start>=start and root.end<=end:
            return root.sum
        return self.query(root.left,start,end)+self.query(root.right,start,end)
class Solution:
    """
    @param A: An integer list
    @param queries: An query list
    @return: The result list
    """
    def intervalSum(self, A, queries):
        # write your code here
        def intervalMinNumber(self, A, queries):
        # write your code here 
        root=SegmentTree.build(0,len(A)-1,A)
        res=[]
        
        for x in queries:
            #res.append(SegmentTree.query(root,x.start,x.end))
            res.append(SegmentTree.query(root,x[0],x[1]))
        return res

A= [1,2,7,8,5]
queries=[(0,4),(1,2),(2,4)]
if __name__ == "__main__":
    print(Solution().intervalMinNumber(A, queries))   

#207. Interval Sum II
class SegmentTree(object):
    def __init__(self,start,end,sum=0):
        self.start=start
        self.end=end
        self.sum=sum
        self.left,self.right=None,None
#Always use self for the first argument to instance methods.

#Always use cls for the first argument to class methods.              
    @classmethod
    def build(cls,start,end,a):
        if start>end:
            return None
        if start==end:
            return SegmentTree(start,end,a[start])
        root=SegmentTree(start,end,a[start])
        mid=(start+end)//2
        root.left=cls.build(start,mid,a)
        root.right=cls.build(mid+1,end,a)
        root.sum=root.left.sum+root.right.sum
        return root
    @classmethod
    def query(cls,root,start,end):
        if root.start>end or root.end<start:
            return 0#make it big so that this one is not select
        if root.start>=start and root.end<=end:
            return root.sum
        return cls.query(root.left,start,end)+cls.query(root.right,start,end)
    @classmethod
    def modify(cls,root, index, value):
        if root.start==index and root.end==index:
            root.sum=value
            return 
        mid=(root.start+root.end)//2
        if mid>=index and root.start<=index:
            cls.modify(root.left,index,value)
        if mid<index and root.end >=index:
            cls.modify(root.right,index,value)
        root.sum=root.left.sum+root.right.sum
            
class Solution:
    """
    @param: A: An integer array
    """
    def __init__(self, A):
        # do intialization if necessary
        self.root=SegmentTree.build(0,len(A)-1,A)

    """
    @param: start: An integer
    @param: end: An integer
    @return: The sum from start to end
    """
    def query(self, start, end):
        
        # write your code here
        return SegmentTree.query(self.root,start,end)
        

    """
    @param: index: An integer
    @param: value: An integer
    @return: nothing
    """
    def modify(self, index, value):
        # write your code here
         SegmentTree.modify(self.root,index, value)
        
        
        
A= [1,2,7,8,5]
#query(0, 2), return 10.
start=0
end=2
if __name__ == "__main__":
    print(Solution().query(start, end))         
        
#209. First Unique Character in a String        
class Solution:
    """
    @param str: str: the given string
    @return: char: the first unique character in a given string
    """
    def firstUniqChar(self, str):
        # Write your code here
        d=[0 for _ in range(256)]
        
        for x in str:
            d[ord(x)]+=1
        for i,x in enumerate(d):
            if  x==1:
                return chr(i)
        
str=     "abaccdeff" #b   
str= "aabc"   
if __name__ == "__main__":
    print(Solution().firstUniqChar( str))   


#211. String Permutation
class Solution:
    """
    @param A: a string
    @param B: a string
    @return: a boolean
    """
    def Permutation(self, A, B):
        # write your code here
        from collections import Counter
        return Counter(A)==Counter(B)



     
        
#212. Space Replacement
class Solution:
    """
    @param: string: An array of Char
    @param: length: The true length of the string
    @return: The true length of new string
    """
    def replaceBlank(self, string, length):
        # write your code here 
        if string is None:
            return length
            
        spaces = 0
        for c in string:
            if c == ' ':
                spaces += 1
        
        L = length + spaces * 2
        index = L - 1
        for i in range(length - 1, -1, -1):
            if string[i] == ' ':
                string[index] = '0'
                string[index - 1] = '2'
                string[index - 2] = '%'
                index -= 3
            else:
                string[index] = string[i]
                index -= 1
        return L
                
string="Mr John Smith" 
length=13               
if __name__ == "__main__":
    print(Solution().replaceBlank(string, length))         
        
        
        
#213. String Compression
class Solution:
    """
    @param str: a string
    @return: a compressed string
    """
    def compress(self, s):
        # write your code here
        if not s:
            return s
        
        res=''
        i=1
        count=1
        s+='#'
        while i<len(s):
            
            
            if i<len(s) and s[i]==s[i-1] :
                count+=1
                
                
            else:
                               
                res+=s[i-count]+str(count)
                count=1
            
            i+=1
        return res if len(res)<len(s)-1 else s[:-1]
            
s='aabcccccaaa' #return a2b1c5a3
s='aabbcc' #return aabbcc
s='aaaa' #return a4        
if __name__ == "__main__":
    print(Solution().compress(s))        
        
        
#221. Add Two Numbers II
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param l1: The first list.
    @param l2: The second list.
    @return: the sum list of l1 and l2.
    """
    def addLists2(self, l1, l2):
        # write your code here
        
        if not l1:
            return l2
        if not l2 :
            return l1
        
        s1=[]
        s2=[]
        
        while l1:
            s1.append(l1.val)
            l1=l1.next
        while l2:
            s2.append(l2.val)
            l2=l2.next
        sm=0
        pre=None
        while s1 or s2:
            if s1:
                sm+=s1.pop()
            if s2:
                sm+=s2.pop()
            
            cur=ListNode(sm%10)
            sm//=10
            cur.next=pre
            pre=cur
            
       
         
        if sm==0:
#           while cur:
#              print(cur.val)
#              cur=cur.next   
           return cur
        else:
           head=ListNode(sm%10)
           head.next=cur
           
#           while head:
#              print(head.val)
#              head=head.next
           
           
           return head
#6->1->7 + 2->9->5
l1=ListNode(6)
l1.next=ListNode(1)
l1.next.next=ListNode(7)

l2=ListNode(3)
l2.next=ListNode(8)
l2.next.next=ListNode(3)
#l2.next.next.next=ListNode(1)



if __name__ == "__main__":
    print(Solution().addLists2(l1, l2))           
             
#222. Setter and Getter
class School:
    def __init__(self):
        
        self.name=None
    '''
     * Declare a setter method `setName` which expect a parameter *name*.
     
    '''   
    # write your code here
    def setName(self,A):
        self.name=A  
        

    '''
     * Declare a getter method `getName` which expect no parameter and return
     * the name of this school
    '''
    # write your code here
    def getName(self):
        return self.name
        

    '''
     * Declare a getter method `getName` which expect no parameter and return
     * the name of this school
    '''
    # write your code here
        

    '''
     * Declare a getter method `getName` which expect no parameter and return
     * the name of this school
    '''
    # write your code here      
#223. Palindrome Linked List
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: A ListNode.
    @return: A boolean.
    """
    def isPalindrome(self, head):
        # write your code here
        if not head:
            return True
        
        fast=head
        slow=head
        while fast and fast.next:
            fast=fast.next.next
            slow=slow.next
        
        pre=None
        cur=slow
        
        while cur:
            temp=cur.next
            cur.next=pre
            pre=cur
            cur=temp
            
        p1=head
        p2=pre
        
        while p1 and p2:
            if p1.val!=p2.val:
                return False
            p1=p1.next
            p2=p2.next
        return True
    
    
#227. Mock Hanoi Tower by Stacks
class Tower:
    """
    @param: i: An integer from 0 to 2
    """
    def __init__(self, i):
        # create three towers
        self.disks=[]
        

    """
    @param: d: An integer
    @return: nothing
    """
    def add(self, d):
        # Add a disk into this tower
        if len(self.disks)>0 and self.disks[-1]<=d:
            print('eror adding disks')
        else:
            self.disks.append(d)

    """
    @param: t: a tower
    @return: nothing
    """
    def moveTopTo(self, t):
        # Move the top disk of this tower to the top of t.
        t.add(self.disks.pop())

    """
    @param: n: An integer
    @param: destination: a tower
    @param: buffer: a tower
    @return: nothing
    """
    def move_disks(self, n, destination, buffer):
        # Move n Disks from this tower to destination by buffer tower
        if n>0:
           self.move_disks(n-1,buffer,destination)
           self.moveTopTo(destination)
           buffer.move_disks(n-1,destination,self)
        

    """
    @return: Disks
    """
    def get_disks(self):
        # write your code here
        return self.disks
        

"""
Your Tower object will be instantiated and called as such:
towers = [Tower(0), Tower(1), Tower(2)]
for i in xrange(n - 1, -1, -1): towers[0].add(i)
towers[0].move_disks(n, towers[2], towers[1])
print towers[0], towers[1], towers[2]
"""
            

#245. Subtree        
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param T1: The roots of binary tree T1.
    @param T2: The roots of binary tree T2.
    @return: True if T2 is a subtree of T1, or false.
    """
    def isSubtree(self, T1, T2):
        # write your code here
        def isEqualTree(h1,h2):
            if not h1 and not h2:
                return True
            if not h1 and h2:
                return False
            if not h2 and h1:
                return False
            
            if h1.val!=h2.val:
                return False
            return isEqualTree( h1.left, h2.left)  and isEqualTree( h1.right, h2.right)
        
        
        def hasSubtree(T1, T2):
            
           if  not T2:
               return True
           if not T1 and  T2:
               return False
           if isEqualTree(T1, T2):
              return True
           else:
               
               if hasSubtree(T1.left, T2):
                   return True
               if hasSubtree(T1.right, T2):
                   return True
           return False
               
        return hasSubtree(T1, T2)
            
        
#247. Segment Tree Query II
"""
Definition of SegmentTreeNode:
class SegmentTreeNode:
    def __init__(self, start, end, count):
        self.start, self.end, self.count = start, end, count
        self.left, self.right = None, None
"""


class Solution:
    """
    @param: root: The root of segment tree.
    @param: start: start value.
    @param: end: end value.
    @return: The count number in the interval [start, end]
    """
    def query(self, root, start, end):
        # write your code here  
        if not root:
            return 0
        if root.start>end or root.end<start:
            return 0#make it big so that this one is not select
        if root.start>=start and root.end<=end:
            return root.count
        return self.query(root.left,start,end)+self.query(root.right,start,end)
        
        
        

#248. Count of Smaller Number        
class Solution:
    """
    @param A: An integer array
    @param queries: The query list
    @return: The number of element in the array that are smaller that the given integer
    """
    def countOfSmallerNumber(self, A, queries):
        # write your code here
        
        def bisearch(start,end,target):
            if start>end:
                return 0
            
            while start<=end:
                mid=(start+end)//2
                
                if A[mid]<target:
                    start=mid+1
                    
                else:
                    end=mid-1
                    res=mid
            return res
        
        
        ans=[]
        A.sort()
        for query in queries:
            ans.append(bisearch(0,len(A)-1,query))
        return ans
A=[1,2,7,8,5]   
queries=[1,8,5]         
if __name__ == "__main__":
    print(Solution().countOfSmallerNumber( A, queries))                  
        
#248. Count of Smaller Number        

class SegmentTree(object):
    def __init__(self,start,end,count=0):
        self.start=start
        self.end=end
        self.count=count
        self.left,self.right=None,None

    
        
        
        
class Solution:
    """
    @param A: An integer array
    @param queries: The query list
    @return: The number of element in the array that are smaller that the given integer
    """
    def countOfSmallerNumber(self, A, queries):
        root=self.build(0, 10000)
        res=[]
        
        for x in A:
            self.modify(root,x, 1)
        
        for i in queries:
            count=0
            if i>0:
                count=self.query(root,0,i-1)
            res.append(count)
        return res
        
        
    def build(self,start,end):
        if start>=end:
            return SegmentTree(start,end,0)
       
        root=SegmentTree(start,end,0)
        mid=(start+end)//2
        root.left=self.build(start,mid)
        root.right=self.build(mid+1,end)
        
        return root
    def modify(self,root, index, value):
        if root.start==index and root.end==index:
            root.count+=value
            return 
        mid=(root.start+root.end)//2
        if mid>=index and root.start<=index:
            self.modify(root.left,index,value)
        if mid<index and root.end >=index:
            self.modify(root.right,index,value)
        root.count=root.left.count+root.right.count
    def query(self,root,start,end):
        if root.start>end or root.end<start:
            return 0#make it big so that this one is not select
        if root.start>=start and root.end<=end:
            return root.count
        return self.query(root.left,start,end)+self.query(root.right,start,end)
        
A=[1,2,7,8,5]   
queries=[1,8,5]    
A=[1,2,3,4,5,6]
queries=[1,2,3,4]     
if __name__ == "__main__":
    print(Solution().countOfSmallerNumber( A, queries)) 


         
        
#249. Count of Smaller Number before itself
        

       
class SegmentTree(object):
    def __init__(self,start,end,count=0):
        self.start=start
        self.end=end
        self.count=count
        self.left,self.right=None,None
class Solution:
    """
    @param A: an integer array
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def countOfSmallerNumberII(self, A):
        # write your code here
        if not A:
            return []
        root=self.build(0, max(A))
        res=[]
        for i in A:
            count=0
            if i>0:
                count=self.query(root,0,i-1)
            res.append(count)
            self.modify(root,i,1)
        return res
        
        
    def build(self,start,end):
        if start>=end:
            return SegmentTree(start,end,0)
       
        root=SegmentTree(start,end,0)
        mid=(start+end)//2
        root.left=self.build(start,mid)
        root.right=self.build(mid+1,end)
        
        return root
    def modify(self,root, index, value):
        if root.start==index and root.end==index:
            root.count+=value
            return 
        mid=(root.start+root.end)//2
        if mid>=index and root.start<=index:
            self.modify(root.left,index,value)
        if mid<index and root.end >=index:
            self.modify(root.right,index,value)
        root.count=root.left.count+root.right.count
    def query(self,root,start,end):
        if root.start>end or root.end<start:
            return 0#make it big so that this one is not select
        if root.start>=start and root.end<=end:
            return root.count
        return self.query(root.left,start,end)+self.query(root.right,start,end)        
        
        
A=[1,2,7,8,5]   
if __name__ == "__main__":
    print(Solution().countOfSmallerNumberII(A))          
                       
            
#249. Count of Smaller Number before itself           
class Solution:
    """
    @param A: an integer array
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def countOfSmallerNumberII(self, A):
        # write your code here
        if not A:
            return []
        import bisect
        
        s=[]
        res=[]
        
        for a in A:
            i=bisect.bisect_left(s,a)
            res.append(i)
            bisect.insort(s,a)
        return res
A=[1,2,7,8,5]  
if __name__ == "__main__":
    print(Solution().countOfSmallerNumberII(A))   



#254. Drop Eggs
class Solution:
    """
    @param n: An integer
    @return: The sum of a and b
    """
    def dropEggs(self, n):
        # write your code here
#https://www.quora.com/You-have-2-eggs-You-are-on-a-100-floor-building-You-drop-the-egg-from-a-particular-floor-It-breaks-or-survives-If-it-survives-you-can-throw-the-same-egg-from-a-higher-floor-How-many-attempts-do-you-need-to-identify-the-max-floor-at-which-the-egg-doesnt-break-when-thrown-down
        for i in range(2*n):
            if i*i+i>=2*n:
                return i
n=100           
if __name__ == "__main__":
    print(Solution().dropEggs( n))          
                        
#360. Sliding Window Median        
class Solution:
    """
    @param nums: A list of integers
    @param k: An integer
    @return: The median of the element inside the window at each moving
    """
    def medianSlidingWindow(self, nums, k):
        # write your code here
        win=nums[:k]
        win.sort()
        median=[]
        import bisect
        for a,b in zip(nums,nums[k:]+[0]):
            print(a,b)
            median.append(win[(k-1)//2])
            #win.remove(a)   TLE   用  POP 比remove 快
            win.pop(bisect.bisect_left(win,a))
            bisect.insort(win,b)
        return median
       
nums=[1,2,7,8,5]  
k = 3        
if __name__ == "__main__":
    print(Solution().medianSlidingWindow(nums, k))         
        
        
#360. Sliding Window Median        
class Solution:
    """
    @param nums: A list of integers
    @param k: An integer
    @return: The median of the element inside the window at each moving
    """
    def medianSlidingWindow(self, nums, k):
        lh=[]
        rh=[]
        rv=[]
        if not nums:
            return []
            
        if k==1:
            return nums
        import heapq
        n=len(nums)
        for i,x in enumerate(nums[:k]):
            heapq.heappush(lh,(-x,i))
        
        for _ in range(k//2):
            heapq.heappush(rh,(-lh[0][0],lh[0][1]))
            heapq.heappop(lh)
        
        
        for i,x in enumerate(nums[k:]):
            rv.append(-lh[0][0])
            if  x>=rh[0][0]:
                heapq.heappush(rh,(x,i+k))
                if nums[i]<=rh[0][0]:
                   
                   heapq.heappush(lh,(-rh[0][0],rh[0][1]))
                   heapq.heappop(rh)
                else:
                    pass
            else:
                heapq.heappush(lh,(-x,i+k))
                if nums[i]>=rh[0][0]:
                   heapq.heappush(rh,(-lh[0][0],lh[0][1]))
                   heapq.heappop(lh)
                else:
                    pass
            while lh and lh[0][1]<=i:
                heapq.heappop(lh)
            while rh and rh[0][1]<=i:
                heapq.heappop(rh)
        rv.append(-lh[0][0])
        return rv

nums=[1,2,7,7,2]
k=1
nums=[1,3,-1,-3,5,3,6,7] 
k = 3     
if __name__ == "__main__":
    print(Solution().medianSlidingWindow(nums, k))             
        

#362. Sliding Window Maximum 
class Solution:
    """
    @param: nums: A list of integers
    @param: k: An integer
    @return: The maximum number inside the window at each moving
    """
    def maxSlidingWindow(self, nums, k):
        # write your code here  
        wh=[]
        
        res=[]
        if not nums:
            return []
            
        if k==1:
            return nums
        import heapq
        n=len(nums)
        for i,x in enumerate(nums[:k]):
            heapq.heappush(wh,(-x,i))
        
        for i, x in enumerate(nums[k:]):
            res.append(-wh[0][0])
            heapq.heappush(wh,(-x,i+k))
            while wh and wh[0][1]<=i:
                heapq.heappop(wh)
        #print(wh)
        res.append(-wh[0][0])
        return res        
            
nums=[1,2,7,7,8]
k=3
if __name__ == "__main__":
    print(Solution().maxSlidingWindow( nums, k))  


#362. Sliding Window Maximum 
class Solution:
    """
    @param: nums: A list of integers
    @param: k: An integer
    @return: The maximum number inside the window at each moving
    """
    def maxSlidingWindow(self, nums, k):
        # write your code here  
        
        
        res=[]
        if not nums:
            return []
            
        if k==1:
            return nums
        from collections import deque
        dq=deque()
        
        for i in range(len(nums)):
            if dq and dq[0]<i-k+1:
                dq.popleft()
            while dq and nums[dq[-1]]<nums[i]:
                dq.pop()
            dq.append(i)
            
            if i>k-2:
                res.append(nums[dq[0]])
        return res
                
nums=[1,2,7,7,8]
k=3
if __name__ == "__main__":
    print(Solution().maxSlidingWindow( nums, k))                        
                    
#363. Trapping Rain Water
class Solution:
    """
    @param heights: a list of integers
    @return: a integer
    """
    def trapRainWater(self, heights):
        # write your code here
#https://leetcode.com/problems/trapping-rain-water/discuss/17395/A-different-O(n)-approach-easy-to-understand-and-simple-code        


#scan A both from left to right and right to left, record the largest seen during the scan; then for each position the water level should be the min of the 2 large value.        
        
        n=len(heights)
        if n==0:
            return 0
        ltr=[0 for _ in range(n)]
        rtl=[0 for _ in range(n)]
        ltr[0]=heights[0]
        rtl[-1]=heights[-1]
        res=0
        for i in range(1,n):
            ltr[i]=max(ltr[i-1],heights[i])
        for i in range(n-2,-1,-1):
            rtl[i]=max(rtl[i+1],heights[i])
        for i in range(1,n-1):
            res+=min(ltr[i],rtl[i])-heights[i]
        return res
heights=[0,1,0,2,1,0,1,3,2,1,2,1]
if __name__ == "__main__":
    print(Solution().trapRainWater(heights))             
        

#364. Trapping Rain Water II
class Solution:
    """
    @param heights: a matrix of integers
    @return: an integer
    """
    def trapRainWater(self, heights):
        # write your code here
#https://leetcode.com/problems/trapping-rain-water-ii/discuss/89466/python-solution-with-heap        
#https://leetcode.com/problems/trapping-rain-water-ii/discuss/89472/Visualization-No-Code
        m=len(heights)     
        n=len(heights[0])
        
        import heapq
        heap=[]
        
        visited=[[0 for _ in range(n)] for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                if i==0 or i==m-1 or j==0 or j==n-1:
                    heapq.heappush(heap,(heights[i][j],i,j))
                    visited[i][j]=1
        res=0
        
        while heap:
            height,i,j=heapq.heappop(heap)
            
            for x,y in ((i-1,j) ,(i+1,j),(i,j-1),(i,j+1)):
                if 0<=x<m  and 0<=y<n and visited[x][y]==0:
                    res+= max(0,height-heights[x][y])
                    heapq.heappush(heap,(     max(heights[x][y],height), x,y))
                    visited[x][y]=1
        return res
heights=[[12,13,0,12],[13,4,13,12],[13,8,10,12],[12,13,12,12],[13,13,13,13]]
heights=[
  [1,4,3,1,3,2],
  [3,2,1,3,2,4],
  [2,3,3,2,3,1]
]                    
if __name__ == "__main__":
    print(Solution().trapRainWater(heights))               
        
#365. Count 1 in Binary
class Solution:
    """
    @param: num: An integer
    @return: An integer
    """
    def countOnes(self, num):
        # write your code here  
        #n & (n - 1) drops the lowest set bit. It's a neat little bit trick.
        #Let's use n = 00101100 as an example. This binary representation has three 1s.

        #If n = 00101100, then n - 1 = 00101011, so n & (n - 1) = 00101100 & 00101011 = 00101000. Count = 1.

        #If n = 00101000, then n - 1 = 00100111, so n & (n - 1) = 00101000 & 00100111 = 00100000. Count = 2.

        #If n = 00100000, then n - 1 = 00011111, so n & (n - 1) = 00100000 & 00011111 = 00000000. Count = 3.

        #n is now zero, so the while loop ends, and the final count (the numbers of set bits) is returned.
#        count=0
#        while num !=0:
#            num=num&(num-1)
#            count+=1
#        return count 在 Python里不能，因为Python不止32位，在JAVA 里可以
        
        count=0
        mask=1
        for _ in range(32):
            if num&mask !=0:
                count+=1
            mask=mask<<1
        return count
        
            
#366. Fibonacci
class Solution:
    """
    @param n: an integer
    @return: an ineger f(n)
    """
    def fibonacci(self, n):
        # write your code here
        #0, 1, 1, 2, 3, 5, 8, 13, 21, 34 ...
#        Given 1, return 0
#Given 2, return 1
#Given 10, return 34
        a=0
        b=1
        if n==1:
            return 0
        if n==2 or n==3:
            return 1
        for _ in range(n-2):
            a,b=b,a+b
            print(a,b)
        return b

n=9          
if __name__ == "__main__":
    print(Solution().fibonacci( n))   

#367. Expression Tree Build            
"""
Definition of ExpressionTreeNode:
class ExpressionTreeNode:
    def __init__(self, symbol):
        self.symbol = symbol
        self.left, self.right = None, None
"""

class myTreeNode():
      def __init__(self,val,s):
          self.left=None
          self.right=None
          self.val=val
          self.exp_node=ExpressionTreeNode(s)
          
      def __str__(self):
        return str(self.val) + '  '+ self.exp_node.symbol
class Solution:
    """
    @param: expression: A string array
    @return: The root of expression tree
    """
    def build(self, expression):
        # write your code here
        root=self.create_tree(expression)
        return self.copy_tree(root)
        
#https://zhengyang2015.gitbooks.io/lintcode/expression_tree_build_367.html        
#观察example，可以看出所有叶节点都为数字。如果给每个元素赋予一个优先级， 和 ／ 为2， ＋ 和 － 为1，
#数字为极大值，然后规定优先级越大的越在下，越小的越在上。这样，这道题就转化为构建*Min Tree，
#和之前的Max Tree做法类似，只是这里维持的是一个递增栈。同时，当遇见“（”时，提高优先级，遇见“）”时，
#降低优先级。
#遍历数组，给每个新来的元素赋予一个val值用以比较优先级。 * 和 ／ 为2， ＋ 和 － 为1， 数字为极大值。
#此时看栈顶元素（若栈为空则直接加入）。为了维持一个递增栈，若栈顶元素比新来元素val大（或相等），则出栈；
#若栈顶元素比新来元素val小，则break。
#若2中栈顶元素出栈，此时若栈为空，则将出栈元素作为新来元素的左节点，并将新来元素加入栈中；若不为空，
#看新栈顶元素，若新栈顶元素比新来元素val小，则将出栈元素作为新来元素的左孩子，并将新来元素加入栈中；
#若新栈顶元素比新来元素val大（或相等），则将出栈元素作为新栈顶元素的右节点，重复2-3，直到栈为空或者栈顶元素
#比新来元素要小，将新来元素加入栈中。
#
#tips：在遍历万整个数组后，多加一个值，将其val赋值为极小，这样所有元素都会出栈并构建成完整的树。  
    def get_val(self,a,base):
        if a=='+' or a=='-':
            if base==float('inf'):
                return base
            return base+1
        elif a=='/' or a=='*':
            if base==float('inf'):
                return base
            return base+2
        return float('inf')
    
    def create_tree(self,expression):
        stack=[]
        base=0
        for i in range(len(expression)):
            if expression[i]=='(':
                if base!=float('inf'):
                    base+=10
                continue
            elif expression[i]==')':
                if base!=float('inf'):
                    base-=10
                continue
            val=self.get_val(expression[i],base)
            node= myTreeNode(val,expression[i])
            print( 'val-->', val,'expression[i]-->', expression[i])
            while stack and node.val <= stack[-1].val:
                print('starting********')
                for i in range(len(stack)):
                      print(stack[i],'   ',stack[i].left,'   ',stack[i].right)
                print('ending********')
                node.left=stack.pop()
                
            if stack:
                stack[-1].right=node
            stack.append(node)
        if not stack:
            return None
        
        return stack[0]
    
    def  copy_tree(self,root):
        if not root:
            return None
        root.exp_node.left=self.copy_tree(root.left)
        root.exp_node.right=self.copy_tree(root.right)
        return root.exp_node
    #def myprint(self,node):
        
  
    
expression=["2","*","6","-","(","23","+","7",")","/","(","1","+","2",")"]          
if __name__ == "__main__":
    print(Solution().build( expression))               
                        

#368. Expression Evaluation
class ExpressionTreeNode:
    def __init__(self, symbol):
        self.symbol = symbol
        self.left, self.right = None, None

class myTreeNode():
    def __init__(self,val,s):
          self.left=None
          self.right=None
          self.val=val
          self.exp_node=ExpressionTreeNode(s)

class Solution:
    """
    @param expression: a list of strings
    @return: an integer
    """
    def build(self, expression):
        # write your code here
        root=self.create_tree(expression)
        return self.copy_tree(root)
    def get_val(self,a,base):
        if a=='+' or a=='-':
            if base==float('inf'):
                return base
            return base+1
        elif a=='/' or a=='*':
            if base==float('inf'):
                return base
            return base+2
        return float('inf')
    
    def create_tree(self,expression):
        stack=[]
        base=0
        for i in range(len(expression)):
            if expression[i]=='(':
                if base!=float('inf'):
                    base+=10
                continue
            elif expression[i]==')':
                if base!=float('inf'):
                    base-=10
                continue
            val=self.get_val(expression[i],base)
            node= myTreeNode(val,expression[i])
            
            while stack and node.val <= stack[-1].val:
                node.left=stack.pop()
            if stack:
                stack[-1].right=node
            stack.append(node)
        if not stack:
            return None
        
        return stack[0]
    
    def  copy_tree(self,root):
        if not root:
            return None
        root.exp_node.left=self.copy_tree(root.left)
        root.exp_node.right=self.copy_tree(root.right)
        return root.exp_node
    
    
    
    def evaluateExpression(self, expression):
        if not  expression:
            return 0
        root=self.build( expression)
        if not root:
            return 0
        def calculate(root):
            if root.symbol not in ('+','-','*','/'):
                return int(root.symbol)
            else:
                if root.symbol=='+':
                    return calculate(root.left)+calculate(root.right)
                if root.symbol=='-':
                    return calculate(root.left)-calculate(root.right)
                if root.symbol=='*':
                    return calculate(root.left)*calculate(root.right)
                if root.symbol=='/':
                    return calculate(root.left)//calculate(root.right)
        return calculate(root)
expression=["2","*","6","-","(","23","+","7",")","/","(","1","+","2",")"]  
expression=[]  
expression=["(","(","(","(","(",")",")",")",")",")"]     
if __name__ == "__main__":
    print(Solution().evaluateExpression( expression))              
        
        

#368. Expression Evaluation
class Solution:
#https://www.jiuzhang.com/solution/expression-evaluation/#tag-other-lang-python
    P={'+':1,'-':1,'*':2,'/':2}
    """
    @param expression: a list of strings
    @return: an integer
    """
    def evaluateExpression(self, expression):
        
        new_e=self.dal2rpn(expression)
        if not new_e:
            return 0
        res=self.eval_rpn(new_e)
        return res

#先将对人友好的中缀表达式 (即 1 + 2)，转为对计算机友好的后缀表达式 (即 1 2 +)
#在用栈做运算
#
#REF: Reverse Polish Notation
    def dal2rpn(self,expression):
        stack=[]
        res=[]
        
        for char in expression:
            if char.isdigit():
                res.append(char)
                continue
            elif char in self.P:
                while stack and stack[-1] in self.P and self.P[stack[-1]]>=self.P[char]:
                    res.append(stack.pop())
                stack.append(char)
            elif char=='(':
                stack.append(char)
            elif char==')':
                while stack and stack[-1]!='(':
                      res.append(stack.pop())
                stack.pop()
        while stack:
            res.append(stack.pop())
        return res
   
        
        
    def eval_rpn(self, expression):
        stack=[]
        
        for char in expression:
            if char.isdigit():
                stack.append(int(char))
                continue
            
            
            b=stack.pop()
            a=stack.pop()
            if char =='+':
                stack.append(a+b)
            elif char =='-':
                stack.append(a-b)
            elif char =='*':
                stack.append(a*b)
            elif char =='/':
                stack.append(a//b)
        return stack[0]
expression=["2","*","6","-","(","23","+","7",")","/","(","1","+","2",")"]  
expression=[]  
expression=["(","(","(","(","(",")",")",")",")",")"]     
if __name__ == "__main__":
    print(Solution().evaluateExpression( expression))              
                        
        
#370. Convert Expression to Reverse Polish Notation        
class Solution:
    P={'+':1,'-':1,'*':2,'/':2}
    """
    @param expression: A string array
    @return: The Reverse Polish notation of this expression
    """
    def convertToRPN(self, expression):
        # write your code here
        stack=[]
        res=[]
        
        for char in expression:
            if char.isdigit():
                res.append(char)
                continue
            elif char in self.P:
                while stack and stack[-1] in self.P and self.P[stack[-1]]>=self.P[char]:
                    res.append(stack.pop())
                stack.append(char)
            elif char=='(':
                stack.append(char)
            elif char==')':
                while stack and stack[-1]!='(':
                      res.append(stack.pop())
                stack.pop()
        while stack:
            res.append(stack.pop())
        return res
expression=["3","-","4","+","5"]
if __name__ == "__main__":
    print(Solution().convertToRPN( expression))


#371. Print Numbers by Recursion
class Solution:
    """
    @param n: An integer
    @return: An array storing 1 to the largest number with n digits.
    """
    def numbersByRecursion(self, n):
        # write your code here
        res=[]
        def myprint(n,res):
            
            if n<1:
                return []
            
            myprint(n-1,res)
     
            begin=10**(n-1)
            end=10**n
            for i in range(begin,end):
                res.append(i)
                
            
        myprint(n,res)
                
        return res
#Given N = 1, return [1,2,3,4,5,6,7,8,9].
#
#Given N = 2, return [1,2,3,4,5,6,7,8,9,10,11,12,...,99]




n=1  
n=2
n=3   
n=5     
if __name__ == "__main__":
    print(Solution().numbersByRecursion( n))

#372. Delete Node in a Linked List
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""


class Solution:
    """
    @param: node: the node in the list should be deletedt
    @return: nothing
    """
    def deleteNode(self, node):
        # write your code here
       
        if node.next==None:
               node=None    
       
        node.val=node.next.val
        node.next=node.next.next
            
        
#Linked list is 1->2->3->4, and given node 3, delete the node in place 1->2->4

head= ListNode(1)
head.next=ListNode(2)
head.next.next=ListNode(3)
head.next.next.next=ListNode(4)
node=head.next.next
#1->2->3->4
if __name__ == "__main__":
    print(Solution().deleteNode(node))        


    
    
#373. Partition Array by Odd and Even
class Solution:
    """
    @param: nums: an array of integers
    @return: nothing
    """
    def partitionArray(self, nums):
        # write your code here
        n=len(nums)
        
        l=0
        r=n-1
        while l<r:
            while l<r and nums[l]%2==1:
                l+=1
            while l<r and nums[r]%2==0:
                r-=1
            
            nums[l],nums[r]=nums[r],nums[l]
        print(nums)
nums=[1, 2, 3, 4]   

nums=[] 
nums=[2]
nums=[2,4,6,8]
if __name__ == "__main__":
    print(Solution().partitionArray( nums))        
    
    
#374. Spiral Matrix
class Solution:
    """
    @param matrix: a matrix of m x n elements
    @return: an integer list
    """
    def spiralOrder(self, matrix):
        # write your code here
        
#https://www.jiuzhang.com/solution/spiral-matrix/#tag-highlight
        
        m=len(matrix)
        if m==0:
            return []
        n=len(matrix[0])
        
        res=[]
        
        dr=[0,1,0,-1]
        dc=[1,0,-1,0]
      
        d=0
        
        x=0
        y=0
        for _ in range(m*n):
                res.append(matrix[x][y])
                matrix[x][y]=-1
                nx=x+dr[d]
                ny=y+dc[d]
                if nx<0 or nx>=m or  ny<0 or ny>=n or  matrix[nx][ny]  ==-1:
                    d=(d+1)%4
                    nx=x+dr[d]
                    ny=y+dc[d]
                x=nx
                y=ny
        return res

 
matrix=[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]

[1,2,3,6,9,8,7,4,5]    
if __name__ == "__main__":
    print(Solution().spiralOrder( matrix))            
    
    
#375. Clone Binary Tree
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of binary tree
    @return: root of new tree
    """
    def cloneTree(self, root):
        # write your code here
        if not root:
            return root
        
        new_root=TreeNode(root.val)
        if root.left:
            new_root.left=self.cloneTree(root.left)
        if root.right:
            new_root.right=self.cloneTree(root.right)
        return new_root
        
#376. Binary Tree Path Sum
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    """
    @param: root: the root of binary tree
    @param: target: An integer
    @return: all valid paths
    """
    def binaryTreePathSum(self, root, target):
        # write your code here
        if not root:
            return []
        
        def search(target,path,res,node):
            if target==0 and not node.left and not node.right:
                print(path)
                res.append(path[:])
                return 

            if node.left:
               search(target-node.left.val,path+[node.left.val],res,node.left)
            if node.right:
               search(target-node.right.val,path+[node.right.val],res,node.right)
      
            
        res=[]
        search(target-root.val,[root.val],res,root)
        return res

#      
#    
#     1
#    / \
#   2   4
#  / \
# 2   3    
#    
#[
#  [1, 2, 2],
#  [1, 4]
#]    
    
                  
root=    TreeNode(1)  
root.left= TreeNode(2)  
root.right= TreeNode(4)  
root.left.left= TreeNode(2)
root.left.right= TreeNode(3)
target=5                
if __name__ == "__main__":
    print(Solution().binaryTreePathSum(root, target)) 

#378. Convert Binary Search Tree to Doubly Linked List
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
Definition of Doubly-ListNode
class DoublyListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = self.prev = next
"""


class Solution:
    """
    @param: root: The root of tree
    @return: the head of doubly list node
    """
    def bstToDoublyList(self, root):
        # write your code here
        if not root:
            return  root
        res=[]
        def inOrder(node,res):
            if not node:
                return 
            inOrder(node.left,res)
            res.append(node.val)
            inOrder(node.right,res)
        inOrder(root,res)
        
        root=DoublyListNode(res[0])
        cur=root
        prev=None
        i=1
        while i<len(res):
             cur.next=DoublyListNode(res[i])
             cur.prev=prev
             
             i+=1
             prev=cur
             cur=cur.next
        cur.prev=prev
        return root
    
root=    TreeNode(1)  
root.left= TreeNode(2)  
root.right= TreeNode(4)  
root.left.left= TreeNode(2)
root.left.right= TreeNode(3)  
# 1<->2<->3<->4<->5      
if __name__ == "__main__":
    print(Solution().bstToDoublyList( root)) 
        
            
#379. Reorder array to construct the minimum number

class Solution:
    """
    @param nums: n non-negative integer array
    @return: A string
    """
    def minNumber(self, nums):
        # write your code here
#        n=len(nums)
#        if n<2:
#            return nums
#        nums=[str(x) for x in nums]
#        class compare(str):
#            def __lt__(x,y):
#              return x+y<y+x
#        
#        nums.sort(key=compare)
#        
#        return ''.join(nums)
        
        #Python 2
        nums.sort(cmp=self.cmp)
        res=''.join([str(x) for x in nums])
        i=0
        while i+1<len(res):
            if res[i]!='0':
                break
            i+=1
        return res[i:]
            
        
        
        
        
    def cmp(self,x,y) :
        if str(x)+str(y)<str(y)+str(x):
            return -1
        elif str(x)+str(y)==str(y)+str(x):
            return 0
        else:
            return 1
        
nums=[3, 32, 321]                
if __name__ == "__main__":
    print(Solution().minNumber( nums)) 

#380. Intersection of Two Linked Lists
"""
Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
"""


class Solution:
    """
    @param: headA: the first list
    @param: headB: the second list
    @return: a ListNode
    """
    def getIntersectionNode(self, headA, headB):
        # write your code here
        
        curA=headA
        curB=headB
        
        while curA and curB:
            curA=curA.next
            curB=curB.next
        if curA:
            curB=headA
            while curA  and curB:
                curA=curA.next
                curB=curB.next
            curA=headB
            while curA  and curB:
                  if curA == curB:                      
                      return curA
                  curA=curA.next
                  curB=curB.next
        if curB:
            curA=headB
            while curA  and curB:
                curA=curA.next
                curB=curB.next
            curB=headA
            while curA  and curB:
                  if curA == curB:                      
                      return curA
                  curA=curA.next
                  curB=curB.next
                
            
#381. Spiral Matrix II
class Solution:
    """
    @param n: An integer
    @return: a square matrix
    """
    def generateMatrix(self, n):
        # write your code here
        
        
        dr=[0,1,0,-1]
        dc=[1,0,-1,0]
        x=0
        y=0
        d=0
        
       
        
        res=[[-1 for _ in range(n)]  for _ in range(n)]
        
        for i in range(n*n):
            res[x][y]=i+1
            nx=x+dr[d]
            ny=y+dc[d]
            if nx<0 or nx>=n or ny<0 or ny>=n or res[nx][ny]!=-1:
                d=(d+1)%4
                nx=x+dr[d]
                ny=y+dc[d]
            x=nx
            y=ny
        return res
        
#[
#  [ 1, 2, 3 ],
#  [ 8, 9, 4 ],
#  [ 7, 6, 5 ]
#] 

n=9       
if __name__ == "__main__":
    print(Solution(). generateMatrix( n))        
        
        
        
        
        
#382. Triangle Count        
class Solution:
    """
    @param S: A list of integers
    @return: An integer
    """
    def triangleCount(self, S):
        # write your code here
        
        n=len(S)
        if n<3:
            return []
#        def dfs(S,res,path):
#            if len(path)==3:
#                if path[0]+path[1]>path[2]:
#                    self.res+=1
#                    return 
#            elif len(path)<3:
#                for i in range(len(S)):
#                    
#                    dfs(S[i+1:],self.res,path+[S[i]])
#        self.res=0
#        S.sort()
#        dfs(S,self.res,[])
#        return self.res
        S.sort()
        count=0
        for i in range(2,n):
            left=0
            right=i-1
            
            while left < right:
                if S[left]+S[right]>S[i]:
                    count+=right-left
                    right-=1
                else:
                    left+=1
        return count
                
                
                
S = [3,4,6,7]     
S = [4,4,4,4] 

if __name__ == "__main__":
    print(Solution().triangleCount( S))         
        
#383. Container With Most Water        
class Solution:
    """
    @param heights: a vector of integers
    @return: an integer
    """
    def maxArea(self, heights):
        # write your code here
        if not heights:
            return 0
        n=len(heights)
        
        l=0
        r=n-1
        ans=0
        while l<r:
            
            if heights[l]<heights[r]:
                area=heights[l]*(r-l)
                l+=1
            else:
                area=heights[r]*(r-l)
                r-=1
            ans=max(ans,area)
        return ans
heights=[1,3,2]    
if __name__ == "__main__":
    print(Solution().maxArea(heights))                   
        
        
#384. Longest Substring Without Repeating Characters
class Solution:
    """
    @param s: a string
    @return: an integer
    """
    def lengthOfLongestSubstring(self, s):
        # write your code here
        if not s:
            return 0
        
        left=0
        last={}
        
        ans=0
        
        for i in range(len(s)):
            if s[i] in last and last[s[i]]>=left:
               
                left=last[s[i]]+1
            last[s[i]]=i
            ans=max(ans,i-left+1)
        return ans
s='abcabcbb'
s='ggggg'
if __name__ == "__main__":
    print(Solution().lengthOfLongestSubstring( s)) 
    
    
    
#385. ArrayList
class ArrayListManager:
    '''
     * @param n: You should generate an array list of n elements.
     * @return: The array list your just created.
    '''
    def create(self, n):
        # Write your code here
        list1=[]
        for i in range(n):
            list1.append(i)
        return list1       
    
    
    '''
     * @param list: The list you need to clone
     * @return: A deep copyed array list from the given list
    '''
    def clone(self, l):
        # Write your code here
        clist=[]
        for i in l:
            clist.append(i)
        return clist
            
    
    
    '''
     * @param list: The array list to find the kth element
     * @param k: Find the kth element
     * @return: The kth element
    '''
    def get(self, l, k):
        # Write your code here
        return l[k]
    
    
    '''
     * @param list: The array list
     * @param k: Find the kth element, set it to val
     * @param val: Find the kth element, set it to val
    '''
    def set(self, l, k, val):
        # write your code here
        l[k]=val
    
    
    '''
     * @param list: The array list to remove the kth element
     * @param k: Remove the kth element
    '''
    def remove(self, l, k):
        # write tour code here
        l.remove(k)
        
   
    
    '''
     * @param list: The array list.
     * @param val: Get the index of the first element that equals to val
     * @return: Return the index of that element
    '''
    def indexOf(self, l, val):
        # Write your code here
        if not l:
            return -1
        
        try:
            ans=l.index(val)
        except   ValueError:
                ans=-1
        return ans
    
       
#386. Longest Substring with At Most K Distinct Characters
class Solution:
    """
    @param s: A string
    @param k: An integer
    @return: An integer
    """
    def lengthOfLongestSubstringKDistinct(self, s, k):
        # write your code here
        left=0
        n=len(s)
        if n==0:
            return 0
        
        count=[0 for _ in range(256)]
        ans=0
        distinct=0
        for i in range(n):
            if count[ord(s[i])]==0:
                distinct+=1
            count[ord(s[i])]+=1
            while distinct>k:
                count[ord(s[left])]-=1
                if  count[ord(s[left])]==0:
                    distinct-=1
                left+=1
            ans=max(ans,i-left+1)
        return ans
                    
                
            
#        from collections import Counter
#        res=0
#        for i in range(n):
#            
#            WC=Counter(s[left:i+1])
#            
#            while len(WC)>k:                
#                
#                WC[s[left]]-=1
#                
#                if WC[s[left]]==0:
#                    del WC[s[left]]
#                left+=1
#            
#            res=max(res,i-left+1)
#        return res
s = "ecccebba"
k = 3
if __name__ == "__main__":
    print(Solution().lengthOfLongestSubstringKDistinct( s, k))            
                 
#387. The Smallest Difference
class Solution:
    """
    @param A: An integer array
    @param B: An integer array
    @return: Their smallest difference.
    """
    def smallestDifference(self, A, B):
        # write your code here
        AA=[(a,0) for a in A]
        BB=[(b,1) for b in B]
        
        AABB= sorted(AA+BB)
        
        n=len(AABB)
        
        
        
        
        i=1
        ans=float('inf')
        while i<n:
            if AABB[i][1]!=AABB[i-1][1]:
                ans=min(ans, abs(AABB[i][0]-AABB[i-1][0]))
            i+=1
        return ans
      
        
A = [3,6,7,4]
B = [2,8,9]        
            
if __name__ == "__main__":
    print(Solution().smallestDifference(A, B))                  
                
  
#388. Permutation Sequence        
class Solution:
    """
    @param n: n
    @param k: the k th permutation
    @return: return the k-th permutation
    """
    def getPermutation(self, n, k):
        # write your code here
        s=[ i for i in range(1,n+1)]
        per=[]
        
        fac=[1]
        for i in range(2,n+1):
            fac.append(fac[-1]*i)
            
        kindex=k-1
            
        for i in range(n):
            
           index=  kindex//fac[n-2-i]
           per.append(s[index])
           s.pop(index)
           kindex=kindex%fac[n-2-i]
        return ''.join([str(x) for x in per])
           
        
n=3
k=4
"123"
"132"
"213"
"231"
"312"
"321"      
if __name__ == "__main__":
    print(Solution().getPermutation( n, k))          
        
  
#389. Valid Sudoku
class Solution:
    """
    @param board: the board
    @return: whether the Sudoku is valid
    """
    def isValidSudoku(self, board):
        # write your code here
        
        m=len(board)
        n=len(board[0])
        sumB=[]
        for i,r in enumerate(board):
            for j,c in enumerate(r):
                if c!='.':
                    sumB+=[(i,c),(c,j),(i//3,j//3,c)]
        return len(sumB)==len(set(sumB))
                    
    
#391. Number of Airplanes in the Sky    
"""
Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    """
    @param airplanes: An interval array
    @return: Count of airplanes are in the sky.
    """
    def countOfAirplanes(self, airplanes):
        # write your code here
        if not airplanes:
            return 0
        events=[]
        for interval in airplanes:
            events+=[(interval.start,'OPEN') , (interval.end,'END')]
        events.sort()
        
        
        res=0
        n=0
        for  x,status in events:
            if status =='OPEN':
                n+=1
                res=max(res,n)
            else:
                n-=1
        return res
            
 
airplanes=[
  (1,10),
  (2,3),
  (5,8),
  (4,7)
] 

#392. House Robber
class Solution:
    """
    @param A: An array of non-negative integers
    @return: The maximum amount of money you can rob tonight
    """
    def houseRobber(self, A):
        # write your code here
        n=len(A)
        if n==0:
            return 0
        if n==1:
            return A[0]
        if n==2:
            return max(A)
        dp=[0 for _ in range(n)]
        dp[0]=A[0]
        dp[1]=max(A[0],A[1])
        
        for i in range(2,n):
            dp[i]=max(dp[i-1],dp[i-2]+A[i])
        return dp[n-1]
A=[3, 8, 4]
if __name__ == "__main__":
    print(Solution().houseRobber( A))         


#393. Best Time to Buy and Sell Stock IV
class Solution:
    """
    @param K: An integer
    @param prices: An integer array
    @return: Maximum profit
    """
    def maxProfit(self, K, prices):
        # write your code here

#Given prices = [4,4,6,1,1,4,2,5], and k = 2, return 6
#https://soulmachine.gitbooks.io/algorithm-essentials/java/dp/best-time-to-buy-and-sell-stock-iv.html        
#设两个状态，global[i][j] 表示i天前最多可以进行j次交易的最大利润，local[i][j]表示i天前最多
#可以进行j次交易，且在第i天进行了第j次交易的最大利润。状态转移方程如下：
#local[i][j] = max(global[i-1][j-1] + max(diff,0), local[i-1][j]+diff)
#global[i][j] = max(local[i][j], global[i-1][j])
#关于global的状态转移方程比较简单，不断地和已经计算出的local进行比较，把大的保存在global中。
#关于local的状态转移方程，取下面二者中较大的一个：
#全局前i-1天进行了j-1次交易，然后然后加上今天的交易产生的利润（如果赚钱就交易，不赚钱就不交易
#，什么也不发生，利润是0）
#局部前i-1天进行了j次交易，然后加上今天的差价（local[i-1][j]是第i-1天卖出的交易，它加
#上diff后变成第i天卖出，并不会增加交易次数。无论diff是正还是负都要加上，否则就不满足
#local[i][j]必须在最后一天卖出的条件了）
#注意，当k大于数组的大小时，上述算法将变得低效，此时可以改为不限交易次数的方式解决，
#即等价于 "Best Time to Buy and Sell Stock II"。        

        n=len(prices)
        if K<1 or n<2:
            return 0
        profit=0
        if K>=n//2:
            
            for i in range(1,n):
                profit+=max(0,prices[i]-prices[i-1])
            return profit
                
        
        localmax=[[0 for _ in range(K+1)] for _ in range(n)]
        globalmax=[[0 for _ in range(K+1)] for _ in range(n)]
        
        for i in range(1,n):
            diff=prices[i]-prices[i-1]
            for j in range(1,K+1):
                
                localmax[i][j]= max(globalmax[i-1][j-1]+max(0,diff),localmax[i-1][j]+diff)
                globalmax[i][j]=max(globalmax[i-1][j],localmax[i][j])
        return globalmax[n-1][K]
    
class Solution:
    """
    @param K: An integer
    @param prices: An integer array
    @return: Maximum profit
    """
    def maxProfit(self, K, prices):
        # write your code here
        n=len(prices)
        if K<1 or n<2:
            return 0
        profit=0
        if K>=n//2:
            
            for i in range(1,n):
                profit+=max(0,prices[i]-prices[i-1])
            return profit
                
        
        localmax=[[0 for _ in range(n)] for _ in range(K+1)]
        globalmax=[[0 for _ in range(n)] for _ in range(K+1)]
        
        for i in range(1,K+1):
            for j in range(1,n):
                diff=prices[j]-prices[j-1]
                localmax[i][j]= max(globalmax[i-1][j-1]+max(0,diff),localmax[i][j-1]+diff)
                globalmax[i][j]=max(globalmax[i][j-1],localmax[i][j])
        return globalmax[K][n-1]    
    
    
    
prices = [4,4,6,1,1,4,2,5]
K= 2        
if __name__ == "__main__":
    print(Solution().maxProfit( K, prices))           
        
#394. Coins in a Line
class Solution:
    """
    @param n: An integer
    @return: A boolean which equals to true if the first player will win
    """
    def firstWillWin(self, n):
        # write your code here
        return n%3!=0
        
#395. Coins in a Line II
class Solution:
    """
    @param values: a vector of integers
    @return: a boolean which equals to true if the first player will win
    """
    def firstWillWin(self, values):
        # write your code here

#http://www.cnblogs.com/grandyang/p/5864323.html
#这道题是之前那道Coins in a Line的延伸，由于每个硬币的面值不同，所以那道题的数学解法就不行了，
#这里我们需要使用一种方法叫做极小化极大算法Minimax，这是博弈论中比较经典的一种思想
#，LeetCode上有一道需要用这种思路解的题Guess Number Higher or Lower II。
#这道题如果没有接触过相类似的题，感觉还是蛮有难度的。我们需要用DP来解，
#我们定义一个一维数组dp，其中dp[i]表示从i到end可取的最大钱数，
#大小比values数组多出一位，若n为values的长度，那么dp[n]先初始化为0。
#我们是从后往前推，我们想如果是values数组的最后一位，及i = n-1时，
#此时dp[n-1]应该初始化为values[n-1]，因为拿了肯定比不拿大，
#钱又没有负面额；那么继续往前推，当i=n-2时，dp[n-2]
#应该初始化为values[n-2]+values[n-1]，应为最多可以拿两个，
#所以最大值肯定是两个都拿；当i=n-3时，dp[n-3]
#应该初始化为values[n-3]+values[n-2]，
#因为此时还剩三个硬币，你若只拿一个，那么就会给对手留两个，
#当然不行，所以自己要拿两个，只能给对手留一个，
#那么到目前位置初始化的步骤就完成了，下面就需要找递推式了：
#
#当我们处在i处时，我们有两种选择，拿一个还是拿两个硬币，我们现在分情况讨论：
#
#1. 当我们只拿一个硬币values[i]时，那么对手有两种选择，拿一个硬币values[i+1]，
#或者拿两个硬币values[i+1] + values[i+2]
#a) 当对手只拿一个硬币values[i+1]时，我们只能从i+2到end之间来取硬币，
#                    所以我们能拿到的最大硬币数为dp[i+2]
#b) 当对手拿两个硬币values[i+1] + values[i+2]时，我们只能从i+3到end之间来取硬币，
#                    所以我们能拿到的最大硬币数为dp[i+3]
#由于对手的目的是让我们拿较小的硬币，所以我们只能拿dp[i+2]和dp[i+3]中较小的一个，
#所以对于我们只拿一个硬币的情况，我们能拿到的最大钱数为values[i] + min(dp[i+2], dp[i+3])
#
#2. 当我们拿两个硬币values[i] + values[i + 1]时，那么对手有两种选择，
#拿一个硬币values[i+2]，或者拿两个硬币values[i+2] + values[i+3]
#a) 当对手只拿一个硬币values[i+2]时，我们只能从i+3到end之间来取硬币，
#                    所以我们能拿到的最大硬币数为dp[i+3]
#b) 当对手拿两个硬币values[i+2] + values[i+3]时，我们只能从i+4到end之间来取硬币，
#                    所以我们能拿到的最大硬币数为dp[i+4]
#由于对手的目的是让我们拿较小的硬币，所以我们只能拿dp[i+3]和dp[i+4]中较小的一个，
#所以对于我们只拿一个硬币的情况，我们能拿到的最大钱数为
#values[i] + values[i + 1] + min(dp[i+3], dp[i+4])
#
#综上所述，递推式就有了
# dp[i] = max(values[i] + min(dp[i+2], dp[i+3]), values[i] + values[i + 1] + min(dp[i+3], dp[i+4]))
#这样当我们算出了dp[0]，知道了第一个玩家能取出的最大钱数，我们只需要算出总钱数，
#然后就能计算出另一个玩家能取出的钱数，二者比较就知道第一个玩家能否赢了，参见代码如下：

        n=len(values)
        if n<=2:
            return True
        dp=[0 for _ in range(n+1)]
        #dp[n] initialized as 0
        
        dp[n-1]=values[n-1]
        dp[n-2]=values[n-1]+values[n-2]
        dp[n-3]=values[n-3]+values[n-2]
        
        for i in range(n-4,-1,-1):
            dp[i]=max( values[i]+min(dp[i+2],dp[i+3]),values[i]+values[i+1] + min(dp[i+4],dp[i+3]))
        
        sumv=sum(values)
        print(dp)
        print(dp[0],sumv-dp[0])
        return dp[0]>sumv-dp[0]
values= [1,2,4]        
if __name__ == "__main__":
    print(Solution().firstWillWin( values))         
        
#396. Coins in a Line III
class Solution:
    """
    @param values: a vector of integers
    @return: a boolean which equals to true if the first player will win
    """
    def firstWillWin(self, values):
        # write your code here
        
#        def get(A):
#            n=len(A)
#            if n==0:
#                return 0
#            if n==1:
#                return A[0]
#            if n==2:
#                return max(A[0],A[1])
#            if n==3:
#                iget=max(A[0],A[-1])
#                if iget==A[0]:
#                    iget+=min(A[1],A[2])
#                else:
#                    iget+=min(A[1],A[0])
#                return iget
#            
#            return max(sum(A[:-1])-get(A[:-1])+A[-1]  ,  sum(A[1:])-get(A[1:])+A[0])
#        return get(values) >sum(values)-get(values)
        A=values
        n=len(values)
        dp=[[0 for _ in range(n+1)] for _ in range(n+1)]
        # dp[0][10] A[0:10] 第一个人能拿到的最大值
        for i in range(n+1):
            dp[i][i]=0
            if i+1<n+1:
              dp[i][i+1]=A[i]
            if i+2<n+1:
                dp[i][i+2]=max(A[i],A[i+1])
        sumV=[0 for _ in range(n+1)]
       
        
        for i in range(1,n+1):
            sumV[i]=A[i-1]+sumV[i-1]
            
        #sum(A[i+1:j])  sumV[j]-sumV[i+1]
        #sum(A[i:j-1])  sumV[j-1]-sumV[i]
        for i in range(n-2,-1,-1):
            for j in range(i+2,n+1):
                dp[i][j]=max(A[i]+sumV[j]-sumV[i+1]-dp[i+1][j],A[j-1]+sumV[j-1]-sumV[i]-dp[i][j-1])
                #dp[i][j]=max(A[i]+sum(A[i+1:j])-dp[i+1][j],A[j-1]+sum(A[i:j-1])-dp[i][j-1])
        return dp[0][n]>sum(A)- dp[0][n]
                
                
values=[1,20,4]
values=[3,2,2]
values=[1,2,4,9,1,2,1,2,2]
values=[1,9999999,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,200,1,1,1,1,1,1,1,1,800,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
values=[1,2,3,4,5,6,7,8,13,11,10,9]
if __name__ == "__main__":
    print(Solution().firstWillWin( values))        
        
        
        
        
        
        
        
        
#397. Longest Continuous Increasing Subsequence        
class Solution:
    """
    @param A: An array of Integer
    @return: an integer
    """
    def longestIncreasingContinuousSubsequence(self, A):
        # write your code here
#For [5, 4, 2, 1, 3], the LICS is [5, 4, 2, 1], return 4.
#
#For [5, 1, 2, 3, 4], the LICS is [1, 2, 3, 4], return 4. 
        n=len(A)
        res=0
        l=1
        r=1
        for i in range(1,n):
            if A[i]>A[i-1]:
                l+=1
                res=max(res,l)
            else:
                l=1
            if A[i]<A[i-1]:
                r+=1
                res=max(res,r)
            else:
                r=1
        return res
A=[5, 4, 2, 1, 3]        
if __name__ == "__main__":
    print(Solution().longestIncreasingContinuousSubsequence( A))         
        
#399. Nuts & Bolts Problem
"""
class Comparator:
    def cmp(self, a, b)
You can use Compare.cmp(a, b) to compare nuts "a" and bolts "b",
if "a" is bigger than "b", it will return 1, else if they are equal,
it will return 0, else if "a" is smaller than "b", it will return -1.
When "a" is not a nut or "b" is not a bolt, it will return 2, which is not valid.
"""


class Solution:
    # @param nuts: a list of integers
    # @param bolts: a list of integers
    # @param compare: a instance of Comparator
    # @return: nothing
    def sortNutsAndBolts(self, nuts, bolts, compare):
        # write your code here 
        if not nuts or not bolts or not compare :
            return 
        if len(nuts)!=len(bolts):
            return 
        
        self.compare=compare
        
        self.quickSort(nuts, bolts,0,len(bolts)-1)
        

    def quickSort(self,nuts, bolts,left,right):
        if left>=right:
            return 
        
        nuts_split_pos= self.partition(nuts,bolts[left],left,right)
        bolts_split_pos= self.partition(bolts,nuts[nuts_split_pos],left,right)
        self.quickSort(nuts, bolts,left,bolts_split_pos-1)
        self.quickSort(nuts, bolts,bolts_split_pos+1,right)
    
    
    def partition(self,items,pivot,left,right):
        if not items or not pivot:
            return 
        
        for i in range(left,right+1):
            if self.compare.cmp(pivot,items[i])==0 or self.compare.cmp(items[i],pivot)==0:
                 items[left],items[i]=items[i],items[left]
                 break
        
        partner_pivot=items[left]
        while left<right:
            while left<right and (self.compare.cmp(items[right],pivot)==1 or self.compare.cmp(pivot,items[right])==-1):
                right-=1
            items[left]=items[right]
            while left<right and (self.compare.cmp(items[left],pivot)==-1 or self.compare.cmp(pivot,items[left])==1):
                left+=1
            items[right]=items[left]
        
        items[left]   =partner_pivot
        return left
    
#400. Maximum Gap
class Solution:
    """
    @param nums: an array of integers
    @return: the maximun difference
    """
    def maximumGap(self, nums):
        # write your code here
        
        n=len(nums)
        
        if n<2:
            return 0
        if n==2:
            return abs(nums[0]-nums[1])
        
        
        imax=max(nums)
        imin=min(nums)
        
        if imin==imax:
            return 0
        
        import math
        gap=math.ceil((imax-imin)/(n-1))
        numbuck=int(math.ceil((imax-imin)/gap))
        
        buckmax=[0 for _ in range(numbuck)]
        buckmin=[float('inf') for _ in range(numbuck)]
        
        maxgap=0
        
        for num in nums:
            if num==imax or num==imin:
                continue
            
            index=int((num-imin)//gap)
            buckmax[index]=max(buckmax[index],num)
            buckmin[index]=min(buckmin[index],num)
         
        prev=imin
        for i in range(numbuck):
            if buckmin[i]==float('inf')  and buckmax[i]==0:
                continue
            maxgap=max(maxgap,buckmin[i]-prev)
            prev=buckmax[i]
            
        maxgap=max(imax-prev,maxgap)
        return maxgap
            
            
nums=[1, 9, 2, 5]       
if __name__ == "__main__":
    print(Solution().maximumGap( nums))               
        
        
#401. Kth Smallest Number in Sorted Matrix        
class Solution:
    """
    @param matrix: a matrix of integers
    @param k: An integer
    @return: the kth smallest number in the matrix
    """
    def kthSmallest(self, matrix, k):
        # write your code here
        m=len(matrix)
        n=len(matrix[0])
        
        
        import heapq
        
        hq=[(matrix[0][0],0,0)]
        
        visited=[[0 for _ in range(n)] for _ in range(m)]
        visited[0][0]=1
        
        for _ in range(k):
            res,x,y=heapq.heappop(hq)
           
            if x+1<m and visited[x+1][y]==0:
                
                 heapq.heappush(hq,(matrix[x+1][y],x+1,y))
                 visited[x+1][y]=1
            if y+1<n and visited[x][y+1]==0:
                
                 heapq.heappush(hq,(matrix[x][y+1],x,y+1))
                 visited[x][y+1]=1
        return res
        
matrix=[
  [1 ,5 ,7],
  [3 ,7 ,8],
  [4 ,8 ,9],
]    
k=4  
if __name__ == "__main__":
    print(Solution().kthSmallest(matrix, k))          
        
#402. Continuous Subarray Sum  
class Solution:
    """
    @param: A: An integer array
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def continuousSubarraySum(self, A):
        # write your code here
        if not A:
            return []
        
        res=[]
        start=0
        end=-1
        ans=float('-inf')
        
        sm=0
        
        for x in A:
            if sm<0:
                start=end+1
                end=start
                sm=x
            else:
                sm+=x
                end+=1
            if ans<sm:
                ans=sm
                res=[start,end]
        return res
A=[-3, 1, 3, -3, 4]      
if __name__ == "__main__":
    print(Solution().continuousSubarraySum( A)) 


#403. Continuous Subarray Sum II
class Solution:
    """
    @param: A: An integer array
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def continuousSubarraySumII(self, A):
        # write your code here
        
       
        
#按I的方法求出的结果
#从整个array中减去中间最小的subarray，需要rotate的array
#思路是这样的，像楼上说的两种情况，不rotate的 和rorate的。不rotate的和Continuous Subarray Sum I做法一样，
#不说了。rotate的，可以这样想，rotate的结果其实相当于是把原来的array中间挖了一段连续的array，那么挖哪一段呢？
#肯定是和最小的一段连续array。这样解法就出来了。
#
#类似Continuous Subarray Sum I，在I里面是找到连续和最大的一段subarray，在这里，不仅找到和最大的一段连续array
#，并且也找到和最小的一段连续array，然后用整个array的sum减去这个最小的和，如果大于不rotate的最大和，
#那么解就是挖去的这段array的（尾+1， 头-1）
#有一个edge case就是当array全部为负时，要挖去的array就是整个array，这个要注意一下。        
                
        
#        sumAA=[0 for _ in range(2*n+1)]
#        
#        
#        for i in range(1,2*n+1):
#            sumAA[i]=sumAA[i-1]+AA[i-1]
#        #print(AA)
#        #print(sumAA)
#            
#        res=float('-inf')
#        start=-1
#        end=-1
#        for i in range(0,2*n+1):
#            for j in range(i,2*n+1):
#                if j-i<n:
#                    if sumAA[j]-sumAA[i]>res:
#                        res=sumAA[j]-sumAA[i]
#                        start=i
#                        end=j-1
        if not A:
            return []
        n=len(A)
        if n==1:
            return [0,0]
        
        totalsum=0
        maxsum=float('-inf')
        curmaxsum=0
        maxstart=0
        maxend=0
        mastart=0
        
        minsum=float('inf')
        curminsum=0
        minstart=0
        minend=0
        mistart=0
        
        for i in range(n):
            totalsum+=A[i]
            
            if curmaxsum<0:
                curmaxsum=A[i]
                mastart=i
            else:
                curmaxsum+=A[i]
            
            if curmaxsum>maxsum:
                maxsum=curmaxsum
                maxstart=mastart
                maxend=i
            
            
            if curminsum>0:
                curminsum=A[i]
                mistart=i
            else:
                curminsum+=A[i]
            
            if curminsum<minsum:
                minsum=curminsum
                minstart=mistart
                minend=i
        print(mastart,maxend)
        if totalsum-minsum>=maxsum  and minstart!=0 and   minend!=n-1:
            start=minend+1
            end=minstart-1
        else:
            start=maxstart
            end=maxend
        return [start,end]
#A=[3, 1, -100, -3, 4]
#A=[3, 1, -1, 3, 4]
#A=[-3, 1, 3, -3, 4]      
if __name__ == "__main__":
    print(Solution().continuousSubarraySumII( A))            
        
#405. Submatrix Sum        
class Solution:
    """
    @param: matrix: an integer matrix
    @return: the coordinate of the left-up and right-down number
    """
    def submatrixSum(self, matrix):
        # write your code here
        
        m=len(matrix)
        n=len(matrix[0])
        if m==0:
            return []
        sum_m=[[0 for _ in range(n+1)] for _ in range(m+1)]
        
        
        for i in range(m+1):
            sum_m[i][0]=0
        for j in range(n+1):
            sum_m[0][j]=0
            
        for i in range(0,m):
            for j in range(0,n):
                sum_m[i+1][j+1]=sum_m[i][j+1]+sum_m[i+1][j]+matrix[i][j]-sum_m[i][j]
                
        res=[]    
        for l in range(0,m):
            for h in range(l+1,m+1):
                dic={}
                for j in range(n+1):
                    diff=sum_m[h][j]-sum_m[l][j]
                    if diff in dic:
                        res.append([l,dic[diff]])
                        res.append(([h-1,j-1]))
                        return res
                    else:
                        dic[diff]=j
                    
      
matrix=[
  [1 ,5 ,7],
  [3 ,7 ,-8],
  [4 ,-8 ,9],
]

matrix=[[0,4],[-4,0]]   
if __name__ == "__main__":
    print(Solution().submatrixSum(matrix))                 
                
            
#406. Minimum Size Subarray Sum        
class Solution:
    """
    @param nums: an array of integers
    @param s: An integer
    @return: an integer representing the minimum size of subarray
    """
    def minimumSize(self, nums, s):
        # write your code here
        if not nums:
            return 0
        
        n=len(nums)
        l=0
        r=0
        ans=n+1
        total=0
        
        while r<n:
            while r<n and total<s:
                total+=nums[r]
                r+=1
            if total<s:
                break
            while l<=r and total>=s:
                #print(l,r)
                
                total-=nums[l]
                l+=1
            ans=min(ans,r-l+1)
        if ans==n+1:
            return -1
        return ans
nums=[2,3,1,2,4,3] 
s = 7        
if __name__ == "__main__":
    print(Solution().minimumSize( nums, s))                 
                
        
#407. Plus One        
class Solution:
    """
    @param digits: a number represented as an array of digits
    @return: the result
    """
    def plusOne(self, digits):
        # write your code here
        res=[]
        n=len(digits)  
        if n==0:
            return []
        carry=1
        for x in digits[::-1]:
            n=x+carry
            carry=n//10
            res.append(n%10)
        if carry==1:
            res.append(carry)
        return res[::-1]
digits=[1,2,3]
digits=[9,9,9]
if __name__ == "__main__":
    print(Solution().plusOne(digits))     
            
#408. Add Binary
class Solution:
    """
    @param a: a number
    @param b: a number
    @return: the result
    """
    def addBinary(self, a, b):
        # write your code here
        la=len(a)
        lb=len(b)
        
        a=list(a)[::-1]
        b=list(b)[::-1]
        carry=0
        
        i=0
        
        res=[]
        while i<la  or i<lb:
            sm=0
            if i<la:
                sm+=int(a[i])
            if i<lb:
                sm+=int(b[i])
            
            sm+=carry
            res.append(sm%2)
            carry=sm//2
            i+=1
        
        if carry==1:
            res.append(carry)
        
        return ''.join(str(x) for x in res[::-1])
a = '11'

b = '1'            
if __name__ == "__main__":
    print(Solution().addBinary( a, b))        
                
#411. Gray Code
class Solution:
    """
    @param n: a number
    @return: Gray code
    """
    def grayCode(self, n):
        # write your code here
        if n==0:
            return [0]
        if n==1:
            return [0,1]
        
        def neighbor(s):
            n=len(s)
            res=[]
            for i in range(n)[::-1]:
                if s[i]=='0':
                   x='1'
                else:
                    x='0'
                res.append(s[:i]+x+s[i+1:])
            return res
                
        def dfs(s,res):
             if len(res)==2**n:
                 return True
             if len(res)>2**n:
                 return False
             for nei in neighbor(s):
                 if nei not in res:
                     res.append(nei)
                     if dfs(nei,res):
                         return True
             return False
        res=[]
        s=''.join(['0'  for _ in range(n)])
        res.append(s)
        dfs(s,res)
        res2=[int(s, 2) for s in res]
        
            
        return res
n=3
if __name__ == "__main__":
    print(Solution().grayCode(n))                    
             
             
                    
#412. Candy
class Solution:
    """
    @param ratings: Children's ratings
    @return: the minimum candies you must give
    """
    def candy(self, ratings):
        # write your code here
        n=len(ratings)
        if n==0:
            return 0
        if n==1:
            return 1
        ratings=[0]+ratings+[0]
        
        res=[1 for _ in range(n+2)]
        res[0]=0
        res[-1]=0
        print(ratings)
        for i in range(1,n+1):
            if ratings[i]>ratings[i-1]:
                 #print(i,res)
                 while res[i]<=res[i-1]:
                       res[i]+=1
            if ratings[i]>ratings[i+1]:
                 #print(res)
                 while res[i]<=res[i+1]:
                       res[i]+=1
        for i in range(n,0,-1):
            if ratings[i]>ratings[i-1]:
                 #print(i,res)
                 while res[i]<=res[i-1]:
                       res[i]+=1
            if ratings[i]>ratings[i+1]:
                 #print(res)
                 while res[i]<=res[i+1]:
                       res[i]+=1               
        #print(res)
        return sum(res)
ratings=  [1, 2, 2]  
ratings=  [1, 1, 1]
ratings=  [1, 2]  
ratings= [5,3,1]                     
if __name__ == "__main__":
    print(Solution().candy( ratings))                       

        
#413. Reverse Integer
class Solution:
    """
    @param n: the integer to be reversed
    @return: the reversed integer
    """
    def reverseInteger(self, n):
        # write your code here
        if n==0:
            return 0
        neg=1
        if n<0:
            n=0-n
            neg=-1
        reverse=0
        while n>0:
            reverse=reverse*10+n%10
            n=n//10
            
        reverse=reverse* neg
        
        if reverse<-(1<<31) or reverse>(1<<31-1):
            return 0
        return reverse
            
#414. Divide Two Integers        
class Solution:
    """
    @param dividend: the dividend
    @param divisor: the divisor
    @return: the result
    """
    def divide(self, dividend, divisor):
        # write your code here
        intmax=2147483647
        if divisor==0:
            return intmax
        
        neg=1
        if (dividend <0 and divisor>0) or  (dividend >0 and divisor<0):
            neg=-1
        
        a=abs(dividend)
        b=abs(divisor)
        
        ans=0
        shift=31
        while shift>=0:
            if a>=b<<shift:
                a-=b<<shift
                ans+=1<<shift
            shift-=1
        if neg==-1:
            ans=0-ans
            
        if ans>intmax or ans <-(intmax+1) :
            return intmax
        
        
        return ans

dividend = 100 
divisor = 9            
if __name__ == "__main__":
    print(Solution().divide( dividend, divisor))                       
        
#415. Valid Palindrome        
class Solution:
    """
    @param s: A string
    @return: Whether the string is a valid palindrome
    """
    def isPalindrome(self, s):
        # write your code here
        n=len(s)
        if n==0 or n==1:
            return True
        l=0
        r=n-1
        while l<r:
            while l<r and not s[l].isalnum():
                l+=1
            while l<r and not s[r].isalnum():
                r-=1
            
            if s[l].lower()!=s[r].lower():
                return False
            l+=1
            r-=1
        return True
        
        
#"A man, a plan, a canal: Panama" is a palindrome.
s='A man, a plan, a canal: Panama' 
s="race a car"        
if __name__ == "__main__":
    print(Solution().isPalindrome( s))  

                        
#417. Valid Number
class Solution:
    """
    @param s: the string that represents a number
    @return: whether the string is a valid number
    """
    def isNumber(self, s):
        # write your code here
        s=s.strip()
        n=len(s)
        if n==0:
            return False
        
        i=0
        
        hasDigit=False
        Eflag=False
        DotFlag=False
        sign=False
        
        
        while i<n:
            
            if s[i].isdigit():
                hasDigit=True
                sign=True
                i+=1
            elif  s[i]=='.'  and not DotFlag:
                DotFlag=True
                sign=True
                i+=1
            elif (s[i]=='e' or s[i]=='E') and not Eflag  and hasDigit:
                i+=1
                Eflag=True
                DotFlag=True
                hasDigit=False
                sign=False
            elif (s[i]=='+' or s[i]=='-')  and not hasDigit and not sign:
                i+=1
                sign=True
            else:
                return False
        if hasDigit:
                return True
        else:
                return False
#\s*[+-]?(\d+\.\d*|\.?\d+)(e[+-]?\d+)?\s*                
s ="0"
s =" 0.1 " 
s ="abc"
s ="1 a"   
s ="2e10"           
if __name__ == "__main__":
    print(Solution().isNumber( s))                   
                
#418. Integer to Roman
class Solution:
    """
    @param n: The integer
    @return: Roman representation
    """
    def intToRoman(self, n):
        # write your code here
#I             1
#V             5
#X             10
#L             50
#C             100
#D             500
#M             1000
        
        nums = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        strs=['M', 'CM',  'D', 'CD',  'C', 'XC', 'L', 'XL', 'X', 'IX', 'V' ,'IV', 'I']
        res=''
        for i ,num  in enumerate(nums):
            while n>=num:
                res+=strs[i]
                n-=num
            if n==0:
                return res
        
n=99 -> XCIX               
if __name__ == "__main__":
    print(Solution().intToRoman( n))                  
            
#419. Roman to Integer                
class Solution:
    """
    @param s: Roman representation
    @return: an integer
    """
    def romanToInt(self, s):
        # write your code here
        #nums = [1000, 900, 500,   400,  100,  90,  50,   40,  10,   9,    5,   4,   1]
        #strs=  ['M', 'CM',  'D', 'CD',  'C', 'XC', 'L', 'XL', 'X', 'IX', 'V' ,'IV', 'I'] 
        def decompse(s):
            if len(s)==1:
                if s=='M':
                    return 1000
                elif s=='D':
                    return 500
                elif s=='C':
                    return 100
                elif s=='L':
                    return 50
                elif s=='X':
                    return 10
                elif s=='V':
                    return 5
                elif s=='I':
                    return 1
            if len(s)==2:
                if s=='CM':
                    return 900
                elif s=='CD':
                    return 400
                elif s=='XC':
                    return 90
                elif s=='XL':
                    return 40
                elif s=='IX':
                    return 9
                elif s=='IV':
                    return 4
                else:
                    return decompse(s[0])+decompse(s[1])
            else:
                if s[:2]  in ('CM',  'CD',  'XC',  'XL', 'IX', 'IV'):
                    return decompse(s[:2])+decompse(s[2:])
                else:
                    return decompse(s[0])+decompse(s[1:])
        return decompse(s)
s='IV'
s='XII'
s='XXI' 
s='XCIX'
if __name__ == "__main__":
    print(Solution().romanToInt( s))                  
                
#420. Count and Say
class Solution:
    """
    @param n: the nth
    @return: the nth sequence
    """
    def countAndSay(self, n):
        # write your code here
        if n==0:
            return ''
        s='1'
        def cal(s):
            count=1
            length=len(s)
            ans=''
            for i,c in enumerate(s):
                if i+1<length and s[i]!=s[i+1]:
                    ans=ans+str(count)+c
                    count=1
                elif i+1<length  and s[i]==s[i+1]:
                    count+=1
            return ans+str(count)+c
                
                
        for _ in range(1,n):
            s=cal(s)
        return s
n=1

n=2               
if __name__ == "__main__":
    print(Solution().countAndSay( n))                  
            
        
#421. Simplify Path       
class Solution:
    """
    @param path: the original path
    @return: the simplified path
    """
    def simplifyPath(self, path):
        # write your code here
        stack=[]
        
        places=[p for p in path.split('/') if p != '' and p!='.']
        
        for p in places:
            if p=='..':
                while stack:
                    stack.pop() 
            else:
                stack.append(p)
        return '/'+'/'.join(stack)
                
#422. Length of Last Word
class Solution:
    """
    @param s: A string
    @return: the length of last word
    """
    def lengthOfLastWord(self, s):
        # write your code here
        if not s:
            return 0
        
        slist=s.strip().split(' ')
        return len(slist[-1])
            
        
        
#423. Valid Parentheses        
class Solution:
    """
    @param s: A string
    @return: whether the string is a valid parentheses
    """
    def isValidParentheses(self, s):
        # write your code here
        if not s:
            return True
        
        n=len(s)
        if n%2:
            return False
        
        stack=[]
        for x in s:
            if x in ('{','[','('):
                stack.append(x)
            else:
                if stack:
                   y=stack.pop()
                   if y=='{'  and x !='}':
                       return False
                   elif y=='['  and x !=']':
                       return False
                   elif y=='('  and x !=')':
                       return False
                   
                else:
                     return False
        return not stack
s="([)]"
if __name__ == "__main__":
    print(Solution().isValidParentheses( s)) 





#424. Evaluate Reverse Polish Notation        
class Solution:
    """
    @param tokens: The Reverse Polish Notation
    @return: the value
    """
    def evalRPN(self, tokens):
        # write your code here
        if not tokens:
            return 0
        stack=[]
        for x in tokens:
            
            if x in ('+','-','*','/' ):
                a=stack.pop()
                b=stack.pop()
                
                if x=='+':
                    stack.append(  str(int(a)+int(b))  )
                elif x=='-':
                    stack.append(  str(-int(a)+int(b))  )
                elif x=='*':
                    stack.append(  str(int(a)*int(b))  )
                elif x=='/':
                    #stack.append(  str(int(b) // int(a) )  )
                    stack.append(  str(  int(int(b) / int(a)) ) )
            else:
                stack.append(x)
            print(stack)
        return int(stack[0])
                
tokens=["2", "1", "+", "3", "*"] -> ((2 + 1) * 3) -> 9
tokens=["4", "13", "5", "/", "+"] -> (4 + (13 / 5)) -> 6                 
tokens=["3","-4","+"]
tokens=["10","6","9","3","+","-11","*","/","*","17","+","5","+"]       
if __name__ == "__main__":
    print(Solution().evalRPN( tokens))         
        

#425. Letter Combinations of a Phone Number
class Solution:
    """
    @param digits: A digital string
    @return: all posible letter combinations
    """
    def letterCombinations(self, digits):
        # write your code here
        if not digits:
            return []
        res=''
        table={
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz'}
        
        length=len(digits)
        if length==1:
            return list(table[digits])
        
            
        def dfs(path,res,n):
            if len(path)==length:
                res.append(path[:])
                return 
            for letter in table[digits[n]]:
                dfs(path+letter,res,n+1)
                
                
                
        res=[]
        dfs('',res,0)
        return res
digits='23'
if __name__ == "__main__":
    print(Solution().letterCombinations(digits))     
           
            
#426. Restore IP Addresses
class Solution:
    """
    @param s: the IP string
    @return: All possible valid IP addresses
    """
    def restoreIpAddresses(self, s):
        # write your code here
#Every part, if treated as an interger, should be in [0,255]

#Every part must not have any leading zeros        
        
        def validate(s,res,path):
            if not s  and len(path)==4:
                res.append(path[:])
                return 
            if not s:
                return 
            if s and len(path)>=4:
                return 
            if s[0]=='0':
                validate(s[1:],res,path+['0'])
                return 
            for i in range(1,min(4,len(s)+1)):
                 
                if int(s[:i])<256 and int(s[:i])>0:
                    validate(s[i:],res,path+[s[:i]]) 
        res=[]
        validate(s,res,[])
        return ['.'.join(l) for l in res]
s="25525511135" 
s="010010"               
if __name__ == "__main__":
    print(Solution().restoreIpAddresses(s))            
        
        
#427. Generate Parentheses
class Solution:
    """
    @param n: n pairs
    @return: All combinations of well-formed parentheses
    """
    def generateParenthesis(self, n):
        # write your code here
        
        if n==0:
            return []
        if n==1:
            return ['()']
        
        def arrange(l,r,res,path):
            if l>r:
                return 
            if l==0 and r==0:
                res.append(path[:])
                return 
            if l>0:
                arrange(l-1,r,res,path+'(')
            if r>0:
                arrange(l,r-1,res,path+')')
                    
                        
        
        res=[]
        l=n
        r=n
        arrange(l,r,res,'')
        return res 
n=1 
n=2
n=5                    
if __name__ == "__main__":
    print(Solution().generateParenthesis(n))                                    
                    

#428. Pow(x, n) 
class Solution:
    """
    @param: x: the base number
    @param: n: the power number
    @return: the result
    """
    def myPow(self, x, n):
        # write your code here
        
        def cal(x,n):
            if x==0:
                return 0
            if n==0:
                return 1
            if n==1:
                return x
            
            if n%2==1:
                return cal(x*x,n//2)*x
            else:
                return cal(x*x,n//2)
            
        return cal(x,n) if n>=0 else 1/cal(x,-n)
x=8.88023
n=3  
x=2.00000
n=-2147483648              
if __name__ == "__main__":
    print(Solution().myPow(x, n))              
        
#430. Scramble String
class Solution:
    """
    @param s1: A string
    @param s2: Another string
    @return: whether s2 is a scrambled string of s1
    """
    def isScramble(self, s1, s2):
        # write your code here
        if s1==s2:
            return True
        if len(s1)!=len(s2):
            return False
        
        from collections import Counter
        if Counter(s1)!=Counter(s2):
            return False
        n=len(s1)
        for i in range(1,len(s1)):
            if self.isScramble( s1[:i], s2[:i])  and self.isScramble( s1[i:], s2[i:]):
                return True
            if self.isScramble( s1[:i], s2[n-i:])  and self.isScramble( s1[i:], s2[:n-i]):
                return True 
        return False
s1= "a"
s2="a"
s1="great"
s2="rgtae"
if __name__ == "__main__":
    print(Solution().isScramble( s1, s2))              
                        
            
#433. Number of Islands        
class Solution:
    """
    @param grid: a boolean 2D matrix
    @return: an integer
    """
    def numIslands(self, grid):
        # write your code here
        m=len(grid)
        if m==0:
            return 0
        n=len(grid[0])
        def dfs(grid,i,j,color):
          
            for x,y in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                if x<m and x>=0 and y<n  and y>=0:
                    if grid[x][y]==1  :
                           grid[x][y]=color
                           dfs(grid,x,y,color)
        color=1
        for i in range(m):
            for j in range(n):
                if grid[i][j]==1:
                    color+=1
                    dfs(grid,i,j,color)
        return color-1
grid=[
  [1, 1, 0, 0, 0],
  [0, 1, 0, 0, 1],
  [0, 1, 0, 1, 1],
  [0, 0, 0, 0, 1],
  [0, 0, 0, 0, 1]
] 
grid=[[0]   ]    
if __name__ == "__main__":
    print(Solution().numIslands(grid))         
        
#434. Number of Islands II
"""
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
    def __str__(self):
        return str(self.x) +'   '+str(self.y)
    
"""

class Solution:
    """
    @param n: An integer
    @param m: An integer
    @param operators: an array of point
    @return: an integer array
    """
    def numIslands2(self, m,n ,positions):
        # write your code here
        
        def node_id(node,n):
            return n*node.x+node.y
        def find_set(x):
            if set[x]!=x:
                set[x]=find_set(set[x])
            return set[x]
        def union_set(x,y):
            root_x,root_y=find_set(x),find_set(y)
            
            set[min(root_x,root_y)]=max(root_x,root_y)
            
        set={}
        number=0
        numbers=[]
        
        direction=[(1,0),(-1,0),(0,1),(0,-1)]
        
        for k,node in enumerate(positions):
          
            if node_id(node,n) in set:
                #print('***')
                numbers.append(number)
                continue
                
            set[node_id(node,n)]=node_id(node,n)
            
            number+=1
            
            
            for  d in direction:
                i,j=node.x+d[0],node.y+d[1]
                if i<m  and i>=0 and j<n and j>=0  and node_id(Point(i,j),n) in set:
                    if find_set(node_id(Point(i,j),n))!=find_set(node_id(node,n)):
                        union_set(node_id(Point(i,j),n),node_id(node,n))
                        number-=1
            numbers.append(number)
        return numbers
                        
n=4
m=5
operators=[Point(1,1),Point(0,1),Point(3,3),Point(3,4)] 


n=3
m=3
positions=[Point(0,0),Point(0,1),Point(2,2),Point(2,2)]                     
                
if __name__ == "__main__":
    print(Solution().numIslands2( n, m, positions))             


#436. Maximal Square
class Solution:
    """
    @param matrix: a matrix of 0 and 1
    @return: an integer
    """
    def maxSquare(self, matrix):
        # write your code here
        
        
   #dp(i, j) represents the length of the square 
   #whose lower-right corner is located at (i, j)
   #dp(i, j) = min{ dp(i-1, j-1), dp(i-1, j), dp(i, j-1) }
        m=len(matrix)
        if m==0:
            return 0
        n=len(matrix[0])
        
        dp=[[0 for _ in range(n+1) ] for _ in range(m+1)]
        
#        for i in range(m):
#            if matrix[i][0]==1:
#                dp[i][0]=1
#        for j in range(n):
#            if matrix[0][j]==1:
#                dp[0][j]=1
        res=0        
        for i in range(1,m+1):
            for j in range(1,n+1):
                if matrix[i-1][j-1]==1:
                    
                   dp[i][j]=min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1
                   res=max(dp[i][j],res)
        return res*res
    
    
matrix=[[1, 0, 1, 0, 0],
[1 ,0 ,1 ,1, 1],
[1 ,1, 1, 1, 1],
[1, 0,0 ,1, 0]]


matrix=[[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]]
if __name__ == "__main__":
    print(Solution().maxSquare( matrix))             
        
        
        

#437. Copy Books
class Solution:
    """
    @param pages: an array of integers
    @param k: An integer
    @return: an integer
    """
    def copyBooks(self, pages, k):
        # write your code here
#https://zhengyang2015.gitbooks.io/lintcode/copy_books_437.html     
#T[i][j]表示前i本书分配给j个人copy
        m=len(pages)
        if m==0:
            return 0
        if k<=0:
            return -1
        
        dp=[[0 for _ in range(k+1)] for  _ in range(m+1)]
        
        summ=0
        for i in range(1,m+1):
            summ+=pages[i-1]
            dp[i][1]=summ
        for j in range(1,k+1):
            dp[1][j]=pages[0]
            
        for i in range(2,m+1):
            for j in range(2,k+1):
                mini=float('inf')
                if j>i:
                    dp[i][j]=dp[i][j-1]
                    continue
                else:
                    for h in range(j-1,i):
                        temp=max(dp[h][j-1],dp[i][1]-dp[h][1])
                        mini=min(mini,temp)
                dp[i][j]=mini
        return dp[i][k]
    
pages = [3,2,4]
k = 2

pages =[8720,1593,4278,1018,9515,5536,8694,869,8423,9920,3425,9689,9908,1404,8371,6019,7531,943,
        9280,8058,1445,7485,9223,5067,736,6065,1724,3269,130,1297,1701,5585,1209,5901,3760,
        3216,5307,8534,9575,7135,1251,3531,5162,7432,8559,2024,9738,2621,7926,3865,9904,4763,
        2031,4561,5870,6033,6442,6405,9886,8455,8970,6746,7923,5222,274,3875,5821,816,3459,9916,
        9447,4252,1871,5784,8272,9526,1531,5647,6676,2416,6281,7878,4382,5376,3618,1158,3717,5573,
        5602,5919,7579,6628,929,6897,8324,8657,8342,8190,7384,2466,8715,2499,9144,7349,3236,2852,
        4613,1917,5345,9288,9205,3363,5008,156,4716,6924,3490,605,3872,577,5393,7941,7390,7042,8242
        ,5376,5939,3910,4727,4053,4850,5531,2658,3876,9379,6157,7544,2845,5499,459,3119,2984,5567,
        5706,1507,5971,8369,8177,5985,1215,8258,2925,9193,7999,334,4761,2586,7770,195,5804,3898,
        7214,6625,2975,4078,5441,9944,4161,9225,7012,6383,4441,4474,7030,7292,6938,48,6253,4707,
        2078,8449,7011,65,7205,4841,2066,5097,3875,6431,2626,6325,789,5047,324,3528,1325,2443,281,
        3094,1908,9686,8720,9310,6817,1856,2137,7718,1180,5988,8766,3408,8439,2555,2214,2678,1991,
        7393,6323,9586,8430,3298,9828,9386,5406,12,7231,8808,3655,208,6314,246,6279,9172,9980,5042,
        5659,7293,5428,2789,6454,2887,5948,6895,847,1443,754,4889,9509,2008,7163,5297,3845,7290,636,
        1900,7035,2718,6348,9216,2577,9852,3935,412,6605,3786,2158,9280,9835,8941,2575,2868,9006,
        6860,8101,3714,7940,268,8199,5597,9898,982,2825,5125,9865,3008,7575,8914,7330,4410,2063,214,
        3625,1588,8894,7882,503,8305,2701,8230,1336,1946,1318,2314,7348,3387,670,3682,7588,9411,
        3812,2509,1106,6148,9162,3716,9041,4980,2057,251,1778,7969,2805,5764,1266,6531,5219,8550,
        1921,9687,8127,826,4642,9398,4097,512,3296,2618,5132,2890,3471,2757,6336,5853,789,980,9109,
        5240,9449,758,8198,3420,4998,4239,995,5805,1997,9124,1265,694,8920,1770,3965,9473,6176,9059,
        6479,3190,8678,7136,9005,6420,1664,8229,8838,6959,8267,3220,8445,7166,7999,6727,4942,8306,
        5704,2005,9106,6139,1954,8219,1707,150,9456,2125,9215,6515,3446,2391,5560,2031,4940,7600,
        5852,6911,6330,571,1691,4217,9403,8555,4183,7995,8398,7931,7509,127,2514,7970,47,7970,2914,
        9730,86,12,723,4736,8464,1108,5422,5102,57,3366,9275,7030,7054,1230,9787,9268,8967,2653,
        1453,5486,7214,3899,9681,315,965,7969,8778,7791,2656,3646,988,261,3800,1317,9497,386,243,
        7713,9333,1563,9973,4759,4799,6150,2865,5293,9948,8505,1550,1240,321,2195,1521,2108,1552,
        3275,6184,2752,9108,3932,1909,5232,2702,3743,1181,7464,1925,4419,8051,796,893,1317,4384,
        8060,4508,4579,5755,6765,1733,117,7343,2719,154,5986,5197,2767,7902,2574,606,2351,558,2739,
        5230,310,3186,9391,6219,8236,6141,668,9697,1218,4273,8812,3986,2631,6865,6216,9427,3287,
        5480,1000,8315,3384,4275,630,9182,8713,9918,3415,3572,9152,6451,8058,9499,1922,1988,1525,
        3348,6806,6996,9026,7077,6816,6489,3395,3714,3595,1134,7325,3761,4924,1157,4308,9439,2307,
        9943,821,4355,5258,5927,9518,6464,3397,942,8028,3266,639,62,2752,5404,2183,2678,8969,7628,
        6666,8509,1246,5546,6889,6537,3432,47,4987,7518,1844,7433,2957,9424,8790,620,6029,7828,8353,
        5974,8560,4415,4788,1794,9941,7061,4465,5854,316,5598,7581,5248,1008,7063,1562,3062,1903,
        7599,3556,3807,7722,7223,1664,8909,4789,1482,9222,2706,1275,3133,5968,588,6682,8964,3264,
        2602,9032,559,7110,974,8997,9697,9367,2028,4764,3815,8957,8214,1710,5591,1100,4609,9684,
        9279,7118,256,5260,633,4223,5345,3597,5404,380,1414,5893,3404,8497,7643,7964,7377,5257,
        1343,5333,9756,8275,9046,3683,3874,6575,4983,1321,9673,8735,624,4445,3940,8732,620,3404,
        3246,6625,4801,3359,8994,7877,2082,1077,8220,8439,3208,362,3282,399,4196,104,2410,9877,9678,
        6326,1443,1647,8627,5171,9869,4092,9068,7882,5249,3254,8011,9659,7355,3520,6863,6891,2744,
        7413,6916,1100,3316,8290,4895,7062,6393,78,8002,3826,8206,1122,2555,2084,5833,6819,9490,
        6316,9488,8609,4711,3897,2446,7384,27,8203,137,5780,1161,9041,8716,3669,3225,1991,9653,
        7780,313,2234,3678,9120,1253,9659,3745,3469,5496,6550,8747,7934,5594,5987,5323,1277,6221,
        6939,9121,4829,331,3666,9932,5244,3512,704,4269,1726,2382,8691,7536,8362,7610,9446,1255,
        2997,7226,1821,1726,7208,6087,3478,5167,1880,4998,5870,9032,3941,7188,1880,8584,179,4042,
        239,3004,2414,4613,8279,6636,2603,9247,3039,7669,2180,7440,3274,5366,2395,2214,3257,4791,
        2081,1631,6292,4254,3733,9828,3644,3522,2550,4424,9989,7654,118,3505,216,5902,4131,2754,
        3333,4072,5624,4975,1160,3900,7783,1200,1361,9290,9284,1251,9817,6255,4832,8680,3667,8141,
        8951,1507,6395,4189,9869,7080,2084,5060,5153,5762,5680,7705,8415,2142,5837,6711,8460,3254,
        5306,869,2364,7737,3892,9618,605,2948,9414,371,4397,3807,1720,9327,961,3598,573,6542,8806,
        3522,2166,3899,9642,8781,1321,6424,7085,5827,3592,9206,8494,9878,8889,8246,3800,7856,4523,
        2859,7029,7764,7891,1825,2165,2544,9758,6080,6399,121,759,1222,5545,2181,9505,7805,6212,
        2743,3059,9952,2826,5660,4951,9826,6066,3340,2614,9186,8971,8866,832,319,4276,1441,5118,
        5409,2587,9550,2542,3995,4005,2197,6265,6070,9324,8711,9977,5444,296,8550,3154,5899,3510,
        1093,6249,8534,4619,1096,3509,5261,6209,1289,8425,3941,4955,3570,5466]
k =165
if __name__ == "__main__":
    print(Solution().copyBooks( pages, k))             
                
                
    
#437. Copy Books
class Solution:
    """
    @param pages: an array of integers
    @param k: An integer
    @return: an integer
    """
    def copyBooks(self, pages, k):
        n=len(pages)
        if n==0:
            return 0
        
        def copierNeed(pages,limit):
            summ=pages[0]
            copiers=1
            for i in range(1,n):
                
                if (summ+pages[i])>limit:
                    copiers+=1
                    summ=0
                summ+=pages[i]
                
            return copiers
        
        maxi=0
        total=0
        for x in pages:
            total+=x
            if x>maxi:
                maxi=x
        start=maxi
        end=total
        
        while start+1<end:
            mid=(start+end)//2
            if copierNeed(pages,mid)<=k:
                end=mid
            else:
                start=mid
        if copierNeed(pages,start)<=k:
                return start
        else:
                return end
        
        
        
#439. Segment Tree Build II
"""
Definition of SegmentTreeNode:
class SegmentTreeNode:
    def __init__(self, start, end, max):
        self.start, self.end, self.max = start, end, max
        self.left, self.right = None, None
"""

class Solution:
    """
    @param A: a list of integer
    @return: The root of Segment Tree
    """
    def build(self, A):
        # write your code here
        
        def building(start,end,A):
            if start>end:
                return None
            node=SegmentTreeNode(start,end,A[start])
             
            if start==end:
                 return node
            
            mid=(start+end)//2
            node.left=building(start,mid,A)
            node.right=building(mid+1,end,A)
            
            if node.left  and node.left.maxi>node.maxi:
                node.maxi=node.left.maxi
            if node.right  and node.right.maxi>node.maxi:
                node.maxi=node.right.maxi
            return node
        return building(0,len(A)-1,A)
            
            

#442. Implement Trie (Prefix Tree) 
class TrieNode:
    def __init__(self):
        self.children={}
        self.hasword=False
    
class Trie:
    
    def __init__(self):
        # do intialization if necessary
        self.root=TrieNode()

    """
    @param: word: a word
    @return: nothing
    """
    def insert(self, word):
        # write your code here
        cur=self.root
        
        for i in range(len(word)):
            if word[i] not in cur.children:
                cur.children[word[i]]=TrieNode()
            cur=cur.children[word[i]]
        cur.hasword=True
                

    """
    @param: word: A string
    @return: if the word is in the trie.
    """
    def search(self, word):
        # write your code here
        cur=self.root
        
        for i in range(len(word)):
            if word[i] not in cur.children:
                return  False
            cur=cur.children[word[i]]
        return cur.hasword

    """
    @param: prefix: A string
    @return: if there is any word in the trie that starts with the given prefix.
    """
    def startsWith(self, prefix):
        # write your code here
        cur=self.root
        
        for i in range(len(prefix)):
            if prefix[i] not in cur.children:
                return  False
            cur=cur.children[prefix[i]]
        return True
        
  
        
        
        
        
            
        
        
                    
        
        
        
        
   
        
        

        
        
        
        
        
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
