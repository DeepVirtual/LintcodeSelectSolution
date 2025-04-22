# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 01:22:05 2018

@author: cz
"""

#445. Cosine Similarity 
class Solution:
    """
    @param: A: An integer array
    @param: B: An integer array
    @return: Cosine similarity
    """
    def cosineSimilarity(self, A, B):
        # write your code here
        if not A and not B:
            return 2.0000
        top=sum([x*y for x , y in zip(A,B)])
        bottom=(sum([x*x for x in A])* sum([x*x for x in B]))**0.5
        
        if bottom==0:
            return 2.0000
        else:
            return top/bottom
        
        
#448. Inorder Successor in BST
"""
Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
"""


class Solution:
    """
    @param: root: The root of the BST.
    @param: p: You need find the successor node of p.
    @return: Successor of p.
    """
    def inorderSuccessor(self, root, p):
        # write your code here
        
        def inorder(node,res):
            if not node:
                return 
            inorder(node.left)
            res.append(node.val)
            inorder(node.right)
        res=[]
        inorder(root,res)
        for i in range(1,len(res)):
            if res[i-1]==p:
                return res[i]
        
            
        
#450. Reverse Nodes in k-Group        
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: a ListNode
    @param k: An integer
    @return: a ListNode
    """
    def reverseKGroup(self, head, k):
        # write your code here
        # D->[1->2->3]->[4->5->6]->7 (k = 3)
        # D->[3->2->1]->[6->5->4]->7
        dummy=ListNode(0)
        dummy.next=head
        prev=dummy
        
        while prev:
            prev=self.reverse_next_k_node(prev,k)
        return dummy.next
    
    def find_kth_node(self,head,k):
        # head -> n1 -> n2 -> ... ->nk
        cur=head
        for _ in range(k):
            if not cur:
                return None
            cur=cur.next
        return cur
    
    def reverse(self,head):
        prev=None
        cur=head
        
        while cur:
            temp=cur.next
            cur.next=prev
            prev=cur
            cur=temp
        return prev
    
            
    def  reverse_next_k_node(self,head,k):
        # head -> n1 -> n2 -> ... ->nk -> nk+1
        # head -> nk -> nk-1 -> ... ->n1 -> nk+1
        n1=head.next
        nk=self.find_kth_node(head,k)
        if not nk:
            return None
        
        nk_plus=nk.next
        nk.next=None# separate the nk and nk+1
        
        nk=self.reverse(n1)
        # Connect head and nk -> nk-1 -> ... ->n1,  n1 and nk+1 -> nk+2 ->..
        head.next=nk
        n1.next=nk_plus
        return n1
        
        
        
#451. Swap Nodes in Pairs        
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: a ListNode
    @return: a ListNode
    """
    def swapPairs(self, head):
        # write your code here
        dummy=ListNode(0)
        dummy.next=head
        prev=dummy
        k=2
        
        while prev:
            prev=self.reverse_next_k_node(prev,k)
        return dummy.next
    
    def find_kth_node(self,head,k):
        # head -> n1 -> n2 -> ... ->nk
        cur=head
        for _ in range(k):
            if not cur:
                return None
            cur=cur.next
        return cur
    
    def reverse(self,head):
        prev=None
        cur=head
        
        while cur:
            temp=cur.next
            cur.next=prev
            prev=cur
            cur=temp
        return prev
    
            
    def  reverse_next_k_node(self,head,k):
        # head -> n1 -> n2 -> ... ->nk -> nk+1
        # head -> nk -> nk-1 -> ... ->n1 -> nk+1
        n1=head.next
        nk=self.find_kth_node(head,k)
        if not nk:
            return None
        
        nk_plus=nk.next
        nk.next=None# separate the nk and nk+1
        
        nk=self.reverse(n1)
        # Connect head and nk -> nk-1 -> ... ->n1,  n1 and nk+1 -> nk+2 ->..
        head.next=nk
        n1.next=nk_plus
        return n1
      
        
#452. Remove Linked List Elements
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: a ListNode
    @param val: An integer
    @return: a ListNode
    """
    def removeElements(self, head, val):
        # write your code here
#1->2->3->3->4->5->3, val = 3, you should return the list as 1->2->4->5  
        
        dummy=ListNode(-1)
        dummy.next=head
        
        cur=dummy
        while cur:
            while cur.next and cur.next.val==val:
                cur.next=cur.next.next
            cur=cur.next
        return dummy.next
        
        
        
        
#453. Flatten Binary Tree to Linked List        
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
    def flatten(self, root):
        # write your code here
        if not root:
            return None
        
        self.flatten(root.left)
        self.flatten(root.right)
        
        if not root.left:
            return 
        
        p=root
        
        p=p.left
        
        while p.right:
            p=p.right
        p.right=root.right
        root.right=root.left
        root.left=None
        
        
#454. Rectangle Area
class Rectangle:

    '''
     * Define a constructor which expects two parameters width and height here.
    '''
    # write your code here
    
    def __init__(self,length,width):
        self.length=length
        self.width=width
    
    
    '''
     * Define a public method `getArea` which can calculate the area of the
     * rectangle and return.
    '''
    # write your code here
    def getArea(self):
        return self.length*self.width


#457. Classical Binary Search
class Solution:
    """
    @param: nums: An integer array sorted in ascending order
    @param: target: An integer
    @return: An integer
    """
    def findPosition(self, nums, target):
        # write your code here
        if not nums:
            return -1
        n=len(nums)
        
        if n==1:
            if target==nums[0]:
                return 0
            else:
                return -1
        l=0
        r=n-1
        
        while l<=r:
            mid=(l+r)//2
            
            if nums[mid]>target:
                r=mid-1
            elif nums[mid]<target:
                l=mid+1
            else:
                return mid
        if l < len(nums) or l>=0 or nums[l]!=target:
            return -1
        else:
            return 1
          
#Given [1, 2, 2, 4, 5, 5]
#For target = 2, return 1 or 2
#For target = 5, return 4 or 5
#For target = 6, return -1
nums=[1, 2, 2, 4, 5, 5]
target=4
nums=[100,156,189,298,299,300,1001,1002,1003,1004]
target=1000
if __name__ == "__main__":
    print(Solution().findPosition(nums, target))             
                
#455. Student ID
class Student:
    def __init__(self, id):
        self.id = id;

class Class:
     def __init__(self, n):
         self.students=[]
         for i in range(n):
             self.students.append(Student(i))
             

    '''
     * Declare a constructor with a parameter n which is the total number of
     * students in the *class*. The constructor should create n Student
     * instances and initialized with student id from 0 ~ n-1
    '''
    # write your code here


#460. Find K Closest Elements
class Solution:
    """
    @param A: an integer array
    @param target: An integer
    @param k: An integer
    @return: an integer array
    """
    def kClosestNumbers(self, A, target, k):
        # write your code here
        def isLeftCloser(A,left,right,target):
            if left <0:
                return False
            if right>=len(A):
                return True
            
            if abs(A[left]-target) <= abs(A[right]-target):
                return True
            else:
                return False
        
        def findLowerClosest(A,target):
            l=0
            r=len(A)-1
            while l+1<r:
                mid=(l+r)//2
                
                if A[mid]<target:
                    l=mid
                else:
                    r=mid
            if A[r]<target:
                return r
            if A[l]<target:
                return l
            return -1
            
        left=findLowerClosest(A,target)
        right=left+1
        
        res=[]
        for i in range(k):
            if isLeftCloser(A,left,right,target):
                res.append(A[left])
                left-=1
            else:
                res.append(A[right])
                right+=1
        return res
A = [1, 2, 3]
target = 2 
k = 3
A = [1, 4, 6, 8]
target = 3 
k = 3
A =[22,25,100,209,1000,1110,1111]
target =15
k =5
if __name__ == "__main__":
    print(Solution().kClosestNumbers(A, target, k))             
                            


#463. Sort Integers
class Solution:
    """
    @param A: an integer array
    @return: nothing
    """
    def sortIntegers(self, A):
        # write your code here
        
        def partition(A,start,end):
            pivot=A[start]
            left=start+1
            right=end
            
            while left<=right:
                while left<=right  and A[left]<=pivot:
                    left+=1
                while left<=right  and A[right]>=pivot:
                    right-=1
                if left<=right:
                    
                    A[left],A[right]=A[right],A[left]
            A[start],A[right]=A[right],A[start]
            return right
        def quicksort(A,start,end):
            if start>=end:
                return 
            partition_point=partition(A,start,end)
            quicksort(A,start,partition_point-1)
            
            quicksort(A,partition_point+1,end)
        
        quicksort(A,0,len(A)-1)   
        print(A)
A=[3, 1, 1, 4, 4]  
A=[1]          
if __name__ == "__main__":
    print(Solution().sortIntegers(A))                                 
            

#464. Sort Integers II
class Solution:
    """
    @param A: an integer array
    @return: nothing
    """
    def sortIntegers2(self, A):
        # write your code here
        def merge(a,b):
            if not a:
                return b
            if not b:
                return a
            
            na=len(a)
            nb=len(b)
            
            pa=0
            pb=0
            
            res=[]
            while pa<na and pb<nb:
                if a[pa]<b[pb]:
                    res.append(a[pa])
                    pa+=1
                else:
                    res.append(b[pb])
                    pb+=1
            while pa<na:
                res.append(a[pa])
                pa+=1
            while pb<nb:
                res.append(b[pb])
                pb+=1
            return res
        
        def mergeSort(A):
            n=len(A)
            if n==1:
                return A
            
            mid=(0+n)//2
            a=mergeSort(A[:mid])
            b=mergeSort(A[mid:])
            
            return merge(a,b)
        
        res=mergeSort(A)   
        
        for i in range(len(A)):
            A[i]=res[i]
        return res
A=[3,2,1,4,5]  
A=[1,1]     
A=[1,3,7,7,8] 
A=[1,3,7,7,8,4,19,11,3]    
if __name__ == "__main__":
    print(Solution().sortIntegers2(A))            


#466. Count Linked List Nodes
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: the first node of linked list.
    @return: An integer
    """
    def countNodes(self, head):
        # write your code here
        if not head:
            return 0
        cur=head
        count=0
        while cur:
            count+=1
            cur=cur.next
        return count
            


#469. Same Tree
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param a: the root of binary tree a.
    @param b: the root of binary tree b.
    @return: true if they are identical, or false.
    """
    def isIdentical(self, a, b):
        # write your code here
        
        def isSame(x,y):
            if not x and y:
                return False
            elif not y and x:
                return False
            elif x.val !=y.val:
                return False
            else:
                return isSame(x.left,y.left)  and isSame(x.right,y.right)
        return isSame(a,b)



#471. Top K Frequent Words
class Solution:
    """
    @param words: an array of string
    @param k: An integer
    @return: an array of string
    """
    def topKFrequentWords(self, words, k):
        # write your code here
        from collections import Counter
        c=Counter(words)
        print(c)
        res=[]
        for key,v in c.items():
            res.append((v,key))
        
        res.sort(key=lambda x:(-x[0],x[1]))
        print(res)
        ans=[]
        print(len(res))
        print(k)
        for i in range( min(len(res),k)):
            ans.append(res[i][1])
        return ans

words=["yes","lint","code","yes","code","baby","you","baby","chrome","safari",
       "lint","code","body","lint","code"]
k=3
if __name__ == "__main__":
    print(Solution().topKFrequentWords(words, k))              




#473. Add and Search Word - Data structure design
class TrieNode:
    def __init__(self):
        self.children={}
        self.hasword=False
class WordDictionary:
    def __init__(self):
        self.root=TrieNode()
    """
    @param: word: Adds a word into the data structure.
    @return: nothing
    """
    def addWord(self, word):
        # write your code here
        cur=self.root
        for w in  word:
            if w not in cur.children:
                cur.children[w]=TrieNode()
            cur=cur.children[w]
        cur.hasword=True

    """
    @param: word: A word could contain the dot character '.' to represent any one letter.
    @return: if the word is in the data structure.
    """
    def search(self, word):
        # write your code here
        def searchFrom(tries,word):
            cur=tries
            for i,w in enumerate(word):
                if w=='.':
                    for k in cur.children:
                        if searchFrom(cur.children[k],word[i+1:]):
                            return True
                    return False
                elif  w not in cur.children:
                    return False
                cur=cur.children[w]
            return cur.hasword
        
        return searchFrom(self.root,word)
ss = WordDictionary() 
ss.addWord("bad")
ss.addWord("dad")
ss.addWord("mad")
ss.search("pad") 
ss.search("bad")
ss.search(".ad") 
ss.search("b..")
ss.search("b...")

#474. Lowest Common Ancestor II
"""
Definition of ParentTreeNode:
class ParentTreeNode:
    def __init__(self, val):
        self.val = val
        self.parent, self.left, self.right = None, None, None
"""


class Solution:
    """
    @param: root: The root of the tree
    @param: A: node in the tree
    @param: B: node in the tree
    @return: The lowest common ancestor of A and B
    """
    def lowestCommonAncestorII(self, root, A, B):
        # write your code here
        if A==root or B==root:
            return root
        if not root:
            return None
        
        if A==B:
            return A
        
        curA=A
        Aparent=[]
        while curA:
            Aparent.append(curA)
            curA=curA.parent
            
        curB=B
        Bparent=[]
        while curB:
            Bparent.append(curB)
            curB=curB.parent
            
        AL=len(Aparent)
        BL=len(Bparent)
        print(AL,BL)
        if BL>AL:
            AL, BL =  BL ,AL
            Aparent ,Bparent= Bparent, Aparent
            
        gap=AL-BL
        
        Ai=gap
        Bi=0
        
        while Ai<AL and Bi<BL:
            if Aparent[Ai]==Bparent[Bi]:
                return Bparent[Bi].val
            Ai+=1
            Bi+=1
            
            
root=  ParentTreeNode(1)
root.left=   ParentTreeNode(2) 
root.right=   ParentTreeNode(3)   

root.left.parent=root 
root.right.parent=root

root.right.right=   ParentTreeNode(4)  
root.right.right.parent= root.right

A=  root.left
B=root.right.right 
if __name__ == "__main__":
    print(Solution().lowestCommonAncestorII(root, A, B))              
            
#476. Stone Game        
class Solution:
    """
    @param A: An integer array
    @return: An integer
    """
    def stoneGame(self, A):
        # write your code here
#        self.ans=float('inf')
#        def add(B,res,memo):
#            n=len(B)
#            
#            if tuple(B) in memo:
#                return memo[tuple(B)]
#                       
#            if n==2:
#                 temp= res+B[0]+B[1]
#                 memo[tuple(B)]=temp
#                 if temp<self.ans:
#                     self.ans=temp
#                 return 
#                     
#                
#            
#            for i in range(n-1):
#                
#                tmp=B[i] +B[i+1] 
#                re=add(B[:i]+[ B[i] +B[i+1] ] +B[i+2:],res+ tmp)
#                
#                
#                #print(temp,B[i] , B[i+1],B[:i]+[ B[i] +B[i+1] ] +B[i+2:]  )
#               
#        add(A,0)
#        return self.ans if self.ans<float('inf') else 0
        
        
        n=len(A)
        if n==0:
            return 0
        
        sumA=[[0 for _ in range(n)] for _ in range(n)]
        dp=[[0 for _ in range(n)] for _ in range(n)]
        
        
        for i in range(n):
            sumA[i][i]=A[i]
            dp[i][i]=0
            for j in range(i+1,n):
                sumA[i][j]= sumA[i][j-1]+A[j]
                
        
        for length in range(2,n+1):
            for i in range(n):
                j=i+length-1
                if j<n:
                    tempmin=float('inf')
                    for k in range(i,j):
#                        print(dp[i][k])
#                        print(dp[k+1][j])
#                        print(sumA[i][j])
#                        print()
                        tempmin=min(tempmin,dp[i][k]+dp[k+1][j])
                        
                    dp[i][j]=tempmin+sumA[i][j]
        return dp[0][n-1]
A=[4, 1, 1, 4]   
A=[1, 1, 1, 1]
A=[4, 4, 5, 9]             
if __name__ == "__main__":
    print(Solution().stoneGame( A))             
        
        
        







#477. Surrounded Regions
class Solution:
    """
    @param: board: board a 2D board containing 'X' and 'O'
    @return: nothing
    """
    def surroundedRegions(self, board):
        # write your code here
#X X X X
#X O O X
#X X O X
#X O X X        
       
       
#X X X X
#X X X X
#X X X X
#X O X X        
        m=len(board)
        if m==0 or m==1  or m==2:
          return 
        n=len(board[0])
        if n==0 or n==1 or n==2:
          return
        from collections import deque
        def fill(board,i,j):
            if board[i][j]!='O':
                return 
            
            q=deque([(i,j)])
            board[i][j]='W'
            while q:
                a,b=q.popleft()
                
                for x , y in ((a+1,b),(a-1,b),(a,b+1),(a,b-1)):
                    if x>=0 and y>=0 and x<len(board)  and y<len(board[0]) and board[x][y]=='O':
                        q.append((x,y))
                        board[x][y]='W'
        board2=[list(row)  for row in board]
        for i in range(m):
            fill(board2,i,0)
            fill(board2,i,n-1)
        for j in range(n):
            fill(board2,0,j)
            fill(board2,m-1,j)
        
        print(board2)
        for i in range(m):
            for j in range(n):
                if board2[i][j]=='W':
                    board2[i][j]='O'
                elif board2[i][j]=='O':
                    board2[i][j]='X'
        for i in range(m):
            board[i]=''.join(board2[i])
            
        print(board)
 
        
board=["XXXX","XOOX","XXOX","XOXX"] 
board=["XXXX","XOOX","XXOX","XOOO"] 
           
if __name__ == "__main__":
    print(Solution().surroundedRegions( board))              


#478. Simple Calculator
class Calculator:
    """
    @param a: An integer
    @param operator: A character, +, -, *, /.
    @param b: An integer
    @return: The result
    """
    def calculate(self, a, operator, b):
        if operator=='+':
            return a+b
        elif operator=='-':
            return a-b
        elif operator=='*':
            return a*b
        else:
            return a/b
        
              
              
#479. Second Max of Array
class Solution:
    """
    @param nums: An integer array
    @return: The second max number in the array.
    """
    def secondMax(self, nums):
        # write your code here
        n=len(nums)
        if n==0 or n==1:
            return None
        
        if n==2:
            return min(nums)
        max1=float('-inf')
        max2=float('-inf')
        max3=float('-inf')
        
        for x in nums:
            if x >= max1:
                if max1>=max2:
                    max2=max1
                max1=x
            elif x>=max2:
                max2=x
            
                
                
        return max2
nums=    [1, 3, 2, 4]
if __name__ == "__main__":
    print(Solution().secondMax( nums))    
            
            
     
#480. Binary Tree Paths                 
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of the binary tree
    @return: all root-to-leaf paths
    """
    def binaryTreePaths(self, root):
        # write your code here
        if not root:
            return []
        if not root.left and not root.right:
            return [str(root.val)]
        
        res=[]
        def walk(node,res,path):
            if not node.left and not node.right:
                res.append(path[:])
                return 
            if node.left:
                walk(node.left,res,path+'->'+str(node.left.val))
            if node.right:
                walk(node.right,res,path+'->'+str(node.right.val))
        walk(root,res,str(root.val))
        return res

            
#484. Swap Two Integers in Array        
class Solution:
    """
    @param A: An integer array
    @param index1: the first index
    @param index2: the second index
    @return: nothing
    """
    def swapIntegers(self, A, index1, index2):
        # write your code here
        A[index1], A[index2] = A[index2] ,A[index1]
                  
        
        
        
        
#486. Merge K Sorted Arrays 
class Solution:
    """
    @param arrays: k sorted integer arrays
    @return: a sorted array
    """
    def mergekSortedArrays(self, arrays):
        # write your code here
        import heapq
        
        hp=[]
        for ai,array in enumerate(arrays):
            if not array:
                continue
            heapq.heappush(hp,(array[0],ai,0))
        res=[]
        while hp:
            value,ai,vi=heapq.heappop(hp)
            res.append(value)
            if vi+1<len(arrays[ai]):
               heapq.heappush(hp,(arrays[ai][vi+1],ai,vi+1))
        return res
        
arrays=[
  [1, 3, 5, 7],
  [2, 4, 6],
  [0, 8, 9, 10, 11]
]                    
if __name__ == "__main__":
    print(Solution().mergekSortedArrays( arrays))         

        
        
#487. Name Deduplication        
class Solution:
    """
    @param names: a string array
    @return: a string array
    """
    def nameDeduplication(self, names):
        # write your code here  
        if not names:
            return []
        nameset=set()
        for name in names:
            nameset.add(name.lower())
        return list(nameset)
names=["James", "james", "Bill Gates", "bill Gates", "Hello World", "HELLO WORLD", "Helloworld"]             
if __name__ == "__main__":
    print(Solution().nameDeduplication(names))         
        
#488. Happy Number
class Solution:
    """
    @param n: An integer
    @return: true if this is a happy number or false
    """
    def isHappy(self, n):
        # write your code here
        if n==1:
            return False
        
        Nset=set()
        
        def digitSum(n):
            res=0
            while n:
                res+=(n%10)**2
                n=n//10
            return res
        
        
        while True:
              Nsum=digitSum(n)
              #print(Nsum)
              if Nsum ==1:
                  return True
              if Nsum not in Nset:
                  Nset.add(Nsum)
                  n=Nsum
                  
              else:
                  return False
n=19             
if __name__ == "__main__":
    print(Solution().isHappy( n))                  
            
            
#491. Palindrome Number
class Solution:
    """
    @param num: a positive number
    @return: true if it's a palindrome or false
    """
    def isPalindrome(self, num):
        # write your code here
        
        def check(s):
            if not s:
                return True
            n=len(s)
            if n==1:
                return True
            if s[0]!=s[-1]:
                return False
            else:
                return check(s[1:-1])
        return check(str(num))
num=12321        
if __name__ == "__main__":
    print(Solution().isPalindrome( num))        
              
#495. Implement Stack
class Stack:
    """
    @param: x: An integer
    @return: nothing
    """
    def __init__(self):
        self.array=[]
    def push(self, x):
        # write your code here
        self.array.append(x)

    """
    @return: nothing
    """
    def pop(self):
        # write your code here
        if not self.isEmpty():
            val=self.array[-1]
            del self.array[-1]
            
            return  val
        else:
            return None 

    """
    @return: An integer
    """
    def top(self):
        # write your code here
        if not self.isEmpty():
          return  self.array[-1]
        else:
            return None

    """
    @return: True if the stack is empty
    """
    def isEmpty(self):
        # write your code here
        if len(self.array)==0:
            return True
        else:
            return False
    
    
    
        
#496. Toy Factory
class Toy:
    def talk(self):
        raise NotImplementedError('This method should have implemented.')

class Dog(Toy):
    # Write your code here
    def talk(self):
        print('Wow')


class Cat(Toy):
    # Write your code here
    def talk(self):
        print('Meow')


class ToyFactory:
    # @param {string} shapeType a string
    # @return {Toy} Get object of the type
    def getToy(self, type):
        # Write your code here
        if type=='Dog':
            return Dog()
        if type=='Cat':
            return Cat()
        return 
            
            

#499. Word Count (Map Reduce)
class WordCount:

    # @param {str} line a text, for example "Bye Bye see you next"
    def mapper(self, _, line):
        # Write your code here
        # Please use 'yield key, value'

        for x in line.split():
            yield x,1
    # @param key is from mapper
    # @param values is a set of value with the same key
    def reducer(self, key, values):
        # Write your code here
        # Please use 'yield key, value'  
        yield key,sum(values)          
        
        
        
        
        
        
        
#501. Design Twitter        
'''
Definition of Tweet:
class Tweet:
    @classmethod
    def create(cls, user_id, tweet_text):
         # This will create a new tweet object,
         # and auto fill id
'''

from collections import defaultdict
import heapq
class MiniTwitter:
    
    def __init__(self):
        # do intialization if necessary
        self.map=defaultdict(list)
        self.time=0
        self.follow_id=defaultdict(list)
        

    """
    @param: user_id: An integer
    @param: tweet_text: a string
    @return: a tweet
    """
    def postTweet(self, user_id, tweet_text):
        # write your code here
        new_tweet=Tweet.create( user_id, tweet_text)
        self.map[user_id].append(( self.time, new_tweet    ))
        self.time+=1
        return new_tweet
        

    """
    @param: user_id: An integer
    @return: a list of 10 new feeds recently and sort by timeline
    """
    def getNewsFeed(self, user_id):
        # write your code here
        heap=[]
        for follow in self.follow_id[user_id]:
            for time , t in self.map[follow]:
                heapq.heappush(heap,(    -time, t ))
        
        for time , t in self.map[user_id]: 
                heapq.heappush(heap,(    -time, t ))
        n=len(heap)
        
        i=0
        res=[]
        while i < 10 and i < n:
            time,tweet=heapq.heappop(heap)
            res.append(tweet)
            i+=1
        return res

            
            
        
       

    """
    @param: user_id: An integer
    @return: a list of 10 new posts recently and sort by timeline
    """
    def getTimeline(self, user_id):
        # write your code here
        heap=[]
        
        
        for time , t in self.map[user_id]: 
                heapq.heappush(heap,(    -time, t ))
        n=len(heap)
        
        i=0
        res=[]
        while i < 10 and i < n:
            time,tweet=heapq.heappop(heap)
            res.append(tweet)
            i+=1
        return res

        
        
    
        

    """
    @param: from_user_id: An integer
    @param: to_user_id: An integer
    @return: nothing
    """
    def follow(self, from_user_id, to_user_id):
        # write your code here
        if to_user_id not in self.follow_id[from_user_id]:
            
           self.follow_id[from_user_id].append(to_user_id )

    """
    @param: from_user_id: An integer
    @param: to_user_id: An integer
    @return: nothing
    """
    def unfollow(self, from_user_id, to_user_id):
        
        # write your code here   
        if to_user_id  in self.follow_id[from_user_id]:
            self.follow_id[from_user_id].remove(to_user_id )
            
       
        
        
        
postTweet(1, "LintCode is Good!!!")
getNewsFeed(1)

follow(2, 1)
getNewsFeed(2)
unfollow(2, 1)
getNewsFeed(2)
        
        

        
#504. Inverted Index (Map Reduce)
'''
Definition of Document
class Document:
    def __init__(self, id, cotent):
        self.id = id
        self.content = content
'''
class InvertedIndex:

    # @param {Document} value is a document
    def mapper(self, _, value):
        # Write your code here
        # Please use 'yield key, value' here


    # @param key is from mapper
    # @param values is a set of value with the same key
    def reducer(self, key, values):
        # Write your code here
        # Please use 'yield key, value' here        
        
        
        
        
#507. Wiggle Sort II
class Solution:
    """
    @param: nums: A list of integers
    @return: nothing
    """
    def wiggleSort(self, nums):
        # write your code here
        
#https://leetcode.com/problems/wiggle-sort-ii/discuss/125429/Python-solution-with-virtual-indexing        
#https://leetcode.com/problems/wiggle-sort-ii/discuss/77682/Step-by-step-explanation-of-index-mapping-in-Java
        
        def partition(A,start,end):
            import random
            pivot_index=random.randrange(start,end+1)
            pivot=A[pivot_index]
            
            A[pivot_index],A[end]=A[end],A[pivot_index]
            partition_index=start
            for i in range(start,end):
                if A[i]>=pivot:
                    A[partition_index], A[i] =A[i],A[partition_index]
                    partition_index+=1
            A[partition_index],A[end]=A[end],A[partition_index] 
            return partition_index
                
        def quickselct(A,start,end,k):
            if start==end:
                return A[start]
            cur_index=partition(A,start,end)
            
            if cur_index==k:
                return A[k]
            elif cur_index>k:
                return quickselct(A,start,cur_index-1,k)
            else:
                return quickselct(A,cur_index+1,end,k)
        
        n=len(nums)
        if n==1:
            return 
        f=lambda i: (1+2*i)%(n|1)
        
        mid=quickselct(nums,0,n-1,n//2)
        
        k=n-1
        i=0
        j=0
        
        while j<=k:
            if nums[f(j)]>mid:
                nums[f(j)],nums[f(i)]=nums[f(i)],nums[f(j)]
                j+=1
                i+=1
            elif nums[f(j)]<mid:
                nums[f(j)],nums[f(k)]=nums[f(k)],nums[f(j)]
                k-=1
            else:
                j+=1
                
        print(mid)
        print(nums)        
nums=[1, 5, 1, 1, 6, 4]
nums=[1, 3, 2, 2, 3, 1]                
if __name__ == "__main__":
    print(Solution().wiggleSort( nums))                 
                
                
#508. Wiggle Sort
class Solution:
    """
    @param: nums: A list of integers
    @return: nothing
    """
    
    def wiggleSort(self, nums):
        # write your code here
        if not nums:
            return 
        def partition(A,start,end):
            import random
            pivot_index=random.randrange(start,end+1)
            pivot=A[pivot_index]
            
            A[pivot_index],A[end]=A[end],A[pivot_index]
            partition_index=start
            for i in range(start,end):
                if A[i]>=pivot:
                    A[partition_index], A[i] =A[i],A[partition_index]
                    partition_index+=1
            A[partition_index],A[end]=A[end],A[partition_index] 
            return partition_index
                
        def quickselct(A,start,end,k):
            if start==end:
                return A[start]
            cur_index=partition(A,start,end)
            
            if cur_index==k:
                return A[k]
            elif cur_index>k:
                return quickselct(A,start,cur_index-1,k)
            else:
                return quickselct(A,cur_index+1,end,k)
        
        n=len(nums)
        if n==1:
            return 
        f=lambda i: (1+2*i)%(n|1)
        
        mid=quickselct(nums,0,n-1,n//2)
        
        k=n-1
        i=0
        j=0
        
        while j<=k:
            if nums[f(j)]>mid:
                nums[f(j)],nums[f(i)]=nums[f(i)],nums[f(j)]
                j+=1
                i+=1
            elif nums[f(j)]<mid:
                nums[f(j)],nums[f(k)]=nums[f(k)],nums[f(j)]
                k-=1
            else:
                j+=1
                
            
#510. Maximal Rectangle
class Solution:
    """
    @param matrix: a boolean 2D matrix
    @return: an integer
    """
    def maximalRectangle(self, matrix):
        # write your code here
        m=len(matrix)
        if m==0:
            return 0
        n=len(matrix[0])
        if n==0:
            return 0
        heights=[0 for _ in range(n+1)]
        
        ans=0
        for row in matrix:
            for i in range(n):
                heights[i]=heights[i]+row[i] if row[i]==1 else 0
                
                stack=[-1]
            for j in range(n+1):
                while heights[j]< heights[stack[-1]]:
                    h=heights[stack.pop()]
                    w=j-stack[-1]-1
                    ans=max(ans,h*w)
                stack.append(j)
        return ans
matrix=[[1,1,0,0,1],[0,1,0,0,1],[0,0,1,1,1],[0,0,1,1,1],[0,0,0,0,1]]                    
if __name__ == "__main__":
    print(Solution().maximalRectangle( matrix))              
        
        
#511. Swap Two Nodes in Linked List
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: a ListNode
    @param v1: An integer
    @param v2: An integer
    @return: a new head of singly-linked list
    """
    def swapNodes(self, head, v1, v2):
        # write your code here
       
        dummy=ListNode(0)
        dummy.next=head
        if not head:
            return None
        if v1==v2:
            return  head
        
        def findPrevs(v1, v2):
            prev1=None
            prev2=None
            cur=dummy
            while cur.next:
                if cur.next.val==v1:
                    prev1=cur
                if cur.next.val==v2:
                    prev2=cur
                cur=cur.next
            return prev1,prev2
        
        # dummy->head->..->prev->node1->node2->post...
        # swap node1 & node2
        def swapAdjcent( prev): 
            node1=prev.next
            node2=node1.next            
            post=node2.next
            
            prev.next=node2
            node2.next=node1
            node1.next=post
        
        
        # dummy->head->..->prev1->node1->post1->...->prev2->node2->post2...
        # swap node1 & node2
        def swapRemote( prev1, prev2):
            node1=prev1.next
            post1=node1.next
            
            node2=prev2.next
            post2=node2.next
            
            prev1.next=node2
            node2.next=post1
            
            prev2.next=node1
            node1.next=post2
        
        prev1,prev2=findPrevs(v1, v2)
        
        if not prev1  or not prev2:
            return head
        
        if prev1.next==prev2:
            swapAdjcent( prev1)
        elif  prev2.next==prev1:
            swapAdjcent( prev2)
        else:
            swapRemote(prev1, prev2)
        return dummy.next
    
    
    
    
    
            
class Solution:
    """
    @param s: a string,  encoded message
    @return: an integer, the number of ways decoding
    """
    def numDecodings(self, s):
        # write your code here
        

#'A' -> 1
#'B' -> 2
#...
#'Z' -> 26        
        n=len(s)
        if  n==0:
            return n  
        if n==1 and s=='0' :
            return 0
        if n==1:
            return n
        dp=[0 for _ in range(n+1)]
        dp[1]=1
        dp[0]=1
        
        for i in range(2,n+1):
            if s[i-1]=='0':
                if s[i-2] in ('3','4','5','6','7','8','9','0'):
                    return 0
                dp[i]=dp[i-2]
            else:
                if s[i-2] in ('0','3','4','5','6','7','8','9'):
                    dp[i]=dp[i-1]
                elif s[i-2]=='1':
                    dp[i]=dp[i-1]+dp[i-2]
                elif s[i-2]=='2':
                    if s[i-1] in ('1','2','3','4','5','6'):
                        dp[i]=dp[i-1]+dp[i-2]
                    elif s[i-1] in ('7','8','9'):
                        dp[i]=dp[i-1]
        return dp[-1]

s='122'
if __name__ == "__main__":
    print(Solution().numDecodings(s))                        
                
                
#513. Perfect Squares            
class Solution:
    """
    @param n: a positive integer
    @return: An integer
    """
    def numSquares(self, n):
        # write your code here
        
        if n==1:
            return 1
        if n==2:
            return 2
        
        dp =[ n for _ in range(n+1)]
        
        i=0
        while i*i<=n:
            dp[i*i]=1
            i+=1
        
        for i in range(1,n+1):
            j=0
            while i+j*j<=n:
                dp[i+j*j]=min(dp[i+j*j],dp[i]+1)
                j+=1
        return dp[n]
            
            
#514. Paint Fence                
class Solution:
    """
    @param n: non-negative integer, n posts
    @param k: non-negative integer, k colors
    @return: an integer, the total number of ways
    """
    def numWays(self, n, k):
        # write your code here
#        根据题意，不能有超过连续两根柱子是一个颜色，也就意味着第三根柱子要么根第一个柱子不是一
#        个颜色，要么跟第二根柱子不是一个颜色。如果不是同一个颜色，计算可能性的时候就要去掉之前的
#        颜色，也就是k-1种可能性。假设dp[1]是第一根柱子及之前涂色的可能性数量，dp[2]是第二根柱子及
#        之前涂色的可能性数量，则dp[3]=(k-1)*dp[1] + (k-1)*dp[2]。
#      post 1,   post 2, post 3
#way1    0         0       1 
#way2    0         1       0
#way3    0         1       1
#way4    1         0       0
#way5    1         0       1
#way6    1         1       0    
        dp=[0,k,k*k]
        
        if n<=2:
            return dp[n]
        if n>2 and k==1:
            return 0
        
        for i in range(3,n+1):
            dp.append((k-1)*(dp[-2]+dp[-1]))
        return dp[n]

n=3
k=2 #return 6
n=0
k=0
if __name__ == "__main__":
    print(Solution().numWays( n, k))                
            
#515. Paint House            
class Solution:
    """
    @param costs: n x 3 cost matrix
    @return: An integer, the minimum cost to paint all houses
    """
    def minCost(self, costs):
        # write your code here
      
        n=len(costs)
        if n==0:
            return 0
        if n==1:
            return min(costs[0]) 
        
        dp=[[0 for _ in range(3)]  for _ in range(n)]
        dp[n-1]=costs[n-1]
        
        for i in range(n-2,-1,-1):
            dp[i][0]=costs[i][0]+min(dp[i+1][1],dp[i+1][2])
            dp[i][1]=costs[i][1]+min(dp[i+1][0],dp[i+1][2])
            dp[i][2]=costs[i][2]+min(dp[i+1][0],dp[i+1][1])
        return min(dp[0])

        
        
costs=[[14,2,11],[11,14,5],[14,3,10]]
costs=[[3,5,3],[6,17,6],[7,13,18],[9,10,18]]
if __name__ == "__main__":
    print(Solution().minCost( costs))                
                               
#516. Paint House II
class Solution:
    """
    @param costs: n x k cost matrix
    @return: an integer, the minimum cost to paint all houses
    """
    def minCostII(self, costs):
        # write your code here
#        n=len(costs)
#        if n==0:
#            return 0
#        if n==1:
#            return min(costs[0]) 
#        
#        k=len(costs[0])
#        
#        dp=[[0 for _ in range(k)]  for _ in range(n)]
#        dp[n-1]=costs[n-1]
#        
#        for i in range(n-2,-1,-1):
#            for j in range(k):
#                dp[i][j]=costs[i][j]+min(dp[i+1][:j]+dp[i+1][j+1:])
#            
#            
#        return min(dp[0])
        n=len(costs)
        if n==0:
            return 0
        if n==1:
            return min(costs[0]) 
        
        K=len(costs[0])
        
        dp=[[0 for _ in range(K)]  for _ in range(n+1)]
       
        
        for i in range(1,n+1):
            a=-1
            b=-1
            
            for k in range(K):
                if a==-1 or dp[i-1][k]<dp[i-1][a]:
                    b=a
                    a=k
                else:
                    if b==-1 or dp[i-1][k]<dp[i-1][b]:
                        b=k
                        
                
            
            
            for j in range(K):
                if a!=j:
                    dp[i][j]=dp[i-1][a]+costs[i-1][j]
                else:
                    dp[i][j]=dp[i-1][b]+costs[i-1][j]
            
            
        return  min(dp[n])
            
costs=[[3,5,3],[6,17,6],[7,13,18],[9,10,18]]        
        
if __name__ == "__main__":
    print(Solution().minCostII( costs))                 
        
        
#517. Ugly Number        
class Solution:
    """
    @param num: An integer
    @return: true if num is an ugly number or false
    """
    def isUgly(self, num):
        # write your code here
        
        if num==1:
            return True
        while num!=1:
            start=num
            if num%2==0:
                num//=2
            if num%3==0:
                num//=3
            if num%5==0:
                num//=5
            end=num
            if start==end  and num!=1:
                return False
            if num==1:
                return True
num=8
num=14
if __name__ == "__main__":
    print(Solution().isUgly( num))                 
                    
            
#518. Super Ugly Number
class Solution:
    """
    @param n: a positive integer
    @param primes: the given prime list
    @return: the nth super ugly number
    """
    def nthSuperUglyNumber(self, n, primes):
        # write your code here
        m=len(primes)
        Pindex=[0 for _ in range(m)]
        import heapq
        ugly=[1]
        
        minlist=[(primes[i]*ugly[Pindex[i]],i) for i in range(m)]
        heapq.heapify(minlist)
        
        while len(ugly)<n:
            value, index=heapq.heappop(minlist)
            Pindex[index]+=1
            if ugly[-1]!=value:
                ugly.append(value)
            heapq.heappush(minlist, (primes[index] * ugly[Pindex[index]],index) )
        return ugly[-1]    
n=6
primes=[2,7,13,19]
n=1
primes=[17,2,3,5,7,97,31]            
if __name__ == "__main__":
    print(Solution().nthSuperUglyNumber( n, primes))

#523. Url Parser
class HtmlParser:
    """
    @param: content: content source code
    @return: a list of links
    """
    def parseUrls(self, content):
        # write your code here
        import re
        links=re.findall(r'\s*(?i)href\s*=\s*("|\')+([^"\'<\s]*)',content,re.I)
        return [link[1] for link in links if  link[1] and not link[1].startswith('#')]
        
content=
if __name__ == "__main__":
    print(Solution().parseUrls(content))                 
                    
#524. Left Pad
class StringUtils:
    """
    @param: originalStr: the string we want to append to
    @param: size: the target length of the string
    @param: padChar: the character to pad to the left side of the string
    @return: A string
    """
    @classmethod
    def leftPad(self, originalStr, size, padChar=' '):
        # write your code here
        n=len(originalStr)
        add=size-n
        if add<0:
            return originalStr
        return padChar*add+originalStr
originalStr="foo"
size= 5
if __name__ == "__main__":
    print(StringUtils().leftPad( originalStr, size, padChar=' '))                 


#526. Load Balancer
class LoadBalancer:
    def __init__(self):
        # do intialization if necessary
        self.stack=[]

    """
    @param: server_id: add a new server to the cluster
    @return: nothing
    """
    def add(self, server_id):
        # write your code here
        self.stack.append(server_id)

    """
    @param: server_id: server_id remove a bad server from the cluster
    @return: nothing
    """
    def remove(self, server_id):
        # write your code here
        if server_id  in self.stack:
            self.stack.remove(server_id)
        else:
            return 
        
        

    """
    @return: pick a server in the cluster randomly with equal probability
    """
    def pick(self):
        # write your code here
        from random import randint
        n=len(self.stack)
        if n==0:
            return 
        
        return self.stack[randint(0,n)]



#528. Flatten Nested List Iterator
"""
This is the interface that allows for creating nested lists.
You should not implement it, or speculate about its implementation

class NestedInteger(object):
    def isInteger(self):
        # @return {boolean} True if this NestedInteger holds a single integer,
        # rather than a nested list.

    def getInteger(self):
        # @return {int} the single integer that this NestedInteger holds,
        # if it holds a single integer
        # Return None if this NestedInteger holds a nested list

    def getList(self):
        # @return {NestedInteger[]} the nested list that this NestedInteger holds,
        # if it holds a nested list
        # Return None if this NestedInteger holds a single integer
"""

class NestedIterator(object):

    def __init__(self, nestedList):
        # Initialize your data structure here.
        
        self.stack=nestedList[::-1]
        
        
    # @return {int} the next element in the iteration
    def next(self):
        # Write your code here
        return self.stack.pop().getInteger()
        
            
            
        
    # @return {boolean} true if the iteration has more element or false
    def hasNext(self):
        # Write your code here
        
        while self.stack:
            top=self.stack[-1]
            if top.isInteger():
                return True
            self.stack=self.stack[:-1]+top.getList()[::-1]
        return False

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())

#532. Reverse Pairs
class Solution:
    """
    @param A: an array
    @return: total of reverse pairs
    """
    def reversePairs(self, A):
        # write your code here
        n=len(A)
        self.tmp=[0 for _ in range(n)]
        
        def mergeSort(A,l,r):
            if l>=r:
                return 0
            mid=(l+r)>>1
            
            ans=mergeSort(A,l,mid)+mergeSort(A,mid+1,r)
            i=l
            j=mid+1
            k=l
            
            while i<=mid and j <=r:
                if A[i]>A[j]:
                    ans+=mid-i+1
                    self.tmp[k]=A[j]
                    j+=1
                else:
                    self.tmp[k]=A[i]
                    i+=1
                k+=1
            while i<=mid:
                self.tmp[k]=A[i]
                k+=1
                i+=1
            while j<=r:
                self.tmp[k]=A[j]
                k+=1
                j+=1
            for x in range(l,r+1):
                A[x]=self.tmp[x]
            return ans
        
                
        return mergeSort(A,0,n-1)
A=[2, 4, 1, 3, 5] 
if __name__ == "__main__":
    print(Solution().reversePairs( A))                 
                
                    
#534. House Robber II
class Solution:
    """
    @param nums: An array of non-negative integers.
    @return: The maximum amount of money you can rob tonight
    """
    def houseRobber2(self, nums):
        # write your code here
        
        m=len(nums)
        if m==0:
            return 0
        if m==1:
            return nums[0]
        if m==2:
            return max(nums[0],nums[1])
        
        def houseRobber(A):
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
            return dp[-1]
        
        a=houseRobber(nums[0:-1])
        b=houseRobber(nums[1:])
        return max(a,b)
            

nums = [3,6,4]# return 6            
if __name__ == "__main__":
    print(Solution().houseRobber2( nums))             
            
            
#535. House Robber III
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
    @return: The maximum amount of money you can rob tonight
    """
    def houseRobber3(self, root):
        # write your code here
        
#        def houseRobber(node):
#            if not node:
#                return 0
#            if not node.left  and not node.right:
#                return node.val
#            if not node.left and node.right:
#                s1=houseRobber(node.right.left)+houseRobber(node.right.right)+node.val
#                s2=houseRobber(node.right)
#                return max(s1,s2)
#            if node.left and not node.right:
#                s1=houseRobber(node.left.left)+houseRobber(node.left.right)+node.val
#                s2=houseRobber(node.left)
#                return max(s1,s2)
#            if node.left and node.right:
#                s1=houseRobber(node.left.left)+houseRobber(node.left.right)+node.val+houseRobber(node.right.left)+houseRobber(node.right.right)
#                s2=houseRobber(node.left)+houseRobber(node.right)
#                return max(s1,s2)
#        return houseRobber(root)
        
        
        def visit(node):
            if not node:
                return 0,0
            
            left_rob,left_not_rob=visit(node.left)
            right_rob,right_not_rob=visit(node.right)
            
            rob=left_not_rob+right_not_rob+node.val
            not_rob=max(left_rob,left_not_rob)+max(right_rob,right_not_rob)
            return rob,not_rob
        
        yes,no=visit(root)
        return max(yes,no)
            
                

#  3
# / \
#2   3
# \   \ 
#  3   1
#
#3 + 3 + 1 = 7
#
#    3
#   / \
#  4   5
# / \   \ 
#1   3   1
#4 + 5 = 9
root=TreeNode(3)
root.left=TreeNode(2)
root.right=TreeNode(3)

root.left.right=TreeNode(3)
root.right.right=TreeNode(1)


root=TreeNode(3)
root.left=TreeNode(4)
root.right=TreeNode(5)

root.left.left=TreeNode(1)
root.left.right=TreeNode(3)
root.right.right=TreeNode(1)


if __name__ == "__main__":
    print(Solution().houseRobber3( root)) 

#539. Move Zeroes
class Solution:
    """
    @param nums: an integer array
    @return: nothing
    """
    def moveZeroes(self, nums):
        # write your code here
        n=len(nums)
        if n==0:
            return 
        if n==1:
            return 

        i0=0
        i1=0
        
        while i0<n  and i1<n:
            while i0<n  and nums[i0]!=0:
                i0+=1
            i1=i0     
            while i1<n  and nums[i1]==0:
                i1+=1
                
           
            #print(i0,i1)
            if i0<n  and i1<n:
               nums[i0]  , nums[i1] = nums[i1], nums[i0]
               
               
            #break
       
               
        print(nums)
                
nums=[0,1,0,3,12]
nums=[0,1,0,3,0]      
nums=[0,0,0,0,0] 
nums=[1,2,3]           
if __name__ == "__main__":
    print(Solution().moveZeroes( nums))             


#540. Zigzag Iterator        
class ZigzagIterator:
    """
    @param: v1: A 1d vector
    @param: v2: A 1d vector
    """
    def __init__(self, v1, v2):
        # do intialization if necessary
        self.v1=v1
        self.v2=v2
        self.pointer1=True
        self.p1=0
        self.p2=0
        self.n1=len(v1)
        self.n2=len(v2)
        

    """
    @return: An integer
    """
    def next(self):
        # write your code here
        #print(self.p1,self.p2,self.pointer1)
        if self.hasNext():
            
            if self.p1< self.n1  and self.p2<self.n2 :
                 if self.pointer1:
                    print('1')
                    val=self.v1[self.p1]
                    self.p1+=1
                    self.pointer1=False
                    return val
                 else:
                     print('2')
                     val=self.v2[self.p2]
                     self.pointer1=True
                     self.p2+=1
                     return val
            elif self.p1< self.n1  and self.p2==self.n2:
                     print('3')
                     val=self.v1[self.p1]
                     self.p1+=1
                     return val
            elif self.p1== self.n1  and self.p2<self.n2:
                     print('4')
                     val=self.v2[self.p2]
                     self.p2+=1
                     return val
                     
                   

    """
    @return: True if has next
    """
    def hasNext(self):
        # write your code here
        if self.p1< self.n1  or self.p2<self.n2 :
            return True
        if self.p1== self.n1  and self.p2==self.n2 :
            return False
        
        
        
        
        

v1 = [1, 2]
v2 = [3, 4, 5, 6]


# Your ZigzagIterator object will be instantiated and called as such:
# solution, result = ZigzagIterator(v1, v2), []
# while solution.hasNext(): result.append(solution.next())
# Output result   

#541. Zigzag Iterator II
class ZigzagIterator2:
    """
    @param: vecs: a list of 1d vectors
    """
    def __init__(self, vecs):
        # do intialization if necessary
        self.queue=[v for v in vecs if v]
 
    """
    @return: An integer
    """
    def next(self):
        # write your code here
        if self.hasNext():
            row=self.queue.pop(0)
            v=row.pop(0)
            if row:
                self.queue.append(row)
            return v
    """
    @return: True if has next
    """
    def hasNext(self):
        # write your code here
       
        return len(self.queue)>0


vecs=[[1,2,3],[4,5,6,7],[8,9]]


     
        
#544. Top k Largest Numbers 
class Solution:
    """
    @param nums: an integer array
    @param k: An integer
    @return: the top k largest numbers in array
    """
    def topk(self, nums, k):
        # write your code here
        import heapq
        hp=[]
        res=[]
        if not nums:
            return []
        
        n=len(nums)
        
        if k>=n:
            return sorted(nums,reverse=True)
        for x in nums:
            heapq.heappush(hp,-x)
        
        for _ in range(k):
            res.append(0-heapq.heappop(hp))
        return res
 
nums=[3,10,1000,-99,4,100]
k = 6
#Return [1000, 100, 10].        
if __name__ == "__main__":
    print(Solution().topk( nums, k))         
        
#545. Top k Largest Numbers II
class Solution:
    """
    @param: k: An integer
    """
    def __init__(self, k):
        # do intialization if necessary
        self.stack=[]
        self.k=k

    """
    @param: num: Number to be added
    @return: nothing
    """
    def add(self, num):
        # write your code here
        self.stack.append(num)

    """
    @return: Top k element
    """
    def topk(self):
        # write your code here
        import heapq
        hp=[]
        res=[]
        if not self.stack:
            return []
        
        n=len(self.stack)
        
        if self.k>=n:
            return sorted(self.stack,reverse=True)
        for x in self.stack:
            heapq.heappush(hp,-x)
        
        for _ in range(self.k):
            res.append(0-heapq.heappop(hp))
        return res
        
        
#547. Intersection of Two Arrays
class Solution:
    
    """
    @param: nums1: an integer array
    @param: nums2: an integer array
    @return: an integer array
    """
    def intersection(self, nums1, nums2):
        # write your code here   
        
        nums1=set(nums1)
        nums2=set(nums2)
        return list(nums1.intersection(nums2))
nums1 = [1, 2, 2, 1]
nums2 = [2, 2]        
if __name__ == "__main__":
    print(Solution().intersection( nums1, nums2))          
        
#548. Intersection of Two Arrays II
class Solution:
    
    """
    @param: nums1: an integer array
    @param: nums2: an integer array
    @return: an integer array
    """
    def intersection(self, nums1, nums2):
        # write your code here
        from collections import Counter
        counts=Counter(nums1)
        res=[]
        for n2 in nums2:
          if counts[n2]>0:
             res.append(n2)
             counts[n2]-=1
        return res
             
#550. Top K Frequent Words II    
class Entry():
    def __init__(self,word,freq):
        # do intialization if necessary
        self.word=word
        self.freq=freq
        self.inTop=False
    def __lt__(self,other):
        if self.freq==other.freq:
            return self.word>other.word
        return self.freq<other.freq

from bisect import bisect_left
                
class TopK:
    """
    @param: k: An integer
    """
    def __init__(self, k):
        # do intialization if necessary
        self.k=k
        self.top_k=[]
        self.mapping={}
        
    """
    @param: word: A string
    @return: nothing
    """
    def add(self, word):
        # write your code here
        if self.k==0:
            return 
        entry=None
        
        if word in self.mapping:
            entry=self.mapping[word]
            if entry.inTop:
                self.removeFromTop(entry)
            entry.freq+=1
        else:
            self.mapping[word]=Entry(word,1)
            entry=self.mapping[word]
        self.addToTop(entry)
        
        if len(self.top_k)>self.k:
            self.top_k[0].inTop=False
            self.top_k.pop(0)
            
    """
    @return: the current top k frequent words.
    """
    def topk(self):
        # write your code here
        if self.k==0:
            return []
        res=[e.word for e in self.top_k]
        res.reverse()
        return res
        
        
        
        
    
    def addToTop(self, entry):
        idx=bisect_left(self.top_k,entry)
        self.top_k.insert(idx,entry)
        entry.inTop=True
    
    def removeFromTop(self, entry):
        idx=bisect_left(self.top_k,entry)
        self.top_k.pop(idx)
        entry.inTop=False
        
        

#551. Nested List Weight Sum            
"""
This is the interface that allows for creating nested lists.
You should not implement it, or speculate about its implementation

class NestedInteger(object):
    def isInteger(self):
        # @return {boolean} True if this NestedInteger holds a single integer,
        # rather than a nested list.

    def getInteger(self):
        # @return {int} the single integer that this NestedInteger holds,
        # if it holds a single integer
        # Return None if this NestedInteger holds a nested list

    def getList(self):
        # @return {NestedInteger[]} the nested list that this NestedInteger holds,
        # if it holds a nested list
        # Return None if this NestedInteger holds a single integer
"""


class Solution(object):
    # @param {NestedInteger[]} nestedList a list of NestedInteger Object
    # @return {int} an integer
    def depthSum(self, nestedList):
        # Write your code here
        
        def decompse(nestedList,res,step):
            n=len(nestedList)
            if n==0:
                return 
            for i in range(n):
                if nestedList[i].isInteger():
                    res.append((nestedList[i].getInteger(),step))
                else:
                    decompse(nestedList[i].getList(),res,step+1)
        res=[]
        decompse(nestedList,res,1)
        
        return sum([num*step for num , step in res])
                    
                

#552. Create Maximum Number
class Solution:
    """
    @param nums1: an integer array of length m with digits 0-9
    @param nums2: an integer array of length n with digits 0-9
    @param k: an integer and k <= m + n
    @return: an integer array
    """
    def maxNumber(self, nums1, nums2, k):
        # write your code here
#https://leetcode.com/problems/create-maximum-number/discuss/77291/Share-my-Python-solution-with-explanation        
#To create the max number from num1 and nums2 with k elements, 
#we assume the final result combined by i numbers (denotes as left) from num1 and 
#j numbers (denotes as right) from nums2, where i+j==k.
#
#Obviously, left and right must be the maximum possible number in num1 and
# num2 respectively. i.e. num1 = [6,5,7,1] and i == 2, then left must be [7,1].
#
#The final result is the maximum possible merge of all left and right.
#
#So there're 3 steps:
#
#iterate i from 0 to k.
#find max number from num1, num2 by select i , k-i numbers, denotes as left, right
#find max merge of left, right
#function maxSingleNumber select i elements from num1 that is maximum. The idea find 
#the max number one by one. i.e. assume nums [6,5,7,1,4,2], selects = 3.
#1st digit: find max digit in [6,5,7,1], the last two digits [4, 2] can not be 
#selected at this moment.
#2nd digits: find max digit in [1,4], since we have already selects 7, we should 
#consider elements after it, also, we should leave one element out.
#3rd digits: only one left [2], we select it. and function output [7,4,2]
#
#function mergeMax find the maximum combination of left, and right.        
#        
        def getmax(nums,nselect):
            #res=[-1]
            n=len(nums)
            if nselect>=n:
                return nums
            
            ans=[]
            for i in range(n):
                while ans and len(ans)+(n-i)>nselect and ans[-1]<nums[i]:
                    ans.pop()
                if len(ans)<nselect:
                    ans.append(nums[i])
            return ans
                
        def mergemax(n1,n2):
            res=[]
            while n1 or n2:
                if n1>n2:
                    res.append(n1[0])
                    n1=n1[1:]
                    
                else:
                    res.append(n2[0])
                    n2=n2[1:]
            return res
        
        l1=len(nums1)
        l2=len(nums2)
        ret=[0 for _ in range(k)]
        for i in range(k+1):
            j=k-i
            if i>l1 or j>l2:
                continue
            left=getmax(nums1,i)
            right=getmax(nums2,j)
            tempret=mergemax(left,right)
            ret=max(ret,tempret)
        return ret
            
        
        
        
nums1 = [3, 4, 6, 5]
nums2 = [9, 1, 2, 5, 8, 3]
k = 5
#[9, 8, 6, 5, 3]

nums1 = [6, 7]
nums2 = [6, 0, 4]
k = 5
#[6, 7, 6, 0, 4]

nums1 = [3, 9]
nums2 = [8, 9]
k = 3
#[9, 8, 9]

if __name__ == "__main__":
    print(Solution().maxNumber( nums1, nums2, k))   


#553. Bomb Enemy
class Solution:
    """
    @param grid: Given a 2D grid, each cell is either 'W', 'E' or '0'
    @return: an integer, the maximum enemies you can kill using one bomb
    """
    def maxKilledEnemies(self, grid):
        # write your code here

        m=len(grid)
        if m==0:
            return 0
        n=len(grid[0])
        
        row=0
        res=0
        col=[0 for _ in range(n)]
        
        for i in range(m):
            for j in range(n):
                if j ==0 or grid[i][j-1]=='W':
                   row=0
                   for c in range(j,n):
                       if grid[i][c]=='W':
                           break
                       if grid[i][c]=='E':
                           row+=1
                if i==0 or grid[i-1][j]=='W':
                    col[j]=0
                    for r in range(i,m):
                        if grid[r][j]=='W':
                           break
                        if grid[r][j]=='E':
                           col[j]+=1
                if grid[i][j]=='0'  and row+col[j]>res:
                    res=row+col[j]
        return res
grid=["0E00","E0WE","0E00"]                        
if __name__ == "__main__":
    print(Solution().maxKilledEnemies(grid))                 
                       
                       
#555. Counting Bloom Filter
from collections import defaultdict
class CountingBloomFilter:
    """
    @param: k: An integer
    """
    def __init__(self, k):
        # do intialization if necessary
        self.capacity=k
        
        self.table=defaultdict(int)
        

    """
    @param: word: A string
    @return: nothing
    """
    def add(self, word):
        # write your code here
        
        self.table[word]+=1
            

    """
    @param: word: A string
    @return: nothing
    """
    def remove(self, word):
        # write your code here
        if self.table[word] >0:
          self.table[word]-=1
          

    """
    @param: word: A string
    @return: True if contains word
    """
    def contains(self, word):
        # write your code here
        if self.table[word]==0:
            return False
        if word not in self.table:
            return False
        
        if self.table[word]>0:
            return True
        
#556. Standard Bloom Filter
class StandardBloomFilter:
    """
    @param: k: An integer
    """
    def __init__(self, k):
        # do intialization if necessary

    """
    @param: word: A string
    @return: nothing
    """
    def add(self, word):
        # write your code here

    """
    @param: word: A string
    @return: True if contains word
    """
    def contains(self, word):
        # write your code here



#557. Count Characters
class Solution:
    """
    @param: : a string
    @return: Return a hash map
    """

    def countCharacters(self, str):
        # write your code here
        from collections import defaultdict
        table=defaultdict(int)
        
        for s in str:
            table[s]+=1
        return table

#564. Combination Sum IV
class Solution:
    """
    @param nums: an integer array and all positive numbers, no duplicates
    @param target: An integer
    @return: An integer
    """
    def backPackVI(self, nums, target):
        # write your code here
        
#        def dfs(target,nums,path,res):
#            if target==0:
#                res.append(path[:])
#            if target<0:
#                return 
#            for x in nums:
#                dfs(target-x,nums,path+[x],res)
#        res=[]
#        dfs(target,nums,[],res)
#        return len(res)
        
        
        n=len(nums)
        if n==0:
            return 0
                
        dp=[0 for _ in range(target+1)]
        dp[0]=1
        
        for i in range(1,target+1):
            for j in range(0,n):
                if i-nums[j]>=0:
                    dp[i]+=dp[i-nums[j]]
        return dp[target]
nums=[1,2,4]
target=4
nums=[1,2,4]
target=32
if __name__ == "__main__":
    print(Solution().backPackVI(nums, target))                 
                       
#569. Add Digit
class Solution:
    """
    @param num: a non-negative integer
    @return: one digit
    """
    def addDigits(self, num):
        # write your code here
#数学推导规律，O(1)时间复杂度。
#
#num = a0 + a1 * 10 + a2 * 100 + ... + ak * 10^k
#= (a0 + a1 + ... + ak) + 9(a1 + a2 + ... + ak) + 99(a2 + ... + ak) + ... + (10^k - 1)ak
#
#其中a0 + a1 + ... + ak是下一步计算要得到的结果，记为num1，重复上述过程：
#
#num = num1 + 9 * x1, 其中 x = (a1 + a2 + ... + ak) + 11 * (a2 + ... + ak) + ...
#num1 = num2 + 9 * x2
#...
#
#直到numl < 10 为止。
#
#则可知 num = numl + 9 * x
#numl = num % 9
        if num==0:
           return 0
        return num%9 if num%9 else 9

#570. Find the Missing Number II
class Solution:
    """
    @param n: An integer
    @param str: a string with number from 1-n in random order and miss one number
    @return: An integer
    """
    def findMissing2(self, n, string):
        # write your code here
        if not string :
            return -1
        
        self.numbers=set()
        
        def dfs(n,string,start,path):
            if len(path)==n-1:
                self.numbers=set(path)
                return 
            
            for i in range(start,len(string)):
                substring=string[start:i+1]
                if len(substring)>2:
                    break
                if substring in path or int(substring)>n or int(substring)<1:
                    continue
                dfs(n,string,i+1,path | set([substring]))
        dfs(n,string,0,set())
        for j in range(1,n+1):
            if str(j) not in self.numbers:
                return j
n = 20
string = '19201234567891011121314151618'            
if __name__ == "__main__":
    print(Solution().findMissing2( n, string))                        
                
            
#573. Build Post Office II
class Solution:
    """
    @param grid: a 2D grid
    @return: An integer
    """
    def shortestDistance(self, grid):
        # write your code here
#        m=len(grid)
#        if m==0:
#            return -1
#        n=len(grid[0])
#        
#        house=set()
#        
#        for i in range(m):
#            for j in range(n):
#                if grid[i][j]==1:
#                    house.add((i,j))
#        
#        from collections import deque
#        
#        def bfs(a,b,house,grid,m,n) :
#            dq=deque([(a,b,0)])
#            temphouse=set()
#            res=0
#        
#            visited=set( ( a,b))
#        
#        
#            while dq:
#               h,k,step=dq.popleft()
#               if grid[h][k]==1  and (h,k) not in temphouse:
#                  temphouse.add((h,k))
#                  res+=step
#                  if len(temphouse)==len(house):
#                     return res
#               if grid[h][k]==0:
#               #for x,y in   ((h+1,k),(h-1,k),(h,k+1),(h,k-1)):
#                  for x,y in   ((h+1,k),(h-1,k),(h,k+1),(h,k-1)):
#                     if x>=0 and x<m  and y>=0 and y<n and (x,y) not in visited and grid[x][y]!=2:
#                        dq.append((x,y,step+1))
#                        visited.add((x,y))
#            return float('inf') 
#        ans=float('inf')
#        for a  in range(m):
#            for b in range(n):
#                if grid[a][b]!=1 and grid[a][b]!=2:
#                   #print(a,b)
#                   tempans=bfs(a,b,house,grid,m,n)
#                   #print(a,b,tempans)
#                   ans=min(ans,tempans)
#        return ans if ans !=float('inf') else -1
        def bfs(i,j,grid):
            queue=[(i,j)]
            _queue=[]
            distance=0
            visited=[[False for _ in range(self.n)] for _ in range(self.m)]
            
            while queue:
                  distance+=1
                  _queue=[]
                  for a, b in queue:
                      for x, y in ((a+1,b),(a-1,b),(a,b+1),(a,b-1)):
                          if  x>=0 and x<self.m  and y>=0 and y<self.n and not visited[x][y] and grid[x][y]==0:
                              visited[x][y]=True
                              self.visitTimes[x][y]+=1
                              self.distance[x][y]+=distance
                              _queue.append((x,y))
                  queue=_queue            
        
        self.m=len(grid)
        self.n=len(grid[0])
        self.visitTimes=[[0 for _ in range(self.n)] for _ in range(self.m)]
        self.distance=[[0 for _ in range(self.n)] for _ in range(self.m)]
        house=0
        for i in range(self.m):
            for j in range(self.n):
                if grid[i][j]==1:
                    house+=1
                    bfs(i,j,grid)
                    
        
        ans=float('inf')
#        print(grid)
#        print(self.visitTimes)
#        print(self.distance)
        
        for i in range(self.m):
            for j in range(self.n):
                if grid[i][j]==0 and self.visitTimes[i][j]==house and self.distance[i][j]<ans:
                    ans=self.distance[i][j]
#        print(ans)
        return ans if ans < float('inf') else -1
        
        
            
  
grid=[[0,1,0,0,0],
      [1,0,0,2,1],
      [0,1,0,0,0]]
grid=[[0,1,0,0],
      [1,0,2,1],
      [0,1,0,0]]

if __name__ == "__main__":
    print(Solution().shortestDistance( grid))                        
                            
        
#575. Decode String 
class Solution:
    """
    @param s: an expression includes numbers, letters and brackets
    @return: a string
    """
    def expressionExpand(self, s):
        # write your code here
        if not s:
            return ''
        
        stack=[]
        
        for x in s:
            if x!=']':
               stack.append(x)
            else:
                temp=''
                while stack[-1]!='[':
                    temp=stack.pop()+temp
                if stack[-1]=='[':
                    stack.pop()
                    num=''
                    while stack and stack[-1].isdigit():
                        num=stack.pop()+num
                    stack.append(temp*int(num))
        #print(stack)
        return ''.join(stack)
s = 'abc3[a]' #return abcaaa
s = '3[abc]' #return abcabcabc
s = '4[ac]dy' # return acacacacdy
s = '3[2[ad]3[pf]]xyz' # return adadpfpfpfadadpfpfpfadadpfpfpfxyz  
s ='5[10[abcd]Ac20[abcde]]'
if __name__ == "__main__":
    print(Solution().expressionExpand(s))               



#577. Merge K Sorted Interval Lists
"""
Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    """
    @param intervals: the given k sorted interval lists
    @return:  the new sorted interval list
    """
    def mergeKSortedIntervalLists(self, intervals):
        # write your code here
        
        import heapq
        heap=[]
                
        for i,row in enumerate(intervals):
            if len(row)>0:
                heapq.heappush(heap,(row[0].start,row[0].end,i,0))
        
        start,end,x,y=heapq.heappop(heap)
        
        if y+1<len(intervals[x]):
            heapq.heappush(heap,(intervals[x][y+1].start,intervals[x][y+1].end,x,y+1))
        res=[]    
        while heap:
            new_start,new_end,new_x,new_y=heapq.heappop(heap)
            if new_start<=end:
                end=max(end,new_end)
            else:
                res.append(Interval(start,end))
                start=new_start
                end=new_end
            x=new_x
            y=new_y
            if y+1<len(intervals[x]):
                heapq.heappush(heap,(intervals[x][y+1].start,intervals[x][y+1].end,x,y+1))
        res.append(Interval(start,end))
        return res

#578. Lowest Common Ancestor III
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        this.val = val
        this.left, this.right = None, None
"""


class Solution:
    """
    @param: root: The root of the binary tree.
    @param: A: A TreeNode
    @param: B: A TreeNode
    @return: Return the LCA of the two nodes.
    """
    def lowestCommonAncestor3(self, root, A, B):
        # write your code here
        
        
        def search(node,A,B):
            if not node:
               return False,False,None
        
            
            left=search(node.left,A,B)
            right=search(node.right,A,B)
            findA=left[0]  or right[0] or node==A
            findB=left[1]  or right[1] or node==B
            
            if node==A or node==B:
                return findA,findB,node
        
            if left[2] and right[2]:
                return findA,findB,node
               
        
            if left[2]:
               return findA,findB,left[2]
            if right[2]:
               return findA,findB,right[2]
            
            return findA,findB,None
        
       
          findA,findB,LCA=search(root,A,B)
          if findA and  findB:
              return LCA
          else:
              return None
            
#582. Word Break II        
class Solution:
    """
    @param: s: A string
    @param: wordDict: A set of words.
    @return: All possible sentences.
    """
    def wordBreak(self, s, wordDict):
        # write your code here
        
        def dfs(s,wordDict,memo):
            if s in memo:
                return memo[s]
            if not s:
                return []
            
            res=[]
               
            for x in wordDict:
                if not s.startswith(x):
                    continue
                
                if len(s)==len(x):
                    res.append(x)
                else:
                    rest=dfs(s[len(x):],wordDict,memo)
                    
                    for item in rest:
                        res.append(x+' '+item)
                    
            memo[s]=res
            return res
        wordDict=[ w for w in wordDict if w]
        return dfs(s,wordDict,{})
s = 'lintcode'
wordDict= ["de", "ding", "co", "code", "lint"]

s = "a"
wordDict=["","t","t"]
wordDict=['']

s ="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
wordDict=["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]
#["lint code", "lint co de"].
if __name__ == "__main__":
    print(Solution().wordBreak( s, wordDict))


#584. Drop Eggs II
class Solution:
    """
    @param m: the number of eggs
    @param n: the number of floors
    @return: the number of drops in the worst case
    """
    def dropEggs2(self, m, n):
        # write your code here
#        dp[i][j]表示有j层、i个鸡蛋时，至少需要尝试多少次。
#现在假设第i个鸡蛋从第k层落下，分两种情况，如果鸡蛋碎了，那么问题变成有k-1层、i-1个鸡蛋，
#也就是dp[i-1][k-1]
#如果鸡蛋没碎，那么问题变成有j-k层、仍是i个鸡蛋，也就是dp[i][j-k]
#因为我们不知道鸡蛋是否碎，所以要取两者的较大值
        egg=m
        floor=n
        
        dp=[[0 for _ in range(floor+1)]  for _ in range(egg+1)]
        
        
        for i in range(1,egg+1):
            dp[i][1]=1
        for j in range(1,floor+1):
            dp[1][j]=j
            
            
            
        for x in range(2,egg+1):
            for y in range(2,floor+1):
                
                dp[x][y]=float('inf')
                
                for k in range(1,y+1):
                    dp[x][y]=min(dp[x][y],  1+max(dp[x-1][k-1],dp[x][y-k]   )     )
        return dp[m][n]
m = 2
n = 100 #return 14
m = 2
n = 36 #return 8                    
if __name__ == "__main__":
    print(Solution().dropEggs2( m, n))                
                
#585. Maximum Number in Mountain Sequence                
class Solution:
    """
    @param nums: a mountain sequence which increase firstly and then decrease
    @return: then mountain top
    """
    def mountainSequence(self, nums):
        # write your code here
        if not nums:
            return 0
        top=num[0]
        for num in nums[1:]:
            if num<top:
                return top
            else:
                top=num
            
                
#587. Two Sum - Unique pairs
class Solution:
    """
    @param nums: an array of integer
    @param target: An integer
    @return: An integer
    """
    def twoSum6(self, nums, target):
        # write your code here
                
                

        nums.sort()
        self.res=0
        def twoSum(nums,target):
              
            for i in range(len(nums)-1):
                if i!= 0 and nums[i]==nums[i-1]:
                   continue
                if target-nums[i] in nums[i+1:]:
                    self.res+=1
        twoSum(nums,target)
        return self.res 
nums = [1,1,2,45,46,46]
target = 
#return 2
#1 + 46 = 47
#2 + 45 = 47                            
if __name__ == "__main__":
    print(Solution().twoSum6(nums, target))                
                                
#588. Partition Equal Subset Sum
class Solution:
    """
    @param nums: a non-empty array only positive integers
    @return: true if can partition or false
    """
    def canPartition(self, nums):
        # write your code here

        n=len(nums)
        nsum=sum(nums)
        if nsum%2:
           return False
        if n==0 or n==1:
           return False
        target=nsum//2
        
        dp=[False for _ in range(20000)]
        
        dp[0]=True
        
        for i in range(n):
            for j in range(target, nums[i]-1,-1  ):
               
                    dp[j]|= dp[j-nums[i]]
        return dp[target]
                
       
#        def dfs(target,nums):
#           if target<0:
#               return 
#           if target==0:
#               return True
#           
#           for i in range(len(nums)):
#               if dfs(target-nums[i],nums[i+1:]):
#                   return True
#           return False    
#        return dfs(target,nums)
nums = [1, 5, 11, 5]# return true
[#1, 5, 5], [11]

nums = [1, 2, 3, 9]# return false
nums = [1,4,5,6,1,2,4,1,3,4,1,2,4,5,1,91,4,5,6,1,2,4,1,3,4,1,2,4,5,1]
nums = [ 2]
if __name__ == "__main__":
    print(Solution().canPartition( nums))           
           
       



#591. Connecting Graph III
    
    
    
    
    
    
    
    
    
    





#594. strStr II
class Solution:
    """
    @param: source: A source string
    @param: target: A target string
    @return: An integer as index
    """
    def strStr2(self, source, target):
        # write your code here
        if source is None or target is None:
            return -1
        m=len(target)
        if m==0:
            return 0
        
        import random
        mod=99999999
        target_value=0
        m26=1
        
        for i in range(m):
            target_value=(target_value*26+ord(target[i])-ord('a'))%mod
            if target_value<0:
                target_value+=mod
        for _ in range(m-1):
            m26=26*m26%mod
          
        value=0
        for i in range(len(source)):
            if i>=m:
                value= value-(ord(source[i-m])-ord('a'))*m26%mod
            value=(value*26+ord(source[i])-ord('a'))%mod
            if value<0:
                value+=mod
            if i>=m-1 and value==target_value:
                return i-m+1
        return -1
source='qwerty'
target='ert'
source="abcdef"
target="bcd"
if __name__ == "__main__":
    print(Solution().strStr2(source, target))                      
                
                

#595. Binary Tree Longest Consecutive Sequence                
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of binary tree
    @return: the length of the longest consecutive sequence path
    """
    def longestConsecutive(self, root):
        # write your code here
        
        if not root:
            return 0
        if not root.right and not root.left:
            return 1
        def walkdown(node,res,path):
            if not node.left  and not node.right:
                res.append(path)
                return 
                
            if node.left:
                if node.left.val==node.val+1:
                    walkdown(node.left,res,path+1)
                else:
                    res.append(path)
                    walkdown(node.left,res,1)
            if node.right:
                if node.right.val==node.val+1:
                    walkdown(node.right,res,path+1)
                else:
                    res.append(path)
                    walkdown(node.right,res,1)
        res=[]
        walkdown(root,res,1)
        return max(res)
#    1
#    \
#     3
#    / \
#   2   4
#        \
#         5            
                
root=TreeNode(1) 

root.right= TreeNode(3) 
 
root.right.left= TreeNode(2)
root.right.right= TreeNode(4)  
root.right.right.right= TreeNode(5)    
            
if __name__ == "__main__":
    print(Solution().longestConsecutive( root)) 


#599. Insert into a Cyclic Sorted List
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""


class Solution:
    """
    @param: node: a list node in the list
    @param: x: An integer
    @return: the inserted new list node
    """
    def insert(self, node, x):
        # write your code here
        
        if node is None:
            #print('**')
            node=ListNode(x)
            node.next=node
            
            return node
        #print('****')
        if node.next==node:
            new_node=ListNode(x)
            new_node.next=node
            node.next=new_node
            return new_node
            
        
        
        cur=node
        
        while cur.next.val>=cur.val:
            cur=cur.next
            if cur==node:
                break
        
        maxv=cur.val
        minv=cur.next.val
        
        if x>maxv or x<minv:
            new_node=ListNode(x)
            new_node.next=cur.next
            cur.next=new_node
        else:
            cur=cur.next
            while cur.next.val<x:
                cur=cur.next
            new_node=ListNode(x)
            new_node.next=cur.next
            cur.next=new_node
        return new_node
            

        
#Given a list, and insert a value 4:
#3->5->1
#Return 5->1->3->4
        
node=None
x=4
if __name__ == "__main__":
    print(Solution().insert(node, x)) 


#600. Smallest Rectangle Enclosing Black Pixels
class Solution:
    """
    @param image: a binary matrix with '0' and '1'
    @param x: the location of one of the black pixels
    @param y: the location of one of the black pixels
    @return: an integer
    """
    def minArea(self, image, x, y):
        # write your code here
#        self.pixels=set()
#        m=len(image)
#        n=len(image[0])
#        def dfs(image,x,y,visited):
#            if image[x][y]=='1':
#                #self.pixels.add((x,y))
#                if x < self.mina:
#                   self.mina=x
#                if x>self.maxa:
#                   self.maxa=x
#                if y < self.minb:
#                   self.minb=y
#                if y>self.maxb:
#                   self.maxb=y
#                
#            for i , j in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
#                if i>=0 and j>=0 and i<m and j <n and (i,j) not in visited:
#                    visited.add((i,j))
#                    dfs(image,i,j,visited)
#        visited=set()
#        
#        self.mina=float('inf')
#        self.minb=float('inf')
#        self.maxa=float('-inf')
#        self.maxb=float('-inf')
#        dfs(image,x,y,visited)
#       
#        return (self.maxa-self.mina+1)*(self.maxb-self.minb+1) 
        
        m=len(image)
        if m==0:
           return 0
        n=len(image[0])
        if n==0:
            return 0
        
       
        
        def checkcol(image,col):
            for i in range(m):
                if image[i][col]=='1':
                    return True
            return False
        def checkrow(image,row):
            for i in range(n):
                if image[row][i]=='1':
                    return True
            return False
        
        #right
        start=y
        end=n-1
        
        while start<end:
            mid=(start+end)//2+1
            if checkcol(image,mid):
                start=mid
            else:
                end=mid-1
        right=start
        #left
        start=0
        end=y
        while start<end:
            mid=(start+end)//2
            if checkcol(image,mid):
                end=mid
            else:
                start=mid+1
            
        left=start
        #down
        start=x
        end=m-1
        while start<end:
            
            mid=(start+end)//2+1
            if checkrow(image,mid):
                start=mid
            else:
                end=mid-1
        down=start
        
        #up
        
        start=0
        end=x
        
        while start<end:
            mid=(start+end)//2
            if checkrow(image,mid):
                end=mid
            else:
                start=mid+1
        
        up=start
        
        return (right-left+1)*(down-up+1)
image=[
  "0010",
  "0110",
  "0100"
]
x = 0
y = 2
#Return 6
if __name__ == "__main__":
    print(Solution().minArea( image, x, y))                
                
                
#601. Flatten 2D Vector                
class Vector2D(object):

    # @param vec2d {List[List[int]]}
    def __init__(self, vec2d):
        # Initialize your data structure here
        self.stack=[]
        self.pointer=0
        for row in vec2d:
                for item in row:
                    self.stack.append(item)
      
            
                    
        
   
    # @return {int} a next element
    def next(self):
        # Write your code here
        if self.hasNext:
            self.pointer+=1
            return self.stack[self.pointer-1]
        

    # @return {boolean} true if it has next element
    # or false
    def hasNext(self):
        # Write your code here
        return not self.pointer==len(self.stack)
        
        
vec2d=[
  [1,2],
  [3],
  [4,5,6]
]
# Your Vector2D object will be instantiated and called as such:
# i, v = Vector2D(vec2d), []
# while i.hasNext(): v.append(i.next())                
                
#602. Russian Doll Envelopes
class Solution:
    """
    @param: envelopes: a number of envelopes with widths and heights
    @return: the maximum number of envelopes
    """
    def maxEnvelopes(self, envelopes):
        # write your code here
        m=len(envelopes)
        if m==0:
            return 0
        n=len(envelopes[0])
        if n==0:
            return 0
        envelopes.sort(key=lambda x:(x[0],-x[1]))

        #就是找最长increasing substring
        h=[]
        from bisect import bisect_left
        
        for e in envelopes:
            j =bisect_left(h,e[1])
            
            if j < len(h):
                h[j]=e[1] #使高度尽量变小，虽然打乱了width的次序，但总长度不变
            else:
                h.append(e[1])
        return len(h)
        
        
        
        
  
envelopes = [[5,4],[6,4],[6,7],[2,3]]
envelopes =[[5,6],[6,4],[6,7],[2,9]]


# 3 ([2,3] => [5,4] => [6,7]).                
if __name__ == "__main__":
    print(Solution().maxEnvelopes( envelopes))                
        
#603. Largest Divisible Subset
class Solution:
    """
    @param: nums: a set of distinct positive integers
    @return: the largest subset 
    """
    def largestDivisibleSubset(self, nums):
        # write your code here
        n=len(nums)
        if n==0 or n==1:
            return []
        
        count=[0 for _ in range(n)]
        
        prev_index=[-1 for _ in range(n)]
        
        res=[]
        nums.sort()
        for i in range(1,n):
            for j in range(i):
                if nums[i]%nums[j]==0:
                    count[i]=count[j]+1
                    prev_index[i]=j
        index= count.index(max(count))
        
        res=[]
        while index!=-1:
            res.append(nums[index])
            index=prev_index[index]
        return res
nums = [1,2,3]
nums = [1,2,4,8]
if __name__ == "__main__":
    print(Solution().largestDivisibleSubset( nums))                  
            
            
                    
                    
#605. Sequence Reconstruction            
class Solution:
    """
    @param org: a permutation of the integers from 1 to n
    @param seqs: a list of sequences
    @return: true if it can be reconstructed only one or false
    """
    def sequenceReconstruction(self, org, seqs):
        # write your code here

#拓扑排序的路数
#
#构建两个字典，一个用来存放入度，一个用来存放邻居，用org里的元素初始化字典
#要防止seqs里面给出不合法的节点，如果发现直接返回False
#确保seqs里的节点个数和orgs相等
        
        degrees={}
        nodes=set()
        edges={}
        
        for x in org:
            edges[x]=[]
            degrees[x]=0
        
        
        for s in seqs:
            nodes|=set(s)
            for i in range(len(s)-1):
                edges[s[i]].append(s[i+1])
                if s[i+1] in degrees:
                   degrees[s[i+1]]+=1
                else:
                    return False
        
        from collections import deque
        q=deque()
        for k,v in degrees.items():
            if v==0:
                q.append(k)
        
        ans=[]
        while len(q)==1:
            
            item=q.popleft()
            ans.append(item)
            for e in edges[item]:
                degrees[e]-=1
                if degrees[e]==0:
                    q.append(e)
                
        return ans==org and len(nodes)==len(org)
org=[1,2,3]
seqs = [[1,2],[1,3]] 
org = [1,2,3]
seqs = [[1,2]]
org = [1,2,3]
seqs = [[1,2],[1,3],[2,3]] 
org = [4,1,5,2,6,3]
seqs = [[5,2,6,3],[4,1,5,2]]          
if __name__ == "__main__":
    print(Solution().sequenceReconstruction(org, seqs))                  
                            
                
#607. Two Sum III - Data structure design            
class TwoSum:
    """
    @param: number: An integer
    @return: nothing
    """
    def __init__(self):
        self.table=[]
        
    def add(self, number):
        # write your code here
        self.table.append(number)

    """
    @param: value: An integer
    @return: Find if there exists any pair of numbers which sum is equal to the value.
    """
    def find(self, value):
        # write your code here
        hashset=set()
        for x in self.table:
            if value-x not in hashset:
                hashset.add()
            else:
                return True
        return False
                
#608. Two Sum II - Input array is sorted            
class Solution:
    """
    @param nums: an array of Integer
    @param target: target = nums[index1] + nums[index2]
    @return: [index1 + 1, index2 + 1] (index1 < index2)
    """
    def twoSum(self, nums, target):
        # write your code here
        import bisect
        
        n=len(nums)
        if n==0 or n==1:
            return []
        
        for i in range(n):
            another=target-nums[i]
            if another<nums[i]:
                break
            if another>nums[-1]:
                continue
            j=bisect.bisect_right(nums,another)
            print(j)
            #if nums[j-1]==another:
            return [i+1,j]
        
  
        
nums = [0,0,3,4]
target = 0
nums =[2,7,11,15]
target =9
#return [1, 2]        
if __name__ == "__main__":
    print(Solution().twoSum(nums, target))  

#611. Knight Shortest Path
"""
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
"""

class Solution:
    """
    @param grid: a chessboard included 0 (false) and 1 (true)
    @param source: a point
    @param destination: a point
    @return: the shortest path 
    """
    def shortestPath(self, grid, source, destination):
        # write your code here
        m=len(grid)
        if m==0:
            return -1
        n=len(grid[0])
        if n==0:
            return -1
        
        if grid[source[0]][source[1]]==1 or grid[destination[0]][destination[1]]==1:
            return -1
        
        source=tuple(source)
        destination=tuple(destination)
        
        from collections import deque
        dq=deque([(source[0],source[1],0)])
        visited=set()
        visited.add(source)
        while dq:
            tempdp=deque()
            for _ in range(len(dq)):
                x,y,step=dq.popleft()
                if (x,y)==destination:
                    return step
                for i ,j in ((x + 1, y + 2), 
                             (x + 1, y - 2),
                             (x - 1, y + 2),
                             (x - 1, y - 2),
                             (x + 2, y + 1),
                             (x + 2, y - 1),
                             (x - 2, y + 1),
                              (x - 2, y - 1)):
                    if i>=0 and j>=0 and i<m and j <n and ( i,j) not in visited and grid[i][j]!=1:
                        tempdp.append((i,j,step+1))
                        visited.add((i,j))
            dq=tempdp
        return -1
                        
                        
                
                
 
        
grid=[[0,0,0],
 [0,0,0],
 [0,0,0]]
source = [2, 0]
destination = [2, 2] #return 2
grid=[[0,1,0],
 [0,0,0],
 [0,0,0]]
source = [2, 0] 
destination = [2, 2] #return 6

grid=[[0,1,0],
 [0,0,1],
 [0,0,0]]
source = [2, 0]
destination = [2, 2] #return -1
if __name__ == "__main__":
    print(Solution().shortestPath( grid, source, destination))  


#612. K Closest Points
"""
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = bDefinition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
"""

class Solution:
    """
    @param points: a list of points
    @param origin: a point
    @param k: An integer
    @return: the k closest points
    """
    def kClosest(self, points, origin, k):
        # write your code here
        import heapq
        n=len(points)
        if n==0:
            return []
        
        hp=[]
        for p in points:
            heapq.heappush(hp,((p.x-origin.x)**2+(p.y-origin.y)**2,p.x,p.y))
            
        
        res=[]
        for _ in range(min(k,n)):
            d,x,y=heapq.heappop(hp)
            res.append(Point(x,y))
        return res
    
points = [[4,6],[4,7],[4,4],[2,5],[1,1]]
origin = [0, 0]
k = 3
#return [[1,1],[2,5],[4,4]]


#614. Binary Tree Longest Consecutive Sequence II
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of binary tree
    @return: the length of the longest consecutive sequence path
    """
    def longestConsecutive2(self, root):
        # write your code here
        
#        def dfs(node,diff):
#            
#            if not node:
#                return 0
#            if not node.left and not node.right:
#                return 1
#            
#            if node.left and node.val-node.left.val==diff:
#                left=1+dfs(node.left,diff)
#            if node.right and node.val-node.right.val==diff:
#                right=1+dfs(node.right,diff)
#            return max(left,right)
#        
#        if not root:
#            return 0
#        
#        if not root.left and not root.right:
#            return 1
#        
#        res=1+dfs(root,-1)+dfs(root,1)
#        
#        return max(res,self.longestConsecutive2(root.left),self.longestConsecutive2(root.right))
        
        def search(node):
            if not node:
                return 0
            dec=1
            inc=1
            
            if node.left:
                cdec,cinc=search(node.left)
                if node.val- node.left.val==1:
                    inc=cinc+1
                elif node.val- node.left.val==-1:
                    dec=cdec+1
            
            if node.right:
                cdec,cinc=search(node.right)
                if node.val- node.right.val==1:
                    inc=max(cinc+1,inc)
                elif node.val- node.right.val==-1:
                    dec=max(dec,cdec+1)
            self.res=max(self.res,inc+dec-1)
            return dec,inc
        self.res=0
        search(root)
        return self.res
                

    1
   / \
  2   0
 /
3

#615. Course Schedule
class Solution:
    """
    @param: numCourses: a total of n courses
    @param: prerequisites: a list of prerequisite pairs
    @return: true if can finish all courses or false
    """
    def canFinish(self, numCourses, prerequisites):
        # write your code here
#        visited=[0 for _ in range(numCourses)]
#        from collections import defaultdict
#        graph=defaultdict(list)
#        for x,y in prerequisites:
#            graph[x].append(y)
#            
#        def dfs(visited,graph,i):
#            if visited[i]==-1:
#              return False
#            if visited[i]==1:
#                return True
#            
#            visited[i]=-1
#            for j in graph[i]:
#                if not dfs(visited,graph,j):
#                    return False
#            visited[i]=1
#            return True
#            
#                
#                
#        for course in range(numCourses):
#            if not dfs(visited,graph,course):
#                return False
#        return True
    
    
    
    #topological sort 
    
        from collections import defaultdict,deque
        ind=defaultdict(list)
        out=[0 for _ in range(numCourses)]
        
        for p in  prerequisites:
            out[p[0]]+=1
            ind[p[1]].append(p[0])
        
        
        dq=deque()
        
        for i in range(numCourses):
            if out[i]==0:
                dq.append(i)
        if len(dq)==0:
            return False
        
        k=0
        while dq:
            x=dq.popleft()
            k+=1
            
            for y in ind[x]:
                out[y]-=1
                if out[y]==0:
                    dq.append(y)
        return k==numCourses
            
            
numCourses = 2
prerequisites = [[1,0]]
numCourses = 2
prerequisites = [[1,0],[0,1]] 
numCourses =10
prerequisites=[[5,8],[3,5],[1,9],[4,5],[0,2],[1,9],[7,8],[4,9]]           
if __name__ == "__main__":
    print(Solution().canFinish(numCourses, prerequisites)) 

#616. Course Schedule II
class Solution:
    """
    @param: numCourses: a total of n courses
    @param: prerequisites: a list of prerequisite pairs
    @return: the course order
    """
    def findOrder(self, numCourses, prerequisites):
        # write your code here
        from collections import defaultdict,deque
        
        ind=defaultdict(list)
        out=[0 for _ in range(numCourses)]
        
        res=[]
        
        for a,b in prerequisites:
            out[a]+=1
            ind[b].append(a)
        dq=  deque()  
        res=[]
        for i in range(numCourses):
            if out[i]==0:
                dq.append(i)
        
        while dq:
            x=dq.popleft()
            res.append(x)
            for j in ind[x]:
                out[j]-=1
                if out[j]==0:
                    dq.append(j)
        return res if len(res)==numCourses else []
numCourses = 2
prerequisites = [[1,0]]  
numCourses = 4
prerequisites = [[1,0],[2,0],[3,1],[3,2]]              
if __name__ == "__main__":
    print(Solution().findOrder(numCourses, prerequisites)) 
            
            
        
#617. Maximum Average Subarray II        
class Solution:
    """
    @param: nums: an array with positive and negative numbers
    @param: k: an integer
    @return: the maximum average
    """
    def maxAverage(self, nums, k):
        # write your code here        
        
#复杂度：O(nlog(max + min))，其中n是nums的长度，max和min分别是nums中的最大值和最小值。
#这里用了“二分答案”思想。
#
#所求的最大平均值一定是介于原数组的最大值和最小值之间，所以我们的目标是用二分法来快速的在这个
#范围内找到我们要求的最大平均值，初始化left为原数组的最小值，right为原数组的最大值，然后mid就是
#left和right的中间值，难点就在于如何得到mid和要求的最大平均值之间的大小关系，从而判断二分的方向。
#我们想，如果我们已经算出来了这个最大平均值maxAvg，那么对于任意一个长度大于等于k的数组，如果让每
#个数字都减去maxAvg，那么得到的累加差值一定是小于等于0的。所以我们通过left和right值算出来的mid，
#可以看作是maxAvg的一个candidate，所以我们就让数组中的每一个数字都减去mid，然后算差值的累加和，
#一旦发现累加和大于0了，那么说明我们mid比maxAvg小，这样就可以判断方向了。
#
#具体步骤：
#1.每次进入循环时，我们建立一个前缀和数组prefixSums，然后求出原数组中最小值赋给left，最大值赋给right，
#题目中说了误差是1e-6，所以我们的循环条件就是right比left大1e-6；
#2.然后我们算出来mid，prefixSumMin初始为0，maxSum初始化为INT_MIN。然后开始遍历数组，
#先更新prefixSums，注意prefixSums是它们和mid相减的差值累加。我们的目标是找长度大于等于k的子数组的
#平均值大于mid，由于我们每个数组都减去了mid，那么就转换为找长度大于等于k的子数组的差累积值大于0。
#然后问题转变成了“最大 >k 的 sum range 要小于0”的问题。如果确实小于0，则end = mid，否则start = mid。

        
        def has_greater(nums,k,mid):
            sm=0.0
            prev_sm=0.0
            prev_min=0.0
            
            for i in range(len(nums)):
                sm+=float(nums[i])-mid
                if i>=k-1 and sm>=0:
                    return True
                if i-k>=0:
                   prev_sm+=float(nums[i-k])-mid
                   prev_min=min(prev_min,prev_sm)
                   if sm-prev_min>=0:
                       return True
            return False
        start=min(nums)
        end=max(nums)
        
        while start+1e-6<end:
            mid=(start+end)/2
            if has_greater(nums,k,mid):
                start=mid
            else:
                end=mid
        return start
                   
#622. Frog Jump  
class Solution:
    """
    @param stones: a list of stones' positions in sorted ascending order
    @return: true if the frog is able to cross the river or false
    """
    def canCross(self, stones):
        # write your code here
#        n=len(stones)
#        if n==0:
#            return True
#        if n==1:
#            return True
#        if stones[1]!=1:
#            return False
#        def dfs(stones,pos,last_jump):
#            if pos==stones[-1]:
#                return True
#            #print(pos)
#            
#            for next_step in (last_jump-1,last_jump,last_jump+1):
#                if next_step==0 or pos+next_step not in stones:
#                    continue
#                if  dfs(stones,pos+next_step,next_step):
#                    return True
#            return False
#        
#        
#        if dfs(stones,1,1):
#            return True
#        else:
#            return False
        
        
        S=set(stones)
        visited=set([(stones[0],0)])
        
        
        from collections import deque
        q=deque([(stones[0],0)])
        
        while q:
            pos,step=q.popleft()
            if pos==stones[-1]:
                return True
            
            for x in (step-1,step,step+1):
                if x>0 and x+pos in S and (x+pos,x) not in visited:
                    visited.add((x+pos,x))
                    q.append((x+pos,x))
        return False
                    
stones = [0,1,3,5,6,8,12,17]
stones = [0,1,2,3,4,8,9,11]        
if __name__ == "__main__":
    print(Solution().canCross(stones)   )      
        
        
#623. K Edit Distance
class Solution:
    """
    @param words: a set of stirngs
    @param target: a target string
    @param k: An integer
    @return: output all the strings that meet the requirements
    """
    def kDistance(self, words, target, k):
        # write your code here
        
        def Distance(w1,w2):
            m=len(w1)
            n=len(w2)
            dp=[[0 for _ in range(n+1)] for _ in range(m+1)]
            
            
            for i in range(1,m+1):
                dp[i][0]=i
            for j in range(1,n+1):
                dp[0][j]=j
                
            for i in range(1,m+1):
                for j in range(1,n+1):
                    if w1[i-1]==w2[j-1]:
                       dp[i][j]=dp[i-1][j-1]
                    else:
                        dp[i][j]=1+min(dp[i-1][j-1],dp[i][j-1],dp[i-1][j])
            return dp[m][n]
        res=[]
        for word in words:
            if Distance(target,word) <=k:
                res.append(word)
        return res
words = ["abc", "abd", "abcd", "adc"] 
target = "ac"
k = 1
#Return ["abc", "adc"]        
if __name__ == "__main__":
    print(Solution().kDistance( words, target, k))      
                
        
        
#626. Rectangle Overlap        
"""
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
"""

class Solution:
    """
    @param l1: top-left coordinate of first rectangle
    @param r1: bottom-right coordinate of first rectangle
    @param l2: top-left coordinate of second rectangle
    @param r2: bottom-right coordinate of second rectangle
    @return: true if they are overlap or false
    """
    def doOverlap(self, l1, r1, l2, r2):
        # write your code here
        
         return not ( (r1.y>l2.y   or r2.y>l1.y)   or (r1.x<l2.x or l1.x>r2.x ))
       
        
#627. Longest Palindrome
class Solution:
    """
    @param s: a string which consists of lowercase or uppercase letters
    @return: the length of the longest palindromes that can be built
    """
    def longestPalindrome(self, s):
        # write your code here
        from collections import Counter
        count=Counter(s)
        
        res=0
        for k,v in count.items():
            if v%2==0:
                res+=v
            else:
                res+=v-1
        return res
                
        
#632. Binary Tree Maximum Nod        
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param: root: the root of tree
    @return: the max node
    """
    def maxNode(self, root):
        # write your code here
        def maxNode(self, root):
        # write your code here
        self.res=None
        def preorder(node):
            if not self.res or  node.val>self.res.val :
                self.res=node
            
            if node.left:
               preorder(node.left)
            if node.right:
                preorder(node.right)
        
                
        if not root:
           return None
        preorder(root)
        return self.res


#633. Find the Duplicate Number
class Solution:
    """
    @param nums: an array containing n + 1 integers which is between 1 and n
    @return: the duplicate one
    """
    def findDuplicate(self, nums):
        # write your code here
        n=len(nums)
        
        table=[0 for _ in range(n)]
        
        for x in nums:
            if table[x-1] >0:
                return x
            else:
                table[x-1]=x
        
            

#634. Word Squares   
class TrieNode:
    def __init__(self):
        self.children={}
        self.words=[]

class Trie:
    def __init__(self):
        self.root=TrieNode()
    
    def insert(self,word):
        cur=self.root
        cur.words.append(word)
        
        for char in  word:
            if char  not in cur.children:
                cur.children[char]=TrieNode()
            cur=cur.children[char]
            cur.words.append(word)
            
    def wordstartwithprefix(self,string):
        cur=self.root
        for char in string:
            if char not in cur.children:
                return []
            cur=cur.children[char]
        return cur.words
        
class Solution:
    
    """
    @param: words: a set of words without duplicates
    @return: all word squares
    """
    def wordSquares(self, words):
        # write your code here
#http://massivealgorithms.blogspot.com/2016/10/leetcode-425-word-squares.html
        
        if words is None or len(words)==0:
            return []
        
        self.wordTrie=self.initTrie(words)
        
        res=[]
        self.getWordSquares(len(words[0]),0,[],res)
        return res
        
    def initTrie(self,words):
        wordTrie=Trie()
        for word in words:
            wordTrie.insert(word)
        return wordTrie
    
    def checkValidPrefix(self,wordLen,level,cur,word):
        for i in range(level+1,wordLen):
            prefix=''.join(cur[row][i] for row in range(level) )
            prefix+=word[i]
            if not self.wordTrie.wordstartwithprefix(prefix):
                return False
        return True
            
    def getWordSquares(self,wordLen,level,cur,res):
        if level==wordLen:
            res.append(cur[:])
            return 
        prefix=''.join(cur[row][level] for row in range(level))
        candidates=self.wordTrie.wordstartwithprefix(prefix)
        
        for word in candidates:
            if not self.checkValidPrefix(wordLen,level,cur,word):
                continue
            self.getWordSquares(wordLen,level+1,cur+[word],res)
words=["abat","baba","atan","atal"]      
if __name__ == "__main__":
    print(Solution().wordSquares( words))      
                    
#636. 132 Pattern        
class Solution:
    """
    @param nums: a list of n integers
    @return: true if there is a 132 pattern or false
    """
    def find132pattern(self, nums):
        # write your code here
        
#the maximum candidate for s3 is always the recently popped number from the stack, because
# if we encounter any 
#entry smaller than the current candidate, the function would already have returned. 
        n=len(nums)
        if n<=2:
            return False
        
        stack=[]
        
        e3=float('-inf')
        for e in reversed(nums):
            
            if e<e3:
                return True
            while stack and e>stack[-1]:
                e3=stack.pop()
                
            stack.append(e)
        return False

                
        
nums = [1, 2, 3, 4]
nums = [3, 1, 4, 2]

#637. Valid Word Abbreviation
class Solution:
    """
    @param word: a non-empty string
    @param abbr: an abbreviation
    @return: true if string matches with the given abbr or false
    """
    def validWordAbbreviation(self, word, abbr):
        # write your code here
        
        def match(word, abbr):
            if not word and not abbr:
                return True
            if not word and abbr:
                return False
            if word and not abbr:
                return False
            m=len(word)
            n=len(abbr)
            
            
            
            
            i=0
            j=0
            
            while i<m and j<n:
                if word[i]==abbr[j]:
                    i+=1
                    j+=1
                else:
                    break
            
            if i==m and j==n:
                return True
            elif i==m and j!=n:
                return False
            elif i!=m and j==n:
                return False
            
            if not abbr[j].isdigit() and  abbr[j] != word[i]:
                return False
            #print(i,j)
            digittemp=''
            for x in range(j,n):
                if abbr[x].isdigit():
                    digittemp+=abbr[x]
                else:
                    break
            if digittemp.startswith('0'):
                return False
            num=int(digittemp)
            if num+i>m:
                return False
            #print(i,num,x)
            return  match(word[i+num:], abbr[j+len(digittemp):])
        return match(word, abbr)
word = "internationalization"
abbr = "i12iz4n" 
word = "apple"
abbr = "a2e"
word ="a"
abbr ="1"
word ="a"
abbr ="01"
word ="aa"
abbr ="a2"   
if __name__ == "__main__":
    print(Solution().validWordAbbreviation(word, abbr))      
                                
                    
#638. Isomorphic Strings        
class Solution:
    """
    @param s: a string
    @param t: a string
    @return: true if the characters in s can be replaced to get t or false
    """
    def isIsomorphic(self, s, t):
        # write your code here
        
        from collections  import Counter
        scount=Counter(s)
        tcount=Counter(t)
        
        return sorted(scount.values())==sorted(tcount.values())
        
s = "egg"
t = "add"
#return true.

s = "foo"
t = "bar"
#return false.

s = "paper"
t = "title"
#return true.        
if __name__ == "__main__":
    print(Solution().isIsomorphic( s, t))            
        
        

#639. Word Abbreviation
class Solution:
    """
    @param dict: an array of n distinct non-empty strings
    @return: an array of minimal possible abbreviations for every word
    """
    def wordsAbbreviation(self, dict):
        # write your code here
        self.dmap={}
        def abbr(word,size):
            if len(word)-size<=3:
                return word
            return word[:size+1]+str(len(word)-size-2)+word[-1]
        
        def solve(dict,size):
            from collections import defaultdict
            
            dlist=defaultdict(list)
            
            for word in dict:
                dlist[abbr(word,size)].append(word)
                #print(dlist)
            
            for ab,wlist in dlist.items():
                if len(wlist)==1:
                    self.dmap[wlist[0]]=ab
                else:
                    #print(wlist)
                    solve(wlist,size+1)
        
        
                    
        solve(dict,0)     
        return list(map(self.dmap.get,dict))
                
dict = ["like", "god", "internal", "me", "internet", "interval", "intension", "face", "intrusion"]        
if __name__ == "__main__":
    print(Solution().wordsAbbreviation( dict))            
                
        
        
#640. One Edit Distance
class Solution:
    """
    @param s: a string
    @param t: a string
    @return: true if they are both one edit distance apart or false
    """
    def isOneEditDistance(self, s, t):
        # write your code here
#        m=len(s)
#        n=len(t)
#        
#        dp=[[0 for _ in range(n+1)] for _ in range(m+1)]
#        
#        for i in range(1,m+1):
#            dp[i][0]=i
#            
#        for j in range(1,n+1):
#            dp[0][j]=j
#        
#        for i in range(1,m+1):
#            for j in range(1,n+1):
#                if s[i-1]==t[j-1]:
#                    dp[i][j]=dp[i-1][j-1]
#                else:
#                    
#                    dp[i][j]=1+min(dp[i-1][j-1],dp[i][j-1],dp[i-1][j])
#        #print(dp)
#        return dp[m][n]==1
        
        m=len(s)
        n=len(t)
        
        #make sure m is the longer one
        
        if abs(m-n)>1:
            return False
        if s==t:
            return False
        # ensure that s is a longer one
        if m<n:
            m,n=n,m
            s,t=t,s
        
        i=0
        
        while i<min(n,m):
            if s[i]==t[i]:
                i+=1
                # find the first different position
            else:
                # insert or delete
                if s[i+1:]==t[i:]:
                    return True
                # replace
                if s[i+1:]==t[i+1:]:
                    return True
                return False
        return True

s= "aDb"
t = "adb"
#return true
s="a"
t ="ab"
if __name__ == "__main__":
    print(Solution().isOneEditDistance(s, t))            
                
#641. Missing Ranges
class Solution:
    """
    @param: nums: a sorted integer array
    @param: lower: An integer
    @param: upper: An integer
    @return: a list of its missing ranges
    """
    def findMissingRanges(self, nums, lower, upper):
        # write your code here
        from bisect import insort,bisect_left
        if not nums:
            if lower==upper:
                return [str(lower)]
            else:
                return [str(lower)+'->'+str(upper)]
        lower_in=True
        upper_in=True
        if lower not in nums:
            insort(nums,lower)
            lower_in=False
        if upper not in nums  :
            insort(nums,upper)
            upper_in=False
            
        index_lower=bisect_left(nums,lower)
        if not lower_in:
            nums[index_lower]=lower-1
            
        index_upper=bisect_left(nums,upper)
        if not upper_in:
            nums[index_upper]=upper+1
            
        
        
        res=[]
        for i in range(index_lower+1,index_upper+1):
            if nums[i]-nums[i-1]==1:
                continue
            elif nums[i]-nums[i-1]==2:
                res.append( str(nums[i]-1))
            elif nums[i]-nums[i-1]>2:
                res.append( str(nums[i-1]+1)+ '->'+ str(nums[i]-1)     )
        return res
                
nums = [0, 1, 3, 50, 75]
lower = 0 
upper = 99
#return ["2", "4->49", "51->74", "76->99"].     
if __name__ == "__main__":
    print(Solution().findMissingRanges(nums, lower, upper))            
                
#642. Moving Average from Data Stream
from collections import deque
class MovingAverage:
    """
    @param: size: An integer
    """
    def __init__(self, size):
        # do intialization if necessary
        self.q=deque()
        self.capacity=size
        self.size=0
        self.sum=0

    """
    @param: val: An integer
    @return:  
    """
    def next(self, val):
        # write your code here
        
        if self.size<self.capacity:
            self.size+=1
            self.q.append(val)
            self.sum+=val
            return self.sum/self.size
        else:
            temp=self.q.popleft()
            self.q.append(val)
            self.sum-=temp
            self.sum+=val
            
            return self.sum/self.size
            


# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param = obj.next(val)           

#643. Longest Absolute File Path
class Solution:
    """
    @param input: an abstract file system
    @return: return the length of the longest absolute path to file
    """
    def lengthLongestPath(self, input):
        # write your code here
        dictionary={}
        
        res=0
        
        fileList=input.split('\n')
        print(fileList)
        
        for file in fileList:
            if '.' not in file:#是文件夹
                key=file.count('\t')#\t 是一个字
                dictionary[key]=len(file)-key
                #print(file,key,dictionary[key])
            else:
                key=file.count('\t')
                filelength=sum( [v for k,v in dictionary.items() if k<key])+len(file.replace('\t',''))+key#还要加斜杠
                res=max(res,filelength)
            print(dictionary)
        return res
input="dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext" 
input="dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"   
if __name__ == "__main__":
    print(Solution().lengthLongestPath(input))    
                
#644. Strobogrammatic Number                
class Solution:
    """
    @param num: a string
    @return: true if a number is strobogrammatic or false
    """
    def isStrobogrammatic(self, num):
        # write your code here
        
        def Strobogrammatic(num):
            n=len(num)
            if n==0:
               return True
            if n==1 and num in ('1','8','0'):
               return True
            elif n==1 and num not in ('1','8','0'):
               return False
            x=num[0]
            y=num[-1]
            if x==y and x in ('1','8','0'):
                return Strobogrammatic(num[1:-1])
            if x!=y:
                if  ((x =='6' and y=='9')  or (x =='9' and y=='6')):
                    return Strobogrammatic(num[1:-1])
                else:
                    return False
        return Strobogrammatic(num)
num = "69"
#return true
num = "68"#
#return false
num ='818'
if __name__ == "__main__":
    print(Solution().isStrobogrammatic( num))


#645. Find the Celebrity                
"""
The knows API is already defined for you.
@param a, person a
@param b, person b
@return a boolean, whether a knows b
you can call Celebrity.knows(a, b)
"""
class Solution:
    # @param {int} n a party with n people
    # @return {int} the celebrity's label or -1
    def findCelebrity(self, n):
        # Write your code here
        
        
        def verify(n,j):
            for y in range(n):
                if y==j:
                    continue
                    
                if not Celebrity.knows(y, j):
                       return False
                if  Celebrity.knows(j,y):
                       return False
            return True
        
        
        if n==1:
            return 0
        for i in range(n):
            for j in range(n):
                if i==j:
                    continue
                
                if Celebrity.knows(i, j):
                    if verify(n,j):
                        return j
        return -1
        
                
#646. First Position Unique Character
class Solution:
    """
    @param s: a string
    @return: it's index
    """
    def firstUniqChar(self, s):
        # write your code here
        n=len(s)
        if n==0:
            return -1
        if n==1:
            return s[0]
        
        for i,x in  enumerate(s):
          if x not in s[:i]:
            if x not in s[i+1:]:
                return i
        return -1
s="lintcode"  
s = "lovelintcode" 
s ="{{;;lintcodelintcode}}"     
if __name__ == "__main__":
    print(Solution().firstUniqChar( s))        
        
        
#647. Find All Anagrams in a String
class Solution:
    """
    @param s: a string
    @param p: a string
    @return: a list of index
    """
    def findAnagrams(self, s, p):
        # write your code here
        if not s:
            return []
        m=len(s)
        n=len(p)
        
        if m<n:
            return []
        res=[]
        from collections import Counter
        pcount=Counter(p)
        print(pcount)
        for i in range(n-1,m):
            
            if i==n-1:
               scount=Counter(s[:i+1])
               if scount==pcount:
                  res.append(i-n+1)
            else:
                
                if s[i] not in scount:
                                       
                   scount[s[i]]=1
                else:
                   scount[s[i]]+=1 
                scount[s[i-n]] -=1 
                if scount[s[i-n]]==0:
                    del scount[s[i-n]]
                print(i,scount,pcount)
                if scount==pcount:
                  res.append(i-n+1) 
                
                
        return res
        
s="abab"
p="ab"
if __name__ == "__main__":
    print(Solution().findAnagrams( s, p))        
                
        
#648. Unique Word Abbreviation  
from collections import  defaultdict
     
class ValidWordAbbr:
    """
    @param: dictionary: a list of words
    """
    def __init__(self, dictionary):
        # do intialization if necessary
        self.table=defaultdict(int)
        self.dictionary=set(dictionary)
        for w in self.dictionary:
            
           
            if len(w)<=2:
                abbr=w
                
            else:
                abbr=w[0]+str(len(w)-2)+w[-1]
            self.table[abbr]+=1
        #print(self.table)
            

    """
    @param: word: a string
    @return: true if its abbreviation is unique or false
    """
    def isUnique(self, word):
        # write your code here
        if len(word)<=2:
                abbr=word
        else:
             abbr=word[0]+str(len(word)-2)+word[-1]
        if abbr not in self.table  or self.table[abbr]==0:
            return True
        elif word in self.dictionary  and self.table[abbr]==1:
            return True
        else:
            return False
        
        
        
dictionary = [ "deer", "door", "cake", "card" ]
obj.isUnique("dear") # return false
obj.isUnique("cart") # return true
obj.isUnique("cane") # return false
obj.isUnique("make") # return true 
       
dictionary =["dog"]
obj.isUnique("dig")
obj.isUnique("dug")
obj.isUnique("dag")
obj.isUnique("dog")
obj.isUnique("doge")        


dictionary =["ValidWordAbbr","isUnique"]
obj = ValidWordAbbr(dictionary)
obj.isUnique("a")
obj.isUnique("")


["a","a"]
isUnique("a")
# Your ValidWordAbbr object will be instantiated and called as such:
# obj = ValidWordAbbr(dictionary)
# param = obj.isUnique(word)        
        
        
#649. Binary Tree Upside Down
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of binary tree
    @return: new root
    """
    def upsideDownBinaryTree(self, root):
        # write your code here
        
        def dfs(node):
            if not node.left:
                return node
            
            newnode=dfs(node.left)
            node.left.right=node
            node.left.left=node.right
            node.left=None
            node.right=None
            return newnode
        if not root:
            return None
        return dfs(root)
            
        
        
        
        
    1
   / \
  2   3
 / \
4   5        
        
        
        
   4
  / \
 5   2
    / \
   3   1          
        
#650. Find Leaves of Binary Tree
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
    @return: collect and remove all leaves
    """
    def findLeaves(self, root):
        # write your code here
        
        from collections import deque
        def depth(node):
            if not node:
                return 0
            return 1+max(depth(node.left),depth(node.right))
        
        self.res=deque()
        
        def preOrder(node):
            if not node:
                return 
            self.res.append((node.val,depth(node.left),depth(node.right)))
            preOrder(node.left)
            preOrder(node.right)
        
        
        preOrder(root) 
        ans=[]
        while self.res:
            temp1=deque()
            temp2=[]
            for _ in range(len(self.res)):
                  node,left,right=self.res.popleft()
                  if left==0 and right==0:
                      temp2.append(node)
                  else:
                      if left>0:
                          left-=1
                      if right>0:
                          right-=1
                      temp1.append((node,left,right))
            ans.append(temp2[:])
            self.res=temp1
                      
                
            
        return ans
        
        
#    1
#   / \
#  2   3
# / \     
#4   5  
#
#
#[[4, 5, 3], [2], [1]]
root=TreeNode(1)
root.left=TreeNode(2)
root.right=TreeNode(3)

root.left.left=TreeNode(4)
root.left.right=TreeNode(5)
if __name__ == "__main__":
    print(Solution().findLeaves(root))       







#651. Binary Tree Vertical Order Traversal
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
from collections import defaultdict,deque
class Solution:
    """
    @param root: the root of tree
    @return: the vertical order traversal
    """
    def verticalOrder(self, root):
        # write your code here
        
        self.res=defaultdict(list)
        if not root:
            return []
        if not root.left and not root.right:
            return [[root.val]]
        dq=deque([(root,0)])
       
        while dq:
           node,level=dq.popleft()
           self.res[level].append(node.val)
           if node.left:
               dq.append((node.left,level-1))
           if node.right:
               dq.append((node.right,level+1))
           
            
        
        
        
        #return self.res
        import heapq
        
        ans=[]
        
        for k,v in self.res.items():
            heapq.heappush(ans,(k,v))
        
        ans2=[]
        while ans:
            
            ans2.append(heapq.heappop(ans)[1])
            
        return ans2

#   3
#  /\
# /  \
# 9  20
#    /\
#   /  \
#  15   7
#[[9],[3,15],[20],[7]]
root=TreeNode(3)
root.left=TreeNode(9)
root.right=TreeNode(20)

root.right.left=TreeNode(15)
root.right.right=TreeNode(7)        
if __name__ == "__main__":
    print(Solution().verticalOrder( root))              
        
#653. Expression Add Operators
class Solution:
    """
    @param num: a string contains only digits 0-9
    @param target: An integer
    @return: return all possibilities
    """
    def addOperators(self, num, target):
        # write your code here

#        res=[]
#        def insert(string,target,path,res):
#            
#            if not string:
#               if  eval(path)==target:
#                   res.append(path[:])
#               return 
#            for operator in ('+','-','*',''):
#                if not operator and (path=='0' or path[-2:] in ('+0','-0','*0')):
#                    continue
#                
#                
#                insert(string[1:],target,path+operator+string[0],res)
#            return 
#        insert(num[1:],target,num[0],res)
#        return res
        
        def dfs(num,path,cur,last,target,res):
            if not num:
                if cur==target:
                    res.append(path[:])
                return 
            
            for i in range(1,len(num)+1):
                if i==1 or (num[0]!='0'):
                    val=int(num[:i])
                    
                    dfs(num[i:],path+'+'+num[:i] ,cur+val,val,target,res  )
                    dfs(num[i:],path+'-'+num[:i] ,cur-val,-val,target,res  )
                    dfs(num[i:],path+'*'+num[:i] ,cur-last+last*val,last*val,target,res)
        
        
        res=[]
        
        for i in range(1,len(num)+1):
                if i==1 or (num[0]!='0'):
                    dfs(num[i:],num[:i],int(num[:i]),int(num[:i]),target,res)
        return res
        
num="123"
target= 6
num="232"
target= 8 #-> ["2*3+2", "2+3*2"]
num="105"
target= 5 #-> ["1*0+5","10-5"]
num="00" 
target=0 #-> ["0+0", "0-0", "0*0"]
num="3456237490"
target= 9191 #-> []
if __name__ == "__main__":
    print(Solution().addOperators(num, target))              
                        
#654. Sparse Matrix Multiplication        
class Solution:
    """
    @param A: a sparse matrix
    @param B: a sparse matrix
    @return: the result of A * B
    """
    def multiply(self, A, B):
        # write your code here
        ma=len(A)
        namb=len(A[0])
        nb=len(B[0])
        
        C=[[0 for _ in range(nb)] for _ in range(ma)]
        
        for i in range(ma):
            for j in range(namb):
                for k in range(nb):
                    C[i][k]+=A[i][j]*B[j][k]
        return C
A = [
  [ 1, 0, 0],
  [-1, 0, 3]
]
B = [
  [ 7, 0, 0 ],
  [ 0, 0, 0 ],
  [ 0, 0, 1 ]
]
if __name__ == "__main__":
    print(Solution().multiply(A, B))          
        
        
#655. Add Strings        
class Solution:
    """
    @param num1: a non-negative integers
    @param num2: a non-negative integers
    @return: return sum of num1 and num2
    """
    def addStrings(self, num1, num2):
        # write your code here
        return str(int(num1)+int(num2))
        
        
#656. Multiply Strings
class Solution:
    """
    @param num1: a non-negative integers
    @param num2: a non-negative integers
    @return: return product of num1 and num2
    """
    def multiply(self, num1, num2):
        # write your code here
        l1=len(num1)
        l2=len(num2)
        l3=l1+l2
        res=[ 0 for _ in range(l3)]

        for i in range(l1-1,-1,-1):
            carry=0
            for j in range(l2-1,-1,-1):
                res[i+j+1]+=carry+int(num1[i])*int(num2[j])
                carry=res[i+j+1]//10
                res[i+j+1]=res[i+j+1]%10
            res[i]=carry
        
        k=0
        while k<l3 and res[k]==0:
            k+=1
        res=res[k:]
        
        return '0' if not res else ''.join([str(x) for x in res])

                
#657. Insert Delete GetRandom O(1)                
class RandomizedSet:
    
    def __init__(self):
        # do intialization if necessary
        self.nums=[]
        self.pos={}

    """
    @param: val: a value to the set
    @return: true if the set did not already contain the specified element or false
    """
    def insert(self, val):
        # write your code here
        if val  not in self.pos:
           self.nums.append(val)
           self.pos[val]=len(self.nums)-1
           return True
        return False
    """
    @param: val: a value from the set
    @return: true if the set contained the specified element or false
    """
    def remove(self, val):
        # write your code here
        if val  in self.pos:
            idx=self.pos[val]
            self.nums[idx]=self.nums[-1]
            self.nums.pop()
            self.pos[self.nums[idx]]=idx
            del self.pos[val]
            return True
        return False
            

    """
    @return: Get a random element from the set
    """
    def getRandom(self):
        # write your code here
        import random
        self.nums[random.randint(0,len(self.nums)-1)]

# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param = obj.insert(val)
# param = obj.remove(val)
# param = obj.getRandom()                
        
#659. Encode and Decode Strings
class Solution:
    """
    @param: strs: a list of strings
    @return: encodes a list of strings to a single string.
    """
    def encode(self, strs):
        # write your code here
        res=''
        for i,string in enumerate(strs):
            for j in range(len(string)):
                if string[j]==':':
                    res=res+string[j]+':'
                else:
                    res=res+string[j]
            res+=':'
        return res[:-1]
                    
                  
            

    """
    @param: str: A string
    @return: dcodes a single string to a list of strings
    """
    def decode(self, str):
        # write your code here
        str+=':'
        res=[]
        temp=''
        for i,char in enumerate(str):
            if char==':'  and i+1<len(str)  and str[i+1]==':':
                temp+=':'
            elif  char!=':':
                temp+=char
            elif  char==':':
                res.append(temp[:])
                temp=''
        return res
strs = ["lint","code","love","you"]        
        
#660. Read N Characters Given Read4 II - Call multiple times        
"""
The read4 API is already defined for you.
@param buf a list of characters
@return an integer
you can call Reader.read4(buf)
"""


class Solution:

    # @param {char[]} buf destination buffer
    # @param {int} n maximum number of characters to read
    # @return {int} the number of characters read
    def __init__(self):
        self.head=0
        self.tail=0
        self.buffer=[0 for _ in range(4)]
    def read(self, buf, n):
        # Write your code here
#The meaning here is that read4() function will read 4 characters at 
#a time from a file and then put the characters that has been read into this buf variable.
#So read() function is reading at most n characters from a file 
#( we don’t know what file and how it’s reading from the file),
# and put x characters into char[] buf.
        i=0
        
        while i<n:
            
            if self.head==self.tail:
                self.head=0
                self.tail=Reader.read4(self.buffer)
                if self.tail==0:
                    break
            
            else:
                while i<n and self.head<self.tail:
                    buf[i]=self.buffer[self.head]
                    i+=1
                    self.head+=1
        return i
                
                
            
                
            






        
        
        
#661. Convert BST to Greater Tree        
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of binary tree
    @return: the new root
    """
    def convertBST(self, root):
        # write your code here
        self.sm=0
        self.last=0
        def inorder(node):
            if not node:
                return 
         
            inorder(node.left)
            self.sm+=node.val
            print(self.sm,node.val)
            #print(node.val)
          
            inorder(node.right)
        
        inorder(root)
        
        
        def inorder2(node):
            if not node:
                return 
            
            inorder2(node.left)
            self.sm-=self.last
            self.last=node.val
            node.val=self.sm
            inorder2(node.right)
        inorder2(root)
        return root
        
              5
            /   \
           2     13 
root=TreeNode(5)  
root.left=   TreeNode(2)
root.right=   TreeNode(13)         
if __name__ == "__main__":
    print(Solution().convertBST(root))             
        
        
#662. Guess Number Higher or Lower
"""
The guess API is already defined for you.
@param num, your guess
@return -1 if my number is lower, 1 if my number is higher, otherwise return 0
you can call Guess.guess(num)
"""


class Solution:
    # @param {int} n an integer
    # @return {int} the number you guess
    def guessNumber(self, n):
        # Write your code here
        
        lo=1
        hi=n
        
        while lo<hi:
            mid=(lo+hi)//2
            
            if Guess.guess(mid)==0:
                return mid
            elif Guess.guess(mid)==1:
                hi=mid-1
            else:
                lo=mid+1
                

#663. Walls and Gates
class Solution:
    """
    @param rooms: m x n 2D grid
    @return: nothing
    """
    def wallsAndGates(self, rooms):
        # write your code here
        
        from collections import deque
        m=len(rooms)
        if m==0:
            return 
        n=len(rooms[0])
        
        
        def bfs(rooms,x,y):
            
            visited=set((x,y))
            dq=deque([(x,y,0)])
            
            
            while dq:
                a,b,step=dq.popleft()
                for i,j in ( (a+1,b), (a-1,b),(a,b+1),(a,b-1)):
                    if i <m and j<n and i>=0 and j>=0 and (i,j) not in visited and rooms[i][j]!=0  and rooms[i][j]!=-1:
                        if rooms[i][j] >step+1:
                            rooms[i][j]=step+1
                            dq.append((i,j,step+1))
                            visited.add((i,j))
        for u in range(m):
            for v in range(n):
                if rooms[u][v]==0:
                    bfs(rooms,u,v)
        #return rooms
rooms=[[2147483647,-1,0,2147483647],
 [2147483647,2147483647,2147483647,-1],
 [2147483647,-1,2147483647,-1],
 [0,-1,2147483647,2147483647]]

if __name__ == "__main__":
    print(Solution().wallsAndGates(rooms))             
        

#664. Counting Bits
class Solution:
    """
    @param num: a non negative integer number
    @return: an array represent the number of 1's in their binary
    """
    def countBits(self, num):
        # write your code here
#        from collections import Counter
#        res=[]
#        for i in range(num):
#            res.append(  bin(i)[2:]   )
#        return res
        
        if num==1:
            return [0,1]
        if num==0:
            return [0]
        if num==2:
            return [0,1,2]
        
        
        
        res=[0,1]
        x=1
        while 2**x <=num:
            res=res+ list(map(lambda x:x+1,res))
            x+=1
        return res[:num+1]
num=4
if __name__ == "__main__":
    print(Solution().countBits( num))  


#665. Range Sum Query 2D - Immutable
class NumMatrix(object):

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        m=len(matrix)
        if m>0:
          n=len(matrix[0])
          
        #self.table=[[0 for _ in range(n+1)]  for _ in range(m+1)]
        self.table=[[0 for _ in range(n+1)]  for _ in range(m+1)]
        for i in range(1,m+1):
            for j in range(1,n+1):
                self.table[i][j]=self.table[i][j-1]+matrix[i-1][j-1]
                #table[i][j]=table[i][j-1]+matrix[i-1][j-1]
                
        for i in range(1,m+1):
            for j in range(1,n+1):
                self.table[i][j]+=self.table[i-1][j]
                #table[i][j]+=table[i-1][j]
            
            
        

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        
        return self.table[row2+1][col2+1]+self.table[row1][col1]-self.table[row2+1][col1]-self.table[row1][col2+1]
        

row1, col1, row2, col2=(2, 1, 4, 3)
matrix=[
  [3, 0, 1, 4, 2],
  [5, 6, 3, 2, 1],
  [1, 2, 0, 1, 5],
  [4, 1, 0, 1, 7],
  [1, 0, 3, 0, 5]
]

# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)


#666. Guess Number Higher or Lower II
class Solution:
    """
    @param n: An integer
    @return: how much money you need to have to guarantee a win
    """
    def getMoneyAmount(self, n):
        # write your code here
#dp[i][j]，代表着如果我们在区间 [i , j] 内进行查找，所需要的最少 cost 来保证找到结果
#可以发现对于区间 [i, j] ，猜测 i <= k <= j 我们可能出现以下三种结果：
#       1. k 就是答案，此时子问题的额外 cost = 0 ，当前位置总 cost  = k + 0;
#       2. k 过大，此时我们的有效区间缩小为 [i , k - 1] 当前操作总 cost  = k + dp[i][k - 1];
#       3. k 过小，此时我们的有效区间缩小为 [k + 1 , j] 当前操作总 cost  = k + dp[k + 1][j];
        
        
        dp=[[0 for _ in range(n+1)] for _ in range(n+1)]
        
        for i in range(1,n+1):
            dp[i][i]=0
        
        for i in range(n-1,0,-1):
            for j in range(i+1,n+1):
                mincost=float('inf')
                for k in range(i,j):
                    mincost=min(mincost,k+max(dp[i][k-1],dp[k+1][j]))
                dp[i][j]=mincost
        print(dp)     
        return dp[1][n]
n=10 
if __name__ == "__main__":
    print(Solution().getMoneyAmount( n))        
       
#667. Longest Palindromic Subsequence       
class Solution:
    """
    @param s: the maximum length of s is 1000
    @return: the longest palindromic subsequence's length
    """
    def longestPalindromeSubseq(self, s):
        # write your code here
#F[i][j] defines the max length of longest palindropmic subsequence from i to j in s (j > i)
#for each case, if (s[i] == s[j]) then we know F[i][j] = 2 + F[i + 1][j-1],
#if (s[i] != s[j]), then two case, either don't consider i, or don't consider j
#so F[i][j] = max(F[i + 1][j], F[i][j-1]);        
        n=len(s)
        if n==0:
            return 0
        
        dp=[[0 for _ in range(n)] for _ in range(n)]
        
        
        for i in range(n):
            dp[i][i]=1
        
        for i in range(n-1,-1,-1):
            for j in range(i+1,n):
                if s[i]==s[j]:
                    dp[i][j]=2+dp[i+1][j-1]
                else:
                    dp[i][j]=max(dp[i+1][j],dp[i][j-1])
        return dp[0][n-1]
       
#668. Ones and Zeroes
class Solution:
    """
    @param strs: an array with strings include only 0 and 1
    @param m: An integer
    @param n: An integer
    @return: find the maximum number of strings
    """
    def findMaxForm(self, strs, m, n):
        # write your code here
        #其中dp[i][j]表示有i个0和j个1时能组成的最多字符串的个数
        from collections import Counter
        res=0
        dp=[[0 for _ in range(n+1)] for _ in range(m+1)]
        for s in strs:
            c =Counter(s)
            zero=c.get('0',0) 
            one=c.get('1',0)
            for i in range(m,zero-1,-1):
                for j in range(n,one-1,-1):
                    dp[i][j]=max(dp[i][j],dp[i-zero][j-one]+1)
        print(dp)
        return dp[-1][-1]
strs = ["10", "0001", "111001", "1", "0"]
m = 5
n = 3  
strs = ["0","11","1000","01","0","101","1","1","1","0","0","0","0","1","0","0110101","0","11","01","00","01111","0011","1","1000","0","11101","1","0","10","0111"]
m =9
n =80  
if __name__ == "__main__":
    print(Solution().findMaxForm( strs, m, n))         
       
#669. Coin Change
class Solution:
    """
    @param coins: a list of integer
    @param amount: a total amount of money amount
    @return: the fewest number of coins that you need to make up
    """
    def coinChange(self, coins, amount):
        # write your code here
        
        
#        self.res=float('inf')
#        def dfs(coins,step,amount):
#            if amount<0:
#                return 
#            if amount==0:
#                self.res=min(self.res,step)
#                return 
#            
#            for coin in coins:
#                dfs(coins,step+1,amount-coin)
#        dfs(coins,0,amount)
#        return self.res if self.res <float('inf') else -1
        dp=[2**31 for i in range(amount+1)]
        dp[0]=0
        
        for i in range(amount+1):
            for coin in coins:
                if i-coin>=0:
                    dp[i]=min(dp[i],dp[i-coin]+1)
        print(dp)
        return dp[amount] if dp[amount]!=2**31  else -1
                    
coins = [1, 2, 5]
amount = 11
coins =[1,2,4]
amount =32000
if __name__ == "__main__":
    print(Solution().coinChange(coins, amount))                
        
#670. Predict the Winner        
class Solution:
    """
    @param nums: nums an array of scores
    @return: check if player 1 will win
    """
    def PredictTheWinner(self, nums):
        # write your code here
        n=len(nums)
        if len(nums)%2==0:
            return True
        
        dp=[[0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            dp[i][i]=nums[i]
        for d in range(1,n):    
            for i in range(0,n-d):
                j=i+d
                dp[i][j]=sum(nums[i:j+1])-min(dp[i+1][j],dp[i][j-1])
                #dp[i][j]= max(nums[i] + sum(nums[i+1:j+1])-dp[i+1][j] , nums[j] +  sum(nums[i:j])- dp[i][j-1])
                #print(dp)
        return dp[0][n-1]> sum(nums)-dp[0][n-1]
nums = [1, 5, 233, 7]
nums = [1, 5, 2]
if __name__ == "__main__":
    print(Solution().PredictTheWinner(nums))              

#671. Rotate Words       
class Solution:
    """
    @param: words: A list of words
    @return: Return how many different rotate words
    """
#    def __init__(self):
#       self.array=[]
#    def find(self,i):
#        if self.array[i]!=i:
#            self.array[i]=self.find(self.array[i])
#        return self.array[i]
#    
#    def union(self,x,y):
#        a=self.find(x)
#        b=self.find(y)
#        self.array[min(a,b)]=max(a,b)
            
    def countRotateWords(self, words):
        # Write your code here
        n=len(words)
        if n==1:
            return 1
        if n==0:
            return 0
        
        hashset=set()
        count=0
        exist=False
        for word in words:
            for i  in range(len(word)):
                new_word=word[i+1:]+word[:i+1]
                #print(word,new_word ,hashset)
                if new_word in hashset:
                    exist=True
                    break
            #print(word,hashset,exist)
            if not exist:
                hashset.add(word)
               
                count+=1
            else:
                exist=False
        return count
                
                
                
            
#        self.array=[i for i in range(n)]
        
#        for i in range(n):
#            if self.array[i]!=i:
#                continue
#            for j in range(i+1,n):
#                if self.array[j]!=j:
#                    continue
#                if  len(words[i]) == len(words[j]):
#                  if  words[i] in words[j]+words[j]:
#                      self.array[j]=i
#        print(self.array)
#        for x in range(1,n):
#            self.find(x)
#        print(self.array)    
#        return len(set(self.array))
        
words = ["picture", "turepic", "icturep", "word", "ordw", "lint"] 
if __name__ == "__main__":
    print(Solution().countRotateWords(words))              
              
#676. Decode Ways II
class Solution:
    """
    @param s: a message being encoded
    @return: an integer
    """
    def numDecodings(self, s):
        # write your code here
        
        mod=1000000007
        one = {'1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1, '*': 9}            
        two = {'10': 1, '11': 1, '12': 1, '13': 1, '14': 1, '15': 1, '16': 1, '17': 1, '18': 1, '19': 1, '20': 1, '21': 1,
       '22': 1, '23': 1, '24': 1, '25': 1, '26': 1, '*0': 2, '*1': 2, '*2': 2, '*3': 2, '*4': 2, '*5': 2, '*6': 2,
       '*7': 1, '*8': 1, '*9': 1, '1*': 9, '2*': 6, '**': 15}
        
        pre=1
        cur=one.get(s[:1],0)
        
        for i in range(1,len(s)):
        
            pre,cur=cur, (cur*one.get(s[i],0)+pre*two.get(s[i-1:i+1],0))%mod
        return cur
                    
                    
#677. Number of Big Islands
class Solution:
    """
    @param grid: a 2d boolean array
    @param k: an integer
    @return: the number of Islands
    """
    def numsofIsland(self, grid, k):
        # Write your code here
        m=len(grid)
        if m==0:
            return 0
        n=len(grid[0])
        
        def dfs(grid,i,j,count):
            for x,y in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
                if x>=0 and y>=0 and x<m and y<n and grid[x][y]==1 and (x,y) not in visited:
                    visited.add((x,y))
                    count=dfs(grid,x,y,count+1)
            return count
        visited=set()
        res=0
        for i in range(m):
            for j in range(n):
                
                if grid[i][j]==1 and (i,j) not in visited:
                    visited.add((i,j))
                    count=dfs(grid,i,j,1)
                    if count>=k:
                        res+=1
        return res
grid=[
  [1, 1, 0, 0, 0],
  [0, 1, 0, 0, 1],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1]
]                    
k=2
if __name__ == "__main__":
    print(Solution().numsofIsland( grid, k))                       
                    
                    
#678. Shortest Palindrome
class Solution:
    """
    @param str: String
    @return: String
    """
    def convertPalindrome(self, str):
        # Write your code here
        def isPalindrome(s):
            n=len(s)
            if n==0 or n==1:
                return True
            if n==2:
                return s[0]==s[1]
            return s[0]==s[-1]  and isPalindrome(s[1:-1])
        
        
        n=len(str)
        if isPalindrome(str):
           return str
        
        for i in range(n-1,-1,-1):
            if isPalindrome(str[:i]):
                return str[i:][::-1]+str
str="aacecaaa"
str="abcd"
if __name__ == "__main__":
    print(Solution().convertPalindrome( str))           
                    
#679. Unique Paths III                
class Solution:
    """
    @param: : an array of arrays
    @return: the sum of all unique weighted paths
    """

    def uniqueWeightedPaths(self, grid):
        # write your codes here
        
#        def travel(grid,i,j,pathsum):
#            #print(i,j)
#            if i+1==len(grid) and j+1==len(grid[0]):
#                #print(i,j)
#                self.res.add(pathsum)
#                return 
#            for x,y in ((i+1,j), (i,j+1)):
#                if x>=0 and y>=0 and x<len(grid) and y<len(grid[0]):
#                    travel(grid,x,y,pathsum+grid[x][y])
#        self.res=set()
#        m=len(grid)
#        if m==0:
#            return 0
#        n=len(grid[0])
#        if n==0:
#            return 0
#            
#        travel(grid,0,0,grid[0][0])
#        return sum(self.res)
        
        m=len(grid)
        if m==0:
            return 0
        n=len(grid[0])
        if n==0:
            return 0
        from collections import defaultdict
        dp=defaultdict(set)
        dp[0].add(grid[0][0])
        
        
        for i in range(1,m):
            cur=i*n
            pre=(i-1)*n
            for dis in dp[pre]:
                dp[cur].add(dis+grid[i][0])
                
        for j in range(1,n):
            cur=j
            pre=j-1
            for dis in dp[pre]:
                dp[cur].add(dis+grid[0][j])
                
                
        for i in range(1,m):
            for j in range(1,n):
                cur=i*n+j
                left=i*n+j-1
                up=(i-1)*n+j
                
                for dis in dp[left]:
                    dp[cur].add(dis+grid[i][j])
                for dis in dp[up]:
                    dp[cur].add(dis+grid[i][j])
        idx=m*n-1
        
        return sum(dp[idx])
                
    
grid=[
  [1,1,2],
  [1,2,3],
  [3,2,4]
]
grid=[[]]
if __name__ == "__main__":
    print(Solution().uniqueWeightedPaths( grid))

                    
#680. Split String                    
class Solution:
    """
    @param: : a string to be split
    @return: all possible split string array
    """

    def splitString(self, s):
        # write your code here
        
        def split(S,path,res):
            if not S:
                res.append(path[:])
                return 
            print(S)
            n=len(S)
            if n==1:
                split(S[1:],path+[S],res)
            elif n>=2:
                split(S[1:],path+[S[0]],res)
                split(S[2:],path+[S[0:2]],res)
            return 
            
        res=[]
        split(s,[],res)
        return res
s= "123"       
if __name__ == "__main__":
    print(Solution().splitString( s))
                    

#681. First Missing Prime Number
class Solution:
    """
    @param nums: an array of integer
    @return: the first missing prime number
    """
    def firstMissingPrime(self, nums):
        # write your code here
        def isPrime(n):
            import math
            if n in (2,3,5,7):
                return True
           
            if n in (4,6,8,9):
                return False
            for i in range(2,  int(math.ceil(n**0.5))+1):
                if n%i==0:
                    return False
            return True
        
        n=len(nums)
        if n==0:
            return 2
        if nums[0]>2:
            return 2
        k=2
        
        nums=set(nums)
        
        while True:
            if k in nums:
                k+=1
                continue
            else:
                if isPrime(k):
                    return k
                else:
                    k+=1
            
                    
nums=[2,3,5,7,11,13,17,23,29]                    
                    
if __name__ == "__main__":
    print(Solution().firstMissingPrime( nums))                    
                    
                    
#683. Word Break III
class Solution:
    """
    @param: : A string
    @param: : A set of word
    @return: the number of possible sentences.
    """

    def wordBreak3(self, s, dict):
        # Write your code here   

        dict=[st.lower()  for st in dict ]
        dict=set(dict)
        s=s.lower()
        
        max_length=0
        for x in dict:
            if len(x)>max_length:
                max_length=len(x)
        
        def dfs(string,dict,max_length,memo):
            if not string:
                return 1
            if string in memo:
                return memo[string]
            
            count=0
            
            for i in range(1,max_length+1):
                if i >len(string):
                    break
                if string[:i] not in dict:
                    continue
                count+=dfs(string[i:],dict,max_length,memo)
            memo[string]=count
            return count
                
        return dfs(s,dict,max_length,{})
        
                   
               

dict=["Cat", "Mat", "Ca", "tM", "at", "C", "Dog", "og", "Do"] 
                
s='CatMat'
s="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
dict=["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]    
if __name__ == "__main__":
    print(Solution().wordBreak3( s, dict))                   
                         
#684. Missing String                     
class Solution:
    """
    @param str1: a given string
    @param str2: another given string
    @return: An array of missing string
    """
    def missingString(self, str1, str2):
        # Write your code here
        str1=str1.split()
        str2=str2.split()
        from collections import Counter
        count=Counter(str2)
        
        for i in range(len(str1)):
            if str1[i] in count  and count[str1[i]]>0:
                str1[i]=''
                count[str1[i]]-=1
        res=[x for x in str1 if x]
        return res
                
        
            
        
        
        
       
str1=    "This is an example"
str2="is example"
str1= "This is an example"
str2=" "
str1="This is an example"
str2="example is"
if __name__ == "__main__":
    print(Solution().missingString( str1, str2))                      
            
#685. First Unique Number In Stream            
class Solution:
    """
    @param nums: a continuous stream of numbers
    @param number: a number
    @return: returns the first unique number
    """
    def firstUniqueNumber(self, nums, number):
        # Write your code here
        from collections import OrderedDict
        numtable=OrderedDict()
        has=False
        for x in nums:
            
            if x in numtable:
               numtable[x]+=1
            else:
                numtable[x]=1
            if x==number:
                has=True
                break
        if not has:
            return -1
        for y in numtable:
            if numtable[y]==1:
                return y
        return -1
nums=[1, 2, 2, 1, 3, 4, 4, 5, 6]            
number=5                        
                    
nums=[1, 2, 2, 1, 3, 4, 4, 5, 6]
number=7                         
if __name__ == "__main__":
    print(Solution().firstUniqueNumber(nums, number))                      
                                    
#688. The Number In Words
class Solution:
    """
    @param number: the number
    @return: the number in words
    """
    def __init__(self):
            self.less20=["","One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten","Eleven","Twelve","Thirteen","Fourteen","Fifteen","Sixteen","Seventeen","Eighteen","Nineteen"]
            self.tens=["","Ten","Twenty","Thirty","Forty","Fifty","Sixty","Seventy","Eighty","Ninety"]
            self.thousands=["","Thousand","Million","Billion"]
    def convertWords(self, number):
        # Write your code here
        if number==0:
            return 'Zero'
        def helper(n):
            if n==0:
                return ''
            if n<20:
               return self.less20[n]+' '
            elif n<100:
               return self.tens[n//10]+' '+self.less20[n%10]
            else:
               #print('&&')
               #print(self.tens[n//100])
               #print(self.tens[n//100]+'Hundred' +helper(n%100))
               return self.less20[n//100]+' Hundred' +' '+helper(n%100)
        
        res=''
        #print('**')
        for i in range(len(self.thousands)):
            if number%1000!=0:
               
               res=helper(number%1000) +' '+self.thousands[i]+' '+res
            number//=1000
            #print(res)
        return res.lower()

number = 125
if __name__ == "__main__":
    print(Solution().convertWords( number))            
        
#689. Two Sum IV - Input is a BST                        
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param: : the root of tree
    @param: : the target sum
    @return: two numbers from tree which sum is n
    """

    def twoSum(self, root, n):
        # write your code here 
        self.table=set()
        
        def inorder(node,n,res,setval):
            if not node:
                return 
            inorder(node.left,n,res,setval | set([node.val]))
            if n-node.val in setval:
                res.append(node.val)
                res.append(n-node.val)
            inorder(node.right,n,res,setval | set([node.val]))
        res=[]
        
        
        inorder(root,n,res,set())
        #print(res)
        return res[:2] if res else None
            
root=  TreeNode(4)     
root.left=  TreeNode(2)
root.right=  TreeNode(5)
root.left.left=  TreeNode(1)
root.left.right=  TreeNode(3)     
 
n = 3
#[1, 2] or [2, 1]                    
#    4
#   / \
#  2   5
# / \
#1   3            
if __name__ == "__main__":
    print(Solution().twoSum(root, n))            
                   
        
        
        
        
#690. Factorial
class Solution:
    """
    @param n: an integer
    @return:  the factorial of n
    """
    def factorial(self, n):
        # write your code here
#https://blog.csdn.net/zhaohengchuan/article/details/78590395
        
        res=[0  for _ in range(5800)]
        
        res[0]=1
        digit=1
        
        for i in range(2,n+1):
            carry=0
            for j in range(digit):
                num=res[j]*i+carry
                #print(res,num)
                res[j]=num%10
                carry=num//10
            while carry:
                res[digit]=carry%10
                carry=carry//10
                digit+=1
   
        start=False
        res2=''
        for  k in range(5800) :
            if start:
                res2+=str(res[5800-k-1])
                continue
                
            if  not start and res[5800-k-1]:
                start=True
                res2+=str(res[5800-k-1])
        
        return res2
n = 2000
if __name__ == "__main__":
    print(Solution().factorial( n))   


#691. Recover Binary Search Tree        
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the given tree
    @return: the tree after swapping
    """
    def bstSwappedNode(self, root):
        # write your code here
        self.prev=TreeNode(float('-inf'))
        self.first=None
        self.second=None
        
        def tranverse(root):
            if not root:
                return 
            tranverse(root.left)
            if not self.first  and self.prev.val>root.val:
                self.first=self.prev
            if self.first  and self.prev.val>root.val:
                self.second=root
            self.prev=root
            tranverse(root.right)
        
        tranverse(root)
        
        if self.second  and self.first:
        
             self.second.val,self.first.val =self.first.val ,self.second.val
        return root

            
#https://leetcode.com/problems/recover-binary-search-tree/discuss/32559/Detail-Explain-about-How-Morris-Traversal-Finds-two-Incorrect-Pointer            
#http://www.cnblogs.com/AnnieKim/archive/2013/06/15/morristraversal.html            
class Solution:
    """
    @param root: the given tree
    @return: the tree after swapping
    """
    def bstSwappedNode(self, root):
        self.prev=None
        self.first=None
        self.second=None
        
        def moristranverse(root):
            if not root:
                return 
            temp=None
            while root:
                if root.left:
                    temp=root.left
                    while temp.right and temp.right!=root:
                         temp=temp.right
                    if temp.right:
                        temp.right=None
                        print(root.val)
                        if self.prev and self.prev.val>root.val:
                            if not self.first:
                                self.first=self.prev
                                self.second=root
                            else:
                                self.second=root
                        self.prev=root
                        root=root.right
                    else:
                        temp.right=root
                        root=root.left
                else:
                    print(root.val)
                    if self.prev and self.prev.val>root.val:
                            if not self.first:
                                self.first=self.prev
                                self.second=root
                            else:
                                self.second=root
                    self.prev=root
                    root=root.right
        moristranverse(root)
        if self.second  and self.first:
               self.second.val,self.first.val =self.first.val ,self.second.val
        return root               
                
#692. Sliding Window Unique Elements Sum
class Solution:
    """
    @param nums: the given array
    @param k: the window size
    @return: the sum of the count of unique elements in each window
    """
    def slidingWindowUniqueElementsSum(self, nums, k):
        # write your code here
        if not nums :
            return 0
        if not k:
            return 0
        n=len(nums)
        from collections import Counter
        res=0
        count={}
        ans=0
        if k>=n:
            count=Counter(nums)
            
            for key,v in count.items():
                if v==1:
                    res+=1
            return res
        
        else:
            for i in range( k ,n+1):
                if not count:
                   count=Counter(nums[i-k:i])
                   
                   for key,v in count.items():
                       if v==1:
                          res+=1
                   #print(count,res)
                   ans+=res
                else:
                    
                    if nums[i-k-1]==nums[i-1]:
                        ans+=res
                        continue
                        
                    if count[nums[i-k-1]]==2  and nums[i-k-1]!=nums[i-1]:
                        res+=1
                    elif count[nums[i-k-1]]==1:
                        #print(count,count[nums[i-k-1]])
                        res-=1
                    #print(res)
                    count[nums[i-k-1]]-=1
                    
                    if nums[i-1] in count:
                        if count[nums[i-1]]==1 :
                            #print(count,count[nums[i-1]])
                            res-=1
                        count[nums[i-1]]+=1
                        
                    else:
                        count[nums[i-1]]=1
                    if count[nums[i-1]]==1:
                        res+=1
                    #print(res)
                    ans+=res
        return ans
                        
nums = [1, 2, 1, 3, 3] 
k = 3   
nums = [1,1,1,1,1]
k =2 
nums =[27,14,60,87,37,53,100,18,51,37,14,57,22,95,50,83,41,43,36,48,52,97,16,46,75,24,47,13,40,40,48,45,56,58,77,3,78,60,31,27,40,53,57,29,30,65,37,77,1,40,89,100,50,49,100,51,22,66,33,33,70,36,64,70,11,27,57,77,17,28,62,70,32,88,12,47,69,30,93,3,47,69,64,88,7,40,38,5,23,4,58,97,19,55,17,23]
k =21                
if __name__ == "__main__":
    print(Solution().slidingWindowUniqueElementsSum( nums, k))   
                 
#696. Course Schedule III
class Solution:
    """
    @param courses: duration and close day of each course
    @return: the maximal number of courses that can be taken
    """
    def scheduleCourse(self, courses):
        # write your code here
        if not courses or len(courses)==0:
            return 0
        start=0
        import heapq
        pq=[]
        for t, end  in sorted(courses,key=lambda x: x[1] ):
        #for t, end in sorted(courses, key = lambda (t, end): end):
            start+=t
            heapq.heappush(pq,-t)
            while start >end:
                start+=heapq.heappop(pq)
        return len(pq)
            
courses=[[100, 200], [200, 1300], [1000, 1250], [2000, 3200]]            
if __name__ == "__main__":
    print(Solution().scheduleCourse(courses))            
        
        
#697. Sum of Square Numbers
class Solution:
    """
    @param num: the given number
    @return: whether whether there're two integers
    """
    def checkSumOfSquareNumbers(self, num):
        # write your code here
        if num<0:
            return False
        import math
        
        cutoff= math.ceil( num**0.5)
        
        hashset=set()
        
        for i in range(cutoff+1):
            hashset.add(i*i)
            if num-i*i in hashset:
                return True
        return False
            
        
        
#698. Maximum Distance in Arrays        
class Solution:
    """
    @param arrs: an array of arrays
    @return: return the max distance among arrays
    """
    def maxDiff(self, arrs):
        # write your code here
        l=[]
        import heapq
        start=[]
        end=[]
        for i,x in enumerate(arrs):
            heapq.heappush(start,(x[0],i))
            heapq.heappush(end,(-x[1],i))
        s,si=heapq.heappop(start)
        e,ei=heapq.heappop(end)
        if si!=ei:
            return abs(s+e)
        else:
            s2,si2=heapq.heappop(start)
            e2,ei2=heapq.heappop(end)
            return max(abs(s2+e),abs(e2+s) )
            
arrs = [[1,2,3], [4,5], [1,2,3]]
arrs = [[2,3,4,5,6,7,8,9],[1,10],[-1,200]]
if __name__ == "__main__":
    print(Solution().maxDiff( arrs))             
        
#699. Check Sum of K Primes        
class Solution:
    """
    @param n: an int
    @param k: an int
    @return: if N can be expressed in the form of sum of K primes, return true; otherwise, return false.
    """
    def isSumOfKPrimes(self, n, k):
        # write your code here
        
        def isPrime(n):
            import math
            if n in (2,3,5,7):
                return True
           
            if n in (4,6,8,9):
                return False
            for i in range(2,  int(math.ceil(n**0.5))+1):
                if n%i==0:
                    return False
            return True
        if n<k*2 or k<1:
            return False
        if k==1:
            return isPrime(n)
        if k==2:
            
            if n%2==0:
                return True
            return isPrime(n-2)
        return True
        
        
#        if n<2:
#            return False
#        if n ==2:
#           primes=set([2])
#        elif n ==3 or n ==4:
#           primes=set([2,3])
#        elif n==5 :
#            primes=set([2,3,5])
#        else:
#            primes=set([2,3,5])
#            for i in range(6,n+1):
#                if isPrime(i):
#                    primes.add(i)
#        dp=[False for _ in range(n+1)]    
#        dp[0]=True        
#        for _ in range(k):
#            temp=[False for _ in range(n+1)] 
#            for x in primes:
#                for y in range(n+1):
#                    if dp[y]==True  and x+y<n+1:
#                        temp[x+y]=True
#            dp=temp
#            
        
n = 10
k = 2
n = 2
k = 2
n =888450280
k =444225140
if __name__ == "__main__":
    print(Solution().isSumOfKPrimes( n, k))             
                        
            
            
#700. Cutting a Rod                    
class Solution:
    """
    @param prices: the prices
    @param n: the length of rod
    @return: the max value
    """
    def cutting(self, prices, n):
        # Write your code here
        prices=[0]+prices
        memo={}
        def cut(prices,k):
            if k in memo:
                return memo[k]
            if k==0:
                memo[0]=0
                return 0
            if k==1:
                memo[1]=prices[1]
                return prices[1]
            if k==2:
                memo[2]=max(prices[1]*2,prices[2])
                return max(prices[1]*2,prices[2])
            res=0
            temp=0
            for i in range(1,k+1):
                
                temp=prices[i]+cut(prices,k-i)
                res=max(res,temp)
            memo[k]=res
            return res
        return cut(prices, n)
prices=[3,5,8,9,10,17,17,20]
n=8
prices= [1, 5, 8, 9, 10, 17, 17, 20]
n=8                    
if __name__ == "__main__":
    print(Solution().cutting(prices, n))                      
                
        
#701. Trim a Binary Search Tree
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: given BST
    @param minimum: the lower limit
    @param maximum: the upper limit
    @return: the root of the new tree 
    """
    def trimBST(self, root, minimum, maximum):
        # write your code here
        def trim(node):
            if not node:
                return 
            if node.val>maximum:
                return trim(node.left)
            elif node.val<minimum:
                return trim(node.right)
            else:
                node.left=trim(node.left)
                node.right=trim(node.right)
            return node
        return trim(root)



#702. Concatenated String with Uncommon Characters of Two Strings
class Solution:
    """
    @param s1: the 1st string
    @param s2: the 2nd string
    @return: uncommon characters of given strings
    """
    def concatenetedString(self, s1, s2):
        # write your code here
        if not s1:
            return s2
        if not s2 :
            return s1
        
        res=''
        
        for s in s1:
            if s not in s2:
                res+=s
        for s in s2:
            if s not in s1:
                res+=s
        return res



#703. Folding Array
class Solution:
    """
    @param nums: the original array
    @param req: the direction each time
    @return: the final folded array 
    """
    def folding(self, nums, req):
        # write your code here
#https://github.com/jiadaizhao/LintCode/blob/master/0703-Folding%20Array/0703-Folding%20Array.cpp        
        n=len(nums)
        l=n//2
        for i in range(len(req)):
            res=[0 for _ in range(n)]
            if req[i]==0:
                k=0
                for j in range(n//2-1,-1,-1):
                    res[j]=nums[k]
                    k+=1
                    if k%l==0:
                        k+=l
                k=l
                for j in range(n//2,n):
                    res[j]=nums[k]
                    k+=1
                    if k%l==0:
                        k+=l
            else:
              k=n-1
              for j in range(n//2):
                res[j]=nums[k]
                k-=1
                if (k+1)%l==0:
                    k-=l
              k=0
              for j in range(n//2,n):
                res[j]=nums[k]
                k+=1
                if (k)%l==0:
                    k+=l
            l//=2
            nums[:]=res[:]
        return res
                    
#req[i] = 0            
#1 2 3 4 5 6 7 8  ==>   4 3 2 1
#                       5 6 7 8  
#If req[i] = 1
#1 2 3 4 5 6 7 8  ==>   8 7 6 5
#                       1 2 3 4                       
#                       
#fold from left to right
#4 3 2 1  ==>  6 5
#5 6 7 8       3 4
#              2 1
#              7 8                       
nums = [1, 2, 3, 4, 5, 6, 7, 8]
req = [0, 0, 1]                       
if __name__ == "__main__":
    print(Solution().folding( nums, req))                      
                                       
#704. Bulb Switcher II
class Solution:
    """
    @param n: number of lights
    @param m: number of operations
    @return: the number of status
    """
    def flipLights(self, n, m):
        # write your code here
                       
#https://leetcode.com/problems/bulb-switcher-ii/discuss/107267/Python-Straightforward-with-Explanation                       
#As before, the first 6 lights uniquely determine the rest of the lights.
# This is because every operation that modifies the xx-th light also modifies 
# the (x+6)(x+6)-th light, so the xx-th light is always equal to the
# (x+6)(x+6)-th light.
#
#Actually, the first 3 lights uniquely determine the rest of the sequence,
# as shown by the table below for performing the operations a, b, c, d:
#
#Light 1 = 1 + a + c + d
#Light 2 = 1 + a + b
#Light 3 = 1 + a + c
#Light 4 = 1 + a + b + d
#Light 5 = 1 + a + c
#Light 6 = 1 + a + b
#So that (modulo 2):
#
#Light 4 = (Light 1) + (Light 2) + (Light 3)
#Light 5 = Light 3
#Light 6 = Light 2                       
        import itertools
        seen=set()
        for cand in  itertools.product((0,1),repeat=4):
              print(cand)
              
              if sum(cand)%2==m%2 and sum(cand)<=m:
                  A=[]
                  for i in range(min(n,3)):
                      light=1
                      light^=cand[0]
                      if i%2==0 :
                         light^=cand[1] 
                      if i%2==1:  
                         light^=cand[2]
                      if i%3==0:#0 index
                         light^=cand[3]
                      A.append(light)
                  seen.add(tuple(A))
        return len(seen)
n=5
m=5
if __name__ == "__main__":
    print(Solution().flipLights( n, m))                      
                                               
#706. Binary Watch
    
class Solution:
    """
    @param num: the number of "1"s on a given timetable
    @return: all possible time
    """
    def binaryTime(self, num):
        # Write your code here
#hours (0-11), and the 6 LEDs on the bottom represent the minutes (0-59)
        from collections import defaultdict
        res=[]
        if num==0:
            return ["0:00"]
        
        gethour=defaultdict(list)
        getmin=defaultdict(list)
        hourList=['0',"1", "2", "3", "4","5", "6","7", "8","9", "10","11"]   
        minList=['00',"01", "02", "03", "04","05", "06","07", "08","09", "10",
                    "11", "12", "13", "14","15", "16","17", "18","19", "20",
                    "21", "22", "23", "24","25", "26","27", "28","29", "30",
                    "31", "32", "33", "34","35", "36","37", "38","39", "40",
                    "41", "42", "43", "44","45", "46","47", "48","49", "50",
                    "51", "52", "53", "54","55", "56","57", "58","59" ]
        for i in range(12):
                gethour[bin(i)[2:].count('1')].append(i)
        for j in range(60):
                getmin[bin(j)[2:].count('1')].append(j)
        print(gethour)
        print(getmin)
        for hour in range(num+1):
            minute=num-hour
            if hour not in gethour  or minute  not in getmin:
                continue
            
            for h in gethour[hour]:
                for m in getmin[minute]:
                    
                    res.append(hourList[h]+':'+minList[m])
        return res
num=1   
num=0 
num=5            
if __name__ == "__main__":
    print(Solution().binaryTime( num))                
                
#707. Optimal Account Balancing  
#这题未作出，need to re do  
#/**************************************************************************************/
#/**************************************************************************************/
#/**************************************************************************************/
#/**************************************************************************************/
#/**************************************************************************************/
#/**************************************************************************************/
#/**************************************************************************************/         

class Solution:
    """
    @param edges: a directed graph where each edge is represented by a tuple
    @return: the number of edges
    """
    def balanceGraph(self, edges):
        # Write your code here
#https://github.com/kamyu104/LeetCode/blob/master/Python/optimal-account-balancing.py
#https://blog.csdn.net/magicbean2/article/details/78582544
#            http://www.cnblogs.com/grandyang/p/6108158.html
#        https://liumingzhang.gitbooks.io/google-questions/optimal_account_balancing.html
#        from collections import defaultdict
#        graph=defaultdict(int)
#        
#        for u,v,w in edges:
#            graph[u]-=w
#            graph[v]+=w
#        account=[]
#        n=0
#        for k,v in graph.items():
#            if v !=0:
#               account.append(v)
#               n+=1
#        memo={}
#        def dfs(account,start,length,cnt,memo):
#            if (tuple(account) , start) in memo:
#                return  memo[(tuple(account) , start)]
#            while start<length and account[start]==0:
#                start+=1
#            res=float('inf')
#            for i in range(start+1,length):
#                if account[start]*account[i]<0:
#                   account[i]+=account[start]
#                   res=min(res,dfs(account,start+1,length,cnt+1,memo))
#                   account[i]-=account[start]
#            memo[(tuple(account) , start)]=res if res!=float('inf') else cnt
#                   
#            return res if res!=float('inf') else cnt
#        return dfs(account,0,n,0,memo)
    
    
        from collections import defaultdict
        graph=defaultdict(int)
        
        for u,v,w in edges:
            graph[u]-=w
            graph[v]+=w
        account=[]
      
        for k,v in graph.items():
            if v !=0:
               account.append(v)
          
        n=1<<len(account)
        dp=[float('inf') for _ in range(n)]
        subset=[]
        if len(account)==0:
            return 0
        for i in range(1,n):
            new_debt=0
            number=0
            for j in range(len(account)):
                if i & 1<<j:#take some random combinations of pairs 
                    new_debt+=account[j]
                    number+=1
            if new_debt==0:
                number-=1
                dp[i]=number
                for s in subset:
                    if (i&s) ==s:#  i&(i-s)==i-s
                        dp[i]=min(dp[i],dp[s]+dp[i-s])# s可以使net_debt  所以i - s 可以使其余的为0，因为总和是0
            subset.append(i)
        return dp[-1]
                    
                    
    
    
   
        
 
edges=[[0,1,10],[2,0,5]]
edges=[[0,1,10],[1,0,1],[1,2,5],[2,0,5]] 
edges=[[7,9,1],[9,8,59],[4,0,46],[7,6,92],[7,6,92],[2,3,93],[1,3,96],[6,8,70],[2,4,36],[3,1,23],[8,9,42],[8,7,45],[2,4,24],[9,8,17],[5,7,89],[0,2,65],[1,0,91],[5,6,2],[8,9,24],[4,1,41]]                 
edges=[[16,15,1],[9,11,59],[0,1,46],[14,15,92],[16,11,37],[14,13,54],[6,5,17],[7,6,72],[5,0,68],[15,11,4],[10,11,74],[5,7,54],[3,4,63],[11,15,24],[15,12,17],[13,14,89],[0,6,65],[6,5,91],[15,13,7],[11,10,30]]                
if __name__ == "__main__":
    print(Solution().balanceGraph( edges))                         
                       
                       
#717. Tree Longest Path With Same Value        
class Solution:
    """
    @param A: as indicated in the description
    @param E: as indicated in the description
    @return: Return the number of edges on the longest path with same value.
    """
    def LongestPathWithSameValue(self, A, E):
        # write your code here
        from collections import defaultdict
#https://github.com/jiadaizhao/LintCode/blob/master/0717-Tree%20Longest%20Path%20With%20Same%20Value/0717-Tree%20Longest%20Path%20With%20Same%20Value.cpp
        #不要用 visited， 只要 cur != prev排除重复检测。 
        graph=defaultdict(list)
        
        for i in range(0,len(E),2):
            graph[E[i]].append(E[i+1])
            graph[E[i+1]].append(E[i])
        print(graph)    
   
        self.visited=set()
    
        A=[0]+A
        def dfs(node,graph,prev):
            if not node :
                return 0
            res=0
         
            for next_node in graph[node]:
                if next_node !=prev:
                    
                    if A[node]==A[next_node]:
                        
                        
                        temp1=1+dfs(next_node,graph,node)
                        if temp1>res:
                           
                           res=temp1
                        
                        print('node',node, 'next_node',next_node, 'res',res)
            
            return res
        ans=0
        prev=0
        for x in range(1,len(A)):
            
                first=0
                second=0
                
                for nei in graph[x]:
                    print('x',x, 'nei',nei)
                    if A[nei] ==A[x]:
                        
                        temp =1+dfs(nei,graph,x)
                        if temp>first:
                            second=first
                            first=temp
                            
                        elif temp>second:
                            second=temp
                        print('first',first ,'second',second)   
                ans=max(ans,first+second)
                            
                            
        return ans
            
 
A = [1, 1, 1 ,2, 2,1,1] 
E =  [1, 2, 1, 3, 2, 4, 2, 5,3,6,6,7]
A = [1, 1, 1 ,2, 2] 
E =  [1, 2, 1, 3, 2, 4, 2, 5]
A =[1,1,1,1,1]
E =[1,2,1,3,2,4,2,5]

                   1 （value = 1）
                 /   \
    (value = 1) 2     3 (value = 1)
               /  \
 (value = 2)  4     5 (value = 2)

if __name__ == "__main__":
    print(Solution().LongestPathWithSameValue( A, E))

class Solution:
    """
    @param A: string A to be repeated
    @param B: string B
    @return: the minimum number of times A has to be repeated
    """
    def repeatedString(self, A, B):
        # write your code here
        na=len(A)
        nb=len(B)
        if na==0:
            return -1
        if nb==0:
            return 0
        
        if B in A:
            return 1
        
        i=2
        while i*na <nb*2:
               
            x=A*i
            if B in x:
                return i
            i+=1
        return -1
      
            
A = 'abcd'
B = 'cdabcdab'

if __name__ == "__main__":
    print(Solution().repeatedString( A, B))



#719. Calculate Maximum Value
class Solution:
    """
    @param str: the given string
    @return: the maximum value
    """
    def calcMaxValue(self, str):
        # write your code here
        if not str:
            return 0
        
        s=str[::-1]
        self.res=0
        
        def decomposite(s):
            if len(s)==0:
                return 0
            if s[0]=='0':
                  return decomposite(s[1:])
            elif s[0]=='1':
                return decomposite(s[1:])+1
            else:
                return max(decomposite(s[1:])+int(s[0]),decomposite(s[1:])*int(s[0]))
            
        return decomposite(s)
str='891'  
str='01231'              
if __name__ == "__main__":
    print(Solution().calcMaxValue(str))
        
#720. Rearrange a String With Integers
class Solution:
    """
    @param str: a string containing uppercase alphabets and integer digits
    @return: the alphabets in the order followed by the sum of digits
    """
    def rearrange(self, string):
        # Write your code here
        if not string:
           return ''
        temp=0
        res=[]
        for x in string:
            if x.isdigit():
                temp+=int(x)
            else:
                res.append(x)
        res.sort()
        return ''.join(res)+str(temp)
                
        
#721. Next Sparse Number            
class Solution:
    """
    @param x: a number
    @return: return the next sparse number behind x
    """
    def nextSparseNum(self, x):
        # write your code here
        
        
#'0b10100010110010001001100110011'            
#'0b10100100000000000000000000000'
        if '11' not in bin(x):
            return x
        def add11(s):
            n=len(s)
            
            for i in range(n-1):
                if s[i:i+2]=='11':
                    break
            if i==0:
                return '1'+'0'*n
            else:
                return s[:i-1]+'1'+'0'*(n-i)
        cur=bin(x)[2:]
        while '11'  in cur:
            
             cur=add11(cur)
        return int(cur,2)

x=341381939
x=6
x=4
x=38
#return 343932928
if __name__ == "__main__":
    print(Solution().nextSparseNum(x))

#723. Rotate Bits - Left
class Solution:
    """
    @param n: a number
    @param d: digit needed to be rorated
    @return: a number
    """
    def leftRotate(self, n, d):
        # write code here
        
        s=bin(n)[2:]
        m=len(s)
        if m<32:
            s='0'*(32-m)+s
        return int( s[d:]+s[:d],2)
if __name__ == "__main__":
    print(Solution().leftRotate( n, d))
        
#724. Minimum Partition
class Solution:
    """
    @param nums: the given array
    @return: the minimum difference between their sums 
    """
    def findMin(self, nums):
        # write your code here
#        n=len(nums)
#        if n==1 :
#            return nums[0]
#        if n==2:
#            return abs(nums[0]-nums[1])
#        sm=sum(nums)
#        nums.sort()   
#        self.res=float('inf')
#        def select(nums,path):
#            if abs(sm/2-path)==0:
#                self.res=0
#                return 
#            if abs(sm-2*path)<self.res:
#                self.res=abs(sm-2*path)
#            for i in range(len(nums)):
#                if path+nums[i]<=sm/2:
#                    select(nums[:i]+nums[i+1:],path+nums[i])
#        select(nums,0)
#        return self.res
        n=len(nums)
        sm=sum(nums)
        if n==1 :
           return nums[0]
        if n==2:
            return abs(nums[0]-nums[1])
        sm=sum(nums)
        dp=[0 for _ in range(sm//2+1)]
        dp[0]=True
        for num in nums:
            for j in range(sm//2,num-1,-1):
                dp[j]=dp[j] or dp[j-num]
        for i in range(sm//2,-1,-1):
            if dp[i]:
                return abs(2*i-sm)
nums=[911,72,268,540,441,328,822,618,132,553,673,189,280,365,157,769,467]
nums = [1, 6, 11,5]
if __name__ == "__main__":
    print(Solution().findMin(nums))
                        
#725. Boolean Parenthesization        
class Solution:
    """
    @param symb: the array of symbols
    @param oper: the array of operators
    @return: the number of ways
    """
    def countParenth(self, symb, oper):
        # write your code here
#        slist=[] # 有重复计算
#        oper=list(oper)
#        for s in symb:
#            if s=='F':
#                slist.append(0)
#            else:
#                slist.append(1)
#        self.res=0
#        def cal(slist,olist):
#            if len(olist)==1:
#                print(slist,olist)
#                if olist[0]=='^':
#                   ans= slist[0] ^ slist[1]
#                elif olist[0]=='&':
#                   ans= slist[0] & slist[1]
#                else:
#                   ans= slist[0] | slist[1]
#                if ans==1:
#                    self.res+=1
#                return 
#            else:
#                for i in range(len(olist)):
#                    print(slist,olist,i)
#                    if olist[i]=='^':
#                       temp=slist[i] ^ slist[i+1]
#                    elif olist[i]=='&':
#                       temp=slist[i] & slist[i+1]
#                       #print(temp,slist[i],slist[i+1])
#                    else:
#                       temp=slist[i] | slist[i+1]
#                    cal(slist[:i]+[temp]+slist[i+2:] ,olist[:i]+olist[i+1:])
#        cal(slist,oper)
#        return self.res
        n=len(symb)
        T=[[0 for _ in range(n)] for _ in range(n)]
        F=[[0 for _ in range(n)] for _ in range(n)]
        
        
        
        for i in range(n):
            if symb[i]=='T':
                T[i][i]=1
            else:
                F[i][i]=1
        
        for j in range(n):
            for i in range(j-1,-1,-1):
                T[i][j]=0
                F[i][j]=0
                for k in range(i,j):
                    if oper[k]=='&':
                         T[i][j]+=T[i][k]*T[k+1][j]
                         F[i][j]+=(T[i][k]+F[i][k])*(T[k+1][j]+F[k+1][j])-T[i][k]*T[k+1][j]
                    elif oper[k]=='|':
                         T[i][j]+=(T[i][k]+F[i][k])*(T[k+1][j]+F[k+1][j])-F[i][k]*F[k+1][j]
                         F[i][j]+=F[i][k]*F[k+1][j]
                    else:
                         T[i][j]+=T[i][k]*F[k+1][j]+F[i][k]*T[k+1][j]
                         F[i][j]+=F[i][k]*F[k+1][j]+T[i][k]*T[k+1][j]
        #print(T)
        return T[0][n-1]
symb = ['T', 'F', 'T']
oper = ['^', '&']
symb = ['T', 'F', 'F']
oper = ['^', '|']
symb ="TFFF"
oper ="^|&"        
if __name__ == "__main__":
    print(Solution().countParenth(symb, oper))
                                
#726. Check Full Binary Tree
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the given tree
    @return: Whether it is a full tree
    """
    def isFullTree(self, root):
        # write your code here
        from collections import deque
        
        if not root:
            return True
        
        dq=deque([root])
        
        while dq:
              node=dq.popleft()
              if (not node.left  and not node.right):
                  continue
              if (not node.left  and node.right)  or (not node.right  and node.left):
                 return False
              dq.append(node.left)
              dq.append(node.right)
              
        return True
        
#727. Chinese Remainder Theorem            
class Solution:
    """
    @param num: the given array
    @param rem: another given array
    @return: The minimum positive number of conditions to meet the conditions
    """
    def remainderTheorem(self, num, rem):
        # write your code here
#https://zh.wikipedia.org/wiki/模反元素
#https://zh.wikipedia.org/wiki/扩展欧几里得算法
        def exd_euclid(a,b):
            if b==0:
                return (1 ,0 , a)
            
            else:
                x,y,q=exd_euclid(b,a%b)#q = gcd(a, b) = gcd(b, a%b)
                x,y=y,x-a//b*y
                return x,y,q
        from functools import reduce 
        modulus=reduce(lambda a,b: a*b,num)
        
        mutiplier=[]
        for mi in num:
            Mi=modulus//mi
            inverse,y,gcd=exd_euclid(Mi,mi)
            mutiplier.append(inverse*Mi%modulus)
        res=0
        for multi, ai in zip(mutiplier,rem):
            res+=multi*ai
        return res%modulus
            
           
                
                
num=[3,4,5]
rem=[2,3,1]    
num=[41,5,23,29,3,2,31,11,7]
rem=[16,4,14,11,2,1,8,1,2]        
if __name__ == "__main__":
    print(Solution().remainderTheorem( num, rem))
        
            
#728. Three Distinct Factors
class Solution:
    """
    @param n: the given number
    @return:  return true if it has exactly three distinct factors, otherwise false
    """
    def isThreeDisctFactors(self, n):
        # write your code here
        
        if n==1 or n==2 :
            return False
        lo=3
        hi=n
        lo=2
        def isPrime(n):
            import math
            if n in (2,3,5,7):
                return True
           
            if n in (4,6,8,9):
                return False
            for i in range(2,  int(math.ceil(n**0.5))+1):
                if n%i==0:
                    return False
            return True
        while lo < hi:
            mid=(lo+hi)//2
            #print(lo,hi,mid)
            if mid*mid==n:
                return isPrime(mid)
            elif mid*mid >n:
                hi=mid-1
            else:
                lo=mid+1
            #print(lo,hi,mid)
        
        
            
        return lo*lo==n and isPrime(lo)
            
            
n=755246849567941769
n=9   
n=550220950190521  
n=10000       
if __name__ == "__main__":
    print(Solution().isThreeDisctFactors( n))            
        
        
#729. Last Digit By Factorial Divide        
class Solution:
    """
    @param A: the given number
    @param B: another number
    @return: the last digit of B! / A! 
    """
    def computeLastDigit(self, A, B):
        # write your code here
        
        if A==B:
            return 1
        if B-A>=5:
            return 0
        res=1
        while B>A:
            res*=B%10
            B=B-1
        return res%10
            
#730. Sum of All Subsets        
class Solution:
    """
    @param n: the given number
    @return: Sum of elements in subsets
    """
    def subSum(self, n):
        # write your code here
        
        return sum(range(1,n+1))**2(n-1)

        
        

#734. Number of Subsequences of Form a^i b^j c^k        
class Solution:
    """
    @param source: the input string
    @return: the number of subsequences 
    """
    def countSubsequences(self, source):
        # write your code here
#他博客里讲的可能比较简略点但大概意思传达到了这里写一下我自己的理解。首先要明确一个思路就是说
#比如当以字符‘c’结尾时有多少种情况是取决于这个c之前的以c结尾的组合数和以b结尾的组合数，当以b
#结尾时有多少种情况取决于这个b之前的以b结尾的组合数和以a结尾的组合数，当以a结尾时有多少种情况则
#仅仅取决于这个a之前的以a结尾的组合数，注意这里说以a结尾或者以b结尾的意思并不是真正的结尾，因为
#结尾一定要以c，这里结尾的意思就是判断到这一位的时候。 
#然后我们设以a结尾的组合数为Sa，以b结尾的组合数为Sb，以c结尾的组合数为Sc，那么这里就有三个层层
#递进的关系了： 
#1. Sa => Sa 
#2. Sa , Sb => Sb 
#3. Sb, Sc => Sc 
#我们最后要得到的答案其实就是Sc。 
#然后接下仔细分析下这三个关系式的具体实现： 
#1. Sa => Sa 
#假设一个a前面的以a结尾的组合数为x，那么又来了个a，这个x会变成多少？答案是2x+1，这个2x+1可以
#其实可以看成 x+x+1，第一份x就是原来的x，即无视掉这个新来的a，第二份x就是对原来的x个组合后面都
#加上这个新来的a，最后那个1就是忽略掉前面的a，只用新来的这个a，即字符串’a’。 当然这个2x+1
#也是可以严格计算出来的，假设新的a之前出现过n个字符a，那么x应该等于 2^n-1，那算是这个
#新来的a之后就出现了n+1个a，那组合数应该就等于 2^(n+1)-1。即x = 2^n-1，设ax+b = 2^(n+1)-1,
#就可以计算出a=2，b=1。 
#2. Sa, Sb => Sb 
#假设这时候来了个新的b，这个b前面以a结尾的组合数为Sa，以b结尾的组合数为Sb，那如何去更新Sb的值？
#很显然 Sb = Sa + 2 * Sb，我们可以再把这个分开来看成 Sb = Sa + Sb + Sb，第一份Sa就是在所有
#以a结尾的组合后面跟上这个新来的b，第二份Sb可以看成在 所有以b结尾的组合后面跟上这个b，第三份
#Sb可以看成直接取所有以b结尾的组合，把新的b不要了。 
#3. Sb, Sc => Sc 
#这个道理跟2一样，同理可得Sc = Sb + 2 * Sc        
        n=len(source)
        a=0
        b=0
        c=0
        
        
        for i in range(n):
            if source[i]=='a':
                a=a+a+1
            elif source[i]=='b':
                b=a+b+b
            else:
                c=b+c+c
        return c
                
#735. Replace With Greatest From Right
class Solution:
    """
    @param nums: An array of integers.
    @return: nothing
    """
    def arrayReplaceWithGreatestFromRight(self, nums):
        # Write your code here.



        n=len(nums)
        maxdict={}
        if n==1:
            return [-1]
        curmax=nums[-1]
        nums[-1]=-1
        for i in range(n-2,-1,-1):
            temp=nums[i]
            nums[i]=curmax
            if curmax<temp:
                curmax=temp
        print(nums)
            
nums=[16, 17, 4, 3, 5, 2]                
#[17, 5, 5, 5, 2, -1]        
if __name__ == "__main__":
    print(Solution().arrayReplaceWithGreatestFromRight( nums))            
                
        
#737. Find Elements in Matrix
class Solution:
    """
    @param Matrix: the input
    @return: the element which appears every row
    """
    def FindElements(self, Matrix):
        # write your code here
        one=set(Matrix[0])
        for row in Matrix[1:]:
            rowset=set(row)
            one=rowset & one
            if len(one)==1:
                return list(one)[0]
        
Matrix= [
  [2,5,3],
  [3,2,1],
  [1,3,5]
]   
Matrix=[[1],[1],[1]]    
if __name__ == "__main__":
    print(Solution().FindElements(Matrix))         
        
        
        
#738. Count Different Palindromic Subsequences        
class Solution:
    """
    @param str: a string S
    @return: the number of different non-empty palindromic subsequences in S
    """
    def countPalindSubseq(self, str):
        # write your code here
#https://leetcode.com/problems/count-different-palindromic-subsequences/discuss/109507/Java-96ms-DP-Solution-with-Detailed-Explanation
        n=len(str)
        dp=[[0 for _ in range(n)] for _ in range(n)]
        
        for k in range(n):
            dp[k][k]=1
        
        for L in range(1,n):
            for i in range(0,n-L):
                j=i+L
                if str[i]!=str[j]:
                    dp[i][j]=dp[i+1][j]+dp[i][j-1]-dp[i+1][j-1]
                else:
                    low=i+1
                    high=j-1
                    
                    while low<=high  and str[low]!=str[i]:
                        low+=1
                    while low<=high  and str[high]!=str[i]:
                        high-=1
                    
                    if low>high:
                        dp[i][j]=dp[i+1][j-1]*2+2
                    elif low==high:
                        dp[i][j]=dp[i+1][j-1]*2+1
                    else:
                        dp[i][j]=dp[i+1][j-1]*2-dp[low+1][high-1]
        return dp[0][n-1]% (10**9 + 7)
    
str='bccb'
str='abcdabcdabcdabcdabcdabcdabcdabcddcbadcbadcbadcbadcbadcbadcbadcba'
str="aacaa"
if __name__ == "__main__":
    print(Solution().countPalindSubseq(str))                           
                        
#739. 24 Game
class Solution:
    """
    @param nums: 4 cards
    @return: whether they could get the value of 24
    """
    def compute24(self, nums):
        # write your code here
        from fractions import Fraction
        
        def compute(array ):
            n=len(array)
            #print(array)
            if n==1:
                if array[0]==24:
                    return True
                return False
               
           
            for i in range(n-1):
                temp1=array[i]+array[i+1]
                temp2=array[i]-array[i+1]
                temp3=array[i]*array[i+1]
                if array[i+1]:
                   temp4=Fraction( array[i], array[i+1]) 
                   if compute(array[:i] +[temp4]+array[i+2:]):
                      return True
                   
                if compute(array[:i] +[temp1]+array[i+2:]):
                    return True
                if compute(array[:i] +[temp2]+array[i+2:]):
                    return True
                if compute(array[:i] +[temp3]+array[i+2:]):
                    return True
            return False
        nums.sort()
        def permute(nums,path,res):
            if not nums:
                res.append(path[:]) 
            
            for i in range(len(nums)):
                if i>0 and nums[i]==nums[i-1]:
                    continue
                permute(nums[:i]+nums[i+1:],path+[nums[i]],res)
        res=[]
        permute(nums,[],res)
        return any([compute(nums )  for nums in res])
nums = [4, 1, 8, 7]
nums = [8, 7,4,1] # return true // 8 * （7 - 4） * 1 = 24
nums = [1, 1, 1, 2]# return false
nums = [3, 3, 8, 8]# return true // 8 / ( 3 - 8 / 3) = 24 
nums = [8, 3, 8, 3]               
if __name__ == "__main__":
    print(Solution().compute24( nums))     
                            
#740. Coin Change 2
class Solution:
    """
    @param amount: a total amount of money amount
    @param coins: the denomination of each coin
    @return: the number of combinations that make up the amount
    """
    def change(self, amount, coins):
        # write your code here
        
        dp=[0 for _ in range(amount+1)]
        dp[0]=1
        for x in coins:
            for i in range(1,amount+1):
                if i>=x:
                    print(i,x)
                    dp[i]+=dp[i-x]
        #print(dp)
        return dp[-1]
        
                
        
amount = 8
coins = [2, 3, 8]# return 3
#8 = 8
#8 = 3 + 3 + 2
#8 = 2 + 2 + 2 + 2        
if __name__ == "__main__":
    print(Solution().change( amount, coins))        
        
        
#742. Self Dividing Numbers
class Solution:
    """
    @param lower: Integer : lower bound
    @param upper: Integer : upper bound
    @return: a list of every possible Digit Divide Numbers
    """
    def digitDivideNums(self, lower, upper):
        # write your code here 
        def isDivid(num):
            cur=num
            while cur:
                temp=cur%10
                if temp==0:
                    return False
                if num%temp!=0:
                    return False
                cur=cur//10
            return True
        res=[]
        for x in range(lower,upper+1):
            if isDivid(x):
                res.append(x)
        return res
                
#743. Monotone Increasing Digits        
class Solution:
    """
    @param num: a non-negative integer N
    @return: the largest number that is less than or equal to N with monotone increasing digits.
    """
    def monotoneDigits(self, num):
        # write your code here
#找到第一个不递增或相等的数，把后面每一位变为0, 然后整体减去1        
        cur=num
        num2=num
        num=[]
        while cur:
            temp=cur%10
            num.append(temp)
            cur=cur//10
        
        num.reverse()
        
        smaller=False
        for i,x in enumerate(num):
            if i>0 and x<num[i-1]:
                smaller=True
                break
        if not smaller:
            return num2
        i=i-1
        res=0
        #print(i)
        while i >0 and num[i]==num[i-1]:
            i-=1
        while i+1<len(num):
            num[i+1]=0
            i+=1
        print(num)
        for j in range(len(num)):
            res=res*10+num[j]
        return res-1  
            
            
                
                
                
num=12234
num=10000
num=111111110
num=1234543
if __name__ == "__main__":
    print(Solution().monotoneDigits( num))     
           
          
#744. Sum of first K even-length Palindrome numbers
class Solution:
    """
    @param k: Write your code here
    @return: the sum of first k even-length palindrome numbers
    """
    def sumKEven(self, k):
#找规律，回文前半部分是顺序递增的数字，需要K个，那么就是1..K，那么就构造11..KK,按序镜像。 
        
        setK=set()
        res=0
        for i in  range(1,k+1):
            half_palindrome=str(i)
            palindrome=half_palindrome+half_palindrome[::-1]
            res+=int(palindrome)
            
        
        return res
            
#745. Palindromic Ranges
class Solution:
    """
    @param L: A positive integer
    @param R: A positive integer
    @return:  the number of interesting subranges of [L,R]
    """
    def PalindromicRanges(self, L, R):
        # test
        dp=[0 for _ in range(R-L+1)]
        def isPalindrome( num):
       
        
            def check(s):
              if not s:
                return True
              n=len(s)
              if n==1:
                return True
              if s[0]!=s[-1]:
                return False
              else:
                return check(s[1:-1])
            return check(str(num))
        for i in range(L,R+1):
            if isPalindrome(i):
                dp[i-L]=1
            if i>L:
                dp[i-L]+=dp[i-L-1]
       
        dp=[0]+dp
        print(dp)
        res=0
        for i in range(R-L+1):
            for j in range(i+1,R-L+2):
                if (dp[j]-dp[i])%2==0:
                    res+=1
        return res
                
L = 1
R = 2
L = 1
R = 7 
L = 87
R = 88               
if __name__ == "__main__":
    print(Solution().PalindromicRanges(L, R))     
                           
       
#749. John's backyard garden            
class Solution:
    """
    @param x: the wall's height
    @return: YES or NO
    """
    def isBuild(self, x):
        # write you code here
        
        dp=[False for _ in range(x+1)]
        
        dp[0]=True
        for brick in (3,7):
            for i in range(1,x+1):
                if i>=3:
                    dp[i]=dp[i] or dp[i-3]
                if i>=7:
                    dp[i]=dp[i] or dp[i-7]
        return 'YES' if dp[-1] else 'NO'
x=10
x=5
x=13
if __name__ == "__main__":
    print(Solution().isBuild( x))             
        
        
#750. Portal
class Solution:
    """
    @param Maze: 
    @return: nothing
    """
    def Portal(self, Maze):
        # 
    
        from collections import deque
        m=len(Maze)
        n=len(Maze[0])
        
        for a in range(m):
            for b in range(n):
                if Maze[a][b]=='S':
                    break
            if Maze[a][b]=='S':
                    break
                
        dq=deque([(a,b,0)])
        visited=set(( a,b))
        while dq:
           i,j,step= dq.popleft()
           
           if Maze[i][j]=='E':
               return step
           
           for x, y in  (( i+1,j),( i,j+1),( i-1,j),( i,j-1)):
               if x >=0 and y >=0 and x<m and y<n and Maze[x][y]!='#' and (x,y) not in visited:
                 dq.append((x,y,step+1))
                 visited.add((x,y))
        return -1
           
        
        
Maze=[
['S','E','*'],
['*','*','*'],
['*','*','*']
]        
#return 1        
        
Maze=[
['S','#','#'],
['#','*','#'],
['#','*','*'],
['#','*','E']
] 
# return -1       
if __name__ == "__main__":
    print(Solution().Portal( Maze))        
        
        
#751. John's business
class SegmentTree(object):
    def __init__(self,start,end,minimum=0):
        self.start=start
        self.end=end
        self.min=minimum
        self.left,self.right=None,None
    
    @classmethod
    def build(cls,start,end,a):
        if start >end:
            return None
        elif start==end:
            return SegmentTree(start,end,a[start])
        else:
            root=SegmentTree(start,end,a[start])
            mid=(start+end)//2
            root.left=cls.build(start,mid,a)
            root.right=cls.build(mid+1,end,a)
            root.min=min(root.left.min,root.right.min)
            return root
        
    @classmethod    
    def query(self,root,start,end):
        if  root.end<start or root.start>end :
            return float('inf')
        elif root.end <= end    and root.start >=start:
            return root.min
        else:
            return min(self.query(root.left,start,end), self.query(root.right,start,end))
        
        
        
    
class Solution:
    """
    @param A: The prices [i]
    @param k: 
    @return: The ans array
    """
    def business(self, A, k):
        # 
        
#        n=len(A)
#        import heapq 
#        hp=      A[:k]   
#        heapq.heapify(hp)
#        res=[]
#        for i,x in enumerate(A):
#            
#            if i<=k:
#                if i+k <n:
#                   heapq.heappush(hp,A[i+k])
#                res.append(x-hp[0])
#            else:
#                hp.remove(A[i-k-1])
#                if i+k <n:
#                   heapq.heappush(hp,A[i+k])
#                #print(x,hp[0])
#                res.append(x-hp[0])
#        return res
#https://github.com/cherryljr/LintCode/blob/master/John's%20business.java
#https://lintcode.com/problem/interval-minimum-number/description 
        
        root=SegmentTree.build(0,len(A)-1,A)
        res=[]
        for i in range(len(A)):
            left=max(0,i-k)
            right=min(len(A)-1,i+k)
            res.append(A[i]-SegmentTree.query(root,left,right))
        return res
        
        
A = [1, 3, 2, 1, 5]
k = 2 # return [0, 2, 1, 0, 4]  
A =[1, 1, 1, 1, 1] 
k =1 
if __name__ == "__main__":
    print(Solution().business( A, k))        
                        
#752. Rogue Knight Sven
class Solution:
    """
    @param n: the max identifier of planet.
    @param m: gold coins that Sven has.
    @param limit: the max difference.
    @param cost: the number of gold coins that reaching the planet j through the portal costs.
    @return: return the number of ways he can reach the planet n through the portal.
    """
    def getNumberOfWays(self, n, m, limit, cost):
        # 
#        self.res=0
#        def search(j,mm,limit,cost):
#            
#            if j==n:
#                self.res+=1
#                return 
#            
#            for i in range(1,limit+1):
#                if j+i <=n and cost[j+i]<=mm:
#                    #print(j+i,mm-cost[j+i],limit,cost)
#                    search(j+i,mm-cost[j+i],limit,cost)
#        search(0, m, limit, cost)
#        return self.res
#F[i][j] defines number of ways when reaching i planet by using j coins ( j <= m)
#F[i][j] = sum(F[i-limit..i-1][j-cost[i]])       
#        dp=[[0 for _ in range(m+1)] for _ in range(n+1)]
#        for j in range(m+1):
#            dp[0][j]=1 #从 dp[n][m] 出发 ，，到 dp[0] 与下面的方法相反
#
#        for i in range(1,n+1):
#                for j in range(m+1):
#                    start=max(i-limit,0)
#                    for k in range(start,i):
#                       if j>=cost[i]:
#                          dp[i][j]+=dp[k][j-cost[i]]
#        print(dp)
#        return dp[n][m]
        dp=[[0 for _ in range(m+1)] for _ in range(n+1)]
        dp[0][m]=1#从 [0][m] 出发 ，，到 dp[n]与上面的方法相反
        for i in range(1,n+1):
            for j in range(m+1):
                start=max(i-limit,0)
                for k in range(start,i):
                    if j+cost[i]<=m:
                        dp[i][j]+=dp[k][j+cost[i]]
        return sum(dp[n])
        
            
                
   
            
n = 1
m = 1
limit = 1
cost = [0, 1]#return 1.


n = 1
m = 1
limit = 1
cost = [0, 2]#return 0.


n = 2
m = 3
limit = 2
cost = [0, 1, 1]#return 2.


n = 2
m = 3
limit = 2
cost = [0, 3, 1]#return 1.

n =37
m =73
limit =20
cost =[0,1,3,1,1,3,2,0,3,2,0,3,1,0,3,3,3,3,0,1,3,1,2,1,0,0,2,0,3,2,1,3,2,2,3,2,0,3] 


n =7
m =7
limit =5
cost =[0,1,3,0,0,3,2,2]

      
if __name__ == "__main__":
    print(Solution().getNumberOfWays( n, m, limit, cost))            
        
#761. Smallest Subset
class Solution:
    """
    @param arr:  an array of non-negative integers
    @return: minimum number of elements
    """
    def minElements(self, arr):
        # write your code here
        n=len(arr)
        
        arr.sort(reverse=True)
        
        N=sum(arr)
        
        res=0
        sm=0
        for a in arr:
            sm+=a
            res+=1
            if sm> N-sm:
                return res
arr=[3,1,7,1] 
arr = [2, 1, 2]       
        
if __name__ == "__main__":
    print(Solution().minElements(arr))         
        
        
#762. Longest Common Subsequence II        
class Solution:
    """
    @param P: an integer array P
    @param Q: an integer array Q
    @param k: the number of allowed changes
    @return: return LCS with at most k changes allowed.
    """
    def longestCommonSubsequenceTwo(self, P, Q, k):
        # Write your code here
        m=len(P)
        n=len(Q)
        
        
        dp=[[[0 for _ in range(k+1)] for _ in range(n+1)] for _ in range(m+1)]
        
        # initialize g=0 
        for i in range(m+1):
            for j in range(n+1):
                if i==0 or j==0:
                    dp[i][j][0]=0
                elif P[i-1]==Q[j-1]:
                    dp[i][j][0]= dp[i-1][j-1][0]+1
                else:
                    dp[i][j][0]= max(    dp[i-1][j][0] , dp[i][j-1][0] )
        
        for i in range(m+1):
            for j in range(n+1):
                for g in range(1,k+1):
                    if i==0 or j==0:
                       dp[i][j][g]=0 
                    
                    elif P[i-1]!=Q[j-1]:
                        dp[i][j][g]= max(    dp[i-1][j][g] , dp[i][j-1][g],  dp[i-1][j-1][g-1]+1)
                    else:
                        dp[i][j][g]= max(    dp[i-1][j][g] , dp[i][j-1][g],  dp[i-1][j-1][g]+1)
                        
        return dp[m][n][k]
P = [8 ,3]
Q = [1, 3]
k = 1

P = [1, 2, 3, 4, 5]
Q = [5, 3, 1, 4, 2]
k = 1      
if __name__ == "__main__":
    print(Solution().longestCommonSubsequenceTwo( P, Q, k))         
                
                    
#763. Hex Conversion                    
class Solution:
    """
    @param n: a decimal number
    @param k: a Integer represent base-k
    @return: a base-k number
    """
    def hexConversion(self, n, k):
        # write your code here
        res=''
        if n==0:
            return '0'
        while n>0:
            t=n%k
            if t<=9:
                c=str(t)
            else:
               c= chr(ord('A')+(t-10))
            res=c+res
            n=n//k
        return res
                
#764. Calculate Circumference And Are        
class Solution:
    """
    @param r: a Integer represent radius
    @return: the circle's circumference nums[0] and area nums[1]
    """
    def calculate(self, r):
        # write your code here
        pi=3.14
        
        nums=[]
        nums.append( round(  r*2*pi,2))
        nums.append( round(  r*r*pi,2))
        return nums
       
       
       
#765. Valid Triangle       
class Solution:
    """
    @param a: a integer represent the length of one edge
    @param b: a integer represent the length of one edge
    @param c: a integer represent the length of one edge
    @return: whether three edges can form a triangle
    """
    def isValidTriangle(self, a, b, c):
        # write your code here
        tri=[a,b,c]
        tri.sort()
        
        if tri[0]+tri[1]>tri[2]:
            return True
        else:
            return False
        

#766. Leap Year       
class Solution:
    """
    @param n: a number represent year
    @return: whether year n is a leap year.
    """
    def isLeapYear(self, n):
        # write your code here
#The year can be evenly divided by 4;
#If the year can be evenly divided by 100, it is NOT a leap year, unless;
#The year is also evenly divisible by 400. Then it is a leap year.
        if n%4==0:
            if n%100==0:
                if n%400==0:
                    return True
                else:
                    return False
            return True
        return False
        
#767. Reverse Array        
class Solution:
    """
    @param nums: a integer array
    @return: nothing
    """
    def reverseArray(self, nums):
        # write your code here
        n=len(nums)
        if n==0 or n==1:
            return nums
        
        for i in range(n//2):
            nums[i],nums[n-i-1]=nums[n-i-1],nums[i]
        print(nums)
nums=[1,2,3]
if __name__ == "__main__":
    print(Solution().reverseArray( nums)) 


#768. Yang Hui Triangle
class Solution:
    """
    @param n: a Integer
    @return: the first n-line Yang Hui's triangle
    """
    def calcYangHuisTriangle(self, n):
        # write your code here
        if n==0:
            return []
        if n==1:
            return [[1]]
        if n==2:
            return [[1],[1,1]]
        res=[[1],[1,1]]
        
        for i in range(2,n):
            temp=[1]
            for j in range(1,i):
                temp.append(res[i-1][j-1]+res[i-1][j])
            temp.append(1)
            res.append(temp[:])
        return res
#[
# [1]
# [1,1]
# [1,2,1]
# [1,3,3,1]
#]
n=6
if __name__ == "__main__":
    print(Solution().calcYangHuisTriangle( n)) 



#769. Spiral Array
class Solution:
    """
    @param n: a Integer
    @return: a spiral array
    """
    def spiralArray(self, n):
        # write your code here
        if n==0:
            return []
        if n==1:
            return [[1]]
        
        direction=[(0,1),(1,0),(0,-1),(-1,0)]
        d=0
        
        grid=[[0 for _ in range(n)] for _ in range(n)]
        
        i=1
        r=0
        c=0
        while i<=n*n:
            grid[r][c]=i
            i+=1
            nr=r+direction[d][0]
            nc=c+direction[d][1]
            if nr>=n or nr<0 or nc>=n or nc<0 or grid[nr][nc]>0:
              d=(d+1)%4
              nr=r+direction[d][0]
              nc=c+direction[d][1]
            r=nr
            c=nc
              
            #print(grid,r,c)
        return grid
n=3
if __name__ == "__main__":
    print(Solution().spiralArray( n)) 
            
            
#770. Maximum and Minimum
class Solution:
    """
    @param matrix: an input matrix 
    @return: nums[0]: the maximum,nums[1]: the minimum
    """
    def maxAndMin(self, matrix):
        # write your code here
        m=len(matrix)
        
        if m==0:
            return []
        imax=float('-inf')
        imin=float('inf')
        for row in matrix:
            if min(row) < imin:
                imin=min(row)
            if max(row) > imax:
                imax=max(row)
        return [imax,imin]
            


#772. Group Anagrams        
class Solution:
    """
    @param strs: the given array of strings
    @return: The anagrams which have been divided into groups
    """
    def groupAnagrams(self, strs):
        # write your code here
        from collections import Counter,defaultdict
        import string 
        
        table=defaultdict(list)
        for s in strs:
            c=Counter(s)
            key=''
            for char in string.ascii_lowercase:
                if char in c:
                    key+=char+str(c[char])
            table[key].append(s)
        res=[]
        return list(table.values())
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]       
#[
#    ["ate", "eat","tea"],
#    ["nat","tan"],
#    ["bat"]
#]       
#       
if __name__ == "__main__":
    print(Solution().groupAnagrams( strs)) 
                   
#774. Repeated DNA
class Solution:
    """
    @param s: a string represent DNA sequences
    @return: all the 10-letter-long sequences 
    """
    def findRepeatedDna(self, s):
        # write your code here
        n=len(s)
        
        if n<=10:
            return []
        print(n)
        sset=set()
        res=[]
        for i in range(n-9):
            if s[i:i+10]  not in sset:
                sset.add(s[i:i+10])
            elif s[i:i+10] not in res:
                res.append(s[i:i+10])
            print(i,sset)
        return res
                
s="AAAAAAAAAAA"
s="AAAAAAAAAAAA"   
s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
#["AAAAACCCCC", "CCCCCAAAAA"]                   
if __name__ == "__main__":
    print(Solution().findRepeatedDna( s))         
        
        

#775. Palindrome Pairs
class Solution:
    """
    @param words: a list of unique words
    @return: all pairs of distinct indices
    """
    def palindromePairs(self, words):
        # Write your code here
        def isPalindrome(s):
            n=len(s)
            if n==0 or n==1:
                return True
            if n==2:
                return s[0]==s[1]
            return s[0]==s[-1]  and isPalindrome(s[1:-1])
        
        n=len(words)
        res=[]
        words={ word : i for i,word in enumerate(words)}
        
        for word,k in words.items():
            n=len(word)
            #print(res,word,k)
            for j in range(n+1):
                pre=word[:j]
                suf=word[j:]
                
                if isPalindrome( pre):
                    front=suf[::-1]
                    if front!=word and front in words:
                        res.append( [ words[front] , k])
                if j!=n and isPalindrome( suf):
                    back=pre[::-1]
                    if back!=word and back in words:
                        res.append( [ k,words[back] ])
        return res
            
        
        
        
        
#        for i in range(n-1):
#            for j in range(i+1,n):
#                if isPalindrome(words[i] + words[j]):
#                    res.append([i,j])
#                if isPalindrome(words[j]+ words[i]):
#                    res.append([j,i])
        return res
words = ["bat", "tab", "cat"] 
words = ["abcd", "dcba", "lls", "s", "sssll"]       
if __name__ == "__main__":
    print(Solution().palindromePairs(words))  

#776. Strobogrammatic Number II
class Solution:
    """
    @param n: the length of strobogrammatic number
    @return: All strobogrammatic numbers
    """
    def findStrobogrammatic(self, n):
        # write your code here
        if n==0:
           return ['']
        if n==1:
            return ['0','1','8']
        if n==2:
            return ["11","69","88","96"]
        res=[]
       
        pair={'1':'1','6':'9','0':'0','8':'8','9':'6'}
        
        def append(left,right,pair,res,n):
            if (len(left)+len(right))==n:
                if left[0]!='0':
                   res.append(left+right)
                return 
            if (len(left)+len(right)) +1 ==n:
                if left[0]!='0':
                   res.append(left+'0'+right)
                   res.append(left+'1'+right)
                   res.append(left+'8'+right)
                return 
            #print(left,right)
            for l,r in pair.items():
                
                append(l+left,right+r,pair,res,n)
                
        append('','',pair,res,n)
        return res
                
n = 2# return ["11","69","88","96"]
n=3 
if __name__ == "__main__":
    print(Solution().findStrobogrammatic( n))  
 
 
#777. Valid Perfect Square
class Solution:
    """
    @param num: a positive integer
    @return: if num is a perfect square else False
    """
    def isPerfectSquare(self, num):
        # write your code here
        
        if num==1 or num==4 or num==9 or num==16:
            return True
        if num<16 :
            return False
        
        left=0
        right=num
        
        while left<=right:
            #print(left,right)
            mid=(left+right)//2
            
            if mid*mid==num:
                return True
            elif mid*mid>num:
                right=mid-1
            else:
                left=mid+1
        if left*left==num:
            return True
        return False
num=25       
if __name__ == "__main__":
    print(Solution().isPerfectSquare( num))  
  
 
#778. Pacific Atlantic Water Flow
class Solution:
    """
    @param matrix: the given matrix
    @return: The list of grid coordinates
    """
    def pacificAtlantic(self, matrix):
        # write your code here
        
        from collections import deque
        m=len(matrix)
        if m==0:
            return []
        n=len(matrix[0])  
        
        pacific=[[False for _ in range(n)] for _ in range(m)]
        atlantic=[[False for _ in range(n)] for _ in range(m)]
        
        
        def bfs(grid,i,j,ocean):
            visited=set()
            visited.add((i,j))
            dq=deque([(i,j)])
            while dq:
                a,b=dq.popleft()
                ocean[a][b]=True
                for x,y in (( a+1,b  ),( a-1,b  ),( a,b+1  ),( a,b -1 )):
                    if x>=0 and y>=0 and x<m and y<n and (x,y) not in visited:
                        if grid[a][b]<=grid[x][y]:
                            visited.add((x,y))
                            dq.append((x,y))
        
        for r in range(m):
            bfs(matrix,r,0,pacific) 
            bfs(matrix,r,n-1,atlantic)
              
          
               
        for c in range(n):
            bfs(matrix,0,c,pacific)
            bfs(matrix,m-1,c,atlantic)
        res=[]
        for i in range(m):
            for j in range(n):
                if pacific[i][j]  and atlantic[i][j] :
                    res.append((i,j))
        return res
Pacific ~   ~   ~   ~   ~ 
       ~  1   2   2   3  (5) *
       ~  3   2   3  (4) (4) *
       ~  2   4  (5)  3   1  *
       ~ (6) (7)  1   4   5  *
       ~ (5)  1   1   2   4  *
          *   *   *   *   * Atlantic 
 
matrix=[[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]

 
if __name__ == "__main__":
    print(Solution().pacificAtlantic(matrix))  
   
 
#779. Generalized Abbreviation 
class Solution:
    """
    @param word: the given word
    @return: the generalized abbreviations of a word
    """
    def generateAbbreviations(self, word):
        # Write your code here
        
        def generate(w,path,abbr,res):
            n=len(w)
            if n==0:
                res.append(path[:])
                return 
            generate(w[1:],path+w[:1],False,res)
            
            if not abbr:
                for i in range(1,n+1):
                    generate(w[i:],path+str(i),True,res)
        res=[]
        generate(word,'',False,res)
        return res[::-1]
        
                    
            
word = "word"
#["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]
if __name__ == "__main__":
    print(Solution().generateAbbreviations( word))  
    

#780. Remove Invalid Parentheses
class Solution:
    """
    @param s: The input string
    @return: Return all possible results
    """
    def removeInvalidParentheses(self, s):
        # Write your code here
        
        from collections import deque
        dq=deque([ s])
        visited=set()
        
        visited.add( s )
        
        def isValid( P):
#            stack=[]
#            for p in P:
#                if p=='(':
#                    stack.append(p)
#                elif p==')':
#                    if not stack:
#                        return False
#                    stack.pop()
#            return not stack
            count=0
            for p in P:
                if p=='(':
                    count+=1
                elif p==')':
                    count-=1
                if count<0:
                    return False
            return count==0
        
            
                
        found=False 
        res=[]
        while dq:
            n=len(dq)
            
            for _ in range(n):
                x=dq.popleft()
                print(x)
                if isValid( x):
                    res.append(x)
                    found=True
                if found:
                    continue
                for i in range(len(x)):
                    if (x[i] =='('  or x[i] ==')' ) and x[:i]+x[i+1:] not in visited: #不能省括号
                        dq.append( x[:i]+x[i+1:])
                        visited.add(x[:i]+x[i+1:])
            if found:
                break
        return res
s="()())()"
s= "(a)())()"
s=  ")("
s= ")((())))))()(((l(((("
if __name__ == "__main__":
    print(Solution().removeInvalidParentheses( s))  
                        
                    
                
#780. Remove Invalid Parentheses            
class Solution:
    """
    @param s: The input string
    @return: Return all possible results
    """
    def removeInvalidParentheses(self, s):
        # Write your code here
        l=0
        r=0
        for p in s:
            if p=='(':
                l+=1
            if p==')':
                if l>0:
                    l-=1
                else:
                    r+=1
            
      
        
        def isValid( P):

            count=0
            for p in P:
                if p=='(':
                    count+=1
                elif p==')':
                    count-=1
                if count<0:
                    return False
            return count==0   
        res=[]
        def dfs(ss, start,l,r,res):  
            if l==0 and r==0 and isValid( ss):
                res.append(ss)
                return 
            
            for i in range(start,len(ss)):
                if i>start  and ss[i-1]==ss[i]:
                    continue
                if ss[i]=='(' and l>0:
                    dfs(ss[:i]+ss[i+1:], i,l-1,r,res)
                if ss[i]==')' and r>0:
                    dfs(ss[:i]+ss[i+1:], i,l,r-1,res)
        
        dfs(s, 0,l,r,res) 
        if not res:
            res=['']
        return res           
s="()())()"
s= "(a)())()"
s=  ")("
s= ")((())))))()(((l(((("
if __name__ == "__main__":
    print(Solution().removeInvalidParentheses( s))                  
                
                
#782. AND and OR                
class Solution:
    """
    @param n: 
    @param nums: 
    @return: return the sum of maximum OR sum, minimum OR sum, maximum AND sum, minimum AND sum.
    """
    def getSum(self, n, nums):
        # write your code here
       
        maxand=nums[0]
        minand=nums[0]
        maxor=nums[0]
        minor=nums[0]
        
        for num in nums:
             maxor|=num
             minand &=num
             
             minor=min(minor,num)
             maxand=max(maxand,num)
        
        return maxand +minand+maxor + minor
n = 3
nums = [1, 2, 3]
n = 3
nums = [0, 0, 1]
n = 5
nums = [12313, 156, 4564, 212, 12]
n = 3
nums = [111111, 333333, 555555]
if __name__ == "__main__":
    print(Solution().getSum( n, nums))                    

#783. Minimum Risk Path
class Solution:
    """
    @param n: maximum index of position.
    @param m: the number of undirected edges.
    @param x: 
    @param y: 
    @param w: 
    @return: return the minimum risk value.
    """
    def __init__( self):
        self.v=[[] for _ in range(1010)]
        self.w=[[] for _ in range(1010)]
        self.visited=[0 for _ in range(1010)]
        self.res=float('inf' )
    
    def dfs(self,start,curvalue,end):
        if start==end:
            return curvalue
        
        if curvalue>=self.res:
            return float('inf' )
        
        tempres=float('inf' )
        
        self.visited[start]=1
        for i in range( len(self.v[start]) ):
            if  self.visited[self.v[start][i]]==1:
                continue
            tempres=min( tempres, self.dfs(self.v[start][i],max( self.w[start][i], curvalue),end))
            self.res=min(tempres,self.res)
                             
        
        
        self.visited[start]=0
        return tempres
            
        
        
    def getMinRiskValue(self, n, m, x, y, w):
        for i in range(m):
            self.v[x[i]].append(y[i])
            self.v[y[i]].append(x[i])
            self.w[x[i]].append(w[i])
            self.w[y[i]].append(w[i])
        self.dfs(0,0,n)
        
        return self.res
            
n = 2
m = 2
x = [0, 1]
y = [1, 2]
w = [1, 2]  

n = 3
m = 4
x = [0, 0, 1, 2]
y = [1, 2, 3, 3]
w = [1, 2, 3, 4] 

n = 4
m = 5
x = [0, 1, 1, 2, 3]
y = [1, 2, 3, 4, 4]
w = [3, 2, 4, 2, 1]

n = 5
m = 7
x = [0, 0, 1, 2, 3, 3, 4]
y = [1, 2, 3, 4, 4, 5, 5]
w = [2, 5, 3, 4, 3, 4, 1]
      
if __name__ == "__main__":
    print(Solution().getMinRiskValue( n, m, x, y, w))          

#784. The Longest Common Prefix II
class Solution:
    """
    @param dic: the n strings
    @param target: the target string
    @return: The ans
    """
    def the_longest_common_prefix(self, dic, target):
        # write your code here
        n=len(dic)
        if n==0:
            return 0
        res=float('-inf')
        for d in dic:
            resd=0
            for x, y in zip( d,target):
                if x==y:
                    resd+=1
            if resd>res:
                res=resd
        return res
dic=["abcba","acc","abwsf"]
target = "abse"#return 2            

#785. Maximum Weighted Sum Path
class Solution:
    """
    @param nums: 
    @return: nothing
    """
    def maxWeight(self, nums):
        # write your code here
        m=len(nums)
        if m==0:
            return 0
        n=len(nums[0])
#        self.res=float('-inf')
#        def dfs(nums,i,j,visited,path):
#            if i==m-1 and j==0:
#                if path>self.res:
#                    self.res=path
#                return 
#            
#            for x,y in ((i,j-1) ,(i+1,j)):
#                if x>=0 and y>=0 and x<m and y<n and (x,y) not in visited:
#                    dfs(nums,x,y,visited | set( [( x,y)]),path+nums[x][y])
#    
#                    
#        dfs(nums,0,n-1,set(),nums[0][n-1])     
#        return self.res
        dp=[[0 for _ in range(n)] for _ in range(m)]
        
        for i in range(m):
            for j in range(n-1,-1,-1):
                if i-1 >=0:
                
                   dp[i][j]= max( dp[i][j],dp[i-1][j]+nums[i][j])
                if j+1 < n:
                   dp[i][j]= max( dp[i][j],dp[i][j+1]+nums[i][j])
        
        return dp[m-1][0] +nums[0][n-1]
            
        


nums=[
[1,2,3,4],
[3,5,6,7],
[9,10,1,2],
[4,4,5,5]
]

nums=[
[1,2,3],
[4,5,6],
[7,9,8]
]
if __name__ == "__main__":
    print(Solution().maxWeight( nums))    
 
 
#787. The Maze
class Solution:
    """
    @param maze: the maze
    @param start: the start
    @param destination: the destination
    @return: whether the ball could stop at the destination
    """
    def hasPath(self, maze, start, destination):
        # write your code here
        import time
        m=len(maze)
        n=len(maze[0])
        def dfs(maze,cur,destination):
            if cur==destination:
                return True
            
            for x, y in ((cur[0]+1,cur[1]),(cur[0]-1,cur[1]),(cur[0],cur[1]+1),(cur[0],cur[1]-1)):
                if   x>=0 and y>=0 and x<m and y<n and maze[x][y]==0:
                    if x==cur[0]+1:
                        while   x<m and  maze[x][y]==0 :
                                x+=1
                        x=x-1
                    elif x==cur[0]-1 :
                        while  x>=0 and maze[x][y]==0:
                                x-=1
                        x=x+1
                    elif y==cur[1]+1 :
                        while  y<n and maze[x][y]==0:
                                y+=1
                        y=y-1
                    elif y==cur[1]-1 :
                        while  y>=0 and maze[x][y]==0:
                                y-=1
                        y=y+1
                    if (x,y)==cur:
                        continue
                    #print(cur,(x,y))
                    #time.sleep(5)
                    if (x,y)  not in self.visited:
                       self.visited.add((x,y))
                       if dfs(maze,(x,y),destination ):
                           return True
            return False
           
        start=tuple(start)
        destination=tuple(destination)
        self.visited=set([ start  ]) 
        return dfs(maze,start,destination )
                    
maze=[[0,0,1,0,0],
      [0,0,0,0,0],
      [0,0,0,1,0],
      [1,1,0,1,1],
      [0,0,0,0,0]]
start = (0, 4) 
destination  = (4, 4)
destination  = (4, 3)
destination  = (3,2)
if __name__ == "__main__":
    print(Solution().hasPath( maze, start, destination)) 
 
#788. The Maze II
class Solution:
    """
    @param maze: the maze
    @param start: the start
    @param destination: the destination
    @return: the shortest distance for the ball to stop at the destination
    """
    def shortestDistance(self, maze, start, destination):
        # write your code here
        import time
        m=len(maze)
        n=len(maze[0])
        
        if start==destination:
            return 0
        
        from collections import deque
        
        dp=[[float('inf') for _ in range(n)] for _ in range(m)]
        dq=deque(  [(start[0],start[1],0 ) ])
        directions=[-1,0,1,0,-1]
        while dq:
            cur_x,cur_y,cur_l=dq.popleft()
            if cur_l >= dp[cur_x][cur_y]:
                continue
            dp[cur_x][cur_y]=cur_l
            #print(x,y,dq)
            #print('#####')
            #print('#####')
            for k in range(4):
                x=cur_x
                y=cur_y
                l=cur_l
                while  x>=0 and y>=0 and x<m and y<n and maze[x][y]==0:
                    x+=directions[k]
                    y+=directions[k+1]
                    l+=1
                    #print(x,y,directions[k],directions[k+1])
                
                x-=directions[k]
                y-=directions[k+1]
                l-=1
            
                #print('****',x,y,l)
               
                dq.append((x,y,l))
                #print(dq)
        return dp[destination[0]][destination[1]] if dp[destination[0]][destination[1]] < float('inf') else -1
        
                    
maze=[[0,0,1,0,0],
      [0,0,0,0,0],
      [0,0,0,1,0],
      [1,1,0,1,1],
      [0,0,0,0,0]]
start = (0, 4) 
destination  = (4, 4)
destination  = (4, 3)
destination  = (3,2)
if __name__ == "__main__":
    print(Solution().shortestDistance( maze, start, destination)) 


#789. The Maze III
class Solution:
    """
    @param maze: the maze
    @param ball: the ball position
    @param hole: the hole position
    @return: the lexicographically smallest way
    """
    def findShortestWay(self, maze, ball, hole):
        # write your code here
        import time
        m=len(maze)
        n=len(maze[0])
        
        if ball==hole:
            return ''
        res=[]
        from collections import deque
        
        dp=[[ (float('inf'),'') for _ in range(n)] for _ in range(m)]
        dq=deque(  [(ball[0],ball[1],0,'' ) ])
        directions=[-1,0,1,0,-1]
        while dq:
            cur_x,cur_y,cur_l,cur_path=dq.popleft()
            if cur_l >= dp[cur_x][cur_y][0]:
                continue
            dp[cur_x][cur_y]=(cur_l,cur_path)
            #print(x,y,dq)
            #print('#####')
            #print('#####')
            for k in range(4):
                x=cur_x
                y=cur_y
                l=cur_l
                path=cur_path
                found=False
                while  x>=0 and y>=0 and x<m and y<n and maze[x][y]==0:
                    if x==hole[0]  and y==hole[1]:
                        found=True
                        break
                    x+=directions[k]
                    y+=directions[k+1]
                    l+=1
                    #print(x,y,directions[k],directions[k+1])
                if not found:
                    x-=directions[k]
                    y-=directions[k+1]
                    l-=1
                if x > cur_x:
                    path+='d'
                elif x<cur_x:
                    path+='u'
                elif y<cur_y:
                    path+='l'
                elif y>cur_y:
                    path+='r'  
                if found:
                    res.append( (l,path))
                #print('****',x,y,l)
               
                if not found:
                    dq.append((x,y,l,path))
                #print(dq)
        return sorted(res)[0][1]  if res else 'impossible'



maze=[[0,0,0,0,0],
      [1,1,0,0,1],
      [0,0,0,0,0],
      [0,1,0,0,1],
      [0,1,0,0,0]]
ball=[4,3]
hole=[0,1]
if __name__ == "__main__":
    print(Solution().findShortestWay( maze, ball, hole)) 

#790. Parser
class Solution:
    """
    @param generator: Generating set of rules.
    @param startSymbol: Start symbol.
    @param symbolString: Symbol string.
    @return: Return true if the symbol string can be generated, otherwise return false.
    """
    def canBeGenerated(self, generator, startSymbol, symbolString):
        # Write  your code here.
        from collections import defaultdict
        graph=defaultdict(list)
        
        for x in generator:
            before ,after=x.split( '→' )
            before=before.strip()
            after=after.strip()
            
            #print(before,after)
            graph[before].append(list(after))
        #print(graph)
        
        def dfs( string,symbolString,graph):
            #print(string)
            if ''.join(string)==symbolString:
                return True
            if len(string) >len(symbolString):
                return 
            for i in range(len(string)):
                if string[i] in graph and string[i]!=startSymbol:
                    for next_s in graph[string[i]]:
                        if dfs( string[:i]+next_s+string[i+1:],symbolString,graph):
                            return True
            return False
        
        for string in graph[startSymbol]:
            if dfs( string,symbolString,graph):
                return True
        return False
            
generator = ["S → abc", "S → aA", "A → b", "A → c"]
startSymbol = 'S'
symbolString = 'ac'

generator = ["S → abcd", "S → A", "A → abc"]
startSymbol = 'S'
symbolString = 'abc'

generator = ["S → abc", "S → aA", "A → b", "A → c"]
startSymbol = 'S'
symbolString = 'a'

generator = ["S → abcd", "S → A", "A → abc"]
startSymbol = 'S'
symbolString = 'ab'


generator =["E → TX", "X → bX", "X → c", "T → a", "T → d"]
startSymbol ='E'
symbolString ='d'
if __name__ == "__main__":
    print(Solution().canBeGenerated( generator, startSymbol, symbolString)) 



#791. Merge Number
class Solution:
    """
    @param numbers: the numbers
    @return: the minimum cost
    """
    def mergeNumber(self, numbers):
        # Write your code here
        res=0
        import heapq
        
        q=[]
        for x in numbers:
            heapq.heappush(q,x)
        
        
        while len(q)>1:
              print(q)
              a=heapq.heappop(q)
              b=heapq.heappop(q)
              res+=a+b
              heapq.heappush(q,a+b)
        return res
              
            
       
        
numbers=[1,2,3,4] #19       
numbers=[2,8,4,1]# 25        
if __name__ == "__main__":
    print(Solution().mergeNumber(numbers))
    
    
    
#792. Kth Prime Number  
import math      
class Solution:
    """
    @param n: the number
    @return: the rank of the number
    """
    def kthPrime(self, n):
        # write your code here
        dic={2:1,3:2,5:3,7:4}
        def isPrime(x):
            
            for t in range(2,int(math.ceil(x**0.5+1))):
                if x%t==0:
                    return False
            return True
        i=8
        index=4
        while i <= 100000:
            if  isPrime(i):
                index+=1
                dic[i]=index
            i+=1
        return dic[n]
        
n = 3 
n=11       
if __name__ == "__main__":
    print(Solution().kthPrime( n))
    
#793. Intersection of Arrays    
class Solution:
    """
    @param arrs: the arrays
    @return: the number of the intersection of the arrays
    """
    def intersectionOfArrays(self, arrs):
        # write your code here
        
        n=len(arrs)
        if n==0:
            return 0
        res=set(arrs[0])
        
        for i in range(1,n):
           res  &= set(arrs[i])
           
        return len(res)
            
    
arrs=[[1,2,3],[3,4,5],[3,9,10]]# return 1        
arrs=[[1,2,3,4],[1,2,5,6,7],[9,10,1,5,2,3]]# return 2        
if __name__ == "__main__":
    print(Solution().intersectionOfArrays( arrs))
            
        
#796. Open the Lock        
class Solution:
    """
    @param deadends: the list of deadends
    @param target: the value of the wheels that will unlock the lock
    @return: the minimum total number of turns 
    """
    def openLock(self, deadends, target):
        # Write your code here        
        
        from collections import deque
        deadends=set(deadends)
        if '0000' in deadends:
            return -1
        
        visited=set()
        visited.add('0000')
        
        dq=deque(['0000'])
        step=0
        while dq:
            
           
            temp=deque([])
    
            for _ in range(len(dq)):
              cur=dq.popleft()
              if cur==target:
                  return step
            
              for i in range(4):
                 n1= str((int(cur[i])+1)%10)
                 n2=str((int(cur[i])-1)%10)
                 for nx in ( cur[:i]+n1+cur[i+1:] , cur[:i]+n2+cur[i+1:]  ):
                    if nx not in deadends and nx not in visited:
                        #print(nx)
                        visited.add( nx)
                        temp.append(nx)
            step+=1
            dq=temp
            
        return -1
                        
 

deadends = ["0201","0101","0102","1212","2002"]
target = "0202"
#Return 6
deadends = ["8888"]
target = "0009"
deadends =["8887","8889","8878","8898","8788","8988","7888","9888"]
target = "8888"

deadends =["2110","2000","0000","2111","1110"]
target ="0012"


#Return 1
if __name__ == "__main__":
    print(Solution().openLock(deadends, target))


#797. Reach a Number
class Solution:
    """
    @param target: the destination
    @return: the minimum number of steps
    """
    def reachNumber(self, target):
        # Write your code here
        import math
        t=abs(target)
        n=math.floor(   (2*t)** 0.5)
        
        while True:
            diff=(1+n)*n/2-t
            if diff>=0:
                if diff%2==0:
                    return int(n)
            n+=1
           
 
target = 3
target = 2    
if __name__ == "__main__":
    print(Solution().reachNumber( target))
    
#802. Sudoku Solver
class Solution:
    """
    @param board: the sudoku puzzle
    @return: nothing
    """
    def solveSudoku(self, board):
        # write your code here   
        
        def isValid( n,i,j,board):
            #row
            if n in board[i]:
                return False
            for a in range(len(board)):
                if board[a][j]==n:
                    return False
            ii=i//3
            jj=j//3
            
            
            for b in range(3):
                for c in range(3):
                    if board[ii*3+b][jj*3+c]==n:
                        return False
            return True
                      
            
        def solve(board):
#            if sum([sum(row) for row in board])==45*9:
#                print(board)
#                return True
            for x in range(len(board)):
               for y in range(len(board[0])):
                 
                
                  if board[x][y]==0:
                    for value in range(1,10):
                        if isValid( value,x,y,board):
                            board[x][y]=value
                            if solve(board):
                                return True
                            else:
                                board[x][y]=0
                    return False # 要在这里加return ，截断不合格的。
                                
            return True
            #print(board)
            
        solve(board)
        print(board)
          
      
    
 
    
board=[[0,0,9,7,4,8,0,0,0],
 [7,0,0,0,0,0,0,0,0],
 [0,2,0,1,0,9,0,0,0],
 [0,0,7,0,0,0,2,4,0],
 [0,6,4,0,1,0,5,9,0],
 [0,9,8,0,0,0,3,0,0],
 [0,0,0,8,0,3,0,2,0],
 [0,0,0,0,0,0,0,0,6],
 [0,0,0,2,7,5,9,0,0]]    
    
if __name__ == "__main__":
    print(Solution().solveSudoku( board))    
    
#803. Shortest Distance from All Buildings    
class Solution:
    """
    @param grid: the 2D grid
    @return: the shortest distance
    """
    def shortestDistance(self, grid):
        # write your code here
        m=len(grid)
        n=len(grid[0])
        #dp=[[0  for _ in range(n)] for _ in range(m)]
        from collections import deque
        res=float('inf')
        building=0
        for i in range(m):
            for j in range(n):
                if grid[i][j]==1:
                    building+=1
                    
                
        for i in range(m):
            for j in range(n):
                if grid[i][j]==0:
                    #BFS
                    q=deque([  (i,j,0 )])
                    visited=set( (i,j ))
                    tempres=0
                    tempbuilding=building
                    while q:
                        x,y,step=q.popleft()
                        if grid[x][y]==1:
                            tempbuilding-=1
                            tempres+=step
                        elif grid[x][y]==0:
                            for a,b in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                                if a>=0 and b>=0 and a<m and b<n and (a,b) not in visited  and grid[a][b]!=2:
                                    visited.add((a,b))
                                    q.append((a,b,step+1))
                    if tempres<res and tempbuilding==0:
                        res=tempres
               
        return res
                    
    
    
    
    
grid=[[1,1,1,1,1,0],
 [0,0,0,0,0,1],
 [0,1,1,0,0,1],
 [1,0,0,1,0,1],
 [1,0,1,0,0,1],
 [1,0,0,0,0,1],
 [0,1,1,1,1,0]]    
    
if __name__ == "__main__":
    print(Solution().shortestDistance(grid))    
        
    
    
#804. Number of Distinct Islands II
class Solution:
    """
    @param grid: the 2D grid
    @return: the number of distinct islands
    """
    def numDistinctIslands2(self, grid):
        # write your code here
        directions=[(-1,0),(1,0),(0,1),(0,-1) ]
        
        def dfs(i,j,grid,island):
            if not (0<= i <len(grid)  and 0<= j <len(grid[0]) and grid[i][j] >0):
                return False
            
            grid[i][j]*=-1
            
            island.append(( i,j))
            for d in directions:
                dfs(i+d[0],j+d[1],grid,island)
            return True
        
        
        def normalize(island):
            shapes=[[]for _ in range(8)]
            
            for x,y in island:
                rotations_and_reflections= [ [-x,y] ,[x,-y] ,[-x,-y] ,[x,y] ,  [y,x], [-y,x],[y,-x],[-y,-x]]
                
                for i in range(len(rotations_and_reflections )):
                    shapes[i].append( rotations_and_reflections[i])
            for shape in shapes:
                shape.sort()
                
                origin=list(shape[0])
                
                for p in shape:
                    p[0]-=origin[0]
                    p[1]-=origin[1]
            print(shapes)
            return min(shapes)
        
        islands=set()
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                island=[]
                if dfs(i,j,grid,island):
                    islands.add(str(normalize(island))  )
        return len(islands)
grid=[[1,1,0,0,0],[1,0,0,0,0],[0,0,0,0,1],[0,0,0,1,1]]  
grid=[[1,1,1,0,0],[1,0,0,0,1],[0,1,0,0,1],[0,1,1,1,0]]                   
if __name__ == "__main__":
    print(Solution().numDistinctIslands2( grid))                
                
        
#813. Find Anagram Mappings
class Solution:
    """
    @param A: lists A
    @param B: lists B
    @return: the index mapping
    """
    def anagramMappings(self, A, B):
        # Write your code here 
        def find(arr,x):
            
            for i in range(len(arr)):
                if arr[i]==x:
                   return i
        
        res=[]
        for x in A:
            res.append(find(B,x))
        return res
        
        
        
A = [12, 28, 46, 32, 50] 
B = [50, 12, 32, 46, 28]#return [1, 4, 3, 2, 0]        
if __name__ == "__main__":
    print(Solution().anagramMappings(A, B))                
                        
#814. Shortest Path in Undirected Graph        
# Definition for a undirected graph node
# class UndirectedGraphNode:
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []
class Solution:
    """
    @param graph: a list of Undirected graph node
    @param A: nodeA
    @param B: nodeB
    @return:  the length of the shortest path
    """
    def shortestPath(self, graph, A, B):
        # Write your code here   
        if A==B:
            return 0
        
        from collections import deque
        q1=deque([A])
        q2=deque([B])
        visited=set([A,B])
        
        dis=0
        
        while q1 and q2:
            dis+=1
            len_q1=len(q1)
            for _ in range(len_q1):
                node=q1.popleft()
                
                
                for neighbor in node.neighbors:
                    if neighbor in q2:
                       return dis
                    if neighbor  not in visited:
                        visited.add(neighbor)
                        q1.append(neighbor)
            dis+=1
            len_q2=len(q2)
            for _ in range(len_q2):
                node=q2.popleft()
                
                
                for neighbor in node.neighbors:
                    if neighbor in q1:
                       return dis
                    if neighbor  not in visited:
                        visited.add(neighbor)
                        q2.append(neighbor)
        return -1
                
                
#817. Range Sum Query 2D - Mutable            
class NumMatrix(object):

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        self.matrix=matrix
        m=len(matrix)
        n=len(matrix[0])
        self.msum=[[0 for _ in range(n+1)]  for _ in range(m+1)]
        for i in range(1,m+1):
            for j in range(1,n+1):
                if j==1:
                    
                   self.msum[i][j]= matrix[i-1][j-1]
                else:
                   self.msum[i][j]=self.msum[i][j-1]+matrix[i-1][j-1]
                if i>1:
                    self.msum[i][j]= self.msum[i-1][j]+self.msum[i][j]
                    
        print(self.msum)
        
#        for i in range(2,m+1):
#            for j in range(1,n+1):
#                self.msum[i][j]= self.msum[i-1][j]+self.msum[i][j]
        print(self.msum)
                      
                      
                
        

    def update(self, row, col, val):
        """
        :type row: int
        :type col: int
        :type val: int
        :rtype: void
        """
        for i in range(row+1,len(self.msum)):
            for j in range(col+1,len(self.msum[0])):
                self.msum[i][j]+= (val-self.matrix[row][col])
        self.matrix[row][col]=val
        #print(self.msum)
        

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        #print(self.msum[row2+1][col2+1],self.msum[row1][col1],self.msum[row1][col2+1],self.msum[row2+1][col1])
        return self.msum[row2+1][col2+1]+self.msum[row1][col1]-self.msum[row1][col2+1]-self.msum[row2+1][col1]
        
        
#817. Range Sum Query 2D - Mutable         
class NumMatrix(object):

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        self.m=len(matrix)
        self.n=len(matrix[0])
        self.A=[[0 for _ in range(self.n+1)] for _ in range(self.m+1)]
        self.C=[[0 for _ in range(self.n+1)] for _ in range(self.m+1)]
        for i in range(self.m):
            for j in range(self.n):
                self.update(i,j,matrix[i][j])
        
        

    def update(self, row, col, val):
        """
        :type row: int
        :type col: int
        :type val: int
        :rtype: void
        """
        i=row+1
        dif=val-self.A[row+1][col+1]
        
        self.A[row+1][col+1]=val
        while i< self.m+1:
            j=col+1
            while j<self.n+1:
                self.C[i][j]+=dif
                j+=self.lowbit(j)
            i+=self.lowbit(i)
                
            
        
    
    def lowbit(self,x):
        return x & (-x)
    
    def prefix_sum(self, row, col):
        i=row+1
        res=0
        while i>0:
            j=col+1
            while j>0:
                res+=self.C[i][j]
                j-=self.lowbit(j)
            i-=self.lowbit(i)
        return res
                
                
        

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        
        return self.prefix_sum(row2,col2)-self.prefix_sum(row2,col1-1)-self.prefix_sum(row1-1,col2)+self.prefix_sum(row1-1,col1-1)
        
matrix = [
  [3, 0, 1, 4, 2],
  [5, 6, 3, 2, 1],
  [1, 2, 0, 1, 5],
  [4, 1, 0, 1, 7],
  [1, 0, 3, 0, 5]
]
matrix = [[1]]
# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# obj.update(row,col,val)
# param_2 = obj.sumRegion(row1,col1,row2,col2) 
obj.update(3, 2, 2)
obj.update(0, 0, -1)
obj.sumRegion(0, 0, 0, 0)
obj.sumRegion(2, 1, 4, 3)       
        
#822. Reverse Order Storage        
class Solution:
    """
    @param head: the given linked list
    @return: the array that store the values in reverse order 
    """
    def reverseStore(self, head):
        # write your code here     
        if not head :
            return []
        cur=head
        res=[]
        while cur:
            res=[cur.val]+res
            cur=cur.next
        return res
            
        
#823. Input Stream
class Solution:
    """
    @param inputA: Input stream A
    @param inputB: Input stream B
    @return: The answer
    """
    def inputStream(self, inputA, inputB):
        # Write your code here   
        
        ra=''
        rb=''
        for a in inputA:
            if a=='<':
                ra=ra[:-1]
            else:
                ra+=a
        for b in inputB:
            if b=='<':
                rb=rb[:-1]
            else:
                rb+=b
        if ra==rb:
            res='YES'
        else:
            res='NO'
        return res
 
inputA = 'abcde<<'
inputB = 'abcd<e<'        
        
inputA = 'a<<bc'
inputB = 'abc<'       
        
if __name__ == "__main__":
    print(Solution().inputStream( inputA, inputB))        
        
#824. Single Number IV        
class Solution:
    """
    @param nums: The number array
    @return: Return the single number
    """
    def getSingleNumber(self, nums):
        # Write your code here  
        
        n=len(nums)
        pre=None
        for i,v in enumerate(nums):
            if i%2==0:
                pre=v
            else:
                if pre!=v:
                    return pre
  
        
nums = [3,3,2,2,4,5,5] 
nums = [2,1,1,3,3]   
if __name__ == "__main__":
    print(Solution().getSingleNumber( nums))         
    
#825. Bus Station
class Solution:
    """
    @param N: The number of buses
    @param route: The route of buses
    @param A: Start bus station
    @param B: End bus station
    @return: Return the minimum transfer number
    """
    def getMinTransferNumber(self, N, route, A, B):
        # Write your code here
        
        from collections import defaultdict,deque
        graph=defaultdict(set)
        Ai=[]
        Bi=[]
        for i,stops in enumerate(route):
            stops=set(stops)
            for stop1 in stops:
                for stop2 in stops:
                    if stop1==A:
                        Ai.append(i)
                    if stop1==B:
                        Bi.append(i)
                    if stop1!=stop2:
                       graph[stop1].add(stop2)
        #print(graph)
                
        q=deque([A])
        visited=set([A])
        transfer=0
        while q:
            temp=deque([])
            
            for _ in range(len(q)):
                stop=q.popleft()
                if stop==B:
                    return transfer
                for nextstop in graph[stop]:
                    if nextstop not in visited:
                       visited.add(nextstop)
                       temp.append( nextstop)
            transfer+=1
            q=temp
        return -1
        
N = 2
route = [[1, 2, 3, 4], [3, 5, 6, 7]]
A = 1
B = 4   

N = 2
route = [[1, 2, 3, 4], [3, 5, 6, 7]]
A = 1
B = 7 
N=20
route =[[7075,7330],[7075,3517],[651,6438],[3517,7075],[7330,83],[10989,5061],[651,6995],[7075,9913],[10989,651],[7437,83],[11189,3897],[6995,3517],[83,5734],[3897,3517],[6995,7330],[7330,10989],[9913,651],[3517,3897],[83,7437],[3897,7075]]
A = 83
B = 6438


N=20
route =[[7075,7330],[7075,3517],[651,6438],[3517,7075],[7330,83],[10989,5061],[651,6995],[7075,9913],[10989,651],[7437,83],[11189,3897],[6995,3517],[83,5734],[3897,3517],[6995,7330],[7330,10989],[9913,651],[3517,3897],[83,7437],[3897,7075]]
A =651
B =3517



if __name__ == "__main__":
    print(Solution().getMinTransferNumber( N, route, A, B))     

#826. Computer Maintenance 
class Solution:
    """
    @param n: the rows of matrix
    @param m: the cols of matrix
    @param badcomputers: the bad computers 
    @return: The answer
    """
    def maintenance(self, n, m, badcomputers):
        # Write your code here    
#        from collections import defaultdict
#        
#        comdict=defaultdict(list)
#        pre=badcomputers[0][0]
#        for i,j in badcomputers:
#            comdict[i].append(j)
#            if i!=pre:
#                comdict[pre].sort()
#                pre=i
#        
#        comdict[i].sort()
#        self.res=float('inf')
#        def walk(x,y,comdict,step):
#            
#            if x==n:
#                if self.res>step:
#                    self.res=step
#                return 
#            
#            if not comdict[x]:
#                walk(x+1,y,comdict,step)
#            else:
#                if y==0:
#                    
#                   walk(x+1,0,comdict,step+comdict[x][-1]*2) 
#                   walk(x+1,m-1,comdict,step+m-1)
#                elif y==m-1:
#                   walk(x+1,m-1,comdict,step+  (m-1-comdict[x][0])*2) 
#                   walk(x+1,0,comdict,step+m-1)
#        walk(0,0,comdict,0)
#        return self.res+n-1
        
        
#826. Computer Maintenance 
class Solution:
    """
    @param n: the rows of matrix
    @param m: the cols of matrix
    @param badcomputers: the bad computers 
    @return: The answer
    """
    def maintenance(self, n, m, badcomputers):
        dp=[[0,0] for _ in range(201)]
        matrix=[[0 for _ in range(201)]for _ in range(201)]
        
        for node in badcomputers:
            matrix[node.x][node.y]=1
        
        for i in range( n):
            most_right=-1
            most_left=-1
            for j in range(m):
                if matrix[i][j]!=0:
                   most_right=max(most_right,j)
                   most_left=max(most_left,m-1-j)
                if i==0:
                    if most_right==-1:
                        dp[0][0]=0
                        dp[0][1]=m-1
                    else:
                        dp[0][0]=most_right*2
                        dp[0][1]=m-1
                    continue
                if most_right!=-1:
                    dp[i][0]=min(dp[i-1][0]+most_right*2, dp[i-1][1] +m-1  )+1
                    dp[i][1]=min(dp[i-1][0]+m-1,most_left*2+dp[i-1][1])+1
                else:
                    dp[i][0]=dp[i-1][0]+1
                    dp[i][1]=dp[i-1][1]+1
        return min(dp[n-1][0],dp[n-1][1])
n = 3
m = 10
badcomputers = [[0,0],[0,9],[1,7]


n = 3
m = 10
badcomputers = [[0,3],[1,7],[1,2]]# return 17
n =30
m =30
badcomputers =[[29,22],[13,1],[12,16],[9,5],[17,21],[0,2],[29,14],[17,8],[23,4],[14,29],[25,27],[6,23],[10,24],[28,22],[1,28],[0,28],[3,23],[9,28],[7,3],[19,25],[29,21],[7,11],[3,19],[5,18],[22,16],[21,7],[10,5],[11,25],[12,26],[15,10],[10,15],[28,23],[25,10],[8,28],[5,25],[28,0],[12,14],[1,4],[2,14],[15,7],[22,2],[22,23],[29,5],[12,20],[7,19],[13,0],[24,26],[9,21],[21,29],[16,24],[14,4],[3,11],[24,12],[13,5],[21,28],[17,10],[29,9],[28,29],[28,11],[25,25],[1,11],[23,14],[10,22],[28,8],[24,5],[3,0],[4,12],[22,4],[10,23],[17,14],[5,0],[25,3],[17,24],[8,8],[28,21],[15,21],[3,27],[23,27],[0,6],[27,20],[20,17],[25,14],[0,20],[7,17],[22,8],[21,23],[23,28],[0,14],[19,14],[3,6],[0,29],[24,19],[8,17],[1,14],[21,9],[28,26],[27,11],[2,24],[14,7],[4,18]]    
if __name__ == "__main__":
    print(Solution().maintenance(n, m, badcomputers))         
    
    

#828. Word Pattern 
class Solution:
    """
    @param pattern: a string, denote pattern string
    @param teststr: a string, denote matching string
    @return: an boolean, denote whether the pattern string and the matching string match or not
    """
    def wordPattern(self, pattern, teststr):
        # write your code here
        mapping={}
        
        for a,b in zip(pattern,teststr.split()):
            if a not in mapping:
                mapping[a]=b
            else:
                if mapping[a]!=b:
                    return False
            if b not in mapping:
                mapping[b]=a
            else:
                if mapping[b]!=a:
                    return False
        return True
         
pattern = "abba"
teststr = "dog cat cat dog"

pattern = "abba"
teststr  = "dog cat cat fish"

pattern ="abba"
teststr  = "dog cat dog cat"
if __name__ == "__main__":
    print(Solution().wordPattern( pattern, teststr))         
    
            
        
#829. Word Pattern II
class Solution:
    """
    @param pattern: a string,denote pattern string
    @param str: a string, denote matching string
    @return: a boolean
    """
    def wordPatternMatch(self, pattern, string):
        # write your code here
      
        
        def search(pattern,string,dictps,dictsp):
            if not pattern  and not string  and len(dictps)==len(dictsp):
                return True
            if not pattern  or not string:
                return False
            
            
            p=pattern[0]
            if p in dictps:
                if string.startswith(dictps[p]):
                    
                    return search(pattern[1:],string[len(dictps[p]):],dictps,dictsp)
                else:
                    return False
            else:
                for i in range(len(string)):
                    
                    if string[:i+1] in dictsp:
                        return False
                    dictps[p]=string[:i+1]
                    
                    dictsp[string[:i+1]]=p
                    if search(pattern[1:],string[i+1:],dictps,dictsp):
                        return True
                    
                    del dictps[p]
                    #print(dictsp,string[:i+1],p)
                    del dictsp[string[:i+1]]
            return False
        dictps={}
        dictsp={}
        return search(pattern,string,dictps,dictsp)
                    

    
pattern = "abab"
string = "redblueredblue"# return true    
pattern = "aaaa"
string = "asdasdasdasd"# return true
pattern = "aabb"
string = "xyzabcxzyabc"# return false
pattern ="lwpstyfsjf"
string ="htkvcxwxkymrvrpcxw"    
if __name__ == "__main__":
    print(Solution().wordPatternMatch( pattern, string))         
    
                
#834. Remove Duplicate Letters
class Solution:
    """
    @param s: a string
    @return: return a string
    """
    def removeDuplicateLetters(self, s):
        # write your code here   
        stack=[]
        from collections import defaultdict
        d=defaultdict(int)
        
        for c in s:
            d[c]+=1
        
        visited={}
        for k in d:
            visited[k]=False
            
        for c in s:
            d[c]-=1
            
            if visited[c]:
                continue
            
            while stack and stack[-1]>c and d[stack[-1]]>0:
                visited[stack[-1]]=False
                stack.pop()
            stack.append(c)
            visited[c]=True
        return ''.join(stack)
   
s="bcabc"    #"abc"
s="cbacdcbc"    #"acdb"
if __name__ == "__main__":
    print(Solution().removeDuplicateLetters( s))             
    
#835. Hamming Distance    
class Solution:
    """
    @param x: an integer
    @param y: an integer
    @return: return an integer, denote the Hamming Distance between two integers
    """
    def hammingDistance(self, x, y):
        # write your code here
        dis=0
        while x!=0  or y!=0:
            
            if x%2 !=y%2:
                dis+=1
            x=x//2
            y=y//2
        return dis
    
x = 1
y = 4                
if __name__ == "__main__":
    print(Solution().hammingDistance( x, y))             
            
        
#836. Partition to K Equal Sum Subsets        
class Solution:
    """
    @param nums: a list of integer
    @param k: an integer
    @return: return a boolean, denote whether the array can be divided into k non-empty subsets whose sums are all equal
    """
    def partitiontoEqualSumSubsets(self, nums, k):
        # write your code here 
        
        total=sum(nums)
        if total%k!=0:
            return False
        
        target=total//k
        
        nums.sort()
        if nums[-1]>target:
            return False
        
        while nums and nums[-1]==target:
            nums.pop()
            k-=1
        
        def search(groups):
            if not nums:
                return True
            
            v=nums.pop()
            
            for i,group in enumerate(groups):
                if group+v<=target:
                    groups[i]=group+v
                    if search(groups):
                        return True
                    groups[i]-=v
                if group==0:
                    break
            nums.append(v)
            return False
        return search([0]*k)
nums = [4, 3, 2, 3, 5, 2, 1]
k = 4  
nums =[3,10,20,36,6,8,6,89,73,80,16,161,90,87,55,160]
k =5                  
if __name__ == "__main__":
    print(Solution().partitiontoEqualSumSubsets( nums, k))                
            
                
#837. Palindromic Substrings       
class Solution:
    """
    @param str: s string
    @return: return an integer, denote the number of the palindromic substrings
    """
    def countPalindromicSubstrings(self, string):
        # write your code here       
        l=len(string)
        dp=[[False for _ in range(l)]  for _ in range(l)]
        count=0
        
        for i in range(l):
            dp[i][i]=True
            count+=1
            
            if i+1< l and string[i]==string[i+1]:
                dp[i][i+1]=True
                count+=1
        
        
        for length in range(3,l+1):
            for i in range(l-length+1):
                j=i+length-1
                if dp[i+1][j-1] and string[i]==string[j]:
                    dp[i][j]=True
                    count+=1
        return count
                    
 
string= "abc"  
string= "aaaa"       
if __name__ == "__main__":
    print(Solution().countPalindromicSubstrings( string))                
                    
        
#838. Subarray Sum Equals K
class Solution:
    """
    @param nums: a list of integer
    @param k: an integer
    @return: return an integer, denote the number of continuous subarrays whose sum equals to k
    """
    def subarraySumEqualsK(self, nums, k):
        # write your code here    
        n=len(nums)
        
        sumn=[0 for _ in range(n+1)]
        for i in range(1,n+1):
            sumn[i]=sumn[i-1]+nums[i-1]
        
        res=0
        
        from collections import defaultdict
        sumdict=defaultdict(int)
        
        for x in sumn:
            sumdict[x]+=1
        #print(sumdict)
        for x in sumn:
            sumdict[x]-=1
            res+=sumdict[x+k]
        return res
            
        
nums = [1,1,1]
k = 2    
if __name__ == "__main__":
    print(Solution().subarraySumEqualsK( nums, k))                
                        
    
#839. Merge Two Sorted Interval Lists    
"""
Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    """
    @param list1: one of the given list
    @param list2: another list
    @return: the new sorted list of interval
    """
    def mergeTwoInterval(self, list1, list2):
        # write your code here
        
        
        if list1 is None or list2 is None:
            return []
        
#        if not list1:
#            return list2
#        if not list2:
#            return list1
        
        i=0
        j=0
        
        res=[]
        def add(interval,res):
            if len(res)==0:
                res.append(interval)
            else:
                if interval.start>res[-1].end:
                    res.append(interval)
                else:
                    res[-1].end=max(res[-1].end,interval.end)
        
        while i < len(list1)  and j < len(list2):
            if lista[i].start<list2[j].start:
                add(list1[i],res)
                i+=1
            else:
                add(list2[j],res)
                j+=1
        
        while i < len(list1):
             add(list1[i],res)
             i+=1
            
            
        while j < len(list2):
             add(list2[j],res) 
             j+=1
        return res
            
        
        
        
        
        
list1 = [(1,2),(3,4)] 
list2 = [(2,3),(5,6)]

list1 =[Interval(1,2),Interval(3,4)]
list2 = [Interval(2,3),Interval(5,6)]
if __name__ == "__main__":
    print(Solution().mergeTwoInterval( list1, list2))                
                        
#840. Range Sum Query - Mutable
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.A=[0 for _ in range(len(nums))]
        self.bits=[0 for _ in range(len(nums)+1)]
        for i in range(len(nums)):
            self.update(i,nums[i])
        

    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: void
        """
        delta=val-self.A[i]
        self.A[i]=val
        
        j=i+1
        
        while j<len(self.bits):
            self.bits[j]+=delta
            j+=self.lowbit(j)
        
        
        
    def lowbit(self,j):
        return j & (-j)
    
    def prefix_sum(self,i):
        j=i
        res=0
        while j>=1:
            res+=self.bits[j]
            j-=self.lowbit(j)
        return res
        

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.prefix_sum(j+1)-self.prefix_sum(i)


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(i,val)
# param_2 = obj.sumRange(i,j)                
    
    
#845. Greatest Common Divisor    
class Solution:
    """
    @param a: the given number
    @param b: another number
    @return: the greatest common divisor of two numbers
    """
    def gcd(self, a, b):
        # write your code here
        
        if a==0:
            return b
        return self.gcd(b%a,a)
a = 15
b = 10    


if __name__ == "__main__":
    print(Solution().gcd( a, b))                
                            
    
#846. Multi-keyword Sort    
class Solution:
    """
    @param array: the input array
    @return: the sorted array
    """
    def multiSort(self, array):
        # Write your code here
        from collections import defaultdict
        dic=defaultdict(list)
        
        for i in range(len(array)):
            dic[array[i][1]].append(array[i][0])
        
        od=sorted (dic.keys(),reverse=True)
        ans=[]
        print(od)
        for score in od:
            for i in sorted(dic[score]):
                ans.append([i,score])
        return ans
        
        
array=[[2,50],[1,50],[3,100]]
if __name__ == "__main__":
    print(Solution().multiSort( array))    
    
#848. Minimize Max Distance to Gas Station
class Solution:
    """
    @param stations: an integer array
    @param k: an integer
    @return: the smallest possible value of D
    """
    def minmaxGasDist(self, stations, k):
        # Write your code here
        
        n=len(stations)
        import math
        def isValid(gap,stations, k):
            count=0
            for i in range(n-1):
              dis=  stations[i+1]-stations[i]
              count+= math.ceil(dis/gap)-1
            return count<=k
        
        left=0
        right=10**8
        
        while right-left>1e-6:
            
            mid=(left+right)/2
            if isValid(mid,stations, k):
                right=mid
            else:
                left=mid
        return mid
            
stations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
k = 9        
if __name__ == "__main__":
    print(Solution().minmaxGasDist( stations, k))        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
