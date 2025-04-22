# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 01:32:57 2018

@author: cz
"""

#849. Basic Calculator III
class Solution:
    """
    @param s: the expression string
    @return: the answer
    """
    def calculate(self, s):
        # Write your code here
        op='+'
        s=s.replace(' ','')
        res=0
        curres=0
        num=0
        n=len(s)
        i=0
        
        while i<n:
            #print(i,num,curres,res)
            c=s[i]
            #print(c)
           
            
            if c>='0' and c<='9':
               num=num*10+ord(c)-ord('0')
               
               #print(num)
            elif c=='(':
                
                cnt=0
                j=i
                
                while i<n:
                    if s[i]=='(':
                        cnt+=1
                    elif s[i]==')':
                        cnt-=1
                    #print(i,j)
                    if cnt==0:
                        break
                    i+=1
                #print(s[j+1:i])
                num=self.calculate(s[j+1:i])
                
            if c=='+' or c=='-' or c=='*' or c=='/' or i==n-1 :
                print(num,op,c,curres,res)
                if op=='+':
                    curres+=num
                elif op=='-':
                    curres-=num
                elif op=='*':
                    curres*=num
                elif op=='/':
                    curres//=num
                
                if c=='+' or c=='-' or i==n-1:
                    res+=curres
                    curres=0
                op=c
                num=0
            i+=1
        return res
                    
s="1+2*9" #= 2 
s=" 6-4 / 2 " 
s="2*(5+5*2)/3+(6/2+8)"
s="(2+6* 3+5- (3*14/7+2)*5)+3"              
if __name__ == "__main__":
    print(Solution().calculate(s))         
            
#851. Pour Water                        
class Solution:
    """
    @param heights: the height of the terrain
    @param V: the units of water
    @param K: the index
    @return: how much water is at each index
    """
    def pourWater(self, heights, V, K):
        # Write your code here
        n=len(heights)
        for i in range(V):
            l=K
            r=K
            while l>0 and heights[l]>=heights[l-1]:
                  l-=1
            while l<K and heights[l]==heights[l+1]:
                  l+=1
            while r<n-1 and heights[r]>=heights[r+1]:
                  r+=1
            while r>K and heights[r]==heights[r-1]:
                  r-=1
            if heights[l]<heights[K]:
                heights[l]+=1
            else:
                heights[r]+=1
        return heights
                
heights = [2,1,1,2,1,2,2]
V = 4
K = 3    
heights = [1,2,3,4]
V = 2
K = 2 
heights = [3,1,3]
V = 5
K = 1           
if __name__ == "__main__":
    print(Solution().pourWater( heights, V, K))                 
            
#853. Number Of Corner Rectangles
class Solution:
    """
    @param grid: the grid
    @return: the number of corner rectangles
    """
    def countCornerRectangles(self, grid):
        # Write your code here   
        rows=[[i  for i,v in enumerate(row) if v] for row in grid]
        N=sum(  sum(row) for row in grid)
        target=N**0.5
        from collections import defaultdict
        count=defaultdict(int)
        ans=0
        import itertools
        
        for r ,row  in enumerate(rows):
            if len(row)>target:
                rset=set(row)
                for r2,row2 in enumerate(rows):
                    if r2<=r and len(row2)>target:
                        continue
                    f=sum( 1 for v in row2 if v in rset)
                    print(f)
                    ans+=f*(f-1)//2
                    
            else:
                
                for pair in itertools.combinations(row,2):
                    ans+=count[pair]
                    count[pair]+=1
        print(count)
        return ans
            
                    
     

grid = [[1, 0, 0, 1, 0],
 [0, 0, 1, 0, 1],
 [0, 0, 0, 1, 0],
 [1, 0, 1, 0, 1]] 
grid = [[1, 1, 1],
 [1, 1, 1],
 [1, 1, 1]]   
grid =[[1, 1, 1, 1]]
if __name__ == "__main__":
    print(Solution().countCornerRectangles( grid))      
    
    
#854. Closest Leaf in a Binary Tree    


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Node:
    def __init__(self, val):
        self.val = val
        self.leftness=0
        self.left, self.right,self.parent = None, None,None
class Solution:
    """
    @param root: the root
    @param k: an integer
    @return: the value of the nearest leaf node to target k in the tree
    """
    def findClosestLeaf(self, root, k):
        # Write your code here
        if not root:
            return None
        node=Node(root.val)
        node.leftness=0
        self.tranverse(node,root)
        k_node=self.find(node,k)
            
        from collections import deque
        q=deque([k_node])
        visited=set([k_node])
        res=[]
        found=False
        while q:
            for _ in range(len(q)):
                anode=q.popleft()
                if not anode.left and not anode.right:
                    found=True
                    res.append((anode.leftness,anode.val ))
                if anode.left  and anode.left not in visited:
                    visited.add(anode.left)
                    q.append(anode.left)
                if anode.right  and anode.right not in visited:
                    visited.add(anode.right)
                    q.append(anode.right)
                if anode.parent  and anode.parent not in visited:
                    visited.add(anode.parent)
                    q.append(anode.parent)
            if found:
                res.sort()
                return res[0][1]
        return None
    def tranverse(self,node,root):
            if not root.left and not root.right:
                return 
            if root.left:
                left=Node(root.left.val)
                left.parent=node
                node.left=left
                left.leftness=node.leftness-1
                self.tranverse(left,root.left)
            if root.right:
                right=Node(root.right.val)
                right.parent=node
                node.right=right
                right.leftness=node.leftness+1
                self.tranverse(right,root.right)
                
        
        
    def find(self,node,k):
            if node is None:
               return None
            if k == node.val:
               return node 
            left = self.find(node.left, k)    
            right = self.find(node.right, k)
            if left:
               return left
            if right:
               return right
            return None    
        
        
                    
root=TreeNode(1)
root.left=TreeNode(3)
root.right=TreeNode(2)
k=1

root = TreeNode(1)
k = 1

root = TreeNode(1)
root.left=TreeNode(2)
root.right=TreeNode(3)
root.left.left=TreeNode(4)
root.left.left.left=TreeNode(5)
root.left.left.left.left=TreeNode(6)
if __name__ == "__main__":
    print(Solution().findClosestLeaf( root, k))             
        
        
#856. Sentence Similarity
class Solution:
    """
    @param words1: a list of string
    @param words2: a list of string
    @param pairs: a list of string pairs
    @return: return a boolean, denote whether two sentences are similar or not
    """
    def isSentenceSimilarity(self, words1, words2, pairs):
        # write your code here 
        if len(words1)!=len(words2):
            return False
        
        for w1,w2 in zip(words1,words2):
            if [w1,w2] not in pairs  and [w2,w1]  not in pairs:
                return False
        return True
        
        
words1=["great","acting","skills"]
words2=["fine","drama","talent"]
pairs=[["great","fine"],["drama","acting"],["skills","talent"]]        
if __name__ == "__main__":
    print(Solution().isSentenceSimilarity( words1, words2, pairs))              
        
        
#857. Minimum Window Subsequence
class Solution:
    """
    @param S: a string
    @param T: a string
    @return: the minimum substring of S
    """
    def minWindow(self, S, T):
        # Write your code here
        m=len(S)
        n=len(T)
        
        dp=[[-1 for _ in range(n+1)] for _ in range(m+1)]
        
        for i in range(m+1):
            dp[i][0]=i
            
        res=float('inf')
        start=-1
        for i in range(1,m+1):
            for j in range(1,min(n,i)+1):
                if S[i-1]==T[j-1]:
                    dp[i][j]=dp[i-1][j-1]
                else:
                    dp[i][j]=dp[i-1][j]
                if dp[i][n]!=-1:
                    length=i-dp[i][n]
                    if res>length:
                       res=length
                       start=dp[i][n]
        if start==-1:
            return ''
        else:
            return S[start:res+start]
S = "abcdebdde"
T = "bde"        
if __name__ == "__main__":
    print(Solution().minWindow( S, T))              
                
#858. Candy Crush
class Solution:
    """
    @param board: a 2D integer array
    @return: the current board
    """
    def candyCrush(self, board):
        # Write your code here
        
        m=len(board)
        if m==0:
            return board
        n=len(board[0])
        
        
        while True:
            deletion=[]
            for i in range(m):
                for j in range(n):
                   if board[i][j]==0:
                        continue
                   x0=i
                   x1=i
                   y0=j
                   y1=j
                   while x0>=0 and x0 >i-3 and board[i][j]==board[x0][j]:
                       x0-=1
                   while x1<m and x1 <i+3 and board[i][j]==board[x1][j]:
                       x1+=1
                   while y0>=0 and y0 >j-3 and board[i][j]==board[i][y0]:
                       y0-=1
                   while y1<n and y1 <j+3 and board[i][j]==board[i][y1]:
                       y1+=1
                   if x1-x0>3 or y1-y0>3:
                       deletion.append((i,j))
            if not deletion:
                break
            for x,y in deletion:
                board[x][y]=0
            
            for j in range(n):
                t=m-1
                for i in range(m-1,-1,-1):
                    if board[i][j]:
                         board[i][j],board[t][j]=board[t][j],board[i][j]
                         t-=1
        return board
 
board=[[110,5,112,113,114],
 [210,211,5,213,214],
 [310,311,3,313,314],
 [410,411,412,5,414],
 [5,1,512,3,3],
 [610,4,1,613,614],
 [710,1,2,713,714],
 [810,1,2,1,1],
 [1,1,2,2,2],
 [4,1,4,4,1014]]        
        
if __name__ == "__main__":
    print(Solution().candyCrush( board))     
    
#860. Number of Distinct Islands    
class Solution:
    """
    @param grid: a list of lists of integers
    @return: return an integer, denote the number of distinct islands
    """
    def numberofDistinctIslands(self, grid):
        # write your code here
        m=len(grid)
        if m==0:
            return 0
        n=len(grid[0])
        from collections import deque
        ans=0
        check=set()
        for i in range(m):
            for j in range(n):
                if grid[i][j]==1:
                    
                     q=deque([(i,j)])
                     grid[i][j]==0
                     path=''
                     
                     while q:
                         x,y=q.popleft()
                         for nx,ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                             if nx>=0 and ny>=0 and nx<m and ny<n and grid[nx][ny]==1:
                                 q.append((nx,ny))
                                 grid[nx][ny]=0
                                 path+=str(nx-i)+str(ny-j)
                     if path not in check:
                        check.add(path)
                        ans+=1
        return ans
grid =[
[1,1,0,0,0],
[1,1,0,0,0],
[0,0,0,1,1],
[0,0,0,1,1]
]
grid =[
[1,1,0,1,1],
[1,0,0,0,0],
[0,0,0,0,1],
[1,1,0,1,1]
]        
if __name__ == "__main__":
    print(Solution().numberofDistinctIslands( grid))                                    
        
        
#861. K Empty Slots
class Solution:
    """
    @param flowers: the place where the flower will open in that day
    @param k:  an integer
    @return: in which day meet the requirements
    """
    def kEmptySlots(self, flowers, k):
        # Write your code here 
        if not flowers:
            return -1
        slot2bloomday=self.getSlot2Bloomdat(flowers)
        segTree=self.buildSegTree(slot2bloomday,0,len(slot2bloomday)-1)
        
        earliest=len(slot2bloomday)+1
        minbloomday=None
        for slot in range(len(slot2bloomday)-k-1):
            minbloomday=self.getMin( segTree, slot+1, slot+k)
            candidate=max( slot2bloomday[slot],  slot2bloomday[slot+k+1]    )
            if candidate<minbloomday:
                if candidate+1 <earliest:
                    earliest=candidate+1
        return earliest if earliest!=len(slot2bloomday)+1 else -1
    
    
    def getSlot2Bloomdat(self,flowers):
        slot2bloomday=[0 for _ in range(len(flowers))]
        for day , slot in enumerate(flowers):
            slot2bloomday[slot-1]=day
        return slot2bloomday
    
    
    
    # The following code is for segment tree
    class Node:
        def __init__(self,minval,start,end,left=None,right=None):
            self.minval=minval
            self.start=start
            self.end=end
            self.left=left
            self.right=right
    def buildSegTree(self,nums,start,end):
        if start>end:
            return None
        if start==end:
            return self.Node(nums[start],start,end)
        
        mid=(start+end)//2
        left=self.buildSegTree(nums,start,mid)
        right=self.buildSegTree(nums,mid+1,end)
        minval=float('inf')
        
        if left:
            minval=min(minval,left.minval)
        if right:
            minval=min(minval,right.minval)
        return self.Node(minval,start,end,left,right)
    
    def getMin(self, node, start, end):
        if start<=node.start and end>=node.end:
            return node.minval
        mid=(node.start+node.end)//2
        lvalue=float('inf')
        rvalue=float('inf')
        if mid>=start:
            lvalue=self.getMin(node.left,start,end)
        if mid+1<=end:
            rvalue=self.getMin(node.right,start,end)
        return min(lvalue,rvalue)
            
            
            
        
        
        
flowers = [1,3,2]
k = 1# return 2.        
        
flowers = [1,2,3]
k = 1# return -1.        


if __name__ == "__main__":
    print(Solution().kEmptySlots( flowers, k))         
        
        
#861. K Empty Slots
class Solution:
    """
    @param flowers: the place where the flower will open in that day
    @param k:  an integer
    @return: in which day meet the requirements
    """
    def kEmptySlots(self, flowers, k): 
        
#https://github.com/kamyu104/LeetCode/blob/master/Python/k-empty-slots.py 
#http://www.cnblogs.com/grandyang/p/8415880.html
        n=len(flowers)
        days=[0 for _ in range(n)]
        for day,position in enumerate(flowers):
            days[position-1]=day+1
        
        right=k+1
        left=0
        i=0
        res=float('inf')
        while right<n:
            if days[i]<days[left]  or days[i] <=days[right]:
                if i==right:
                    res=min(res,  max(days[left],days[right]))
                left=i
                right=i+k+1
            i+=1
        return res if res<float('inf') else -1
                
                    
        
            
            
        
flowers = [1,3,2]
k = 1# return 2.        
        
flowers = [1,2,3]
k = 1# return -1.        


if __name__ == "__main__":
    print(Solution().kEmptySlots( flowers, k))              
        
#862. Next Closest Time        
class Solution:
    """
    @param time: the given time
    @return: the next closest time
    """
    def nextClosestTime(self, time):
        # write your code here   
        
        h,m=time.split(':')
        curr=int(h)*60+int(m)
        
        for t in range(curr+1,curr+1441):
            res=t%1440
            h=res//60
            m=res%60
            result="%02d:%02d" %(h,m)
            
            if set(result)<=set(time):
                return result
        
time = "19:34"    
time = "23:59"    
if __name__ == "__main__":
    print(Solution().nextClosestTime(time))                      
        
#863. Binary Tree Path Sum IV
class Solution:
    """
    @param nums: a list of integers
    @return: return an integer
    """
    def pathSum(self, nums):
        # write your code here    
        dmap={1:0}
        
        leaves=set([1])
        
        for num in nums:
            path,value=num//10,num%10
            level,pos=path//10,path%10
            
            parent=(level-1)*10+(pos+1)//2
            
            dmap[path]=dmap[parent]+value
            
            leaves.add(path)
            
            if parent in leaves:
                leaves.remove(parent)
        return sum(dmap[v] for v in leaves)
            
        
        
        
#864. Equal Tree Partition
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: a TreeNode
    @return: return a boolean
    """
    def checkEqualTree(self, root):
        # write your code here
        
        def treeSum(node,res):
            if not node:
                return res
            if not node.left and not node.right:
                return node.val+res
            if not node.left:
                return node.val+res+treeSum(node.right,0)
            if not node.right:
                return node.val+res+treeSum(node.left,0)
           
            return node.val+res+treeSum(node.left,0)+treeSum(node.right,0)
        allsum=treeSum(root,0)
        if allsum%2==1:
            return False
        target=allsum//2
        
        from collections import deque
        if not root:
            return False
        
        q=deque([])
        
        if not root.left and not root.right:
            return False
        
        if root.left:
            q.append(root.left)
        if root.right:
            q.append(root.right)
        
        while q:
           cur= q.popleft()
           if treeSum(cur,0)==target:
               return True
           if cur.left:
               q.append(cur.left)
           if cur.right:
               q.append(cur.right)
        return False
           
   
        
root=TreeNode(5)        
root.left=    TreeNode(10) 
root.right=    TreeNode(10)        
root.right.left=  TreeNode(2)        
root.right.right=   TreeNode(3)        

    1
   / \
  2  10
    /  \
   2   20    

root=TreeNode(1)        
root.left=    TreeNode(2) 
root.right=    TreeNode(10)        
root.right.left=  TreeNode(2)        
root.right.right=   TreeNode(20)     
    
if __name__ == "__main__":
    print(Solution().checkEqualTree(root))                      
                
#865. Remove 9
class Solution:
    """
    @param n: an integer
    @return: return an long integer
    """
    def newInteger(self, n):
        # write your code here
        #0，1，2，3，4，5，6，7，8 （移除了9）
#10，11，12，13，14，15，16，17，18 （移除了19）
#.....
#
#80，81，82，83，84，85，86，87，88 （移除了89）
#（移除了 90 - 99 ）
#100，101，102，103，104，105，106，107，108 （移除了109）
#我们可以发现，8的下一位就是10了，18的下一位是20，88的下一位是100，
#实际上这就是九进制的数字的规律，那么这道题就变成了将十进制数n转为九进制数，这个就没啥难度了
#，就每次对9取余，然后乘以base，n每次自除以9，base每次扩大10倍，参见代码如下：
        res=0
        base=1
        
        while n:
            res+=n%9 * base
            n=n//9
            base=base*10
        return res
 
        
n=10 
n=100
n=1000               
if __name__ == "__main__":
    print(Solution().newInteger(n))                      
                                  
#866. Coin Path        
class Solution:
    """
    @param A: a list of integer
    @param B: an integer
    @return: return a list of integer
    """
    def cheapestJump(self, A, B):
        # write your code here
        Adict={}
        N=len(A)
        for i,a in enumerate(A):
            Adict[i+1]=a
        
        from collections import deque,defaultdict
        
        q=deque([ (1, '1',Adict[1]) ])
        res=float('inf')
        resDict=defaultdict(list)
        while q:
            
            index,path,curSum=q.popleft()
            if index==N:
                if curSum <= res:
                    res=curSum
                    resDict[res].append(path)
            else:
                for jump in range(1,B+1):
                    if index+jump <=N  and Adict[index+jump]!=-1:
                        q.append( ( index+jump,path  + str(index+jump),curSum+Adict[index+jump] ))
        if res!=float('inf'):
           return [int(x) for x in resDict[res][0]]
        else:
            return []
        
A = [1,2,4,-1,2]
B = 2

A = [1,2,4,-1,2]
B = 1
A =[36,33,18,55,98,14,77,43,6,97,49,72,62,48,68,65,22,18,63,44,14,4,99,52,52,23,47]
B = 50
if __name__ == "__main__":
    print(Solution().cheapestJump( A, B))                     
                    
            
#866. Coin Path        
class Solution:
    """
    @param A: a list of integer
    @param B: an integer
    @return: return a list of integer
    """
    def cheapestJump(self, A, B): 
        N=len(A)
        cost=[float('inf')  for _ in range(N+1)]
        path=[[] for _ in range(N+1) ]
        
        cost[1]=A[0]
        path[1]=[1]
        
        for x in range(2,N+1):
            if A[x-1]==-1:
                continue
            for y in range(1,B+1):
                z=x-y
                if z>=1:
                   if A[z-1]==-1:
                      continue
                   if cost[x]>cost[z]+A[x-1]  or ( cost[x]==cost[z]+A[x-1] and path[x]>path[z]+[x]):
                       cost[x]=cost[z]+A[x-1]
                       path[x]=path[z]+[x]
        return path[-1]
        
A = [1,2,4,-1,2]
B = 2

A = [1,2,4,-1,2]
B = 1
A =[36,33,18,55,98,14,77,43,6,97,49,72,62,48,68,65,22,18,63,44,14,4,99,52,52,23,47]
B = 50
if __name__ == "__main__":
    print(Solution().cheapestJump( A, B))           
        
#867. 4 Keys Keyboard
class Solution:
    """
    @param N: an integer
    @return: return an integer
    """
    def maxA(self, N):
        # write your code here  
        
        dp=[0 for _ in range(N+1)]
        dp[1]=1
        dp[2]=2
        dp[3]=3

        for i in range(4,N+1):
            dp[i]=i
            prev=i-3
            count=2
            while prev>0:
                dp[i]=max(dp[i],dp[prev]*count)
                prev-=1
                count+=1
        return dp[N]
            
        
        
N = 3  
N = 7# return 9      
if __name__ == "__main__":
    print(Solution().maxA( N))           
            
#868. Maximum Average Subarray
class Solution:
    """
    @param nums: an array
    @param k: an integer
    @return: the maximum average value
    """
    def findMaxAverage(self, nums, k):
        # Write your code here
        n=len(nums)
        
        cursum=sum(nums[:k-1])
        res=float('-inf')
        for i in range(k-1,n):
            cursum+=nums[i]
            res=max(res,cursum/k)
            cursum-=nums[i-k+1]
        return res
nums = [1,12,-5,-6,50,3]
k = 4
if __name__ == "__main__":
    print(Solution().findMaxAverage(nums, k))            
            
        
#869. Find the Derangement of An Array        
class Solution:
    """
    @param n: an array consisting of n integers from 1 to n
    @return: the number of derangement it can generate
    """
    def findDerangement(self, n):
        # Write your code here
        
#n = 1 时有 0 种错排
#n = 2 时有 1 种错排 [2, 1]
#n = 3 时有 2 种错排 [3, 1, 2], [2, 3, 1]
#我们来想n = 4时该怎么求，我们假设把4排在了第k位，这里我们就让k = 3吧，
#那么我们就把4放到了3的位置，变成了：
#x x 4 x
#我们看被4占了位置的3，应该放到哪里，这里分两种情况，如果3放到了4的位置，那么有：
#x x 4 3
#那么此时4和3的位置都确定了，实际上只用排1和2了，那么就相当于只排1和2，就是dp[2]的值，
#是已知的。那么再来看第二种情况，3不在4的位置，那么此时我们把4去掉的话，就又变成了：
#x x x
#这里3不能放在第3个x的位置，在去掉4之前，这里是移动4之前的4的位置，那么实际上这又变成了排1，2，3
#的情况了，就是dp[3]的值。
#再回到最开始我们选k的时候，我们当时选了k = 3，其实k可以等于1,2,3，也就是有三种情况，
#所以dp[4] = 3 * (dp[3] + dp[2])。
#那么递推公式也就出来了：
#dp[i] = (i - 1) * (dp[i - 1] + dp[i - 2])
        
#        dp=[0 for _ in range(n+1)]
#        dp[1]=0
#        dp[2]=1
#        for i in range(3,n+1):
#            dp[i]=(i-1)*(dp[i-1]+dp[i-2])
#        return dp[n]%(10**9+7)
        
#我们假设 e[i] = dp[i] - i * dp[i - 1]
#
#递推公式为:  dp[i] = (i - 1) * (dp[i - 1] + dp[i - 2])
#
#将递推公式带入假设，得到：
#
#e[i] = -dp[i - 1] + (n - 1) * dp[i - 2] = -e[i - 1]
#
#从而得到 e[i] = (-1)^n
#
#那么带回假设公式，可得: dp[i] = i * dp[i - 1] + (-1)^n        
        
    
        if n==1:
            return 0
        if n==2:
            return 1
    
        first=0
        second=1
        temp=1
        for i in range(3,n+1):
            temp*=(-1)
            third=(i*second+temp)%(10**9+7)
            
            second=third
        return third%(10**9+7)


#871. Minimum Factorization
class Solution:
    """
    @param a: a positive integer
    @return: the smallest positive integer whose multiplication of each digit equals to a
    """
    def smallestFactorization(self, a):
        # Write your code here
        
        def decompose(x):
            if x<10:
              return [x]
        
            else:
              for i in range(9,1,-1):
                if x%i==0:
                    ans=decompose(x//i)+[i]
                    return ans
        ans=[]
        
        ans=decompose(a)
        print
        res=0
        if not ans:
            return 0
        else:
            ans.sort()
            if ans[-1]>9:
                return 0
            else:
                for j in range(len(ans)):
                    res*=10
                    res+=ans[j]
                
                return res if res<2**31 else 0
a=48
a=15     
a=18000000               
if __name__ == "__main__":
    print(Solution().smallestFactorization( a))            


#872. Kill Process
class Solution:
    """
    @param pid: the process id
    @param ppid: the parent process id
    @param kill: a PID you want to kill
    @return: a list of PIDs of processes that will be killed in the end
    """
    def killProcess(self, pid, ppid, kill):
        # Write your code here
        from collections import defaultdict
        
        graph=defaultdict(list)
        
        for s,p in zip(pid,ppid):
            graph[p].append(s)
            
        ans=[kill]
        
        def getChild(k):
            
            for c in graph[k]:
                ans.append(c)
                getChild(c)
        getChild(kill)
        return ans
              
            

pid = [1, 3, 10, 5]
ppid = [3, 0, 5, 3]
kill = 5# return [5,10]
           3
         /   \
        1     5
             /
            10


if __name__ == "__main__":
    print(Solution().killProcess(pid, ppid, kill))            


#873. Squirrel Simulation
class Solution:
    """
    @param height: the height
    @param width: the width
    @param tree: the position of tree
    @param squirrel: the position of squirrel
    @param nuts: the position of nuts
    @return: the minimal distance 
    """
    def minDistance(self, height, width, tree, squirrel, nuts):
        # Write your code here
        m=height
        n=width
        
        res=0
        maxdiff=float('-inf')
        
        for nut in nuts:
            dist=abs(tree[0]-nut[0])+abs(tree[1]-nut[1])
            res+=dist*2
            
            maxdiff=max(maxdiff,abs(tree[0]-nut[0])+abs(tree[1]-nut[1])-abs(squirrel[0]-nut[0])-abs(squirrel[1]-nut[1])  )
        return res-maxdiff


#那么正确思路应该是，假设小松树最先应该去粟子i，那么我们假设粟子i到树的距离为x，
#小松鼠到粟子i的距离为y，那么如果小松鼠不去粟子i，累加步数就是2x，如果小松鼠去粟子i，
#累加步数就是x+y，我们希望x+y尽可能的小于2x，那么就是y尽可能小于x，即x-y越大越好 



             
height = 5
width = 7
tree = [2,2]
squirrel = [4,4]
nuts = [[3,0], [2,5]]              
           
height = 1
width =3
tree =[0,1]
squirrel =[0,0]
nuts = [[0,2]]

height = 2
width =2
tree =[0,0]
squirrel =[1,1]
nuts =[[1,0]]

height =5
width =5
tree =[3,2]
squirrel =[0,1]
nuts =[[2,0],[4,1],[0,4],[1,3],[1,0],[3,4],[3,0],[2,3],[0,2],[0,0],[2,2],[4,2],[3,3],[4,4],[4,0],[4,3],[3,1],[2,1],[1,4],[2,4]]


if __name__ == "__main__":
    print(Solution().minDistance(height, width, tree, squirrel, nuts))            



#874. Maximum Vacation Days
class Solution:
    """
    @param flights: the airline status from the city i to the city j
    @param days: days[i][j] represents the maximum days you could take vacation in the city i in the week j
    @return: the maximum vacation days you could take during K weeks
    """
    def maxVacationDays(self, flights, days):
        # Write your code here
#http://bookshadow.com/weblog/2017/04/30/leetcode-maximum-vacation-days/        
        N=len(days)
        k=len(days[0])
        
        dp=[[-1 for _ in range(N)]  for _ in range(k+1)]
        
        dp[0][0]=0
        
        for w in range(k):
            
            for sc in range(N):
                if dp[w][sc]<0:
                    continue
                for tc in range(N):
                    if sc==tc  or flights[sc][tc]!=0:
                        dp[w+1][tc]=max(dp[w+1][tc] ,dp[w][sc] +days[tc][w] )
        return max(dp[k])
        
flights = [[0,1,1],[1,0,1],[1,1,0]]
days = [[1,3,1],[6,0,3] ,[3,3,3]]       
if __name__ == "__main__":
    print(Solution().maxVacationDays( flights, days))                    
                    
            
#875. Longest Line of Consecutive One in Matrix                
class Solution:
    """
    @param M: the 01 matrix
    @return: the longest line of consecutive one in the matrix
    """
    def longestLine(self, M):
        # Write your code here
        
        m=len(M)
        if m==0:
            return 0
        n=len(M[0])
        def isHbegin( x,y ):
            if y==0 or M[x][y-1]==0:
                return True
            return False
        
        def isVbegin( x,y ):
            if x==0 or M[x-1][y]==0:
                return True
            return False
        
        def isDbegin( x,y ):
            if x==0 or y==0 or  M[x-1][y-1]==0:
                return True
            return False
        
        def isADbegin( x,y ):
            if x==0 or y==n-1 or  M[x-1][y+1]==0:
                return True
            return False
        
        def HDFS(x,y):
            res=0
            while y<n and M[x][y]==1:
                y+=1
                res+=1
            return res
        
        def VDFS(x,y):
            res=0
            while x<m and M[x][y]==1:
                x+=1
                res+=1
            return res
        
        def DDFS(x,y):
            res=0
            while x<m and y<n and M[x][y]==1:
                x+=1
                y+=1
                res+=1
            return res
                
        def ADDFS(x,y):
            res=0
            while x<m and y>=0 and M[x][y]==1:
                x+=1
                y-=1
                res+=1
            return res       
        ans=float('-inf')
        for i in range(m):
            for j in range(n):
                tempH=0
                tempV=0
                tempD=0
                tempAD=0
                
                if M[i][j]==1:
                
                   if isHbegin( i,j ):
                       temp=HDFS( i,j)
                       if temp>ans:
                           ans=temp
                           
                   if isVbegin(  i,j ):
                       temp=VDFS( i,j)
                       if temp>ans:
                           ans=temp
                
                   if isDbegin(  i,j ):
                       temp=DDFS( i,j)
                       if temp>ans:
                           ans=temp
                   if isADbegin(  i,j):
                       temp=ADDFS( i,j)
                       if temp>ans:
                           ans=temp
        return ans if ans > 0 else 0

M =[
    [0,1,1,0],
    [0,1,1,0],
    [0,0,0,1]
]
if __name__ == "__main__":
    print(Solution().longestLine( M))       



#876. Split Concatenated Strings
class Solution:
    """
    @param strs: a list of string
    @return: return a string
    """
    def splitLoopedString(self, strs):
        # write your code here
        res='a'
        S=''
        
        for s in strs:
            temp=s[::-1]
            S+=max(s,temp)
        
        curIndex=0
        for i in range(len(strs)):
            tempIndex=len(strs[i])
            p1=strs[i]
            p2=p1[::-1]
            
            body=S[curIndex+tempIndex:]+S[:curIndex]
            curIndex+=tempIndex
            
            for j in range(tempIndex):
                if p1[j]>=res[0]:
                    res=max(p1[j:]+body+p1[:j] ,res)
                if p2[j]>=res[0]:
                    res=max(p2[j:]+body+p2[:j] ,res)
        return res
strs = ["abc", "xyz"]        

if __name__ == "__main__":
    print(Solution().splitLoopedString( strs))       



#877. Split Array with Equal Sum
class Solution:
    """
    @param nums: a list of integer
    @return: return a boolean
    """
    def splitArray(self, nums):
        # write your code here
        n=len(nums)
        lasttarget=None
        for i in range(1,n-3):
            target=sum(nums[:i])
            if lasttarget==target:
              continue
            lasttarget=target
            sumj=0
            for j in range(i+1,n-2):
                #print(sumj)
                sumj+=nums[j]
                if sumj > target:
                    break
                elif sumj==target:
                    sumk=0
                    for k in range(j+2,n-1):
                        sumk+=nums[k]
                        #print(i,j,sumj,sumk)
                        if sumk > target:
                           break
                        elif sumk==target:
                             if sum(nums[k+2:])==target:
                                 return True
                             else:
                                 break
        return False
                        
    
#0 < i, i + 1 < j, j + 1 < k < n - 1            
#(0, i - 1), (i + 1, j - 1), (j + 1, k - 1) (k + 1, n - 1)         
  
nums = [1,2,1,2,1,2,1]
nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]        
if __name__ == "__main__":
    print(Solution().splitArray( nums))       
        
#878. Boundary of Binary Tree
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: a TreeNode
    @return: a list of integer
    """
    def boundaryOfBinaryTree(self, root):
        # write your code here
        
        self.left=[]
        self.right=[]
        self.leaves=[]
        
        def find_left(node):
            self.left.append(node.val)
            if node.left and not is_leaves(node.left):
                find_left(node.left)
            elif not node.left and not is_leaves(node.right):
                find_left(node.right)
                
                
            
        
        def find_right(node):
            self.right.append(node.val)
            if node.right and not is_leaves(node.right):
                find_right(node.right)
            elif not node.right and not is_leaves(node.left):
                find_right(node.left)
        
        def find_leaves(node):
            
            if node.left:
                find_leaves(node.left)
            if is_leaves(node):
                self.leaves.append(node.val)
            if node.right:
                find_leaves(node.right)
        
        def is_leaves(node):
            return not node.left and not node.right
        
        if not root:
            return []
        
        if root.left and not is_leaves(root.left):
            find_left(root.left)
        if root.right and not is_leaves(root.right):
            find_right(root.right)
        
        if not is_leaves(root):
            find_leaves(root)
        
        return [root.val]+self.left+self.leaves+list(reversed(self.right))
            
            
        
        
        
        
        
        
        
        
            
            
                
        


#879. Output Contest Matches
class Solution:
    """
    @param n: a integer, denote the number of teams
    @return: a string
    """
    def findContestMatch(self, n):
        # write your code here








        
#880. Construct Binary Tree from String        
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param s: a string
    @return: a root of this tree
    """
    def str2tree(self, s):
        # write your code here
        
        def parse(st):
            count=0
            left=''
            if not st:
                return None,None
            for i,c in enumerate(st):
                if c=='(':
                   count+=1
                elif c==')':
                   count-=1
                if count==0:
                    
                    break
            left=None
            right=None
            if len(st[:i+1])>0:
                left=st[1:i]
            if len(st[i+1:])>0:
                right=st[i+2:-1]
            return left,right
        
        def build(S):
            if not S:
                return None
            print(S)
            if S.isdigit():
                root=TreeNode(int(S))
                return root
            
            i=0
            while i<len(S) and (S[i]=='-' or S[i].isdigit()):
                i+=1
            root=TreeNode(int(S[:i]))
            left,right=parse(S[i:])
            
            left_child=build(left)
            right_child=build(right)
            root.left=left_child
            root.right=right_child
            return root
        return build(s)
                
                


s = "4(2(3)(1))(6(5))"
s = "-4(2(3)(1))(6(5))"
s = "1(2(3(4(5(6(7(8)))))))(9(10(11(12(13(14(15)))))))"
s = "351(821(568(621)(725))(516(622)(250)))(387(576(568))(607(835)(97)))"

       4
     /   \
    2     6
   / \   / 
  3   1 5  

if __name__ == "__main__":
    print(Solution().str2tree( s))  


#879. Output Contest Matches
class Solution:
    """
    @param n: a integer, denote the number of teams
    @return: a string
    """
    def findContestMatch(self, n):
        # write your code here
        S=[]
        for i in range(1,n+1):
            S.append(str(i))
        
        while n>1:
            for i in range(n//2):
                S[i]='('+S[i]+','+S[n-1-i]+')'
            n=n//2
            #print(S)
        
        return S[0]
n=8
if __name__ == "__main__":
    print(Solution().findContestMatch( n))  


#883. Max Consecutive Ones II        
class Solution:
    """
    @param nums: a list of integer
    @return: return a integer, denote  the maximum number of consecutive 1s
    """
    def findMaxConsecutiveOnes(self, nums):
        # write your code here 
        temp=[]
        zero=0
        one=0
        i=0
        while i < len(nums):
            
            while i < len(nums) and  nums[i]==0:
                zero+=1
                i+=1
            if zero >0:
                if zero>1:
                   temp.append('2')
                else:
                   temp.append('1')
                zero=0
                
            
            while i < len(nums) and  nums[i]==1:
                one+=1
                i+=1
            if one>0:
                temp.append(one)
                one=0
        print(temp)
        
        res=float('-inf')
        cur=0
        
        for j in range(len(temp)):
            if temp[j]=='1':
                if j>0:
                    cur+=temp[j-1]
                if j<len(temp)-1:
                    cur+=temp[j+1]
            res=max(res,cur+1)
            cur=0
        return res
            
            
                
nums = [1,0,1,1,0]# return 4              
                
if __name__ == "__main__":
    print(Solution().findMaxConsecutiveOnes(nums))  
            
            
#884. Find Permutation            
class Solution:
    """
    @param s: a string
    @return: return a list of integers
    """
    def findPermutation(self, s):
        # write your code here 
        n=len(s)
        res=[i for i in range(1,n+2)]
        
        i=0
        while i< n:
            
            if s[i]=='D':
               j=i
               
               while i<n and s[i]=='D':
                   i+=1
               res[j:i+1]=res[j:i+1][::-1]
               i-=1
               
            i+=1
               
        return res
            
            
        
        
s="DI"        
s='DDIIDI'       
if __name__ == "__main__":
    print(Solution().findPermutation( s))  
                    
#885. Encode String with Shortest Length        
class Solution:
    """
    @param s: a string
    @return: return a string
    """
    def encode(self, s):
        # write your code here
        #https://segmentfault.com/a/1190000008341304
        #其中dp[i][j]表示s在[i, j]范围内的字符串的缩写形式(如果缩写形式长度大于子字符串，那么还是保留子字符串)
        n=len(s)
        dp=[['' for _ in range(n)]for _ in range(n)]
        for i in range(n):
            dp[i][i]=s[i]
        
        for step in range(1,n+1):
            for i in range(n-step):
                j=i+step
                
                for k in range(i,j):
                    left=len(dp[i][k])
                    right=len(dp[k+1][j])
                    if not dp[i][j] or left+right<len(dp[i][j]):
                        dp[i][j]=dp[i][k]+dp[k+1][j]
                t=s[i:j+1]
              
                pos=(t+t).find(t,1)
                if pos<len(t):
                    t=str(len(t)//pos)+'['+dp[i][i+pos-1]+']'
                    
                if not dp[i][j] or len(t)<len(dp[i][j]):
                        dp[i][j]=t
        print(dp)
        return dp[0][n-1]
                        
                        
s = "aaa"                
s = "aaaaa"  
s = "aaaaaaaaaa" 
s = "aabcaabcd"   
s = "abbbabbbcabbbabbbc" 
if __name__ == "__main__":
    print(Solution().encode( s)) 


#886. Convex Polygon
class Vector:
    def __init__(self,x,y):
        self.x=x
        self.y=y
class Solution:
    """
    @param point: a list of two-tuples
    @return: a boolean, denote whether the polygon is convex
    """
    def isConvex(self, point):
        # write your code here
#已知A、B两点的坐标，求向量AB、向量BA的坐标：（1）A（3，5）B（6，9）
#AB=(6,9)-(3,5)=(3,4)
#BA=(3,5)-(6,9)=(-3,-4)
#验证边向量叉乘，当aXb<0时，b对应的线段在a的顺时针方向；当aXb=0时，a、b共线；当aXb>0时，
#b在a的逆时针方向。
#
#注意的是我们一开始不知道逆时针顺时针顺序，所以只需要所有的 相邻向量乘积都是同号即可
#
#先建立边的向量组，再两两验证即可
#叉积的运用（此处在之后的凸包和极角排序会用用到）：
#叉积运算结果为一个向量
#例如： a=（x1,y1）,b=(x2,y2) n为向量
#则 a×b=（x1y2-x2y1)n;
#a×b>0 则说明 b在a的左上方
#a×b<0 则说明b在a的右下方
        n=len(point)
        vectors=[None for _ in range(n) ]
        
        for i in range(n-1):
            vectors[i]=Vector(point[i+1][0]-point[i][0],point[i+1][1]-point[i][1])
            
        
        vectors[n-1]=Vector(point[0][0]-point[n-1][0],point[0][1]-point[n-1][1])
        
        cur=0
        prev=0
        
        for i in range(n-1):
            cur=vectors[i].x*vectors[i+1].y-vectors[i+1].x*vectors[i].y
            #print(prev,cur)
            if cur!=0:
                if cur*prev<0:
                    return False
                else:
                    prev=cur
        cur=vectors[n-1].x*vectors[0].y-vectors[0].x*vectors[n-1].y
        if cur*prev<0:
            return False
        else:
            return True
        
point = [[0, 0], [0, 1], [1, 1], [1, 0]]    
point = [[0, 0], [0, 10], [10, 10], [10, 0], [5, 5]]    
if __name__ == "__main__":
    print(Solution().isConvex( point))         

#887. Ternary Expression Parser
class Solution:
    """
    @param expression: a string, denote the ternary expression
    @return: a string
    """
    def parseTernary(self, expression):
        # write your code here
        
        def parse(s):
            n=len(s)
            if n==1:
                return s
            count1=0
            count2=0
            for i in range(n-1):
                if s[i]=='?':
                    count1+=1
                elif s[i]==':':
                    count2+=1
                    if count1==count2:
                        if s[0]=='T':
                            return parse(s[2:i])
                        else:
                            return parse(s[i+1:])
                
        return parse(expression)
                
expression = "T?2:3" # "2"
expression = "F?1:T?4:5" #"4"
expression = "T?T?F:5:3"#"F"
if __name__ == "__main__":
    print(Solution().parseTernary( expression))         


#888. Valid Word Square
class Solution:
    """
    @param words: a list of string
    @return: a boolean
    """
    def validWordSquare(self, words):
        # Write your code here
        m=len(words)
        col=[]
        
        rawl=[]
        coll=[]
        
        for i,row in enumerate(words):
            for j in range(len(row)):
                if i >= len(words[j]):
                    return False
                if words[i][j]!=words[j][i]:
                    return False
        return True
        
words=[
  "abcd",
  "bnrt",
  "crmy",
  "dtye"
]        
        
words=[
  "abcd",
  "bnrt",
  "crm",
  "dt"
]        
        
words=[
  "ball",
  "area",
  "read",
  "lady"
]        
if __name__ == "__main__":
    print(Solution().validWordSquare( words)) 

#889. Sentence Screen Fitting
class Solution:
    """
    @param sentence: a list of string
    @param rows: an integer
    @param cols: an integer
    @return: return an integer, denote times the given sentence can be fitted on the screen
    """
    def wordsTyping(self, sentence, rows, cols):
        # Write your code here
        
#        row=0
#        col=-1
#        n=len(sentence)
#        count=0
#        
#        temp={}
#        for i in range(n):
#            temp[i]=sentence[i]
#        sentence=temp
#            
#        
#        while True:
#        
#            for i in range(n):
#                
#                l=len(sentence[i])
#                #print(row,col,sentence[i],l,count)
#                if col!=-1:
#                  if col+1+l==cols-1:
#                     col=-1
#                     row+=1
#                  elif col+1+l>cols-1:
#                     col=l-1
#                     if col == cols-1:
#                         col=-1
#                         row+=2
#                     else:              
#                         
#                         row+=1
#                  else:
#                    col=col+l+1
#                else:
#                    if col+l==cols-1:
#                       col=-1
#                       row+=1
#                    elif col+l>cols-1:
#                       col=l-1
#                       if col == cols-1:
#                         col=-1
#                         row+=2
#                       else:              
#                         
#                         row+=1
#                    else:
#                       col=col+l
#            if col==-1 and row==rows:
#                    return count+1
#                
#            elif row>=rows :
#                    return count
#                
#            count+=1
#        

#https://medium.com/@rebeccahezhang/leetcode-418-sentence-screen-fitting-9d6258ce116e
#0      7     13          25
#abc de f abc de f abc de f
#XXXXXX
#       XXXXXX 
#             XXXXXX
#                  XXXXXX 
#                         X....
#abc-de
#f-abc-
#de-f--
#abc-de
#f...

        string=''
        for s in sentence:
            string+=s+' '
        n=len(string)
        start=0
        for row in range(rows):
            start+=cols
            if string[start%n]==' ':
                start+=1
            else:
                while start>0 and string[(start-1)%n]!=' ':
                    start-=1
        return start//n
                
            
            


   
rows = 2
cols = 8
sentence = ["hello", "world"]

#hello---
#world---


rows = 3
cols = 6
sentence = ["a", "bcd", "e"]
#a-bcd- 
#e-a---
#bcd-e-

rows = 4
cols = 5
sentence = ["I", "had", "apple", "pie"]


#I-had
#apple
#pie-I
#had--
sentence =["bcgqp", "xlqayc", "jzsxzhu", "ycxbxpxllq", "xqhz", "xtkegmw", "rtmye", "sxszyk", "mogkdakn", "tul", "jfn", "wh", "lldk", "schxgncgw", "jfdosso", "vnmxlag", "vkfo", "pzn", "nvyhr", "cqkerpihgn", "rrlggse"]
rows = 868
cols =942

sentence =["dfasaje", "yq", "nutwaqrxr", "hib", "fuoek", "msmlym", "rxkb", "g", "kxudip", "mt", "ezgdoxrjta", "xal", "ozfzdpp", "iqibu", "tuggjitblt", "jp", "m", "eqrkedg", "ojsg", "ksopshzvy", "xsukuxlqvo", "ln", "km", "osyq", "jeapard", "suq", "kgawxc", "ycpxhxzx", "iyz", "yfbqgcpl", "qfcqz", "nd", "wzgfu", "u", "trsn", "wutobt", "tdyz", "emqavunxf", "iok", "mtjclq", "gjbniqx", "wkuvit", "yalyp", "oqsbo", "zierybnyv", "rqyawhshit", "fpyexnqjnu", "djc", "tllwsfaei", "xv", "afp", "g", "jrjv", "cmtnkszzm", "fvbtrwaom", "rpyvzmzzni", "x", "cdqeitxbl", "zmvlow", "zhwus", "qe", "rzabtpalr", "c", "mbdbde", "d", "inisv", "pjwrunw", "yqnjztb", "bpp", "qqnnzrwvna", "fa", "bfq", "nwon", "ddklo", "iaxoozc", "nqn", "rwxdosoya", "qsxh", "nqq", "bj", "wgjf", "ekjerybaxq", "jbdudsyqne", "psohf", "prmj", "frpaxvra", "bjr", "fisirjwkq", "lily", "eyldhxrj", "bjuwf", "kvt", "glqa", "z", "rkn", "sgf", "k", "uwgda", "edtcfou", "hc"]
rows =20000
cols =20000
if __name__ == "__main__":
    print(Solution().wordsTyping(sentence, rows, cols)) 
        
#891. Valid Palindrome II
class Solution:
    """
    @param s: a string
    @return: nothing
    """
    def validPalindrome(self, s):
        # Write your code here
       
        
        def check(s):
               if not s:
                  return True
               n=len(s)
               if n==1:
                  return True
               left=0
               right=n-1
               while left<right:
                   if s[left]!=s[right]:
                       return False
                   left+=1
                   right-=1
               return True
                   
            
        if  check(s):
            return True
        n=len(s)
        left=0
        right=n-1
        while left<right:
            if s[left]!=s[right]:
#                print(s[left],s[right])
#                print(s[:left]+s[left:])
#                print(s[:right]+s[right:])
#                print(check(s[:left]+s[left:]))
#                print(check(s[:right]+s[right:]))
                if check(s[:left]+s[left+1:])   or check(s[:right]+s[right+1:]) :
                   
                    return True
                else:
                    return False
            left+=1
            right-=1
s = "aba" #return true
s = "abca" #return true
if __name__ == "__main__":
    print(Solution().validPalindrome( s))

        
#892. Alien Dictionary
class Solution:
    """
    @param words: a list of words
    @return: a string which is correct order
    """
    def alienOrder(self, words):
        # Write your code here
        from collections import defaultdict
        graph={ch : [] for word in words for ch in word}
        indegree={ch:0  for word in words for ch in word}
        
        for i in range(len(words)-1):
            for pos in range(min(len(words[i]),len(words[i+1]))):
                pre=words[i][pos]
                nxt=words[i+1][pos]
                if pre!=nxt:
                    indegree[nxt]+=1
                    graph[pre].append(nxt)
                    break
        
        import heapq
        print(indegree)
        heap=[w for w in indegree if indegree[w]==0]
        heapq.heapify(heap)
        order=[]
        while heap:
            for _ in range(len(heap)):
                word=heapq.heappop(heap)
                order.append(word)
                for child in graph[word]:
                    indegree[child]-=1
                    if indegree[child]==0:
                        heapq.heappush(heap,child)
        print(order)
        if len(order)!=len(indegree):
            return ''
        return ''.join(order)
                        
words=["wrt","wrf","er","ett","rftt"]        
if __name__ == "__main__":
    print(Solution().alienOrder(words)) 
                
        
#900. Closest Binary Search Tree Value        
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the given BST
    @param target: the given target
    @return: the value in the BST that is closest to the target
    """
    def closestValue(self, root, target):
        # write your code here
        
        self.mindif=float('inf')
        self.res=None
        def tranverse(node):
            
            if node.left:
                tranverse(node.left)
            dif=abs(node.val-target)
            if dif < self.mindif:
                self.res=node.val
                self.mindif=dif
            if node.right:
                tranverse(node.right)
        tranverse(root)
        return self.res
                
        
        
#901. Closest Binary Search Tree Value II
 """
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the given BST
    @param target: the given target
    @param k: the given k
    @return: k values in the BST that are closest to the target
    """
    def closestKValues(self, root, target, k):
        # write your code here
       
        self.res=[]
        self.count=0
        def tranverse(node):
            if self.count>2*k:
                return 
            
            if node.left:
                tranverse(node.left)
            dif=abs(node.val-target)
            
            self.res.append((dif,node.val))
            self.count+=1
            if self.count>2*k:
                    return 
                
            if node.right:
                tranverse(node.right)
        tranverse(root)
        self.res.sort()
        ans=[]
        for i in range(k):
            ans.append(self.res[i][1])
        return self.ans
        
        
        
#902. Kth Smallest Element in a BST
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the given BST
    @param k: the given k
    @return: the kth smallest element in BST
    """
    def kthSmallest(self, root, k):
        # write your code here  
        self.count=0
        self.res=None
        def tranverse(node):
            
            if self.count <k and node.left:
                tranverse(node.left)
            self.count+=1
            if self.count==k:
                self.res=node.val
                return 
            if self.count <k and node.right:
                tranverse(node.right)
        tranverse(root)
        return self.res
        
#903. Range Addition                
class Solution:
    """
    @param length: the length of the array
    @param updates: update operations
    @return: the modified array after all k operations were executed
    """
    def getModifiedArray(self, length, updates):
        # Write your code here  
        array=[0 for _ in range(length)]   
        add=[0 for _ in range(length+1)]
        
        for start,end,step in updates:
            add[start]+=step
            add[end+1]-=step
        
        array[0]=add[0]
        for i in range(1,length):
            array[i]=array[i-1]+add[i]
        return array
        
#        for start,end,step in updates:
#            for i in range(start,end+1):
#                array[i]+=step
#        return array
length = 5 
updates =[
[1,  3,  2],
[2,  4,  3],
[0,  2, -2]
]
#return [-2, 0, 3, 5, 3]        
if __name__ == "__main__":
    print(Solution().getModifiedArray(length, updates)) 
                        
        
#904. Plus One Linked List        
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: the first Node
    @return: the answer after plus one
    """
    def plusOne(self, head):
        # Write your code here
        def add(node):
            if not node:
                return 1
            carry=add(node.next)
            sm=carry+node.val
            node.val=sm%10
            return sm//10
        
        carry=add(head)
        if carry==1:
            newhead=ListNode(1)
            newhead.next=head
            return newhead
        else:
            return head


#906. Sort Transformed Arra
class Solution:
    """
    @param nums: a sorted array
    @param a: 
    @param b: 
    @param c: 
    @return: a sorted array
    """
    def sortTransformedArray(self, nums, a, b, c):
        # Write your code here
        def cal(x):
            return a*x*x+b*x+c
        
        n=len(nums)
        res=[0 for _ in range(n)]
        
        if a==0 and b==0:
            return [c for _ in range(n)]        
        elif  a==0 and b!=0:
            
            for i in range(n):
                res[i]=cal(nums[i])
            return res if b>0 else res[::-1]
        else:
            left=0
            right=n-1
            if a>0:
                idx=n-1
            else:
                idx=0
                
            while left<=right:
                #print(left,right)
                left_val=cal(nums[left])
                right_val=cal(nums[right])
                #print(left_val,right_val)
                if a>0:
                   
                   if left_val>right_val:
                      res[idx]=left_val
                      left+=1
                      idx-=1
                   else:
                      res[idx]=right_val
                      right-=1
                      idx-=1
                else:
                    
                    if left_val<right_val:
                       res[idx]=left_val
                       left+=1
                       idx+=1
                    else:
                       res[idx]=right_val
                       right-=1
                       idx+=1
                #print(res)
        return res
                    
nums = [-4, -2, 2, 4]
a = 1
b = 3
c = 5# return [3, 9, 15, 33] 

nums = [-4, -2, 2, 4]
a = -1
b = 3
c = 5# return [-23, -5, 1, 7]           
if __name__ == "__main__":
    print(Solution().sortTransformedArray(nums, a, b, c))                     
            
                
#908. Line Reflection
class Solution:
    """
    @param points: n points on a 2D plane
    @return: if there is such a line parallel to y-axis that reflect the given points
    """
    def isReflected(self, points):
        # Write your code here
        
        mi=float('inf')
        ma=float('-inf')
        
        from collections  import defaultdict 
        m=defaultdict(list)
        for point in points:
            mi=min(mi,point[0])
            ma=max(ma,point[0])
            m[point[0]].append(point[1])
        
        y=(mi+ma)/2
        
        #print(m)
        for point in points:
            t=(mi+ma)-point[0]
            #print(point[0],t)
            m[point[0]].sort()
            m[t].sort()
            
            if m[point[0]]!=m[t]:
                return False
        return True
            
            
            
        
                
  
            
points = [[1,1],[-1,1]]# return true
points = [[1,1],[-1,-1]]# return false
points = [[0,0],[1,0]]        
points =[[1,1],[-3,1]]        
if __name__ == "__main__":
    print(Solution().isReflected(points))               
        
        
        
#909. Android Unlock Patterns        
class Solution:
    """
    @param m: an integer
    @param n: an integer
    @return: the total number of unlock patterns of the Android lock screen
    """
    def numberOfPatterns(self, m, n):
        # Write your code here
        visited=[False for _ in range(10)]
#| 1 | 2 | 3 |
#| 4 | 5 | 6 |
#| 7 | 8 | 9 | 
#http://www.cnblogs.com/grandyang/p/5541012.html
        jumps=[[0 for _ in range(10)]  for _ in range(10)]
        
        jumps[1][3]=2
        jumps[3][1]=2
        jumps[4][6]=5
        jumps[6][4]=5
        jumps[7][9]=8
        jumps[9][7]=8
        
        jumps[1][7]=4
        jumps[7][1]=4
        jumps[2][8]=5
        jumps[8][2]=5
        jumps[3][9]=6
        jumps[9][3]=6
              
              
        jumps[1][9]=5
        jumps[9][1]=5
        jumps[3][7]=5
        jumps[7][3]=5
        
        
        def dfs(num,l,res,m,n,visited,jumps):
            if l>=m :
                res+=1
            l+=1
            
            if l>n :
                return res
            visited[num]=True
            for nxt in range(1,10):
                jump=jumps[num][nxt]
                if not visited[nxt]  and ( jump==0 or visited[jump]):
                    res=dfs(nxt,l,res,m,n,visited,jumps)
            visited[num]=False
            return res
        
        ans=0
        ans+=dfs(1,1,0,m,n,visited,jumps)*4
        ans+=dfs(2,1,0,m,n,visited,jumps)*4
        ans+=dfs(5,1,0,m,n,visited,jumps)
        return ans
m=1
n=1

m=1
n=2

if __name__ == "__main__":
    print(Solution().numberOfPatterns( m, n))                       
                
        
#911. Maximum Size Subarray Sum Equals k        
class Solution:
    """
    @param nums: an array
    @param k: a target value
    @return: the maximum length of a subarray that sums to k
    """
    def maxSubArrayLen(self, nums, k):
        # Write your code here   
        from collections import defaultdict
        table=defaultdict(list)
        maxindex=0
        total=0
        table[0]=[-1]
        for i , num in enumerate(nums):
            total+=num
            if total-k in table:
                maxindex=max(maxindex,i-table[total-k][0])
            table[total].append(i)
            #print(i,table, total)
        return maxindex
nums = [1, -1, 5, -2, 3]
k = 3
nums = [-2, -1, 2, 1]
k = 1
if __name__ == "__main__":
    print(Solution().maxSubArrayLen( nums, k))                
        

#912. Best Meeting Point
class Solution:
    """
    @param grid: a 2D grid
    @return: the minimize travel distance
    """
    def minTotalDistance(self, grid):
        # Write your code here

        
#[思路] 二维的等于一维的相加, 一维的最小点必在median点(用反证法可以证明)   
        
        def calculate(array):
            array.sort()
            
            res=0
            
            n=len(array)
            left=0
            right=n-1
            
            while left<right:
                
                res+=array[right]-array[left]
                left+=1
                right-=1
            return res
        if len(grid)==0:
            return 0
        x=[]
        y=[]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]==1:
                    x.append(i)
                    y.append(j)
        return calculate(x)+calculate(y)
grid=[[1,0,0,0,1],[0,0,0,0,0],[0,0,1,0,0]]            
if __name__ == "__main__":
    print(Solution().minTotalDistance(grid))              
            
#913. Flip Game II            
class Solution:
    """
    @param s: the given string
    @return: if the starting player can guarantee a win
    """
    memo={}
    def canWin(self, s):
        # write your code here 
        if s in self.memo:
            return self.memo[s]
        
        for i in range(len(s)-1):
            if s[i:i+2]=='++':
                temp=s[:i]+'--'+s[i+2:]
                flag=self.canWin(temp)
                if not flag:
                    return True
                self.memo[temp]=flag
        return False
s='++++'            
if __name__ == "__main__":
    print(Solution().canWin(s))              
                       
#914. Flip Game            
class Solution:
    """
    @param s: the given string
    @return: all the possible states of the string after one valid move
    """
    def generatePossibleNextMoves(self, s):
        # write your code here
        n=len(s)
        if n<2:
            return []
        
        res=[]
        for i in range(n-1):
            if s[i]=='+'  and s[i+1]=='+':
                res.append(s[:i]+'--'+s[i+2:])
        return res
            
            
            
s = "++++"
[
  "--++",
  "+--+",
  "++--"
]            
if __name__ == "__main__":
    print(Solution().generatePossibleNextMoves( s))              

#915. Inorder Predecessor in BST                                   
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the given BST
    @param p: the given node
    @return: the in-order successor of the given node in the BST
    """
    def inorderSuccessor(self, root, p):
        # write your code here
        self.prev=None
        self.res=False
        def tranverse(node):
            if self.res:
                return 
            
            if node.left:
                tranverse(node.left)
            if node==p:
                self.res=self.prev
                return 
            else:
                self.prev=node
            if node.right:
                tranverse(node.right)
        tranverse(root)
        return self.res
        
        
#916. Palindrome Permutation
class Solution:
    """
    @param s: the given string
    @return: if a permutation of the string could form a palindrome
    """
    def canPermutePalindrome(self, s):
        # write your code here
        from collections import Counter
        
        count=Counter(s)
        
        odd=0
        for v in count.values():
            if v%2==1:
                odd+=1
            if odd==2:
                return False
        return True

s = "code"
s = "aab"
s = "carerac"
if __name__ == "__main__":
    print(Solution().canPermutePalindrome(s))     


#917. Palindrome Permutation II
class Solution:
    """
    @param s: the given string
    @return: all the palindromic permutations (without duplicates) of it
    """
    def generatePalindromes(self, s):
        # write your code here
        
        if not s:
            return []
        
        from collections import Counter
        count=Counter(s)
        
        if len(count)==1:
            return [s]
        odd=''
        string=''
        oddcount=0
        for k,v in count.items():
            if v%2==1:
                odd=k
                oddcount+=1
            string+=k*(v//2)
        
        #print(string)
        if oddcount>1:
            return []
        ans=set()
        
        def permutation(cur,s,odd):
            if not s:
                ans.add(cur[:] +odd+cur[::-1])
                
            for i in range(len(s)):
                permutation(cur+s[i],s[:i]+s[i+1:],odd)
        permutation('',string,odd)
        return list(ans)
            
            
#918. 3Sum Smaller
class Solution:
    """
    @param nums:  an array of n integers
    @param target: a target
    @return: the number of index triplets satisfy the condition nums[i] + nums[j] + nums[k] < target
    """
    def threeSumSmaller(self, nums, target):
        # Write your code here
        
        n=len(nums)
        if n==0:
            return []
        if n<3:
            return []
        
        nums.sort()
        count=0
        
        for i in range(n-2):
            
           start=i+1
           end=n-1
           while start<end:
               total=nums[i]+nums[start]+nums[end]
               if total<target:
                   count+=end-start
                   start+=1
               else:
                   end-=1
        return count
nums = [-2,0,1,3]
target = 2# return 2            
if __name__ == "__main__":
    print(Solution().threeSumSmaller( nums, target))     

        
#919. Meeting Rooms II
"""
Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    """
    @param intervals: an array of meeting time intervals
    @return: the minimum number of conference rooms required
    """
    def minMeetingRooms(self, intervals):
        # Write your code here
        
        def meeting(intervals):
             n=len(intervals)
             if n==0:
                 return 0
             if n==1:
                return 1
             intervals.sort(key=lambda x: (x.start,x.end) )
             newmeeting=[]
             i=0
             j=i+1
             
             while j<n:
                 
                 while j<n and  intervals[i].end>intervals[j].start:
                     newmeeting.append(intervals[j])
                     j+=1
                 i=j
                 j=i+1
             return 1+meeting(newmeeting)
        return meeting(intervals)
intervals=[(0,30),(5,10),(15,20)]   
intervals=[(567707,730827),(166232,719216),(634677,756013),(285191,729059),(237939,915914),(201296,789707),(578258,585240),(164298,218749),(37396,968315),(666722,934674),(742749,824917),(141115,417022),(613457,708897),(343495,994363),(380250,428265),(214441,493643),(588487,811879),(97538,262361)]     
if __name__ == "__main__":
    print(Solution().minMeetingRooms( intervals))     
                
        
#920. Meeting Rooms        
"""
Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    """
    @param intervals: an array of meeting time intervals
    @return: if a person could attend all meetings
    """
    def canAttendMeetings(self, intervals):
        # Write your code here
        
        n=len(intervals)
        if n<2:
            return True
        intervals.sort(key=lambda x: (x.start,x.end) )
        for i in range(n-1):
            if intervals[i].end>intervals[i+1].start:
                return False
        return True
            
        
        
        
intervals = [[0,30],[5,10],[15,20]]        
intervals =  [(465,497),(386,462),(354,380),(134,189),(199,282),(18,104),(499,562),(4,14),(111,129),(292,345)]       
if __name__ == "__main__":
    print(Solution().canAttendMeetings( intervals))     
        
#921. Count Univalue Subtrees
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
    @return: the number of uni-value subtrees.
    """
    def countUnivalSubtrees(self, root):
        # write your code here
        self.count=0
        
        def dfs(node):
            if not node:
                return True
            
            left=dfs(node.left)
            right=dfs(node.right)
            
            if left and right:
                if node.left and node.left.val!=node.val:
                    return False
                if node.right and node.right.val!=node.val:
                    return False
                self.count+=1
                return True
            return False
        dfs(root)
        return self.count
                
                    
#927. Reverse Words in a String II
class Solution:
    """
    @param str: a string
    @return: return a string
    """
    def reverseWords(self, string):
        # write your code here
          return ' '.join(string.split()[::-1])            
        
        
#937. How Many Problem Can I Accept
class Solution:
    """
    @param n: an integer
    @param k: an integer
    @return: how many problem can you accept
    """
    def canAccept(self, n, k):
        # Write your code here   
        target=2*n/k
        import math
        
        start=int(math.floor(target**0.5))
        
        while start*(start+1)<=target:
            start+=1
        return start-1
n = 30
k = 1            
if __name__ == "__main__":
    print(Solution().canAccept( n, k))          
        
        
#941. Sliding Puzzle
class Solution:
    """
    @param board: the given board
    @return:  the least number of moves required so that the state of the board is solved
    """
    def slidingPuzzle(self, board):
        # write your code here 
        cur=''.join([ str(i) for i in  board[0]+board[1]])
        swap={0:(1, 3) ,1:(0, 2, 4),2:(1, 5),3:(0, 4),4:(1, 3, 5),5:(2, 4)}
        from collections import deque
        
        cur_list=deque([cur])
        
        
        visited=set([cur])
        
        
        step=0
        while cur_list:
            #print(cur_list)
            nx_list=deque([])
            for _ in range(len(cur_list)):
                cur=cur_list.popleft()
                if cur=='123450':
                    return step
                
                idx=cur.index('0')
                for pos in swap[idx]:
                    cur_L=list(cur)
                    cur_L[pos],cur_L[idx]=cur_L[idx],cur_L[pos]
                    nx=''.join(cur_L)
                    if nx not in visited:
                        visited.add(nx)
                        nx_list.append(nx)
            cur_list=nx_list
            step+=1
        return -1
        
board=[[1,2,3],[4,0,5]]# return 1 
board = [[1,2,3],[5,4,0]] # return -1     
board = [[4,1,2],[5,0,3]]# return 5        
                               
if __name__ == "__main__":
    print(Solution().slidingPuzzle( board))          
        
                        
#943. Range Sum Query - Immutable            
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.sums=[0 for _ in range(len(nums)+1)]
        for i in range(1,len(nums)+1):
            self.sums[i]=self.sums[i-1]+nums[i-1]
        #print(self.sums)
            
        

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.sums[j+1]-self.sums[i]
        
nums = [-2, 0, 3, -5, 2, -1]

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)   
obj.sumRange(0,2)       
obj.sumRange(2, 5)     
obj.sumRange(0, 5)    
        
       
#944. Maximum Submatrix
class Solution:
    """
    @param matrix: the given matrix
    @return: the largest possible sum
    """
    def maxSubmatrix(self, matrix):
        # write your code here   
        m=len(matrix)
        if m==0:
            return 0
        n=len(matrix[0])
        
        sum1=[[0 for _ in range(n+1)] for _ in range(m+1)]
        
        for i in range(1,m+1):
            for j in range(1,n+1):
                sum1[i][j]=sum1[i-1][j]+matrix[i-1][j-1]
        
        
        maximum=float('-inf')
        for i in range(m):
            for j in range(i+1,m+1):
                cur=0
                for k in range(n+1):
                    cur+=sum1[j][k]-sum1[i][k]
                    maximum=max(maximum,cur)
                    cur=max(0,cur)
        return maximum
                    
                
        
   
matrix = [
[1,3,-1],
[2,3,-2],
[-1,-2,-3]
]  

#[[1,3],
#[2,3]
#]
if __name__ == "__main__":
    print(Solution().maxSubmatrix( matrix)) 

#945. Task Scheduler
class Solution:
    """
    @param tasks: the given char array representing tasks CPU need to do
    @param n: the non-negative cooling interval
    @return: the least number of intervals the CPU will take to finish all the given tasks
    """
    def leastInterval(self, tasks, n):
        # write your code here  
#Imagine task A appeared 4 times, others less than 4. And n=2. 
#You only need to arrange A in the way that doesn't violate the rule first,
#then insert other tasks in any order:
#A - - A - - A - - A
#It's obvious that we need 6 other tasks to fill it. If other tasks are less 6, we need 
#(4 - 1) * (n + 1) + 1 = 10 tasks in total, if other tasks are equal to or more than 6, 
#tasks.length will be our result.
#Now if we have more than one tasks have the same max occurrence, the scheduling will
#look like this:
#A B - A B - A B - A B
#So we only need to modify the formula by replacing 1 with the different amount of tasks 
#that has the max occurrence: (4 - 1) * (n + 1) + taskCountOfMax = 11 
        from collections import Counter
        count=Counter(tasks)
        l=list(count.values())
        l.sort(reverse=True)
 
        i=0
        while i < len(l)-1  and l[i]==l[i+1]:
            i+=1
        return max( len(tasks ) ,  (l[0]-1)*(n+1)+i+1)
tasks = ['A','A','A','B','B','B']
n = 2             
if __name__ == "__main__":
    print(Solution().leastInterval(tasks, n)) 
    
    
#946. 233 Matrix
class Solution:
    """
    @param X: a list of integers
    @param m: an integer
    @return: return an integer
    """
    def calcTheValueOfAnm(self, X, m):
        # write your code here
        n=len(X)
        
        
        A=[[0 for _ in range(m+1)]  for _ in range(n+1)]
        if m>0:
           A[0][1]=233
        for j in range(2,m+1):
            A[0][j]=A[0][j-1]*10+3
        
        for i in range(n):
            A[i+1][0]=X[i]
        
        for j in range(1,m+1):
          for i in range(1,n+1):
              
              A[j][i]=A[j-1][i]+A[j][i-1]% 10000007
        return A[n][m] % 10000007
X=[1]  
m=1  

X=[0,0]
m=2


#（矩阵快速幂）
#https://blog.csdn.net/Fusheng_Yizhao/article/details/79170197
if __name__ == "__main__":
    print(Solution().calcTheValueOfAnm( X, m)) 
    
    
    
    
#947. Matrix Power Series
    
    
    
    
#949. Fibonacci II
class Solution:
    """
    @param n: an integer
    @return: return a string
    """
    def lastFourDigitsOfFn(self, n):
        # write your code here 
        base=[[1,1],[1,0]]
        
        def fast_power(base,n):
            if n==0:
                return [[1,0],[0,1]]
            if n==1:
                return [[1,1],[1,0]]
            
            temp=fast_power(base,n//2)
            temp2=matrix_muti(temp,temp)
            if n%2==0:
                return temp2
            else:
                return matrix_muti(temp2,base)
            
            
        def matrix_muti(a,b):
            row=len(a)
            col=len(a[0])
            res=[[0 for _ in range(col)] for _ in range(row)]
            for i in range(row):
                for j in range(col):
                    for k in range(row):
                        res[i][j]+=a[i][k]*b[k][j]
                        res[i][j]%=10000
            return res
        
        if n==0:
            return '0'
        elif n==1:
            return '0001'
        
        else:
            result=fast_power(base,n)
        
        return '{:04d}'.format( result[0][1])
n=9                        
if __name__ == "__main__":
    print(Solution().lastFourDigitsOfFn( n))        
        
        
        
#954. Insert Delete GetRandom O(1) - Duplicates allowed
class RandomizedCollection(object):

    def __init__(self):
        """
        Initialize your data structure here.
        
        """
        self.map={}
        self.nums=[]
        

    def insert(self, val):
        """
        Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        self.nums.append(val)
        if val in self.map:
            self.map[val].append(len(self.nums)-1)
            return False
        else:
            self.map[val]=[len(self.nums)-1]
            return True
        
            
        

    def remove(self, val):
        """
        Removes a value from the collection. Returns true if the collection contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.map:
            pos=self.map[val].pop()
            if not self.map[val]:
                del self.map[val]
            if pos!=len(self.nums)-1:
                self.map[self.nums[-1] ][-1]=pos
                
                self.nums[pos],self.nums[-1]=self.nums[-1],self.nums[pos]
            self.nums.pop()
            return True
        else:
            return False

    def getRandom(self):
        """
        Get a random element from the collection.
        :rtype: int
        """
        import random
        
        return random.choice(self.nums)
        


# Your RandomizedCollection object will be instantiated and called as such:
# obj = RandomizedCollection()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()   
        
class CircularQueue:
    def __init__(self, n):
        # do intialization if necessary
        
        self.arr=[None for _ in range(n)]
        self.head=0
        self.tail=-1
        self.size=n
    """
    @return:  return true if the array is full
    """
    def isFull(self):
        # write your code here
        return self.head==(self.tail+1)%self.size  and self.arr[self.tail ] is not None

    """
    @return: return true if there is no element in the array
    """
    def isEmpty(self):
        # write your code here
        return not self.arr[self.tail]

    """
    @param element: the element given to be added
    @return: nothing
    """
    def enqueue(self, element):
        # write your code here
        self.tail=(self.tail+1)%self.size
        self.arr[self.tail]=element
        

    """
    @return: pop an element from the queue
    """
    def dequeue(self):
        # write your code here
        ele=self.arr[self.head]
        self.arr[self.head]=None
        self.head=(self.head+1)%self.size
        return ele
        
#960. First Unique Number in a Stream II
from collections import OrderedDict
class DataStream:
    
    def __init__(self):
        # do intialization if necessary
        self.d=OrderedDict()
        
    """
    @param num: next number in stream
    @return: nothing
    """
    def add(self, num):
        # write your code here
        if num in self.d:
            self.d[num]+=1
        else:
            self.d[num]=1

    """
    @return: the first unique number in stream
    """
    def firstUnique(self):
        # write your code here 
        
        for k,v in self.d.items():
            if v==1:
                return k
        
        
        
    
obj=    DataStream()
obj.add(1)
obj.add(2)
obj.firstUnique() #1
obj.add(1)
obj.firstUnique() # 2

#973. 1-bit and 2-bit Characters
class Solution:
    """
    @param bits: a array represented by several bits. 
    @return: whether the last character must be a one-bit character or not
    """
    def isOneBitCharacter(self, bits):
        # Write your code here
        #0,10 , 11
        def decompose(bits):
            n=len(bits)
            if n==1:
                return 
            if n==0:
                return 
            if len(bits)==2:
                if bits[0]==1 and bits[0]==0:
                    self.res=False
                if bits[0]==1 and bits[0]==1:
                    self.res=False
                return 
            
            if bits[0]==0:
                decompose(bits[1:])
            else:
                decompose(bits[2:])
        self.res=True
        if len(bits)==1:
            return True
        decompose(bits)
        return self.res 
bits = [1, 0, 0]
bits = [1, 1, 1, 0]
if __name__ == "__main__":
    print(Solution().isOneBitCharacter( bits))        
                
                
#974. 01 Matrix                    
class Solution:
    """
    @param matrix: a 0-1 matrix
    @return: return a matrix
    """
    def updateMatrix(self, matrix):
        # write your code here        
        from collections import deque
        m=len(matrix)
        n=len(matrix[0])
        
        res=[[0 for _ in range(n)] for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j]==1:
                    visited=set( [( i,j)])
                    q=deque(  [ ( i,j,0)])
                    
                    while q:
                        x,y,step=q.popleft()
                        if matrix[x][y]==0:
                            res[i][j]=step
                            break
                        else:
                            for a, b in (x+1,y),(x-1,y),(x,y+1),(x,y-1):
                                if a>=0 and b>=0 and a<m and b <n  and (a,b) not in visited:
                                    q.append((  a,b,step+1))
                                    visited.add(  (  a,b))
        return res
                                    
 
matrix=[[0,0,0],
[0,1,0],
[1,1,1]
]

#[
#[0,0,0],
#[0,1,0],
#[1,2,1]
#]

if __name__ == "__main__":
    print(Solution().updateMatrix(matrix))        
                


#975. 2 Keys Keyboard
class Solution:
    """
    @param n: The number of 'A'
    @return: the minimum number of steps to get n 'A'
    """
    def minSteps(self, n):
        # Write your code here

#https://leetcode.com/problems/2-keys-keyboard/discuss/105899/Java-DP-Solution
        
        dp=[i for i in range(n+1)]
        
        for i in range(2,n+1):
            for j in range( i-1,1,-1):
                if i%j==0:
                    dp[i]=dp[j]+i//j
                    break
        return dp[n]
n=3
if __name__ == "__main__":
    print(Solution().minSteps( n))        
                

#976. 4Sum II
class Solution:
    """
    @param A: a list
    @param B: a list
    @param C: a list
    @param D: a list
    @return: how many tuples (i, j, k, l) there are such that A[i] + B[j] + C[k] + D[l] is zero
    """
    def fourSumCount(self, A, B, C, D):
        # Write your code here
        
        dictAB={}
        dictCD={}
        
        for i in range(len(A)):
            for j in range(len(B)):
                
                if A[i]+B[j] not in dictAB:
                    dictAB[A[i]+B[j]]=1
                else:
                    dictAB[A[i]+B[j]]+=1
        res=0           
        for k in range(len(C)):
            for l in range(len(D)):
             
                if -(C[k]+D[l]) in dictAB:
                    res+=dictAB[-(C[k]+D[l])]
        return res
                
A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]
if __name__ == "__main__":
    print(Solution().fourSumCount( A, B, C, D))  
    
    
#977. Base 7    
class Solution:
    """
    @param num: the given number
    @return: The base 7 string representation
    """
    def convertToBase7(self, num):
        # Write your code here
        if num<0:
            symbol=-1
        else:
            symbol=1
        res=[]
        num=abs(num)
        if num<7 :
            return symbol*num
        
        else:
            
            while num>0:
                remainder=num%7
                num=num//7
                res.append(str(remainder))
        
        if symbol==-1:
            res.append('-')
        res=res[::-1]
        return ''.join(res)
num = 100
#return "202"
num = -7#
#return "-10"            
if __name__ == "__main__":
    print(Solution().convertToBase7( num))    
    
    
    
#978. Basic Calculator    
class Solution:
    """
    @param s: the given expression
    @return: the result of expression
    """
    def calculate(self, s):
        # Write your code here
        
        res=0
        sign=1
        number=0
        stack=[]
        
        for i,c in enumerate(s):
            if c.isdigit():
                number=number*10+int(c)
            elif c=='+':
                res+=sign*number
                number=0
                sign=1
            elif c=='-':
                res+=sign*number
                number=0
                sign=-1
            elif c=='(':
                
                stack.append(res)
                stack.append(sign)
                sign=1
                res=0
            elif c==')':
                res+=sign*number
                number=0
                res*=stack.pop()
                res+=stack.pop()
               
                
        if number!=0:
            res+=sign*number
        return res 
s='1+1'
s="(1+(4+5+2)-3)+(6+8)"
s=" 2-1 + 2 "
if __name__ == "__main__":
    print(Solution().calculate( s))    
                    
#979. Additive Number                
class Solution:
    """
    @param num: a string
    @return: Is it a valid additive number
    """
    def isAdditiveNumber(self, num):
        # Write your code here
        
        def decompose(s,first,second):
            if s.startswith(  str(first+ second   )):
                if s==str(first+ second   ):
                    return True
                else:
                    l=len(str(first+second))
                    return decompose(s[l:],second,first+second)
            else:
                return False
            
        n=len(num)   
        if n==3:
            return int(num[0])+int(num[1])==int(num[2])
        for i in range(1,n-2):
            if num[:i].startswith('0') and num[:i]!='0':
                continue
            first=int(num[:i])
            for j in range(i+1,n-1):
                if num[i:j].startswith('0') and num[i:j]!='0':
                   continue
                second=int(num[i:j])
                if num[j:].startswith(str(first+second)):
                    
                    if num[j:]==str(first+second):
                            return True
                    else:
                            l=len(str(first+second))
                            
                            if decompose(num[j+l:],second,first+second):
                                return True
        return False
num="112358"   
num="199100199" 
num="123"                        
if __name__ == "__main__":
    print(Solution().isAdditiveNumber( num))        
           
                
                            
#980. Basic Calculator II                   
class Solution:
    """
    @param s: the given expression
    @return: the result of expression
    """
    def calculate(self, s):
        # Write your code here 
        
        number=0
        stack=[]
        sign='+'
        for i, c in enumerate(s):
            
            if c.isdigit():
                number=number*10+int(c)
            if c!=' ' and not c.isdigit()  or i==len(s)-1:
                if sign=='+':
                    stack.append(number)
                elif sign=='-':
                    stack.append(-number)
                elif sign=='*':
                    last=stack.pop()
                    print( last,number)
                    stack.append(last*number)
                elif sign=='/':
                    last=stack.pop()
                    print( last,number)
                    stack.append(   int(last/number))
                number=0
                sign=c
        print(stack)
        
        return sum(stack)
s="3+2*1" #= 7        
s= " 3/2 "
s= "1*2-3/4+5*6-7*8+9/10"       
if __name__ == "__main__":
    print(Solution().calculate( s))        
           
                                
#981. Basic Calculator IV
            
        
#982. Arithmetic Slices
class Solution:
    """
    @param A: an array
    @return: the number of arithmetic slices in the array A.
    """
    def numberOfArithmeticSlices(self, A):
        # Write your code here

        curr=0
        res=0
        for i in range(2,len(A)):
            if A[i]-A[i-1]==A[i-1]-A[i-2]:
                curr+=1
                res+=curr
            else:
                curr=0
        return res
        
        
        



#983. Baseball Game
class Solution:
    """
    @param ops: the list of operations
    @return:  the sum of the points you could get in all the rounds
    """
    def calPoints(self, ops):
        # Write your code here
        res=0
        stack=[]
        
        for op in ops:
            if op.isdigit() or op.startswith('-'):
                stack.append(op)
                res+=int(op)
            elif op=='C':
                c=stack.pop()
                res-=int(c)
            elif op=='D':
                c=stack[-1]
                score=2*int(c)
                res+=score
                stack.append(score)
            elif op=='+':
                res+=int(stack[-1])+int(stack[-2])
                stack.append(int(stack[-1])+int(stack[-2]))
            print(stack)
        return res
ops=["5","2","C","D","+"]
ops=["5","-2","4","C","D","9","+","+"]               
if __name__ == "__main__":
    print(Solution().calPoints(ops))                            
                
            

#984. Arithmetic Slices II - Subsequence            
class Solution:
    """
    @param A: an array
    @return: the number of arithmetic subsequence slices in the array A
    """
    def numberOfArithmeticSlices(self, A):
        # Write your code here   
        from collections import defaultdict
        total=0
        n=len(A)
        if n<3:
            return 0
        dp=[defaultdict(int) for _ in range(n)]
        
        for i in range(n):
            for j in range(i):
                dp[i][A[i]-A[j]]+=1
                if A[i]-A[j] in dp[j]:
                    dp[i][A[i]-A[j]]+=dp[j][A[i]-A[j]]
                    total+=dp[j][A[i]-A[j]]
        return total
                    
A=[2, 4, 6, 8, 10]
if __name__ == "__main__":
    print(Solution().numberOfArithmeticSlices( A))                            
                
#985. Can I Win                    
class Solution:
    """
    @param maxChoosableInteger: a Integer
    @param desiredTotal: a Integer
    @return: if the first player to move can force a win
    """
    def canIWin(self, maxChoosableInteger, desiredTotal):
        # Write your code here  
        
        if sum( range( 1, maxChoosableInteger+1)) <desiredTotal:
            return False
        
        memo={}
        def dfs( nums, desiredTotal   ):
            
            hashn=str(nums)
            
            if nums[-1]>=desiredTotal:
                return True
                
            if hashn in memo:
                return memo[hashn]
            
            for i in range( len( nums)):
                if not dfs( nums[:i]+nums[i+1:], desiredTotal-nums[i]     ):
                    memo[str(nums)]=True
                    return True
            
            memo[str(nums)]=False
            return False
        
        return dfs( list(  range( 1, maxChoosableInteger+1)  ), desiredTotal   )
maxChoosableInteger=10
desiredTotal = 11    
if __name__ == "__main__":
    print(Solution().canIWin( maxChoosableInteger, desiredTotal))        
        
        
        
#986. Battleships in a Board
class Solution:
    """
    @param board: the given 2D board
    @return: the number of battle ships
    """
    def countBattleships(self, board):
        # Write your code here    
        m=len(board)
        if m==0:
            return 0
        n=len(board[0])
        
        self.visited=set()
        
        def dfs(board,i,j):
            
            
            for x,y in (i+1,j), (i-1,j), (i,j+1), (i,j-1):
                if x>=0 and y>=0 and x<m and y<n and board[x][y]=='X' and ( x,y) not in self.visited:
                    self.visited.add(( x,y))
                    dfs(board,x,y)
        res=0
        for i in range(m):
            for j in range(n):
                if board[i][j]=='X' :
                    if (i,j) in self.visited:
                        continue
                    else:
                        res+=1
                        self.visited.add(( i,j))
                        dfs(board,i,j)
        return res
                     
        
board=["X..X",
 "...X",
 "...X"] 
if __name__ == "__main__":
    print(Solution().countBattleships( board))             
        
        
#987. Binary Number with Alternating Bits
class Solution:
    """
    @param n: a postive Integer
    @return: if two adjacent bits will always have different values
    """
    def hasAlternatingBits(self, n):
        # Write your code here
        
        
        last='X'
        
        while n:
            temp=n%2
            if last=='X':
                last=temp
            else:
                if last==temp:
                    return False
                last=temp
            n=n//2
        return True
n=5
n=7
if __name__ == "__main__":
    print(Solution().hasAlternatingBits( n))             
                        
#988. Arranging Coins
class Solution:
    """
    @param n: a non-negative integer
    @return: the total number of full staircase rows that can be formed
    """
    def arrangeCoins(self, n):
        # Write your code here
        
        if n==1:
            return 1
        if n==2:
            return 1
        if n==3:
            return 2
        if n==4:
            return 2
        if n==5:
            return 2
        if n==6:
            return 3
        
        l=0
        r=n
        
        target=2*n
        
        while l+1<r:
            
            mid=(l+r)//2
            
            if mid*(mid+1)==target:
                return mid
            elif mid*(mid+1)>target:
                r=mid
            else:
                l=mid
       
        if l*(l+1)==target:
            return l
        
        if l*(l+1) <target and (l+2)*(l+1)>target:
            return l
        
        
        if l*(l+1) <target and (l+2)*(l+1)<target  and (l+2)*(l+3)>target:
            return l+1
n=5 
n=8
n=100       
if __name__ == "__main__":
    print(Solution().arrangeCoins( n))             
                                    
            
#989. Array Nesting
class Solution:
    """
    @param nums: an array
    @return: the longest length of set S
    """
    def arrayNesting(self, nums):
        # Write your code here   
        self.visited=set()
        self.res=float('-inf')
        
        
        def dfs(i):
            temp=set()
            temp
            
          
            
            while nums[i] not in temp:
                temp.add(nums[i])
                self.visited.add(i)
                i=nums[i]
                
                #print(temp)
            if len(temp)>self.res:
                self.res=len(temp)
            
        for i,num in enumerate(nums):
            if i in self.visited:
                continue
            self.visited.add(num)
            dfs(i)
        return self.res
    
nums = [5,4,0,3,1,6,2]        

if __name__ == "__main__":
    print(Solution().arrayNesting( nums))   

#990. Beautiful Arrangement
class Solution:
    """
    @param N: The number of integers
    @return: The number of beautiful arrangements you can construct
    """
    def countArrangement(self, N):
        # Write your code here
        
        integers=set(range(1,N+1))
        #print(integers)
      
        self.res=0
        
        def place(integers,cur,res):
            
            if cur==N+1:
                print(integers,cur,res)
                self.res+=1
                return 
                
            for integer in integers:
                if cur % integer==0 or integer % cur==0:
                    
                    place(integers-set([integer]),cur+1,res+[integer])
                     
               
        place(integers,1,[]) 
        return self.res
N=3
N=2
N=4
if __name__ == "__main__":
    print(Solution().countArrangement( N))               

#991. Bulb Switcher
class Solution:
    """
    @param n: a Integer
    @return: how many bulbs are on after n rounds
    """
    def bulbSwitch(self, n):
        # Write your code here
        
        table = [0 for _ in range(n)]
        
        for i in range(n):# 第几轮
            for j in range(n): # 第几个
                if (j+1) % (i+1)==0:
                    table[j]=1-table[j]
            print(table)
        return sum(table)
    
        
    
    
    
        return int(n**0.5)
        
n=3        
if __name__ == "__main__":
    print(Solution().bulbSwitch( n))               
    
#992. Beautiful Arrangement II
class Solution:
    """
    @param n: the number of integers
    @param k: the number of distinct integers
    @return: any of answers meet the requirment
    """
    def constructArray(self, n, k):
        # Write your code here
#    i++ j-- i++ j--  i++ i++ i++ ...
#out: 1   9   2   8    3   4   5   6   7
#dif:   8   7   6   5    1   1   1   1  
        
#https://leetcode.com/problems/beautiful-arrangement-ii/discuss/106948/C++-Java-Clean-Code-4-liner 

        res=[]
        l=1
        r=n
        
        if k%2:
            
          i=0
        else:
            i=1
        
        while len(res)!=n:
          if i<k:
            if i%2==0:
                res.append(l)
                l+=1
            else:
                res.append(r)
                r-=1
            i+=1
          else:
              res.append(l)
              l+=1
        return res
n=9
k=5  

n=5
k=2          
if __name__ == "__main__":
    print(Solution().constructArray( n, k))            




#993. Array Partition I
class Solution:
    """
    @param nums: an array
    @return: the sum of min(ai, bi) for all i from 1 to n
    """
    def arrayPairSum(self, nums):
        # Write your code here
        
        nums.sort()
        
        res=0
        
        for i,num in enumerate(nums):
            if i%2==0:
                res+=num
        return res
        
        
#994. Contiguous Array        
class Solution:
    """
    @param nums: a binary array
    @return: the maximum length of a contiguous subarray
    """
    def findMaxLength(self, nums):
        # Write your code here
        n=len(nums)
        nsum={}
        nsum[0]=-1      
        for i in range(n):
            if nums[i]==0:
                nums[i]=-1
                
        res=0
        s=0
   
        for i in range(n):
            s+=nums[i]
            if s in nsum:
                res=max(res,i-nsum[s])
            else:
                nsum[s]=i
        return res
                
            
nums= [0,1,0,1,1,1,0,0]
nums=[0,1]            
if __name__ == "__main__":
    print(Solution().findMaxLength(nums))            
            

#995. Best Time to Buy and Sell Stock with Cooldown
class Solution:
    """
    @param prices: a list of integers
    @return: return a integer
    """
    def maxProfit(self, prices):
        # write your code here
        
        n=len(prices)
        
        if n==0:
            return 0
        sell=[0]*n
        do_nothing=[0]*n
        
        sell[1]=prices[1]-prices[0]
        
        
        for i in range(2,n):
            #sell on i day
            sell[i]=max(sell[i-1],do_nothing[i-2])+prices[i]-prices[i-1]
            do_nothing[i]=max(do_nothing[i-1],sell[i-1])
        return max(do_nothing[n-1],sell[n-1])
prices = [1, 2, 3, 0, 2] 
prices = [3,3,5,0,0,3,1,4]       
if __name__ == "__main__":
    print(Solution().maxProfit( prices))                 
        
        
#1000. Best Time to Buy and Sell Stock with Transaction Fee
class Solution:
    """
    @param prices: a list of integers
    @param fee: a integer
    @return: return a integer
    """
    def maxProfit(self, prices, fee):
        # write your code here
        n=len(prices)
        
        if n==0:
            return 0
        sell=[0]*n
        own=[0]*n
        own[0]=-prices[0]
        
        for i in range(1,n):
            sell[i]=max(sell[i-1],own[i-1]+prices[i]-fee)
            own[i]=max(own[i-1],sell[i-1]-prices[i])
        return sell[n-1]
        
     
prices = [1, 3, 2, 8, 4, 9]
fee = 2
if __name__ == "__main__":
    print(Solution().maxProfit( prices,fee))    


#1001. Asteroid Collision
class Solution:
    """
    @param asteroids: a list of integers
    @return: return a list of integers
    """
    def asteroidCollision(self, asteroids):
        # write your code here
        
#        def collision(ls):
#            n=len(ls)
#            if n==1 or n==0:
#                return ls
#                
#            for i in range(1,n):
#                if ls[i] * ls[i-1]<0:
#                    if  (ls[i-1]>0 and  ls[i]<0) :
#                    
#                      if abs(ls[i])== abs(ls[i-1]):
#                          return collision(ls[:i-1]+ls[i+1::])
#                      elif abs(ls[i])> abs(ls[i-1]):
#                          return collision(ls[:i-1]+ls[i:])
#                      else :
#                          return collision(ls[:i]+ls[i+1::])
#            return ls
#        return collision(asteroids)
        
        
        ls=asteroids
        if True:
            stack=[asteroids[0]]
            
            for i in range(1,len(asteroids)):
                if (stack[-1]>0 and  ls[i]<0):
                    if abs(stack[-1])> abs(ls[i]):
                        pass
                    elif abs(stack[-1])< abs(ls[i]):
                         while stack and (stack[-1]>0 and  ls[i]<0)  and abs(stack[-1])< abs(ls[i]):
                              stack.pop()
                         if not stack:
                             stack.append(ls[i])
                             
                         elif abs(stack[-1])== abs(ls[i]):
                             stack.pop()
                         elif stack[-1]<0:
                             stack.append(ls[i])
                             
                        
                    else:
                        
                        stack.pop()
                else:
                    stack.append(ls[i])
        return stack
            
                
            
            
            
asteroids = [5, 10, -5]
asteroids = [5, 10, -11,-5]
asteroids = [5, 10, -10,-5]
asteroids =[-2,-1,1,2]
if __name__ == "__main__":
    print(Solution().asteroidCollision(asteroids))    

#1002. Bus Routes
class Solution:
    """
    @param routes:  a list of bus routes
    @param S: start
    @param T: destination
    @return: the least number of buses we must take to reach destination
    """
    def numBusesToDestination(self, routes, S, T):
        # Write your code here
        n=len(routes)
        
        if S==T:
            return 0
        
        from collections import defaultdict,deque
        
        graph=defaultdict(set)
        
        routes=list(map(set,routes))
        
        for i,stopi in enumerate(routes):
            for j in range(i+1,n):
                if any( [r  for r in routes[j] if r in stopi]):
                    graph[i].add(j)
                    graph[j].add(i)
        
        seen=set()
        target=set()
        
        for i in range(n):
            if S in routes[i]:
               seen.add(i)
            if T in routes[i]:
               target.add(i)
        q=deque([(r,1) for r in seen])
        
        while q:
            node,depth=q.popleft()
            
            if node in target:
                return depth
            
            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    q.append(( nei,depth+1))
        return -1
            
routes = [[1, 2, 7], [3, 6, 7]]
S = 1
T = 6            
if __name__ == "__main__":
    print(Solution().numBusesToDestination( routes, S, T))                
                
        
 
#1003. Binary Tree Pruning
"""
Definition of TreeNode
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root
    @return: the same tree where every subtree (of the given tree) not containing a 1 has been removed
    """
    def pruneTree(self, root):
        # Write your code here
        
        def delete(node):
            if not node:
                return None
            if not node.left and not node.right:
                if node.val==0:
                    return None
                else:
                    return node
            if node.left:
                left=delete(node.left)
                node.left=left
            if node.right:
                right=delete(node.right)
                node.right=right
            return node
        def equal(node1,node2):
            if not node1 and not node2:
                return True
            if not node1 and node2:
                return False
            if not node2 and node1:
                return False
            if node1.val==node2.val:
                return equal(node1.left,node2.left)  and equal(node1.right,node2.right)
            else:
                return False
            
            
        while  not equal(root,delete(root)):
               root=delete(root)
        return root

#1005. Largest Triangle Area
class Solution:
    """
    @param points: List[List[int]]
    @return: return a double
    """
    def largestTriangleArea(self, points):
        # write your code here
        from itertools import combinations
        
        res=float('-inf')
        
        for i,j,k in combinations(points,3):
            res=max(res, abs(i[0]*j[1]+j[0]*k[1]+k[0]*i[1] - i[1]*j[0]-j[1]*k[0] - k[1]*i[0] ))
        return round(res/2,2)
points = [[0,0],[0,1],[1,0],[0,2],[2,0]]
points =[[1,0],[0,0],[0,1]]
if __name__ == "__main__":
    print(Solution().largestTriangleArea(points))                
                

#1006. Subdomain Visit Count
class Solution:
    """
    @param cpdomains: a list cpdomains of count-paired domains
    @return: a list of count-paired domains
    """
    def subdomainVisits(self, cpdomains):
        # Write your code here
        from collections import defaultdict
        d=defaultdict(int)
        for i in range( len(cpdomains)):
            count,domain=cpdomains[i].split()
            #print(count,domain)
            count=int(count)
            d[domain]+=count
            
            domain=domain+'.'
            
            while domain:
            
                 index=domain.index('.')
                 nxt=domain[index+1:-1]
                 if nxt:
                    d[nxt]+=count
                 domain=domain[index+1:]
        res=[]
        
        for k,v in d.items():
            res.append(str(v)+' '+k)
        return res
cpdomains=["9001 discuss.lintcode.com"]        
cpdomains=["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]        
        
if __name__ == "__main__":
    print(Solution().subdomainVisits( cpdomains))                
                
        
#1007. Chalkboard XOR Game        
class Solution:
    """
    @param nums: a list of integers
    @return: return a boolean
    """
    def xorGame(self, nums):
        # write your code here  
        
        xor=0
        for i in nums:
            xor^=i
        
        return xor==0 or len(nums)%2==0
        
        
        
#1008. Expressive Words
class Solution:
    """
    @param S: a string
    @param words: a list of strings
    @return: return a integer
    """
    def expressiveWords(self, S, words):
        # write your code here 
        def countbegining(word):
            n=len(word)
            for i in range(1,n):
                if word[i]!=word[i-1]:
                    return i
        
        def gettable(S):
            tableS=[]
            S=S+'#'
            while S!='#':
                count=countbegining(S) 
                tableS.append( (S[0] ,count))
                S=S[count:]
            return tableS
        res=0
        Stable=gettable(S)
        n=len(Stable)
        for word in words:
            wtable=gettable(word)
            temp=True
            if len(wtable)==n:
                
                for (Sl,Sc ) , (wl,wc) in zip(Stable,wtable):
                    if not (Sl==wl and (Sc==wc  or ( wc<=2 and  Sc>=3) or (wc >=3 and  Sc>=wc) )):
                        temp=False
            else:
                continue
            if temp:
                res+=1
        return res
                    
            
S = "heeellooo"
words = ["hello", "hi", "helo"]        
if __name__ == "__main__":
    print(Solution().expressiveWords( S, words))                
                        
        
#1010. Max Increase to Keep City Skyline        
class Solution:
    """
    @param grid: a 2D array
    @return: the maximum total sum that the height of the buildings can be increased
    """
    def maxIncreaseKeepingSkyline(self, grid):
        # Write your code here
        
        rowmax=[]
        colmax=[]
        sumgrid=0
        
        for row in grid:
            rowmax.append(max(row))
            sumgrid+=sum(row)
        for x in  zip(*grid):
            colmax.append(max(x))
        
        m=len(grid)
        if m==0:
            return 0
        n=len(grid[0])
        
        aftersum=0
        for i in range(m):
            for j in range(n):
                grid[i][j]= min(rowmax[i],colmax[j])
                aftersum+=grid[i][j]
        return aftersum-sumgrid    
grid = [[3,0,8,4],[2,4,5,7],[9,2,6,3],[0,3,1,0]]        
if __name__ == "__main__":
    print(Solution().maxIncreaseKeepingSkyline( grid))                
                        
#1011. Number of Lines To Write String
class Solution:
    """
    @param widths: an array
    @param S: a string
    @return: how many lines have at least one character from S, and what is the width used by the last such line
    """
    def numberOfLines(self, widths, S):
        # Write your code here
        
        ord('f')-ord('a')
        
        line=0
        curlen=0
        
        for s in S:
            curlen+=widths[ord(s)-ord('a')]
            if curlen>100:
                curlen=widths[ord(s)-ord('a')]
                line+=1
            elif curlen==100:
                line+=1
                last=curlen
                curlen=0
        return [line+1,curlen] if curlen>0  else [line,last]
widths = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
S = "abcdefghijklmnopqrstuvwxyz"                


widths = [4,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
S = "bbbcccdddaaa"
if __name__ == "__main__":
    print(Solution().numberOfLines( widths, S))                
                        

#1013. 独特的摩尔斯编码        
class Solution:
    """
    @param words: the given list of words
    @return: the number of different transformations among all words we have
    """
    def uniqueMorseRepresentations(self, words):
        # Write your code here
        hashset=set()

        dictionary=[".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]    
        
        for word in words:
            temp=''
            for c in word:
                temp+=dictionary[ord(c)-ord('a')]
            hashset.add(temp)
        return len(hashset)
words = ["gin", "zen", "gig", "msg"]
if __name__ == "__main__":
    print(Solution().uniqueMorseRepresentations( words))                     
        
        
#1014. Bricks Falling When Hit        
class Solution:
    """
    @param grid: a grid
    @param hits: some erasures order
    @return: an array representing the number of bricks that will drop after each erasure in sequence
    """
    def hitBricks(self, grid, hits):
        # Write your code here    
        
        m=len(grid)
        n=len(grid[0])
        
        def dfs(i,j):
            if  not ( i>=0 and j>=0 and i<m and j < n  )  or grid[i][j]!=1:
                return 0
            res=1
            grid[i][j]=2
            
            res+= sum( dfs(x,y) for x ,y in  ((i+1,j),  (i-1,j),(i,j+1),(i,j-1)))   
            
            return res
        def is_connect(i,j):
            if i==0 or any( [x >=0 and y>=0 and x <m and y < n and grid[x][y]==2 for x,y in ((i+1,j),  (i-1,j),(i,j+1),(i,j-1)) ]):
                return True
            return False
        
        for i,j in hits:
            grid[i][j]-=1
        
        for i in range(n):
            dfs(0,i)
        
        res=[0 for _ in range( len(hits))]
        
        for k in range( len(hits)-1,-1,-1):
            
            x,y=hits[k]
            grid[x][y]+=1
            
            if grid[x][y]==1 and is_connect(x,y):
                res[k]=dfs(x,y)-1
        return res
            
        
grid = [[1,0,0,0],
        [1,1,1,0]]
hits = [[1,0]]
#Output: [2]   

grid = [[1,0,0,0],[1,1,0,0]]
hits = [[1,1],[1,0]]     
if __name__ == "__main__":
    print(Solution().hitBricks( grid, hits))                     
                
        
#1015. Find Eventual Safe States
class Solution:
    """
    @param graph: a 2D integers array
    @return: return a list of integers
    """
    def eventualSafeNodes(self, graph):
        # write your code here
        n=len(graph)
        out_degree=[0 for _ in range(n)]
        from collections import defaultdict,deque
        in_node=defaultdict(list)
        
        
        queue=[]
        
        for i in range(n):
            out_degree[i]=len(graph[i])
            if out_degree[i]==0:
                queue.append(i)
            for j in graph[i]:
                in_node[j].append(i)
        for term_node in queue:
            for innode in in_node[term_node]:
                out_degree[innode]-=1
                if out_degree[innode]==0:
                    queue.append(innode)
        return list(sorted(queue))
graph = [[1,2],[2,3],[5],[0],[5],[],[]]
if __name__ == "__main__":
    print(Solution().eventualSafeNodes( graph))                     
                        
#1016. Minimum Swaps To Make Sequences Increasing        
class Solution:
    """
    @param A: an array
    @param B: an array
    @return: the minimum number of swaps to make both sequences strictly increasing
    """
    def minSwap(self, A, B):
        # Write your code here
        n=len(A)
        swap=[n]*n
        not_swap=[n]*n
        if n==1:
            return 0
        swap[0]=1
        not_swap[0]=0
        
        for i in range(1,n):
            if  A[i-1]<A[i]  and B[i-1]<B[i]:
                not_swap[i]=not_swap[i-1]
                swap[i]=swap[i-1]+1
            if  A[i-1]<B[i]  and B[i-1]<A[i]:
                not_swap[i]=min( not_swap[i],swap[i-1])
                swap[i]=min( swap[i],not_swap[i-1]+1)
        return min(swap[-1],    not_swap[-1])
        
A = [1,3,5,4]
B = [1,2,3,7]    
if __name__ == "__main__":
    print(Solution().minSwap( A, B))                  
                
#1017. Similar RGB Color
class Solution:
    """
    @param color: the given color
    @return: a 7 character color that is most similar to the given color
    """
    def similarRGB(self, color):
        # Write your code here 
#很容易发现 shorthand color 就是 RGB 都可以被17整除的颜色。
#
#所以只需要分别对 RGB 部分除以 17 取整，就可以得到对应的值，然后格式化为十六进制即可    
        red=       int(color[1:3],16 )
        green=       int(color[3:5],16 )
        blue=       int(color[5:7],16 )
        
        r=round(red/17)*17
        g=round(green/17)*17
        b=round(blue/17)*17
        
        return '#{:02x}{:02x}{:02x}'.format(r,g,b)
color = "#09f166"
if __name__ == "__main__":
    print(Solution().similarRGB( color))         

        
#1018. Champagne Tower
class Solution:
    """
    @param poured: an integer
    @param query_row: an integer
    @param query_glass: an integer
    @return: return a double
    """
    def champagneTower(self, poured, query_row, query_glass):
        # write your code here
        dp=[[0.00 for _ in range(i)]  for i in range(1,query_row+2)]
        dp[0][0]=poured
        
        for i in range(query_row):
            for j in range(i+1):
                if dp[i][j]>1:
                   dp[i+1][j]+=(dp[i][j]-1)/2.0
                   dp[i+1][j+1]+=(dp[i][j]-1)/2.0
        return round(dp[query_row][query_glass] if dp[query_row][query_glass]<=1 else 1.00,2)


        
#1019. Smallest Rotation with Highest Score        
class Solution:
    """
    @param A: an array
    @return: the smallest index K that corresponds to the highest score we could receive
    """
    def bestRotation(self, A):
        # Write your code here    
#        table={}
#        n=len(A)
#        
#        for i, a in enumerate(A):
#            table[i]=a
#            
#        res=float('-inf')
#        resk=0
#        
#        
#        for i in range(n):
#            temp=0
#            for idx,v in table.items():
#                if v<= (idx-i)%n:
#                    temp+=1
#            if temp>res:
#                res=temp
#                resk=i
#            
#        return resk
#https://leetcode.com/problems/smallest-rotation-with-highest-score/discuss/118725/C++JavaPython-Solution-with-Explanation        
        n=len(A)
        change=[1]*n
        
        for i in range(n):
            change[(i-A[i]+n+1)%n]-=1
        for i in range(1,n):
            change[i]+=change[i-1]
            
        return change.index(max(change))
            
        
        
        
        
A=[2, 3, 1, 4, 0]
A=[1, 3, 0, 2, 4]
if __name__ == "__main__":
    print(Solution().bestRotation( A))             


#1020. All Paths From Source to Target
class Solution:
    """
    @param graph: a 2D array
    @return: all possible paths from node 0 to node N-1
    """
    def allPathsSourceTarget(self, graph):
        # Write your code here
        self.res=[]
        
        def connecting(graph,cur,path,visited):
            if cur==len(graph)-1:
                self.res.append(path[:])
            
            for nx in graph[cur]:
                if nx not in visited:
                    connecting(graph,nx,path+[nx],visited|set([nx]))
        connecting(graph,0,[0],set([0]))
            
        return self.res
graph=[[1,2], [3], [3], []]         
if __name__ == "__main__":
    print(Solution().allPathsSourceTarget( graph))             
        
        
#1021. Number of Subarrays with Bounded Maximum
class Solution:
    """
    @param A: an array
    @param L: an integer
    @param R: an integer
    @return: the number of subarrays such that the value of the maximum array element in that subarray is at least L and at most R
    """
    def numSubarrayBoundedMax(self, A, L, R):
        # Write your code here  
#https://leetcode.com/problems/number-of-subarrays-with-bounded-maximum/discuss/117595/Short-Java-O(n)-Solution
        n=len(A)
        left=0
        count=0
        res=0
        for right in range(n):
            if A[right]>=L and A[right]<=R:
                count=right-left+1
                res+=count
            elif A[right]<L:
                res+=count
            else:
                left=right+1
                count=0
        return res
A = [2, 1, 4, 3]     
L=2
R=3  
A =[2,9,2,5,6]
L=2
R=8 
if __name__ == "__main__":
    print(Solution().numSubarrayBoundedMax( A, L, R))             
        

#1022. Valid Tic-Tac-Toe State
class Solution:
    """
    @param board: the given board
    @return: True if and only if it is possible to reach this board position during the course of a valid tic-tac-toe game
    """
    def validTicTacToe(self, board):
        # Write your code
        
        def isWin(board,c):
            for i in range(3):
                if board[i]==c*3:
                    return True
            
            for i in range(3):
                if board[0][i]==c and board[1][i]==c and board[2][i]==c:
                    
                    return True
            
            if board[0][0] ==c and board[1][1] ==c and board[2][2] ==c:
                return True
            if board[0][2] ==c and board[1][1] ==c and board[2][0] ==c:
                return True
            return False
        
        
        count_x=0
        count_o=0
        
        for i in range(3):
            for j in range(3):
                if board[i][j]=='X':
                    count_x+=1
                if board[i][j]=='O':
                    count_o+=1
                    
        if count_o>count_x or count_x>count_o+1:
                    return False
                
#                if ( isWin(board,'X')  and count_x >= count_o) or ( isWin(board,'O')  and count_x > count_o):
#                    return False
        if  count_x==count_o  and isWin(board,'X')  or isWin(board,'O')  and count_x==count_o+1:
                    return False
        return True
board = ["O  ", "   ", "   "]
board = ["XOX", " X ", "   "]
board = ["XXX", "   ", "OOO"]
board = ["XOX", "O O", "XOX"]            
if __name__ == "__main__":
    print(Solution().validTicTacToe( board))                 
                    
#1023. Preimage Size of Factorial Zeroes Function                
class Solution:
    """
    @param K: an integer
    @return: how many non-negative integers x have the property that f(x) = K
    """
    def preimageSizeFZF(self, K):
        # Write your code here
        def nzero(n):
            f=5
            count=0
            while f<=n:
                count+=n//f
                f*=5
            return count
        
        if K==0:
            return 5
        
        l=0
        r=K*5
        
        while l<r:
            mid=(l+r)//2
            if nzero(mid)<K:
                l=mid+1
            else:
                r=mid
        
        
        if nzero(l)!=K:
            return 0
        else:
            return 5
K = 5            
if __name__ == "__main__":
    print(Solution().preimageSizeFZF( K))                       
                
        
#1024. Number of Matching Subsequences
        
class Solution:
    """
    @param S: a string
    @param words: a dictionary of words
    @return: the number of words[i] that is a subsequence of S
    """
    def numMatchingSubseq(self, S, words):
        # Write your code here  
        from collections import defaultdict
        waiting=defaultdict(list)
        
        for it in map(iter,words):
            waiting[next(it,())].append(it)
        
        for c in S:
            for it in waiting.pop(c,()):
                waiting[next(it,None)].append(it)
        return len(waiting[None])


#1026. Domino and Tromino Tiling
class Solution:
    """
    @param N: a integer
    @return: return a integer
    """
    def numTilings(self, N):
        # write your code here

#https://s3-lc-upload.s3.amazonaws.com/users/yuweiming70/image_1519549786.png
#https://leetcode.com/problems/domino-and-tromino-tiling/discuss/116506/Python-recursive-DP-solution-with-cache-w-Explanation
       
        cacheD={}
        cacheT={}
        
        def tilingD(N):
            if N in cacheD:
                return cacheD[N]
            if N==0:
                return 1
            if N==1:
                return 1
            if N==2:
                return 2
            
            cacheD[N]=tilingD(N-1)+tilingD(N-2)+2*tilingT(N-1)
            return cacheD[N]
        
        def tilingT(N):
            if N in cacheT:
                return cacheT[N]
            if N==0:
                return 1
            if N==1:
                return 0
            if N==2:
                return 1
            
            cacheT[N]=tilingD(N-2)+tilingT(N-1)
            return cacheT[N]
        
        return tilingD(N)

N=3
if __name__ == "__main__":
    print(Solution().numTilings( N))               

#1027. Escape The Ghosts
class Solution:
    """
    @param ghosts: a 2D integer array
    @param target: a integer array
    @return: return boolean
    """
    def escapeGhosts(self, ghosts, target):
        # write your code here
        
        distance=abs(target[0])+abs(target[1])
        
        for r, c in ghosts:
            if abs(r-target[0])+abs(c-target[1])<=distance:
                return False
        return True
        


#1028. Rotated Digits
class Solution:
    """
    @param N: a positive number
    @return: how many numbers X from 1 to N are good
    """
    def rotatedDigits(self, N):
        # write your code here
        
        res=0
        for i in range(2,N+1):
            ns=str(i)
            temp=''
            for c in ns:
               if c =='0'  or c=='1' or c=='8':
                   temp+=c
               elif c=='6':
                   temp+='9'
               elif c=='9':
                   temp+='6'
               elif c=='2':
                   temp+='5'
               elif c=='5':
                   temp+='2'
               else:
                   break
            if temp!=ns and len(temp)==len(ns):
                   print(ns,temp)
                   res+=1
        return res
                   
N=10                   
if __name__ == "__main__":
    print(Solution().rotatedDigits( N))                    
                   
#1029. Cheapest Flights Within K Stops                   
class Solution:
    """
    @param n: a integer
    @param flights: a 2D array
    @param src: a integer
    @param dst: a integer
    @param K: a integer
    @return: return a integer
    """
    def findCheapestPrice(self, n, flights, src, dst, K):
        # write your code here
        
        from collections import defaultdict,deque
        
        graph=defaultdict(dict)
        
        for i,j,v in flights:
            graph[i][j]=v
        
        q=deque([(src,0)])
        res=float('inf')
        table={}
        while  q and K>=-1: 
          temp=deque()
          for _ in range(len(q)):
              cur,price=q.popleft()
              if cur==dst:
                if price <res:
                    res=price
                continue
              if cur not in table:
                  table[cur]=price
              elif table[cur] < price:
                   continue
              for nx in graph[cur]:
                 if nx not in table or table[nx]>price+graph[cur][nx]:
                     temp.append((nx,price+graph[cur][nx]))
          q=temp
          #print(q,res)
          K-=1
        return res if res<float('inf') else -1
n = 3
flights = [[0,1,100],[1,2,100],[0,2,500]]
src = 0
dst = 2
K = 1

n = 3
flights = [[0,1,100],[1,2,100],[0,2,500]]
src = 0
dst = 2
K = 0
if __name__ == "__main__":
    print(Solution().findCheapestPrice( n, flights, src, dst, K)) 



#1029. Cheapest Flights Within K Stops                   
class Solution:
    """
    @param n: a integer
    @param flights: a 2D array
    @param src: a integer
    @param dst: a integer
    @param K: a integer
    @return: return a integer
    """
    def findCheapestPrice(self, n, flights, src, dst, K):
        # write your code here
        
        from collections import defaultdict
        import heapq
        graph=defaultdict(list)
        
        for i,j,v in flights:
            graph[i].append(( v,j ))
        if src not in graph:
            return -1
            
        hq=[]
        
        for cost, stop in graph[src]:
            heapq.heappush(hq,( cost,stop,0      ))
            
            
        while hq:
            culcost,cur,level=heapq.heappop(hq)
            if level>K:
                continue
            if cur==dst:
                return culcost
            
            if cur in graph:
                for cost,nx in graph[cur]:
            
                    heapq.heappush( hq, ( cost+ culcost,nx,level+1        ))
        return -1
n = 3
flights = [[0,1,100],[1,2,100],[0,2,500]]
src = 0
dst = 2
K = 1

n = 3
flights = [[0,1,100],[1,2,100],[0,2,500]]
src = 0
dst = 2
K = 0
if __name__ == "__main__":
    print(Solution().findCheapestPrice( n, flights, src, dst, K)) 

#1030. K-th Smallest Prime Fraction
class Solution:
    """
    @param A: a list of integers
    @param K: a integer
    @return: return two integers
    """
    def kthSmallestPrimeFraction(self, A, K):
        # write your code here
        import bisect
        l=0
        r=1
        N=len(A)
        
        while True:
            
            m=(l+r)/2
            border=[bisect.bisect_left(A,A[i]/m) for i in range(N)]
            cur=sum(N-i for i in border)
            if cur>K:
                r=m
            elif cur<K:
                l=m
            else:
                return max( [(A[i],A[j]) for i, j in enumerate(border) if j<N] ,key=lambda x: x[0]/x[1])
 
A = [1, 2, 3, 5]
K=3
if __name__ == "__main__":
    print(Solution().kthSmallestPrimeFraction( A, K))


#1031. Is Graph Bipartite?
class Solution:
    """
    @param graph: the given undirected graph
    @return:  return true if and only if it is bipartite
    """
    def isBipartite(self, graph):
        # Write your code here
        set1=set()
        set2=set()
        for node , neiList in enumerate(graph):
            if node not in set1 and node not in set2:
                for nei in neiList:
                    if nei in set1:
                        set2.add(node)
                        
                    else:
                        set1.add(node)
                
            if node in set1:
                for  nei in neiList:
                    if nei in set1:
                        return False
                    set2.add(nei)
            if node in set2:
                for  nei in neiList:
                    if nei in set2:
                        return False
                    set1.add(nei)
        return True
graph=[[1,3], [0,2], [1,3], [0,2]]
graph=[[1,2,3], [0,2], [0,1,3], [0,2]]
               
                   
if __name__ == "__main__":
    print(Solution().isBipartite( graph))
        
        

#1032. Letter Case Permutation
class Solution:
    """
    @param S: a string
    @return: return a list of strings
    """
    def letterCasePermutation(self, S):
        # write your code here
        res=[]
        
        def adding(cur,S):
            if not S:
                res.append(cur)
                return 
            
            c=S[0]
            if c.isdigit():
               adding(cur+c,S[1:])
            else:
               adding(cur+c.lower(),S[1:]) 
               adding(cur+c.upper(),S[1:]) 
        adding('',S)
        return res
S = "a1b2"
S = "3z4"
S = "12345"
if __name__ == "__main__":
    print(Solution().letterCasePermutation(S))
        


#1033. Minimum Difference Between BST Nodes
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root
    @return: the minimum difference between the values of any two different nodes in the tree
    """
    def minDiffInBST(self, root):
        # Write your code here
        self.pre=None
        self.res=float('inf')
        def inorder(node):
            if not node:
                return 
            
            if node.left:
                inorder(node.left)
            if self.pre and abs(self.pre-node.val) <self.res:
                self.res=abs(self.pre-node.val)
                
            self.pre=node.val
            
            if node.right:
                inorder(node.right)
        
                
        inorder(root)
        return self.res
                
        
        
        
#1034. Transform to Chessboard
class Solution:
    """
    @param board: the given board
    @return: the minimum number of moves to transform the board into a "chessboard"
    """
    def movesToChessboard(self, board):
        # Write your code here
#https://leetcode.com/problems/transform-to-chessboard/discuss/114847/Easy-and-Concise-Solution-with-Explanation-C++JavaPython  
        rowswap=0
        colswap=0
        n=len(board)
        if any(board[0][0]^board[i][0]^board[0][j]^board[i][j]  for i in range(n) for j in range(n)):
            return -1
        if  not n//2<=sum(board[0])<=(n+1)//2:
            return -1
        if not n//2<=sum(board[j][0] for j in range(n))<=(n+1)//2:
            return -1
        
        for i in range(n):
            if board[0][i]==(i%2):
                rowswap+=1
            if board[i][0]==(i%2):
                colswap+=1
        
        
        if n%2:
            if colswap%2:
                colswap=n-colswap
            if rowswap%2:
                rowswap=n-rowswap
        else:
            colswap=min(n-colswap,colswap)
            rowswap=min(n-rowswap,rowswap)
        return  (colswap+rowswap)//2
        
        
#1035. Rabbits in Forest        
class Solution:
    """
    @param answers: some subset of rabbits (possibly all of them) tell 
    @return: the minimum number of rabbits that could be in the forest.
    """
    def numRabbits(self, answers):
        # write your code here   
#来看一个比较tricky的例子，[0, 0, 1, 1, 1]，前两只兔子都说森林里没有兔子和其颜色相同了，
#那么这两只兔子就是森林里独一无二的兔子，且颜色并不相同，所以目前已经确定了两只。
#然后后面三只都说森林里还有一只兔子和其颜色相同，那么这三只兔子就不可能颜色都相同了，
#但我们可以让两只颜色相同，另外一只颜色不同，那么就是说还有一只兔子并没有在数组中，
#所以森林中最少有6只兔子。分析完了这几个例子，我们可以发现，如果某个兔子回答的数字是x，
#那么说明森林里共有x+1个相同颜色的兔子，我们最多允许x+1个兔子同时回答x个，一旦超过了x+1个兔子，
#那么就得再增加了x+1个新兔子了。所以我们可以使用一个HashMap来建立某种颜色兔子的总个数和在数组中还
#允许出现的个数之间的映射，然后我们遍历数组中的每个兔子，如果该兔子回答了x个，若该颜色兔子的总个
#数x+1不在HashMap中，或者映射为0了，我们将这x+1个兔子加入结果res中，然后将其映射值设为x，
#表示在数组中还允许出现x个也回答x的兔子；否则的话，将映射值自减1即可，参见代码如下： 
        from collections import defaultdict
        dd =defaultdict(int)
        
        res=0
        
        for num in answers:
            if num+1 not in dd or dd[num+1]==0:
                dd[num+1]=num
                res+=num+1
            else:
                dd[num+1]-=1
        return res
                
        
        
#1036. Reaching Points
class Solution:
    """
    @param sx: x for starting point
    @param sy: y for starting point
    @param tx: x for target point 
    @param ty: y for target point
    @return: if a sequence of moves exists to transform the point (sx, sy) to (tx, ty)
    """
    def reachingPoints(self, sx, sy, tx, ty):
        # write your code here
        if sx==tx and sy==ty:
            return True
        if sx>=tx or sy>=ty:
            return False
        def move(i,j):
            
            if i==tx and j==ty:
                
                return True
            if (i == tx and  j < ty ) :
                if move(i,i+j):
                    return True
            elif (i < tx and  j == ty ):
                if move(i+j,j):
                    return True
            elif (i < tx and  j < ty ):
                if move(i,i+j) or  move(i+j,j):
                    return True
            return False
        
        return move(sx,sy)
sx = 1
sy = 1
tx = 3
ty = 5  
 
sx = 1
sy = 1
tx = 2
ty = 2 

sx = 1
sy = 1
tx = 1
ty = 1        
if __name__ == "__main__":
    print(Solution().reachingPoints(sx, sy, tx, ty))
        

#1036. Reaching Points
class Solution:
    """
    @param sx: x for starting point
    @param sy: y for starting point
    @param tx: x for target point 
    @param ty: y for target point
    @return: if a sequence of moves exists to transform the point (sx, sy) to (tx, ty)
    """
    def reachingPoints(self, sx, sy, tx, ty):
        # write your code here
        if sx==tx and sy==ty:
            return True
        if sx>=tx or sy>=ty:
            return False
        
        while sx<tx  and sy<ty:
            tx,ty=tx%ty,ty%tx
           
        if sx==tx and (ty-sy)%sx==0:
            return True
        elif sy==ty and (tx-sx)%sy==0:
            return True
        else:
            return False
    
sx = 1
sy = 1
tx = 3
ty = 5  
 
sx = 1
sy = 1
tx = 2
ty = 2 

sx = 1
sy = 1
tx = 1
ty = 1        
if __name__ == "__main__":
    print(Solution().reachingPoints(sx, sy, tx, ty))
        

#1037. Global and Local Inversions
class Solution:
    """
    @param A: an array
    @return: is the number of global inversions is equal to the number of local inversions
    """
    def isIdealPermutation(self, A):
        # Write your code here
        n=len(A)
        
        if n==1:
            return True
        if n==2:
            return True
        
        leftmost=float('-inf')
        
        for i in range(n-2):
            leftmost=max(leftmost,A[i])
            if leftmost>A[i+2]:
                return False
        return True
        

#1038. Jewels And Stones
class Solution:
    """
    @param J: the types of stones that are jewels
    @param S: representing the stones you have
    @return: how many of the stones you have are also jewels
    """
    def numJewelsInStones(self, J, S):
        # Write your code here
        
        return sum(1  for s in S if s in J)
J = "aA"
S = "aAAbbbb"
if __name__ == "__main__":
    print(Solution().numJewelsInStones(J, S))
    
    
    
#1039. Max Chunks To Make Sorted
class Solution:
    """
    @param arr: a permutation of N
    @return: the most number of chunks
    """
    def maxChunksToSorted(self, arr):
        # write your code here
        temp=sorted(arr)
        d={}
        
        for i ,x in enumerate(temp):
            if i==0:
                d[x]=i
            elif temp[i]==temp[i-1]:
                d[x]=d[temp[i-1]]
            else:
                d[x]=i
        
        arr=[d[x]+1  for x in arr]
        res=0
        leftmax=arr[0]
        
        
        
        for i,x in enumerate(arr):
            leftmax=max(leftmax,x)
            if leftmax<=i+1:
                res+=1
        return res




arr = [4,3,2,1,0]
arr = [1,0,2,3,4]

if __name__ == "__main__":
    print(Solution().maxChunksToSorted(arr))
    

#1040. Max Chunks To Make Sorted II
class Solution:
    """
    @param arr: an array of integers
    @return: number of chunks
    """
    def maxChunksToSorted(self, arr):
        # Write your code here
        
        temp=sorted(arr)
        d={}
        
        for i ,x in enumerate(temp):
            if i==0:
                d[x]=i
            elif temp[i]==temp[i-1]:
                d[x]=d[temp[i-1]]
            else:
                d[x]=i
        
        arr=[d[x]+1  for x in arr]
        res=0
        leftmax=arr[0]
        
        
        
        for i,x in enumerate(arr):
            leftmax=max(leftmax,x)
            if leftmax<=i+1:
                res+=1
        return res
arr=[5,4,3,2,1]   

arr = [2,1,3,4,4]

arr =[27060055,4149524,8328754,79457994,30081343,11566671,30491837,71497332,16256213,93710529,93175212,3693641,44884302,48127487,161953,92240077,28000502,31990617,76344525,76896676,49059253,28358998,98536606,60361334,94956361,55843085,25273835,26664886,58306390,96119848,88024294,96932082,74856864,13303607,54392395,28161577,5435331,37308156,37841081,30070833,71605463,29887588,65756334,33143104,59246664,38742699,90550314,91628578,51532923,8980040,4022527,63178924,40854697,75390619,96878630,46200002,86125370,18706146,49430368,63088205,49029921,52728510,36694190,73193003,81749342,22331904]     

if __name__ == "__main__":
    print(Solution().maxChunksToSorted( arr))   

#1042. Toeplitz Matrix
class Solution:
    """
    @param matrix: the given matrix
    @return: True if and only if the matrix is Toeplitz
    """
    def isToeplitzMatrix(self, matrix):
        # Write your code here
        
        m=len(matrix)
        n=len(matrix[0])
        def isTrue(i,j):
            
            if i+1 > m-1 or j+1 >n-1:
                return True
            elif matrix[i][j] !=  matrix[i+1][j+1] :
                return False
            else:
                
               return isTrue(i+1,j+1)
           
        for i in range(m):
            if not isTrue(i,0):
                return False
        for j in range(m):
            if not isTrue(0,j):
                return False
        return True
        
matrix = [[1,2,3,4],[5,1,2,3],[9,5,1,2]]
matrix = [[1,2],[2,2]]
    
if __name__ == "__main__":
    print(Solution().isToeplitzMatrix( matrix))       
    
    
    
    
            
#1043. Couples Holding Hands       
class Solution:
    """
    @param row: the couples' initial seating
    @return: the minimum number of swaps
    """
    def minSwapsCouples(self, row):
        # Write your code here
        
        swap=0
        d={ x:i for i,x in enumerate(row) }
        
        for i,x in enumerate(row):
            partner=x^1
            
            j=d[partner]
            
            if abs(i-j)>1:
                
                row[i+1],row[j]=row[j],row[i+1]
                d[row[i+1]]=i+1
                d[row[j]]=j
                swap+=1
        return swap
row = [0, 2, 1, 3]    
if __name__ == "__main__":
    print(Solution().minSwapsCouples( row))
                    

        
        
        
#1044. Largest Plus Sign
class Solution:
    """
    @param N: size of 2D grid
    @param mines: in the given list
    @return: the order of the plus sign
    """
    def orderOfLargestPlusSign(self, N, mines):
        # Write your code here
        banned=set((x,y) for x, y in mines)    
        
        dp=[[0 for _ in range(N)]  for _ in range(N)]
        for r in range(N):
            count=0
            for c in range(N):
                count=0 if (r,c) in banned else count+1
                
                dp[r][c]=count
            count=0
            for c in range(N-1,-1,-1):
                count=0 if (r,c) in banned else count+1
                if count<dp[r][c]:
                    dp[r][c]=count
        
        ans=0
        for c in range(N):
            count=0
            
            for r in range(N):
                count=0 if (r,c) in banned else count+1
                if count<dp[r][c]:
                    dp[r][c]=count
            count=0
            
            for r in range(N-1,-1,-1):
                count=0 if (r,c) in banned else count+1
                if count<dp[r][c]:
                    dp[r][c]=count
                if ans<dp[r][c]:
                    ans=dp[r][c]
        return ans
                    
  
        
N = 5
mines = [[4, 2]]     

if __name__ == "__main__":
    print(Solution().orderOfLargestPlusSign( N, mines))   
        
        
#1045. Partition Labels
class Solution:
    """
    @param S: a string
    @return: a list of integers representing the size of these parts
    """
    def partitionLabels(self, S):
        # Write your code here
        res=[]
        
        for i in range(len(S)):
           if not  any(True for x in set(S[:i])  if x in set(S[i:]) ):
               res.append(i)
        res.append(len(S))
        return [res[i]-res[i-1]  for i in range(1,len(res))]
S = "ababcbacadefegdehijhklij"               
if __name__ == "__main__":
    print(Solution().partitionLabels( S))   
    
class Solution:
    """
    @param S: a string
    @return: a list of integers representing the size of these parts
    """
    def partitionLabels(self, S):
        # Write your code here
        location={c:i for i,c in enumerate(S)}
        left=0
        right=0
        res=[]
        for i in range(len(S)):
            
            right=max(right, location[S[i]])
            if i==right:
                res.append(right-left+1)
                left=i+1
        return res
        
S = "ababcbacadefegdehijhklij"               
if __name__ == "__main__":
    print(Solution().partitionLabels( S))   
                
        
#1046. Prime Number of Set Bits in Binary Representation        
class Solution:
    """
    @param L: an integer
    @param R: an integer
    @return: the count of numbers in the range [L, R] having a prime number of set bits in their binary representation
    """
    def countPrimeSetBits(self, L, R):
        # Write your code here
        
        def isPrime(y):
            if y==1:
                return False
            if y==2 or y==3 or y==5 or y==7:
                return True
            
            if y==4 or y==6 or y==8 or y==9:
                return False
            
            if y%2==0:
                return False
            
            for i in range(2,y//2):
                if y%i==0:
                    return False
            return True
        
        res=0
        for x in range(L,R+1):
          n=bin(x).count('1') 
          if isPrime(n):
              res+=1
        return res
L = 6
R = 10    
L = 10
R = 15          
if __name__ == "__main__":
    print(Solution().countPrimeSetBits( L, R))           


        
            
#1047. Special Binary String        
class Solution:
    """
    @param S: a string
    @return: return a string
    """
    def makeLargestSpecial(self, S):
        # write your code here
#https://leetcode.com/problems/special-binary-string/discuss/113211/Easy-and-Concise-Solution-with-Explanation-C++JavaPython
        res=[]
        i=0
        count=0        
        for j , v in enumerate(S):
            if v=='1':
                count+=1
            else:
                count-=1
            if count==0:
                
               res.append('1'+ self.makeLargestSpecial( S[i+1:j])+'0'     )
               i=j+1
        return ''.join(sorted(res)[::-1])
S = "11011000"            
if __name__ == "__main__":
    print(Solution().makeLargestSpecial( S))                   

#1048. Set Intersection Size At Least Two
class Solution:
    """
    @param intervals: List[List[int]]
    @return: return an integer
    """
    def intersectionSizeTwo(self, intervals):
        # write your code here
        intervals.sort(key= lambda x : ( x[1],-x[0]))
        res=0
        
        left=intervals[0][1]-1
        right=intervals[0][1]
        res+=2
        for i in range(1,len(intervals)):
            cur=intervals[i]
            
            if cur[0]>left and cur[0]<=right:
                res+=1
                left=right
                right=cur[1]
            elif cur[0]>right:
                left=cur[1]-1
                right=cur[1]
                res+=2
        return res
        
        
#1049. Pyramid Transition Matrix
class Solution:
    """
    @param bottom: a string
    @param allowed: a list of strings
    @return: return a boolean
    """
    def pyramidTransition(self, bottom, allowed):
        # write your code here  
        
        from collections import defaultdict
        from itertools import product
        
        f=defaultdict( lambda: defaultdict(list))
        
        for a,b ,c in allowed:
            f[a][b].append(c)
            
        
        def build(bottom):
            if len(bottom)==1:
                return True
            
            for i in product(  *(f[a][b]  for a,b in  zip(bottom[:-1],bottom[1:] )      )):
                if build(i):
                    return True
            return False
        return build(bottom)
    A
   / \
  D   E
 / \ / \
X   Y   Z
bottom = "XYZ"
allowed = ["XYD", "YZE", "DEA", "FFF"]

bottom = "XXYX"
allowed = ["XXX", "XXY", "XYX", "XYY", "YXZ"]

if __name__ == "__main__":
    print(Solution().pyramidTransition( bottom, allowed)) 


#1051. Contain Virus
class Solution:
    """
    @param grid: the given 2-D world
    @return: the number of walls
    """
    def containVirus(self, grid):
        # Write your code here
        R=len(grid)
        C=len(grid[0])
        def neighbor(i,j):
            
            for x ,y in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                if x>=0 and x<R and y>=0 and y<C:
                    yield (x,y)
        
        
        
        def dfs(i,j):
            if (i,j) not in seen:
                seen.add(( i,j))
                region[-1].add((i,j))
                for nr,nc in neighbor(i,j) :
                   if grid[nr][nc]==1:
                      dfs(nr,nc)
                   elif grid[nr][nc]==0:
                      frontiers[-1].add((nr,nc))
                      perimeter[-1]+=1
        ans=0
        
        while True:
            region=[]
            frontiers=[]
            seen=set()
            
            perimeter=[]
            
            for i,row in enumerate(grid):
                for j,c in enumerate(row):
                    if grid[i][j]==1  and ( i,j) not in seen:
                        region.append(set())
                        frontiers.append(set())
                        perimeter.append(0)
                        dfs(i,j)
            
            if not region:
                break
            
            triage_index=frontiers.index(  max(frontiers,key=len  ))
            ans+=perimeter[triage_index]
           # print('region',region)
            #print('frontiers',frontiers)
            #print('perimeter',perimeter)
            
            
            
            
            for i , reg in enumerate(region):
                if i==triage_index:
                    for r,c in reg:
                        grid[r][c]=-1
                else:
                    for r,c in reg:
                        for nr,nc in neighbor(r,c):
                            #print('***')
                            if grid[nr][nc]==0:
                                grid[nr][nc]=1
            #print('grid',grid)
        return ans
grid = [[0,1,0,0,0,0,0,1],
 [0,1,0,0,0,0,0,1],
 [0,0,0,0,0,0,0,1],
 [0,0,0,0,0,0,0,0]]   

grid = [[1,1,1],
 [1,0,1],
 [1,1,1]]
grid = [[1,1,1,0,0,0,0,0,0],
 [1,0,1,0,1,1,1,1,1],
 [1,1,1,0,0,0,0,0,0]]                             
if __name__ == "__main__":
    print(Solution().containVirus( grid)) 
                        
                        
#1052. Shortest Completing Word
class Solution:
    """
    @param licensePlate: a string
    @param words: List[str]
    @return: return a string
    """
    def shortestCompletingWord(self, licensePlate, words):
        # write your code here 
        licensePlate=licensePlate.lower()
        words.sort(key=len)
        
        from collections import Counter
        
        LP_count=Counter(licensePlate)
        print(LP_count)
        for word in words:
            WD_count=Counter(word)
            isbreak=False
            for k,v in   LP_count.items():
                if k.isalpha()   :
                    if not (k in WD_count and WD_count[k]>=v):
                        isbreak=True
                        break
                else:
                    continue
            if isbreak:
                continue
            
                        
            return word
            
            
                    
licensePlate = "1s3 PSt"
words = ["step", "steps", "stripe", "stepple"]            
            
                
licensePlate = "1s3 456"
words = ["looks", "pest", "stew", "show"]        
if __name__ == "__main__":
    print(Solution().shortestCompletingWord(licensePlate, words)) 
                                
#1054. Min Cost Climbing Stairs
class Solution:
    """
    @param cost: an array
    @return: minimum cost to reach the top of the floor
    """
    def minCostClimbingStairs(self, cost):
        # Write your code here        
        #dp[i] to get i min cost
        n=len(cost)
        dp=[float('inf') for _ in range(n+1)]
        dp[0]=0
        dp[1]=0
        
        if n==0:
            return 0
        if n==1:
            return 1
        if n==2:
            return min(cost[0],cost[1])
        
        dp[0]=0
        dp[1]=0
       
        
        for i in range(2,n+1):
            dp[i]= min(dp[i-1]+cost[i-1],dp[i-2]+cost[i-2])
        #print(dp)
        return dp[-1]
cost = [10, 15, 20]        
        
        
cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]        
        
if __name__ == "__main__":
    print(Solution().minCostClimbingStairs(cost))         
        
        
        

#1056. Find Smallest Letter Greater Than Target        
class Solution:
    """
    @param letters: a list of sorted characters
    @param target: a target letter
    @return: the smallest element in the list that is larger than the given target
    """
    def nextGreatestLetter(self, letters, target):
        # Write your code here      
        for letter in letters:
            if letter > target:
                return letter
        
        
        
        
        
#1057. Network Delay Time
class Solution:
    """
    @param times: a 2D array
    @param N: an integer
    @param K: an integer
    @return: how long will it take for all nodes to receive the signal
    """
    def networkDelayTime(self, times, N, K):
        # Write your code here   
        from collections import defaultdict,deque
        import heapq
        graph=defaultdict(dict)
        
        for start,target,time in times:
            graph[start][target]=time
        
        q=[]
        heapq.heappush( q,(   0, K))
        seen=set()
        res=0
        while q and len(seen)!=N:
            curtime,cur=heapq.heappop(q)
            seen.add(cur)
            res=curtime
            
            for nx,time in graph[cur].items():
                if nx not in seen:
                    heapq.heappush(q,( curtime+ time,nx    ))
        return res if len(seen)==N else-1
        
       
      
            
times=[[2,1,1],[2,3,1],[3,4,1]]
N=4
K=2  

times=[[1,2,1],[2,3,7],[1,3,4],[2,1,2]]      
N=3
K=1

times=[[1,2,1],[2,3,7],[1,3,4],[2,1,2]]
N=3
K=2
if __name__ == "__main__":
    print(Solution().networkDelayTime( times, N, K))         
        
                
        
#1058. Cherry Pickup        
class Solution:
    """
    @param grid: a grid
    @return: the maximum number of cherries possible
    """
    def cherryPickup(self, grid):
        # Write your code here
        dp={}
        
        n=len(grid)
        
        def twoWalker(i1,j1,i2,j2,dp,grid ):
            if (i1,j1,i2,j2) in dp:
                return dp[i1,j1,i2,j2]
            if i1==n-1 and j1==n-1 and i2==n-1 and j2==n-1:
                return grid[-1][-1]
            if i1>=n or j1>=n or i2>=n or j2>=n :
                return float('-inf')
            if grid[i1][j1]==-1 or grid[i2][j2]==-1:
                return float('-inf')
            ans=0
            best=max( twoWalker(i1+1,j1,i2+1,j2,dp,grid ) ,twoWalker(i1,j1+1,i2,j2+1,dp,grid ),twoWalker(i1,j1+1,i2+1,j2,dp,grid ),twoWalker(i1+1,j1,i2,j2+1,dp,grid )    )
            ans+=best
            ans+=grid[i1][j1] if (i1,j1)==(i2,j2)  else grid[i1][j1]+grid[i2][j2]
            dp[i1,j1,i2,j2]=ans
            return ans
        res=twoWalker(0,0,0,0,dp,grid )
        return res if res>float('-inf') else 0
grid =[[0, 1, -1],
 [1, 0, -1],
 [1, 1,  1]]        
        
if __name__ == "__main__":
    print(Solution().cherryPickup( grid))         
                
        
        
#1059. Delete and Earn
class Solution:
    """
    @param nums: a list of integers
    @return: return a integer
    """
    def deleteAndEarn(self, nums):
        # write your code here
        n=10001
        values=[0 for _ in range(n)]
        
        for num in nums:
            values[num]+=num
        
        take=0
        skip=0
        
        for i in range(1,n):
            takei=skip+values[i]
            skipi=max(take,skip)
            skip=skipi
            take=takei
        return max(skip,take)
                

#1060. Daily Temperatures
class Solution:
    """
    @param temperatures: a list of daily temperatures
    @return: a list of how many days you would have to wait until a warmer temperature
    """
    def dailyTemperatures(self, temperatures):
        # Write your code here
        n=len(temperatures)
        ans=[0  for _ in range(n)]
        stack=[]
        
        for i in range(n-1,-1,-1):
            
            while stack and temperatures[i]>=temperatures[stack[-1]]:
                stack.pop()
            if stack:
                ans[i]=stack[-1]-i
            stack.append(i)
        return ans


#1061. Parse Lisp Expression        
class Solution:
    """
    @param expression:  a string expression representing a Lisp-like expression 
    @return: the integer value of
    """
    def evaluate(self, expression):
        # write your code here  
        tokens=expression.split(' ')
        scope=[{}]
        
        def continue_let(i):
            return 'a'<=tokens[i][0]<='z'  and tokens[i][-1]!=')'
        
                
        def helper(start):
            if start>len(tokens)-1:
                return 0,start
            operator=tokens[start]
            
            if operator[0]=='(':
                operator=operator[1:]
                scope.append(dict( scope[-1]))
            closing_brackets=0
            while operator[len(operator)-1-closing_brackets]==')':
                closing_brackets+=1
            if closing_brackets>0:
               operator=operator[:-closing_brackets]
            
            if operator.isdigit() or ( operator[0]=='-' and operator[1:].isdigit()):
                result= int(operator),start+1
            elif operator=='add':
                    left,nexti=helper(start+1)
                    right,nexti=helper(nexti)
                    result=(left+right,nexti)
            
            elif operator=='mult':
                    left,nexti=helper(start+1)
                    right,nexti=helper(nexti)
                    result=(left*right,nexti)
            
            
            elif operator=='let':   
                nexti=start+1
                while continue_let(nexti):
                    variable=tokens[nexti]
                    expression,nexti=helper(nexti+1)
                    scope[-1][variable]=expression
                result= helper(nexti)
            else:
                result=(scope[-1][operator],start+1)
            
            
            while closing_brackets>0:
                closing_brackets-=1
                scope.pop()
            return result
        return helper(0)[0]
expression='(add 1 2)'  
expression='(mult 3 (add 2 3))' 
expression='(let x 2 (mult x 5))'  
expression='(let x 2 (mult x (let x 3 y 4 (add x y))))'        
expression='(let x 3 x 2 x)'     
expression='(let x 1 y 2 x (add x y) (add x y))'
expression='(let x 2 (add (let x 3 (let x 4 x)) x))'
expression='(let a1 3 b2 (add a1 1) b2)'
if __name__ == "__main__":
    print(Solution().evaluate( expression))         
                                    
                
#1062. Flood Fill
class Solution:
    """
    @param image: a 2-D array
    @param sr: an integer
    @param sc: an integer
    @param newColor: an integer
    @return: the modified image
    """
    def floodFill(self, image, sr, sc, newColor):
        # Write your code here
        m=len(image)
        n=len(image[0])
        color=image[sr][sc]
        def dfs(i,j,color,newColor):
            #print(image)
            image[i][j]=newColor
            #print(image)
            for x,y in  ((i-1, j),(i+1, j),(i, j+1),(i,j-1)):
                if x>=0 and y>=0 and x <m and y<n:
                    if image[x][y]==color:
                        
                        dfs(x,y,color,newColor)
        
        
                        
        dfs(sr,sc,color,newColor)  
        return  image              
                    
image = [[1,1,1],[1,1,0],[1,0,1]]                
sr = 1
sc = 1
newColor = 2


image =[[0,0,1,2,0,0,9,5,6,5],[4,5,4,3,1,0,3,9,3,6],[6,2,4,1,2,6,5,0,5,7],[0,6,7,8,9,1,6,4,1,9],[2,2,8,6,0,7,6,4,8,2],[0,0,2,0,6,4,0,6,6,8],[4,8,7,4,9,9,7,1,2,7],[9,8,7,0,0,5,5,8,6,8],[3,0,2,1,2,8,5,2,3,9],[0,4,9,2,2,6,1,6,2,5]]                
       
sr = 9
sc = 0
newColor =9
if __name__ == "__main__":
    print(Solution().floodFill( image, sr, sc, newColor)) 
    
#1063. My Calendar III
import bisect
class MyCalendarThree(object):

    def __init__(self):
        self.time=[]
        

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: int
        """
        bisect.insort(self.time,(  start,1))
        bisect.insort(self.time,(  end,-1))
        
        
        ans=0
        maxv=0
        for _,x in self.time:
            ans+=x
            maxv=max(ans,maxv)
        return maxv
            
        


# Your MyCalendarThree object will be instantiated and called as such:
# obj = MyCalendarThree()
# param_1 = obj.book(start,end)    
    
#1064. My Calendar II
class MyCalendarTwo(object):

    def __init__(self):
        self.calendar=[]
        self.overlap=[]
        

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: bool
        """
        for i ,j in self.overlap:
            if start<j and end>i:
                return False
        for i ,j in self.calendar:
            if start<j and end>i:
                self.overlap.append( ( max( i,start   ),min(j,end) )   )
        self.calendar.append( (start, end  )  )
        return True
            
        


# Your MyCalendarTwo object will be instantiated and called as such:
# obj = MyCalendarTwo()
# param_1 = obj.book(start,end)    
            
            
#1065. My Calendar I            
class MyCalendar(object):

    def __init__(self):
        self.table=[]
        

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: bool
        """
        
        import bisect
        if not self.table:
            self.table.append( (start, end))
            return True
        elif len(self.table)==1:
            if start>=self.table[0][1]  :
                self.table.append((start,end))
                return True
                
            elif end<self.table[0][0]:
                self.table= [(start,end)]+self.table  
                return True
            else:
                return False
        else:
            idx=bisect.bisect_left(self.table,(start,end))
            if start>=self.table[-1][-1]:
                self.table.append((start,end))
                return True
            if idx==0:
                if end<=self.table[idx][0]:
                    self.table= [(start,end)]+self.table
                    return True
                return False
            elif idx==len(self.table):
                 return False
            else:
                if end<=self.table[idx][0] and start>=self.table[idx-1][1] :
                    self.table= self.table[:idx]  + [(start,end)]+self.table[idx:]
                    
                    return True
                else:
                    return False
                    
            
            
        


# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end) 

obj.book(23,32)
obj.book(42,50)
obj.book(6,14)
obj.book(0,7)
obj.book(21,30)
obj.book(26,31)
obj.book(46,50)
obj.book(28,36)
obj.book(0,6)
obj.book(27,36)
obj.book(6,11)
obj.book(20,25)
obj.book(32,37)
obj.book(14,20)
obj.book(7,16)
obj.book(13,22)
obj.book(39,47)
obj.book(37,46)
obj.book(42,50)
obj.book(9,17)
obj.book(49,50)
obj.book(31,37)
obj.book(43,49)
obj.book(2,10)
obj.book(3,12)
obj.book(8,14)
obj.book(14,21)
obj.book(42,47)
obj.book(43,49)
obj.book(36,43)
            
#1066. Verify Preorder Serialization of a Binary Tree
class Solution:
    """
    @param preorder: a string
    @return: return a bool
    """
    def isValidSerialization(self, preorder):
        # write your code here
        p=preorder.split(',')
        
        slot=1
        #print(p)
        
        for x in p:
            if slot==0:
                return False
            if x=='#':
                 slot-=1
            else:
                slot+=1
        #print(slot)
        return slot==0
preorder='#'            
if __name__ == "__main__":
    print(Solution().isValidSerialization( preorder))            
            
            
            
#1068. Find Pivot Index
class Solution:
    """
    @param nums: an array
    @return: the "pivot" index of this array
    """
    def pivotIndex(self, nums):
        # Write your code here
        n=len(nums)
        if n==0 or n==1:
            return -1
        if n==2:
            return False
        
        total=sum(nums)
        
        cur=0
               
        for i,x in enumerate(nums):
            if (total-x)/2==cur:
                return i
            cur+=x
            if i==0:
                continue
        return -1
            
nums=[1, 7, 3, 6, 5, 6]     
nums=[-1,-1,0,1,1,0] 
      
if __name__ == "__main__":
    print(Solution().pivotIndex( nums))            
            
            
#1069. Remove Comments
class Solution:
    """
    @param source: List[str]
    @return: return List[str]
    """
    def removeComments(self, source):
        # write your code here

        ans=[]
        in_block=False
        
        for line in source:
            i=0
            if not in_block:
               new_line=[]
            while i<len(line):
                if line[i:i+2]=='/*' and not in_block:
                    in_block=True
                    i+=1
                elif line[i:i+2]=='*/' and  in_block:
                    in_block=False
                    i+=1
                elif line[i:i+2]=='//' and  not in_block:
                    break
                elif not in_block:
                    new_line.append(line[i])
                i+=1
            if not in_block and new_line :
                ans.append(''.join(new_line))
            
        return ans
source = ["/*Test program */", "int main()", "{ ", "  // variable declaration ", "int a, b, c;", "/* This is a test", "   multiline  ", "   comment for ", "   testing */", "a = b + c;", "}"]
source =["a/*comment", "line", "more_comment*/b"]
if __name__ == "__main__":
    print(Solution().removeComments( source))    
                    
                    
               
#1070. Accounts Merge
class Solution:
    """
    @param accounts: List[List[str]]
    @return: return a List[List[str]]
    """
    def accountsMerge(self, accounts):
        # write your code here
#https://leetcode.com/problems/accounts-merge/discuss/175269/My-Python-DFS-and-Union-Find-solutions-beats-98.7-and-100
        n=len(accounts)
        parent=[i for i in range(n)]
        
        def find(i):
            while i!=parent[i]:
                parent[i]=parent[ parent[i]]
                i=parent[i]
            return i

        d= {}
        
        for i,a in enumerate(accounts)  :
            
            for email in a[1:]:
                if email in d:
                    r1,r2=find(i),find(d[email])
                    parent[r2]=r1
                else:
                    d[email]=i
        
        
        from collections import defaultdict
        res0=defaultdict(set)
        print(parent)
        
        for i in range(n):
            res0[find(i)] |= set(accounts[i][1:])
            
        res=[]
        
        for k,v in res0.items():
            res.append([accounts[k][0]]+ sorted(v))
        return res
accounts=[
  ["John", "johnsmith@mail.com", "john00@mail.com"],
  ["John", "johnnybravo@mail.com"],
  ["John", "johnsmith@mail.com", "john_newyork@mail.com"],
  ["Mary", "mary@mail.com"]
]  

accounts=[["David","David0@m.co","David1@m.co"],["David","David3@m.co","David4@m.co"],["David","David4@m.co","David5@m.co"],["David","David2@m.co","David3@m.co"],["David","David1@m.co","David2@m.co"]]          
if __name__ == "__main__":
    print(Solution().accountsMerge( accounts))            
        
            
#1071. Longest Word in Dictionary
class Solution:
    """
    @param words: a list of strings
    @return: the longest word in words that can be built one character at a time by other words in words
    """
    def longestWord(self, words):
        # Write your code here
        
        words=set(words)
        ans=''
        if not words:
            return 0
        
        for word in words:
            if len(word)>len(ans)  or ( len(word)==len(ans)  and word<ans):
                if all(  word[:k]  in words for k in range(1,len(word))):
                    ans=word
        return ans


words = ["w","wo","wor","worl", "world"]
words = ["a", "banana", "app", "appl", "ap", "apply", "apple"]


#1074. Range Module
from bisect import bisect_left,bisect_right
class Solution(object):

    def __init__(self):
        self.intervals=[]
        

    def addRange(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: void
        """
        intervals=self.intervals
        
        lo=bisect_left(intervals,left)
        hi=bisect_right(intervals,right)
        
        if lo%2==1:
           lo-=1
           left=intervals[lo]
        if hi%2==1:
            right=intervals[hi]
            hi+=1
           
        
        self.intervals=intervals[:lo]+[left,right]+intervals[hi:]
        print(self.intervals)
        
        
        

    def queryRange(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: bool
        """
        
        idx=bisect_right(self.intervals,left)
        
        return idx%2==1 and idx<len(self.intervals)  and self.intervals[idx-1]<=left <right<=self.intervals[idx]
        
    def removeRange(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: void
        """
        intervals=self.intervals
        new=[]
        
        lo=bisect_left(intervals,left)
        hi=bisect_right(intervals,right)
        
        if lo%2==1:
           lo-=1
           new.extend([intervals[lo] , left])
          
        if hi%2==1:
            new.extend([right,intervals[hi] ])
            right=intervals[hi]
            hi+=1
           
        
        self.intervals=intervals[:lo]+new+intervals[hi:]
        print(self.intervals)



         
obj=Solution()         
obj.addRange(10,20)
obj.removeRange(14,16)
obj.queryRange(10,14)
obj.queryRange(13,15)
obj.queryRange(16,17)            
            
            
            
            
            
            
        
#1075. Subarray Product Less Than K        
class Solution:
    """
    @param nums: an array
    @param k: an integer
    @return: the number of subarrays where the product of all the elements in the subarray is less than k
    """
    def numSubarrayProductLessThanK(self, nums, k):
        # Write your code here    
        n=len(nums)
        
        if n==0:
            return 0
        if k<=1:
            return 0
        
        ans=0
        left=0
        
        product=1
        
        for i in range(n):
            product*=nums[i]
            
            while product>=k:
                product/=nums[left]
                left+=1
            ans+=i-left+1
        return ans
        
        
        
nums=[10,5,2,6]  
k=100      
if __name__ == "__main__":
    print(Solution().numSubarrayProductLessThanK( nums, k))            
                
        
#1076. Minimum ASCII Delete Sum for Two Strings
class Solution:
    """
    @param s1: a string
    @param s2: a string
    @return: the lowest ASCII sum of deleted characters to make two strings equal
    """
    def minimumDeleteSum(self, s1, s2):
        # Write your code here
        n1=len(s1)
        n2=len(s2)
        
        dp=[[0 for _ in range(n2+1)]  for _ in range(n1+1)  ]
        
        for i in range(1,n1+1):
            dp[i][0]=dp[i-1][0]+ord(s1[i-1])
        for j in range(1,n2+1):
            dp[0][j]=dp[0][j-1]+ord(s2[j-1])
        
        
        for i in range(1,n1+1):
            for j in range(1,n2+1):
                if s1[i-1]==s2[j-1]:
                    dp[i][j]=dp[i-1][j-1]
                else:
                    
                    dp[i][j]=min(dp[i-1][j]+ord(s1[i-1]),dp[i][j-1]+ord(s2[j-1]))
        return dp[n1][n2]
s1 = "sea"
s2 = "eat"   
s1 = "delete"
s2 = "leet"     
if __name__ == "__main__":
    print(Solution().minimumDeleteSum( s1, s2))          
    
    
#1077. Falling Squares
class Solution:
    """
    @param positions: a list of (left, side_length)
    @return: return a list of integer
    """
    def fallingSquares(self, positions):
        # write your code here
        
        h={}
        ans=[]
        maxh=float('-inf')
        
        for position,side in positions:
            left=position
            right=position+side-1
            
            nearby=[h[key]  for key in h if  not (left>key[1]  or right <key[0]) ]
            
            if len(nearby)>0:
                cur=side+max(nearby)
            else:
                cur=side
            h[(left,right)]=cur
            maxh=max(maxh,cur)
            ans.append( maxh)
        return ans
        
positions=[[1, 2], [2, 3], [6, 1]]            
if __name__ == "__main__":
    print(Solution().fallingSquares( positions))          
        
    
#971. Flip Binary Tree To Match Preorder Traversal    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def flipMatchVoyage(self, root, voyage):
        """
        :type root: TreeNode
        :type voyage: List[int]
        :rtype: List[int]
        """
        res=[]
        self.i=0
        def dfs(node):
            if not node:
                return True
            if node.val!=voyage[self.i]:
                return False
            self.i+=1
            if node.left and node.left.val!=voyage[self.i]:
                res.append(node.val)
                node.left,node.right=node.right,node.left
            return dfs(node.left)  and dfs(node.right)
        return res if dfs(root)  else [-1]
        
        
        
        
        
        
        
        
        
        
root = [1,2]
voyage = [2,1] 

root = [1,2,3]
voyage = [1,3,2]

root = [1,2,3]
voyage = [1,2,3]

       
    