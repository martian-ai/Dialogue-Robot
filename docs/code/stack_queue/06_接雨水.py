# 登录 AlgoMooc 官网获取更多算法图解
# https://www.algomooc.com
# 作者：程序员吴师兄
# 代码有看不懂的地方一定要私聊咨询吴师兄呀
# 接雨水（ LeetCode 42 ）:https://leetcode-cn.com/problems/trapping-rain-water/
class Solution:
    def trap(self, height: List[int]) -> int:
         # 构建一个栈，用来存储对应的柱子的下标
        # stack 存储的是下标而非高度
        stack = list()

        # 一开始水的面积是 0
        result = 0

        # 获取数组的长度
        n = len(height)

        # 从头开始遍历整个数组
        for i, h in enumerate(height):
            # 如果栈为空，那么直接把当前索引加入到栈中
            if not stack :
                # 把当前索引加入到栈中
                stack.append(i)

             # 否则的话，栈里面是有值的，我们需要去判断此时的元素、栈顶元素、栈顶之前的元素能否形成一个凹槽
             # 情况一：此时的元素小于栈顶元素，凹槽的右侧不存在，无法形成凹槽
            elif height[i] < height[stack[-1]]:
                # 把当前索引加入到栈中
                stack.append(i)

               # 情况二：此时的元素等于栈顶元素，也是无法形成凹槽
            elif height[i] == height[stack[-1]]:

                # 把当前索引加入到栈中
                stack.append(i)

               # 情况三：此时的的元素大于栈顶元素，有可能形成凹槽
               # 注意是有可能形成，因为比如栈中的元素是 2 、2 ，此时的元素是 3，那么是无法形成凹槽的
            else :   

                # 由于栈中有可能存在多个元素，移除栈顶元素之后，剩下的元素和此时的元素也有可能形成凹槽
                # 因此，我们需要不断的比较此时的元素和栈顶元素
                # 此时的元素依旧大于栈顶元素时，我们去计算此时的凹槽面积
                # 借助 while 循环来实现这个操作
                while stack and height[i] > height[stack[-1]]:

                    # 1、获取栈顶的下标，bottom 为凹槽的底部位置
                    bottom = stack[-1]

                    # 将栈顶元素推出，去判断栈顶之前的元素是否存在，即凹槽的左侧是否存在
                    stack.pop()

                    # 2、如果栈不为空，即栈中有元素，即凹槽的左侧存在
                    if stack :

                        # 凹槽左侧的高度 height[stack[-1] 和 凹槽右侧的高度 height[i] 
                        # 这两者的最小值减去凹槽底部的高度就是凹槽的高度  
                        h = min(height[stack[-1]], height[i]) - height[bottom]

                        # 凹槽的宽度等于凹槽右侧的下标值 i 减去凹槽左侧的下标值 stack.top 再减去 1
                        w = i - stack[-1] - 1 

                        # 将计算的结果累加到最终的结果上去
                        result += h * w


                # 栈中和此时的元素可以形成栈的情况在上述 while 循环中都已经判断了
                # 那么，此时栈中的元素必然不可能大于此时的元素，所以可以把此时的元素添加到栈中
                stack.append(i)

        # 最后返回结果即可
        return result