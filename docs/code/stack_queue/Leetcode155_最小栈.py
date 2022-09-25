# 最小栈（ LeetCode 155 ）:https://leetcode-cn.com/problems/min-stack/
class MinStack:
    def __init__(self):
        # 首先定义好两个栈

        # 一个栈叫做 stack，负责栈的正常操作
        self.stack = []

        # 一个栈叫做 min_stack，负责获取 stack 中的最小值，它等价于遍历 stack 中的所有元素，把升序的数字都删除掉，留下一个从栈底到栈顶降序的栈
        self.min_stack = []

    def push(self, x: int) -> None:

        # 新添加的元素添加到 stack 中
        self.stack.append(x)

        # 判断 min_stack 是否为空，如果为空，直接同时把新添加的元素添加到 minStack 中
        # 如果 min_stack 不为空
        if self.min_stack:
           # 获取 min_stack 的栈顶元素
           top = self.min_stack[-1]

           # 只有新添加的元素不大于 top 才允许添加到 minStack 中，目的是为了让 minStack 从栈底到栈顶是降序的
           if x <= top :
              self.min_stack.append(x)

        # min_stack 中没有元素，所以直接把新添加的元素添加到 min_stack 中
        else:
            self.min_stack.append(x)


    def pop(self) -> None:

        # 让 stack 执行正常的 pop 操作就行
        pop =  self.stack[-1]

        self.stack.pop()

        # 由于 minStack 中的所有元素都是来自于 stack 中，所以 stack 删除元素后，minStack 也要考虑是否需要删除元素
        # 否则的话，minStack 有可能保存一个 stack 中不存在的元素

        # 首先，获取 minStack 的栈顶元素
        top = self.min_stack[-1]

        # 再判断 top 这个栈顶元素是否和 stack 移除的元素相等，如果相等，那么需要把 minStack 中的栈顶元素一并移除 
        if pop == top:
            # 移除 min_stack 的栈顶元素
            self.min_stack.pop()

    def top(self) -> int:
        # 返回 stack 的栈顶元素
        return self.stack[-1]
    def getMin(self) -> int:
        # 返回 min_stack 的栈顶元素
        return self.min_stack[-1]