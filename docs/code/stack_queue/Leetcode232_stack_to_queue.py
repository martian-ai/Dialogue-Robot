# 用栈实现队列 （ LeetCode 232 ）:https://leetcode-cn.com/problems/implement-queue-using-stacks/
class MyQueue:
    def __init__(self):
        # 一个栈叫做 stackIn，负责进栈操作，相当于队列 queue 中的入队操作
        self.stackIn = []
        # 一个栈叫做 stackOut，负责出栈操作，相当于队列 queue 中的出队操作
        self.stackOut = []


    def push(self, x: int) -> None:
        # 新添加的元素添加到 stackIn 中
        self.stackIn.append(x)


    def pop(self) -> int:
        # 如果 stackOut 为空，首先需要将 stackIn 中的所有元素添加到 stackOut 中
        #注意 stackIn 是栈，栈的性质是先进后出，后进先出，所以是不断的将 stackIn 中的栈顶元素添加进 stackOut 中
        if not self.stackOut:
            # 通过 while 循环将 stackIn 中的所有元素都取出
            while self.stackIn:
                # stackOut 不断的添加 stackIn 的栈顶元素
                self.stackOut.append(self.stackIn.pop())
        # 此时，stackIn 已经为空，直接「移除」 stackOut 的栈顶元素
        return self.stackOut.pop()


    def peek(self) -> int:
        # peek 和 pop 的区别在于是返回栈顶元素而非删除栈顶元素
        # 如果 stackOut 为空，首先需要将 stackIn 中的所有元素添加到 stackOut 中
        # 注意 stackIn 是栈，栈的性质是先进后出，后进先出，所以是不断的将 stackIn 中的栈顶元素添加进 stackOut 中
        if not self.stackOut:
            # 通过 while 循环将 stackIn 中的所有元素都取出
            while self.stackIn:
                # stackOut 不断的添加 stackIn 的栈顶元素
                self.stackOut.append(self.stackIn.pop())

        # peek 和 pop 的区别在于是返回栈顶元素而非删除栈顶元素
        # 此时，stackIn 已经为空，直接「返回」 stackOut 的栈顶元素
        return self.stackOut[-1]


    def empty(self) -> bool:
        # 队列是否为空，判断 stackIn 和 stackOut 是否同时不存在元素
        return not self.stackIn and not self.stackOut