# 请计算在每一天需要等几天才会有更高的温度。如果气温在这之后都不会升高，请在该位置用 0 来代替

# 可以维护一个存储下标的单调栈，从栈底到栈顶的下标对应的温度列表中的温度依次递减。如果一个下标在单调栈里，则表示尚未找到下一次温度更高的下标。
#https://leetcode-cn.com/problems/daily-temperatures/solution/mei-ri-wen-du-by-leetcode-solution/

# 每日温度（ LeetCode 739 ）：https://leetcode-cn.com/problems/daily-temperatures
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:

        # 构建一个栈,用来存放每日温度的下标
        stack = []

        # 获取数组长度
        length = len(temperatures)

        # 构建一个数组，用来保存输出结果
        res = [0] * length

        # 从头开始遍历每天的温度
        for i in range(length):
            #  拿到当天的温度，不断的和栈顶元素进行比较
            temperature = temperatures[i]

            # 如果栈顶元素存在并且当天的温度大于栈顶元素
            # 意味着栈顶元素等到了第一个温度比它更高的那一天
            # 它们的下标差就是栈顶元素等了多少天等到的更高温度的结果
            while stack and temperature > temperatures[stack[-1]]:

                # 首先获取栈顶元素的值，也就是每日温度的下标值
                preIndex = stack.pop()

                # 它们的下标差就是栈顶元素等了多少天等到的更高温度的结果，保存到输出数组 res 中
                res[preIndex] = i - preIndex


            # 再把当天的温度的下标值存放到栈中
            # 注意：是下标索引值
            stack.append(i)

        # 最后输出 res 数组即可
        return res