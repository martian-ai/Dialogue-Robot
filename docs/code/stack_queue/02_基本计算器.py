# 登录 AlgoMooc 官网获取更多算法图解
# https://www.algomooc.com
# 作者：程序员吴师兄
# 代码有看不懂的地方一定要私聊咨询吴师兄呀
# 基本计算器（ LeetCode 224 ）:https://leetcode-cn.com/problems/basic-calculator
class Solution:
    def calculate(self, s: str) -> int:

        # 使用栈来储存字符串表达式中的数字
        stack = list()

        # 为了方便计算，所有的操作都视为加法操作
        # 那么原先的减法操作就相当于是加一个负数
        # 默认都是正数
        sign = 1

        # 保存计算的结果
        res = 0

        # 获取字符串的长度，然后获取里面的每个字符
        length = len(s)

        # 从 0 开始访问字符串中的每个字符
        i = 0

        # 获取字符串里面的每个字符
        while i < length:
            # 获取此时的字符
            ch = s[i]

            if ch == ' ' :
                i += 1
            # 如果当前字符是数字的话
            elif ch.isdigit() :
                # 那么把获取到的数累加到结果 res 上
                value = 0

                # 去查看当前字符的后一位是否存在
                # 如果操作并且后一位依旧是数字，那么就需要把后面的数字累加上来
                while i < length and s[i].isdigit():
                    value = value * 10 + ord(s[i]) - ord('0')
                    i += 1

                # 那么把获取到的数累加到结果 res 上
                res += value * sign

              # 如果是 '+'
            elif ch == '+' :
                # 那么说明直接加一个正数
                sign = 1

                i += 1

              # 如果是 '-'
            elif ch == '-' :

                # 那么说明加一个负数
                sign = -1

                i += 1

              # 如果是 '('
            elif ch == '(' :
                # 根据数学计算规则，需要先计算括号里面的数字
                # 而什么时候计算呢？
                # 遇到 ) 为止
                # 所以，在遇到 ) 之前需要把之前计算好的结果存储起来再计算
                # ( ) 直接的计算规则和一开始是一样的

                # 1、先把 ( 之前的结果存放到栈中
                stack.append(res)

                # 2、重新初始化 res 为 0
                res = 0
                # 3、把 ( 左边的操作符号存放到栈中
                # 比如如果是 5 - （ 2 + 3 ） ，那么就是把 -1 存放进去
                # 比如如果是 5 +（ 2 + 3 ） ，那么就是把 1 存放进去
                stack.append(sign)

                # 4、为了方便计算，所有的操作都视为加法操作
                # 那么原先的减法操作就相当于是加一个负数
                # 默认都是正数
                sign = 1

                i += 1

                # 如果是 ')'
            elif ch == ')' :

                # 那么就需要把栈中存放的元素取出来了
                # 在 ch == '（' 这个判断语句中，每次都会往栈中存放两个元素
                # 1、先存放左括号外面的结果
                # 2、再存放左括号外面的符号

                # 1、先获取栈顶元素，即左括号外面的符号，查看是 + 还是 -
                # 把栈顶元素弹出
                formerSign = stack.pop()

                # 2、再获取栈顶元素，即左括号结果
                # 把栈顶元素弹出
                formerRes = stack.pop()

                # 那结果就是括号外面的结果 + 括号里面的结果，至于是加正数还是负数，取决于左括号外面的符号
                res = formerRes +  formerSign * res 

                i += 1


        # 返回计算好的结果
        return res