# // 有效的括号（ LeetCode 20 ）:https://leetcode-cn.com/problems/valid-parentheses
class Solution:
    def isValid(self, s: str) -> bool:
        # 当字符串长度为奇数的时候，属于无效情况，直接返回 False
        if len(s) % 2 == 1:
             # 无效情况，返回 False
             return False


        # 构建一个栈，用来存储括号
        stack = list()


        # 遍历字符串数组中的所有元素
        for c in s : 

            # 如果字符为左括号 ( ，那么就在栈中添加对左括号 （
            if c == '(' :

               # 添加对左括号 （
               stack.append('(')

             # 如果字符为左括号 [ ，那么就在栈中添加对左括号 [
            elif c == '[' :

               # 添加对应的右括号 ]
               stack.append('[')

             # 如果字符为左括号 { ，那么就在栈中添加对左括号 {
            elif c == '{' :

               # 添加对应的右括号 }
               stack.append('{')

               # 否则的话，说明此时 c 是 ）] } 这三种符号中的一种
            else :

               # 如果栈已经为空，而现在遍历的字符 c 是 ）] } 这三种符号中的一种
               # 找不到可以匹配的括号，返回 False
               # 比如这种情况  }{，直接从右括号开始，此时栈为空
               if not stack : 
                  return False

               # 如果栈不为空，获取栈顶元素
               top = stack[-1]

               # 将栈顶元素和此时的元素 c 进行比较，如果相同，则将栈顶元素移除
               if (top == '(' and c == ')' ) or (top == '[' and c == ']' ) or (top == '{' and c == '}')  :
                    # 移除栈顶元素
                    stack.pop()
               else :
                   # 如果不相同，说明不匹配，返回 False
                   return False


        # 遍历完整个字符数组，判断栈是否为空
        # 如果栈为空，说明字符数组中的所有括号都是闭合的
        # 如果栈为空，说明有未闭合的括号
        return not stack