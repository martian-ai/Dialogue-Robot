# python的内置数据结构list可以用来实现栈，用append()向栈顶添加元素, pop() 可以以后进先出的顺序删除元素

stack = []
#append() fuction to push
#element in list 
stack.append('hello')
stack.append('world')
stack.append('!')
print('Initial stack')
print(stack)
#pop() function to pop element
#from stack in LIFO order
 
print('\nElement poped from stack')

print(stack.pop())
print(stack.pop())
print(stack.pop())
print(stack)