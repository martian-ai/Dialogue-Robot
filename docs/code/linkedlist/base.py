class Node():
    def __init__(self, val) -> None:
        self.val = val
        self.next = None
    
class ListNode():
    def __init__(self) -> None:
        self.head = None
        self.length = 0

    def is_empty(self):
        return self.length == 0
    
    def PrintList(self):
        if self.length ==0: # 边界条件
            return None
        else:
            p = self.head
            while p.next:
                print(p.val,'-->',end = '') # 打印不换行技巧
                p = p.next
            print(p.val)
    
    def append(self, node):
        if isinstance(node, Node):
            pass
        else:
            node = Node(node)
        if self.is_empty():
            self.head = node
        else:
            p = self.head 
            while p.next:
                p = p.next
            p.next = node
        self.length += 1

    def index_insert(self, node, index):
        if index > self.length:
            return 'error'

        if isinstance(node, Node):
            pass
        else:
            node = Node(node)

        if index == 0:
            node.next = self.head
            self.head = node
        else:
            # index 3
            #                p
            # 0 -> 1 -> 2 -> 3
            p = self.head
            # while index: # 该元素之后插入
            while index - 1: # 该元素之前插入
                p = p.next
                index -= 1
            pre_next = p.next
            p.next = node
            node.next = pre_next

        self.length = self.length + 1

    def index_delete(self, index):
        if index > self.length:
            return 'error'
        p = self.head
        if index == 0:
            self.head = self.head.next
        else:
            while index-1:
                p = p.next
                index -= 1
            p.next = p.next.next
            self.length -= 1
    
    def index_update(self, index, val):
        if not 0 <= index < self.length: # 边界条件
            return 'Error'
        if index == 0:
            self.head.val = val
        else:
            p = self.head
            while index: # 不需要减1
                p = p.next
                index-=1
            p.val = val
    def get_index_data(self, index):
        if not 0 <= index < self.length: # 边界条件
            return 'Error'
        if index == 0:
            return self.head.val
        else:
            p = self.head
            while index: # 不需要减1
                p = p.next
                index-=1
            return p.val
    def get_length(self):
        return self.length
    def clear(self):
        self.length = 0
        self.head = None