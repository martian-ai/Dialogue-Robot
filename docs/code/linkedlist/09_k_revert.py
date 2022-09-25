# K 个一组翻转链表（ LeetCode 25 ） ：https://leetcode-cn.com/problems/reverse-nodes-in-k-group/
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        # 一开始设置一个虚拟节点，它的值为 -1，它的值可以设置为任何的数，因为我们根本不需要使用它的值
        dummy = ListNode(-1)

        # 虚拟头节点的下一节点指向 head 节点
        # 如果原链表是  1 -->  2 -->  3
        # 那么加上虚拟头节点就是  -1 -->  1 -->  2 -->  3

        dummy.next = head

        # 设置一个指针，指向此时的虚拟节点，pre 表示每次要翻转的链表的头结点的【上一个节点】
        # pre: -1 -->  1 -->  2 -->  3
        pre = dummy

        # 设置一个指针，指向此时的虚拟节点，end 表示每次要翻转的链表的尾节点
        # end: -1 -->  1 -->  2 -->  3
        end = dummy

        # 通过 while 循环，不断的找到翻转链表的尾部
        while end.next != None :

            # 通过 for 循环，找到【每一组翻转链表的尾部】
            # 由于原链表按照 k 个一组进行划分会可能出现有一组的长度不足 k 个
            # 比如原链表 1 -->  2 -->  3 -->  4 -->  5
            # k = 2，划分了三组 1 -->  2， 3 -->  4， 5
            # 所以得确保 end 不为空才去找它的 next 指针，否则 None.next 会报错
            for i in range(k) :
                if end is None:
                   break
                else:
                   # end 不断的向后移动，移动 k 次到达【每一组翻转链表的尾部】
                   end = end.next 

            # 如果发现 end == None，说明此时翻转的链表的节点数小于 k ，保存原有顺序就行
            if end is None:
                # 直接跳出循环，只执行下面的翻转操作
                break


            # next 表示【待翻转链表区域】里面的第一个节点
            next = end.next

            # 【翻转链表区域】的最尾部节点先断开
            end.next = None 

            # start 表示【翻转链表区域】里面的第一个节点
            start = pre.next



            # 【翻转链表区域】的最头部节点和前面断开
            pre.next = None

            # 这个时候，【翻转链表区域】的头节点是 start，尾节点是 end
            # 开始执行【反转链表】操作
            # 原先是 start --> ...--> end
            # 现在变成了 end --> ...--> start

            # 要翻转的链表的头结点的【上一个节点】的 next 指针指向这次翻转的结果
            pre.next = self.reverseList(start)

            # 接下来的操作是在为【待翻转链表区域】的反转做准备

            # 原先是 start --> ...--> end
            # 现在变成了 end --> ...--> start
            # 【翻转链表区域】里面的尾节点的 next 指针指向【待翻转链表区域】里面的第一个节点
            start.next = next 

            # 原先是 start --> ...--> end
            # 现在变成了 end --> ...--> start
            # pre 表示每次要翻转的链表的头结点的【上一个节点】
            pre = start

            # 将 end 重置为【待翻转链表区域】的头结点的上一个节点。
            end = start

        return dummy.next


    # 反转链表的代码
    def reverseList(self,head:ListNode) -> ListNode:
        # 寻找递归终止条件
        # 1、head 指向的结点为 None 
        # 2、head 指向的结点的下一个结点为 None 
        # 在这两种情况下，反转之后的结果还是它自己本身
        if head == None or head.next == None:
           return head

        # 不断的通过递归调用，直到无法递归下去，递归的最小粒度是在最后一个节点
        # 因为到最后一个节点的时候，由于当前节点 head 的 next 节点是空，所以会直接返回 head
        cur = self.reverseList(head.next)


        # 比如原链表为 1 --> 2 --> 3 --> 4 --> 5
        # 第一次执行下面代码的时候，head 为 4，那么 head.next = 5
        # 那么 head.next.next 就是 5.next ，意思就是去设置 5 的下一个节点
        # 等号右侧为 head，意思就是设置 5 的下一个节点是 4

        # 这里出现了两个 next
        # 第一个 next 是「获取」 head 的下一节点
        # 第二个 next 是「设置」 当前节点的下一节点为等号右侧的值
        head.next.next = head


        # head 原来的下一节点指向自己，所以 head 自己本身就不能再指向原来的下一节点了
        # 否则会发生无限循环
        head.next = None

        # 我们把每次反转后的结果传递给上一层
        return cur