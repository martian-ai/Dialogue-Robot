# 移除链表元素（ LeetCode 203 ）:https://leetcode-cn.com/problems/remove-linked-list-elements/
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        # 边界情况，如果链表为空，那么没有元素可以删除，直接返回空
        if head == None :
            return None

        # 一开始设置一个虚拟节点，它的值为 -1，它的值可以设置为任何的数，因为我们根本不需要使用它的值
        # 设置虚拟节点的目的是为了让原链表中所有节点就都可以按照统一的方式进行移除
        # 因为如果不设置虚拟节点，如果删除的元素是原链表中的头节点，那么需要额外的做一些判断，比较繁琐
        dummy =  ListNode(-1)

        # 虚拟头节点的下一节点指向 head 节点
        # 如果原链表是  1 -->  2 -->  3
        # 那么加上虚拟头节点就是  -1 -->  1 -->  2 -->  3
        dummy.next = head

        # 设置一个指针，指向此时的虚拟节点
        # pre: -1 -->  1 -->  2 -->  3
        pre = dummy

        # 设置一个指针，指向原链表 head
        cur = head

        # 让 cur 不断的向后移动，直到移动到链表的最尾部，指向 None 的那个位置
        # 此时 pre 还在指向 dummy
        # 也就是说一开始 pre 落后 cur 一个节点
        while cur != None :

            # 移动的过程中，如果发现当前的节点值和目标值一样
            # 我们就让指针 pre 所指向的节点的下一节点跳过这个值
            if cur.val == val :
                # 让指针 pre 所指向的节点的下一节点跳过这个值
                pre.next = cur.next
            else :
                # 否则的话，pre 跟上 cur 的位置
                pre = cur
                    # 判断完当前的节点情况后，让 cur 向后移动
            cur = cur.next

        # 最后返回 dummy 节点的下一节点
        # 因为这个时候 dummy 指向的是一个我们设置的节点，它的下一节点才是原链表中的节点
        return dummy.next