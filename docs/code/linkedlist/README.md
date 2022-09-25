# Linked List

### Data Structure
```
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
```
```
class Node{
	Node next = null;
    	int data;
    	public Node(int data){this.data = data;}
}
```
### 链表的增加删除
```
public class MyLinkedList{
	Node head = null;
	/*add a data to LinkedList*/
	public void addNode(int d){
		Node newNode = new Node(d);
		if ( head == null){
			head = newNode;
		}
		Node tmp = head;
		while(tmp.next != null){
			tmp = tmp.next;
		}	
		//add to the end
		tmp.next = newNode;
	}
	/* delete the index th node*/
	public Boolean deleteNode(int index){
		if( index < 1 || index > length()){
			return false;
		}
		//delete the first node
		if (index == 1){
			head = head.next;
			return true;
		}
		int i = 1;
		Node preNode = head;
		Node curNode = preNode.next;
		while(curNode != null){
			if(i == index){
				preNode.next = curNode.next;
				return true;
			}
			preNode = curNode;
			curNode = curNode.next;
			i++;
		}
	}
	/* return the length of LinkedList*/
	public int length(){
		Node tmp = head;
		int len = 0;
		while(tmp != null){
			len++;
			tmp = tmp.next;
		}
		return len;
	}
	/* sort and return the LinkedList*/
	public Node orderList(){
		Node nextNode = null;
		Node curNode = head;
		int tmp = 0;
		while(curNode != null){
			nextNode = curNode.next;
			while(nextNode != null){
				if(curNode.data > nextNode.data){
					tmp = curNode.data;
					curNode.data = nextNode.data;
					nextNode.data = tmp;
				}
				nextNode = nextNode.next;
			}
			curNode = curNode.next;
		}
		return head;
	}
}
```
### 链表删除重复元素
```
// double pointer
// pointer p : return pointer
// pointer q : traverse the linkedlist; find the repeated node and delete it 
// tips : p, q operate one linkedlist
public void deleteDuplecate(Node head){
	Node p = head;
	while(p != null){
		Node q = p;
		while(q.next != null){
			if(p.data == q.next.data){
				q.next = q.next.next;
			}else{
				q = q.next;
			}
		}
		p = p.next
	}
}
```
### 找出单链表的倒数第K个元素
```
//正向可查找第 length-k：问题是需要遍历两次列表
public findElem(Node head, int k){
	if(k < 1 || k > length()){
        	return null
	}
	Node p1 = head;
	Node p2 = head;
        for (int iCount = 0; i < k ; i++){
        	p1 = p1.next
        }
        while(p1.next != null){
        	p1 = p1.next;
		p2 = p2.next;
	}
	return p2;
}
```
### 实现链表的反转
```
public Node reverseLinkedList(Node head){
	Node pReverHead = Head;
	Node pNode = head;
 	Node pPrev = null;
	while(pNode != null){
        	Node pNext = pNode.next;
		if(pNext == null){
            		pReverHead = pNode;
		}
		// 第一次的时候 pPrev 是 null
            	pNode.next = pPrev;
            	pPrev = pNode;
            	pNode = pNext;
        }
        this.head = pReverHead;
}
```
### 从尾到头输出单链表
```
 public void printReverse(Node head){
 	if(head.next != null){
     	printReverse(head.next);
     	System.out.println(head.next.data);
     }
 }
```
http://www.cnblogs.com/kira2will/p/4109985.html
### 寻找单链表的中间节点
+ 设置两个指针，快指针一次跳两步，慢指针一次跳一步
+ 快指针到尾部时，慢指针为中间节点（奇数个为慢指针指向的那个，偶数时为慢指针指向的那个及下一个）
### 检测链表是否有环
+ 设置两个指针，快指针一次跳两步，慢指针一次跳一步
+ 使用两个slow, fast指针从头开始扫描链表。指针slow 每次走1步，指针fast每次走2步。如果存在环，则指针slow、fast会相遇；如果不存在环，指针fast遇到NULL退出。
### 判断有环链表的起始节点
+ 方法一： 判断节点有环后，使用（快慢指针相交的节点）确定环的长度N，从原有头节点开始遍历，如果当前Node 与 Node + N 第一次相等，则此节点为环的起始节点
+ 方法二： 从头节点和相遇节点分别向前走，相遇得点就是环得起始位置
### 不知道头指针的情况下，删除指定节点

### 判断两个链表是否相交
+ 链表相交的含义（？？？）
+ 求两个链表的长度差，长的链表的头指针先移动这个长度差，之后判断移动后的长链表与短链表是否相同
+ 判断是否有相同的尾节点