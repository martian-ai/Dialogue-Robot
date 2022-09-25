
import re

# text.split() 只能按照一个分个词切分
# re.split("", text) 可以按照多个分割词切分

def text_segment(text):
    return re.split('[/()（）]', text)


if __name__ == "__main__":
    print(text_segment('A/B(CC)'))
    print(text_segment('行驶/作业选择开关'))
    print(text_segment('多路阀线束（X3）'))