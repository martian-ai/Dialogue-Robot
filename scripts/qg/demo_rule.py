import argparse
import sys
sys.path.append("../..")

from solutions.qg.rule import question_generate_by_rule

if __name__ == "__main__":
    print(question_generate_by_rule('国务院规定农名工的工资应当按时发放'))
    print(question_generate_by_rule('建筑公司负责农民工的工资应当按时发放'))
    print(question_generate_by_rule('商务部制定本条例的目的是保护农民工的权益'))