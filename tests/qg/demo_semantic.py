import argparse
import sys
sys.path.append("../..")

from solutions.qg.semantic import question_generate_by_semantic

if __name__ == "__main__":
    print(question_generate_by_semantic('国务院规定农名工的工资应当按时发放')) 