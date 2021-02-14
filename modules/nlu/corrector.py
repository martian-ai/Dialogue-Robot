import pycorrector

"""
Error Correction 
"""

def error_correction(text):
    """
    使用pycorrector 进行纠错
    """
    correct_str , _ = pycorrector.correct(text)
    return correct_str

def error_correction_by_transformer(text):
    """
    使用transformer 进行纠错
    """
    return text