# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 19:57:29 2016

@author: ldy
"""

import sys
sys.path.append("code")
from Sentiment_svm import svm_predict
from Sentiment_lstm import lstm_predict
argvs_lenght = len(sys.argv)
if argvs_lenght != 3:
    print '参数长度错误！'
argvs = sys.argv

sentence  = argvs[-1]

if argvs[1] == 'svm':
    svm_predict(sentence)
    
elif argvs[1] == 'lstm':
    lstm_predict(sentence)
    
else:
    print '选择svm或lstm！'
    
