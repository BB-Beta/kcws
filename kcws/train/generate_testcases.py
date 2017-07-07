#coding=utf8
'''
Created on 2017年7月6日

@author: wuling
'''

def loadFile(path, result):
    with open(path, 'r') as f_input:
        for line in f_input:
            attrs = line.strip().split('\t')
            result[attrs[1]] = attrs[0]
    
def process():
    w2v_dict, label_dict = dict(), dict()
    loadFile('kcws/models/pos_vocab.txt', label_dict)
    loadFile('kcws/models/word_vocab.txt', w2v_dict)
    with open('test.txt', 'r') as f_input:
        with open('result.txt', 'w') as f_output:
            for line in f_input:
                elems = line.strip().split(' ')
                words = elems[:50]
                labels = elems[300:]
                for word in words:
                    if word != '0':
                        f_output.write('%s\t'%(w2v_dict[word]))
                    else:
                        f_output.write('<pad>\t')
                f_output.write('\n')
                for elem in labels:
                    if elem != '0':
                        f_output.write('%s\t'%(label_dict[elem]))
                    else:
                        f_output.write('<pad>\t')
                f_output.write('\n')
    
if __name__ == '__main__':
    process()
