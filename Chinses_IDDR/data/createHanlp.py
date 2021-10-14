import hanlp
import torch
import sys
import os
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)

def _text2i(texts, max_sent_len):
        vocab_dict = {'Name':123456}
        vocab_temp = []
        with open('./bpe/vocab_wyh.txt') as f:
            vocab_temp = f.readlines()
        for i in range(len(vocab_temp)):
            temp = vocab_temp[i].strip('\n')
            vocab_dict[temp] = i

        l = len(texts)
        tensor = torch.LongTensor(l, max_sent_len).zero_()
        for i in range(l):
            s = texts[i] + ['</s>']
            minlen = min(len(s), max_sent_len)
            for j in range(minlen):
                if s[j] in vocab_dict.keys():
                    tensor[i][j] = vocab_dict[s[j]]
                else:
                    tensor[i][j] = 0
        return tensor

if __name__=="__main__":
    os.chdir(sys.path[0])
    train_arg1 = []
    train_arg2 = []    
    for line in open('./exam.txt','r',encoding='utf-8'):
        temp = line.split('|||')[0]
        temp = temp.replace(" ", "")
        try:
            sentence = HanLP([temp])["tok/coarse"]
            train_arg1.append(sentence[0])
        except:
            train_arg1.append([' ']) 
        temp = line.split('|||')[1]
        temp = temp.replace(" ", "")
        try:
            sentence = HanLP([temp])["tok/coarse"]
            train_arg2.append(sentence[0])
        except:
            train_arg2.append([' '])

    train_data = [
            _text2i(train_arg1, 100),
            _text2i(train_arg2, 100),
        ]
    print('saving data...')
    torch.save(train_data, './processed/hanlp/exam.pt')