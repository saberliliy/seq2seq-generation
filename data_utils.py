#! /usr/bin/env python
#-*- coding:utf-8 -*-

import codecs
import os
import random
from functools import reduce
import numpy as np

from cnt_words import get_pop_quatrains
from rank_words import get_word_ranks
from segment import Segmenter
from utils import DATA_PROCESSED_DIR, embed_w2v, apply_one_hot, apply_sparse, pad_to, SEP_TOKEN, PAD_TOKEN
from vocab import ch2int, VOCAB_SIZE, sentence_to_ints
from word2vec import get_word_embedding
import jieba.posseg as peg
sxhy_path = os.path.join(DATA_PROCESSED_DIR, 'sxhy_dict.txt')
train_path = os.path.join(DATA_PROCESSED_DIR, 'train.txt')
cangtou_train_path = os.path.join(DATA_PROCESSED_DIR, 'cangtou_train.txt')
kw_train_path = os.path.join(DATA_PROCESSED_DIR, 'kw_train.txt')
test_path=os.path.join(DATA_PROCESSED_DIR,"test.txt")

def random_int_list(start, stop, length):
  start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
  length = int(abs(length)) if length else 0
  random_list = []
  for i in range(length):
    random_list.append(random.randint(start, stop))
  return random_list
def fill_np_matrix(vects, batch_size, value):
    max_len = max(len(vect) for vect in vects)
    res = np.full([batch_size, max_len], value, dtype=np.int32)
    for row, vect in enumerate(vects):
        res[row, :len(vect)] = vect
    return res


def fill_np_array(vect, batch_size, value):
    result = np.full([batch_size], value, dtype=np.int32)
    result[:len(vect)] = vect
    return result


def _gen_train_data():
    sampled_poems = np.array(random_int_list(1, 70000, 4000))
    segmenter = Segmenter()  #generation   sxhy dict
    poems = get_pop_quatrains() #获得较为流行的10万首诗
    random.shuffle(poems)       #重新排序
    ranks = get_word_ranks()    #Textrank  word  -rank_number
    print( "Generating training data ...")
    data = []
    kw_data = []
    test_data=[]
    for idx, poem in enumerate(poems):
        sentences = poem['sentences']
        if len(sentences) == 4:
            flag = True
            test_flag=True
            rows = []
            kw_row = []
            test_row=[]
            if idx in sampled_poems:
                test_flag=False
            for sentence in sentences:
                rows.append([sentence])
                test_row.append([sentence])
                segs = list(filter(lambda seg: seg in ranks, segmenter.segment(sentence)))
                if 0 == len(segs):
                    flag = False
                    break
                keyword = reduce(lambda x,y: x if ranks[x] < ranks[y] else y, segs)  #选取权重比较大的keywords
                kw_row.append(keyword)
                rows[-1].append(keyword)
            if flag and test_flag:
                data.extend(rows)
                kw_data.append(kw_row)
            if flag and test_flag is False:
                test_data.extend(test_row)

        if 0 == (idx+1)%2000:
            print ("[Training Data] %d/%d poems are processed." %(idx+1, len(poems)))
    print(test_data)
    with codecs.open(train_path, 'w', 'utf-8') as fout:
        for row in data:
            fout.write('\t'.join(row)+'\n')
    with codecs.open(kw_train_path, 'w', 'utf-8') as fout:
        for kw_row in kw_data:
            fout.write('\t'.join(kw_row)+'\n')
    with codecs.open(test_path, 'w', 'utf-8') as fout:
        for  test_row in test_data:
            fout.write('\t'.join(test_row)+'\n')
    print( "Training data is generated.")


# TODO(vera): find a better name than cangtou...
def _gen_cangtou_train_data():
    poems = get_pop_quatrains()
    random.shuffle(poems)
    with codecs.open(cangtou_train_path, 'w', 'utf-8') as fout:
        for idx, poem in enumerate(poems):
            for sentence in poem['sentences']:
                fout.write(sentence + "\t" + sentence[0] + "\n")
            if 0 == (idx + 1) % 2000:
                print ("[Training Data] %d/%d poems are processed." %(idx+1, len(poems)))
    print ("Cangtou training data is generated.")


def get_train_data(cangtou=False):
    train_data_path =train_path  #选择是否生成藏头诗
    if not os.path.exists(train_data_path):
        if cangtou:
            _gen_cangtou_train_data()
        else:
            _gen_train_data()

    data = []
    with codecs.open(train_data_path, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            toks = line.strip().split('\t')
            data.append({'sentence':toks[0], 'keyword':toks[1]})
            line = fin.readline()
    return data

def get_kw_train_data():
    if not os.path.exists(kw_train_path):
        _gen_train_data()
    data = []
    with codecs.open(kw_train_path, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            data.append(line.strip().split('\t'))
            line = fin.readline()
    return data


def batch_train_data(batch_size):
    """Get training data in poem, batch major format

    Args:
        batch_size:

    Returns:
        kw_mats: [4, batch_size, time_steps]
        kw_lens: [4, batch_size]
        s_mats: [4, batch_size, time_steps]
        s_lens: [4, batch_size]
    """
    if not os.path.exists(train_path):
        _gen_train_data()
    with codecs.open(train_path, 'r', 'utf-8') as fin:
        stop = False
        while not stop:
            batch_s = [[] for _ in range(4)]
            batch_kw = [[] for _ in range(4)]
            # NOTE(sdsuo): Modified batch size to remove empty lines in batches
            for i in range(batch_size * 4):
                line = fin.readline()
                if not line:
                    stop = True
                    break
                else:
                    toks = line.strip().split('\t')
                    # NOTE(sdsuo): Removed start token
                    batch_s[i%4].append([ch2int[ch] for ch in toks[0]])
                    batch_kw[i%4].append([ch2int[ch] for ch in toks[1]])
            if batch_size != len(batch_s[0]):
                print( 'Batch incomplete with size {}, expecting size {}, dropping batch.'.format(len(batch_s[0]), batch_size))
                break
            else:
                kw_mats = [fill_np_matrix(batch_kw[i], batch_size, VOCAB_SIZE-1) \
                        for i in range(4)]
                kw_lens = [fill_np_array(list(map(len, batch_kw[i])), batch_size, 0) \
                        for i in range(4)]
                s_mats = [fill_np_matrix(batch_s[i], batch_size, VOCAB_SIZE-1) \
                        for i in range(4)]
                s_lens = [fill_np_array([len(x) for x in batch_s[i]], batch_size, 0) \
                        for i in range(4)]
                yield kw_mats, kw_lens, s_mats, s_lens


def process_sentence(sentence, rev=False, pad_len=None, pad_token=PAD_TOKEN):
    if rev:
        sentence = sentence[::-1]   #将句子反转

    sentence_ints = sentence_to_ints(sentence)   #word-idx

    if pad_len is not None:
        result_len = len(sentence_ints)
        for i in range(pad_len - result_len):    #填充pad_token
            sentence_ints.append(pad_token)

    return sentence_ints


def prepare_batch_predict_data(keyword, previous, prev=True, rev=False, align=False):
    # previous sentences
    content_tmp = [SEP_TOKEN]
    for sentence in previous:
        # if previous=[] 整个for 循环都不执行
        sentence_ints = process_sentence(sentence, rev=rev, pad_len=7 if align else None)
        content_tmp += [SEP_TOKEN] + sentence_ints

    # keywords
    keywords_ints = process_sentence(keyword, rev=rev, pad_len=4 if align else None)
    keyword_len = len(keywords_ints)
    content_input=content_tmp if prev else []
    content_input_length=len(content_input)
    keyword_inputs = fill_np_matrix([keywords_ints], 1, PAD_TOKEN)
    keyword_length = np.array([keyword_len])
    content_inputs_length = np.array([content_input_length])
    content_inputs = fill_np_matrix([content_input], 1, PAD_TOKEN)
    return content_inputs, content_inputs_length,keyword_inputs,keyword_length


def gen_batch_train_data(batch_size, prev=True, rev=False, align=False, cangtou=False):
    """
    Get training data in batch major format, with keyword and previous sentences as source,
    aligned and reversed

    Args:
        batch_size:

    Returns:
        source: [batch_size, time_steps]: keywords + SEP + previous sentences
        source_lens: [batch_size]: length of source
        target: [batch_size, time_steps]: current sentence
        target_lens: [batch_size]: length of target
    """
    train_data_path = train_path
    if not os.path.exists(train_data_path):
            _gen_train_data()

    with codecs.open(train_data_path, 'r', 'utf-8') as fin:
        stop = False
        while not stop:
            keyword = []
            keyword_length = []
            content=[]
            content_length=[]
            target = []
            target_length = []
            content_tmp = []
            for i in range(batch_size):
                line = fin.readline()
                if not line:
                    stop = True
                    break
                else:
                    line_number = i % 4
                    sentence, keywords = line.strip().split('\t')
                    if line_number == 0:
                        content_tmp=[SEP_TOKEN]
                    sentence_ints = process_sentence(sentence, rev=rev, pad_len=7 if align else None)   #将句子转换成数字，并且进行填充
                    keywords_ints = process_sentence(keywords, rev=rev, pad_len=4 if align else None)
                    keyword.append(keywords_ints)
                    keyword_length.append(len(keywords_ints))
                    content.append(content_tmp)
                    content_length.append(len(content_tmp))
                    target.append(sentence_ints)
                    target_length.append(len(sentence_ints))

                    # Always append to previous sentences
                    content_tmp += [SEP_TOKEN] + sentence_ints

            if len(keyword) == batch_size:
                keywordes = fill_np_matrix(keyword, batch_size, PAD_TOKEN)
                contents = fill_np_matrix(content, batch_size, PAD_TOKEN)
                targets = fill_np_matrix(target, batch_size, PAD_TOKEN)
                keyword_lengths = np.array(keyword_length)
                content_lengths = np.array(content_length)
                target_lengths=np.array(target_length)

                yield keywordes, keyword_lengths,contents,content_lengths,targets,target_lengths


def main():
    train_data = get_train_data()         #sentence  keyword
    print( "Size of the training data: %d" %len(train_data))
    kw_train_data = get_kw_train_data()      # keyword
    print ("Size of the keyword training data: %d" %len(kw_train_data))
    assert len(train_data) == 4 * len(kw_train_data)          #kw_train_data  四行古诗的关键词组成一行


if __name__ == '__main__':
    main()
