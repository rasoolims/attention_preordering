import re, codecs, sys, random, gzip, pickle
import numpy as np
from collections import defaultdict

reload(sys)
sys.setdefaultencoding('utf8')

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
urlRegex = re.compile("((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)")

def normalize(word):
    return '<num>' if numberRegex.match(word) else ('<url>' if urlRegex.match(word) else word.lower())

def normalize_sent(sent):
    words, tags = get_words_tags(sent)
    return ' '.join([normalize(words[i])+'_'+tags[i] for i in range(len(words))])

def get_words_tags(sent):
    words, tags = [], []
    for sen_t in sent.strip().split():
        r = sen_t.rfind('_')
        words.append(sen_t[:r])
        tags.append(sen_t[r + 1:])
    return words, tags

def vocab(path):
    word_counts = defaultdict(int)
    tags = set()
    for line in codecs.open(path, 'r'):
        ws, ts = get_words_tags(normalize_sent(line.strip()))
        for w in ws:
            word_counts[w] += 1
        for t in ts:
            tags.add(t)
    return [w for w in word_counts.keys() if word_counts[w]>2], list(tags)

def read_data(train_path, output_path):
    t1 = codecs.open(train_path, 'r')
    t2 = codecs.open(output_path, 'r')
    data = []
    l1 = t1.readline()
    while l1:
        l2 = t2.readline()
        data.append((get_words_tags(normalize_sent(l1.strip())), [int(l)-1 for l in l2.split()]))
        l1 = t1.readline()
    return data