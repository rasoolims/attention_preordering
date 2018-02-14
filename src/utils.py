import re, codecs, sys, random
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


def vocab(path, min_count=2):
    word_counts = defaultdict(int)
    tags = set()
    for line in codecs.open(path, 'r'):
        ws, ts = get_words_tags(normalize_sent(line.strip()))
        for w in ws:
            word_counts[w] += 1
        for t in ts:
            tags.add(t)
    return [w for w in word_counts.keys() if word_counts[w]>min_count], list(tags)

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


def create_string_output_from_order(order_file, dev_file, outfile):
    lines = codecs.open(order_file, 'r').read().strip().split('\n')
    dev_lines = codecs.open(dev_file, 'r').read().strip().split('\n')
    outputs = []
    for i in range(len(lines)):
        o = [int(l) for l in lines[i].strip().split()]
        ws = dev_lines[i].strip().split()
        words = [ws[j-1] for j in o]
        outputs.append(' '.join(words))

    open(outfile, 'w').write('\n'.join(outputs))

def eval_trigram(gold_file, out_file):
    r1 = open(gold_file, 'r')
    r2 = open(out_file, 'r')

    ac_c, all_c = 0.0, 0
    l1 = r1.readline()
    while l1:
        l2 = r2.readline()
        spl1 = ['<s>', '<s>'] + l1.strip().split() + ['</s>', '</s>']
        spl2 = ['<s>', '<s>'] + l2.strip().split() + ['</s>', '</s>']
        #assert len(spl1) == len(spl2)
        gc, oc = set(), set()
        for i in range(2, len(spl1)-2):
            s1, s2 = ' '.join(spl1[i:i+3]), ' '.join(spl2[i:i+3])
            gc.add(s1)
            oc.add(s2)
        for s in oc:
            if s in gc:
                ac_c += 1
        all_c += len(gc)

        l1 = r1.readline()
    return round(ac_c * 100.0/all_c, 2)