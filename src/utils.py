import re, codecs, sys, random
import numpy as np
from collections import defaultdict
from dep_tree import DepTree

reload(sys)
sys.setdefaultencoding('utf8')

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
urlRegex = re.compile("((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)")

def normalize(word):
    return '<num>' if numberRegex.match(word) else ('<url>' if urlRegex.match(word) else word.lower())

def normalize_sent(sent):
    words, tags = get_words_tags(sent, False)
    return ' '.join([normalize(words[i])+'_'+tags[i] for i in range(len(words))])

def get_words_tags(sent, add_eos=True):
    words, tags = [], []
    if add_eos:
        words, tags = ['<EOS>'], ['<EOS>']
    for sen_t in sent.strip().split():
        r = sen_t.rfind('_')
        words.append(sen_t[:r])
        tags.append(sen_t[r + 1:])
    if add_eos:
        words.append('<EOS>')
        tags.append('<EOS>')
    return words, tags

def vocab(train_data, min_count=2):
    word_counts = defaultdict(int)
    tags, chars = set(), set()
    output_words = set(['<EOS>'])
    for data in train_data:
        ws, ts = data[0]
        for w in ws:
            word_counts[w] += 1
            for c in list(w):
                chars.add(c)
        for t in ts:
            tags.add(t)
        for w in data[2]:
            output_words.add(w)
    return [w for w in word_counts.keys() if word_counts[w]>min_count], list(tags), list(chars), list(output_words)


def read_tree_as_data(tree_path):
    trees = DepTree.load_trees_from_conll_file(tree_path)
    data = []
    for tree in trees:
        words, tags = tree.lemmas, tree.tags
        sen = ' '.join([words[i]+'_'+tags[i] for i in range(len(words))])
        data.append((get_words_tags(normalize_sent(sen)), [0] + [int(l) for l in range(len(words))] + [len(words) + 1], ['<EOS>'] + words + ['<EOS>']))
    assert len(data) == len(trees)
    return trees, data


def read_data(train_path, output_path, word_output_path):
    t1 = codecs.open(train_path, 'r')
    t2 = codecs.open(output_path, 'r')
    t3 = codecs.open(word_output_path, 'r')
    data = []
    l1 = t1.readline()
    while l1:
        l2 = t2.readline()
        l3 = t3.readline()
        words, tags = get_words_tags(normalize_sent(l1.strip()))
        data.append(((words, tags), [0]+[int(l) for l in l2.split()]+[len(l2.split())+1], ['<EOS>'] + l3.strip().split() + ['<EOS>']))
        l1 = t1.readline()
    return data


def split_data(train_path, output_path, word_output_path):
    t1 = codecs.open(train_path, 'r')
    t2 = codecs.open(output_path, 'r')
    t3 = codecs.open(word_output_path, 'r')
    tdata, ddata = [], []
    l1 = t1.readline()
    while l1:
        l2 = t2.readline()
        l3 = t3.readline().lower()
        words, tags = get_words_tags(normalize_sent(l1.strip()))
        w_output = ['<EOS>'] + l3.strip().split() + ['<EOS>']
        if random.randint(0, 9)==9:
            ddata.append(((words, tags), [0] + [int(l) for l in l2.split()] + [len(l2.split()) + 1], w_output))
        else:
            tdata.append(((words, tags), [0]+[int(l) for l in l2.split()]+[len(l2.split())+1], w_output))
        l1 = t1.readline()
    return tdata, ddata


def get_batches(buckets, model, is_train):
    d_copy = [buckets[i][:] for i in range(len(buckets))]
    if is_train:
        for dc in d_copy:
            random.shuffle(dc)
    mini_batches = []
    batch, cur_len, cur_c_len = [], 0, 0
    for dc in d_copy:
        for d in dc:
            if (is_train and len(d[1])<=100) or not is_train:
                batch.append(d)
                cur_c_len = max(cur_c_len, max([len(w) for w in d[0][0]]))
                cur_len = max(cur_len, len(d[0][0]))

            if cur_len * len(batch) >= model.options.batch:
                add_to_minibatch(batch, cur_c_len, cur_len, mini_batches, model)
                batch, cur_len, cur_c_len = [], 0, 0

    if len(batch)>0:
        add_to_minibatch(batch, cur_c_len, cur_len, mini_batches, model)
        batch, cur_len = [], 0
    if is_train:
        random.shuffle(mini_batches)
    return mini_batches


def add_to_minibatch(batch, cur_c_len, cur_len, mini_batches, model):
    words = np.array([np.array(
        [model.w2int.get(batch[i][0][0][j], 0) if j < len(batch[i][0][0]) else model.w2int[model.EOS] for i in
         range(len(batch))]) for j in range(cur_len)])
    pwords = np.array([np.array(
        [model.evocab.get(batch[i][0][0][j], 0) if j < len(batch[i][0][0]) else model.w2int[model.EOS] for i in
         range(len(batch))]) for j in range(cur_len)])
    pos = np.array([np.array(
        [model.t2int.get(batch[i][0][1][j], 0) if j < len(batch[i][0][1]) else model.t2int[model.EOS] for i in
         range(len(batch))]) for j in range(cur_len)])
    positions = np.array([np.array([batch[i][1][j] if j < len(batch[i][1]) else model.PAD for i in range(len(batch))]) for j in range(cur_len)])
    output_words = np.array([np.array(
        [model.w2int.get(batch[i][0][0][batch[i][1][j]], 0) if j < len(batch[i][0][0]) else model.w2int[model.EOS] for i in
         range(len(batch))]) for j in range(cur_len)])
    foreign_words = np.array([np.array(
        [model.o2int.get(batch[i][2][j], 0) if j < len(batch[i][2]) else model.o2int[model.EOS] for i in
         range(len(batch))]) for j in range(cur_len)])

    chars = [list() for _ in range(cur_c_len)]
    for c_pos in range(cur_c_len):
        ch = [model.PAD] * (len(batch) * cur_len)
        offset = 0
        for w_pos in range(cur_len):
            for sen_position in range(len(batch)):
                if w_pos < len(batch[sen_position]) and c_pos < len(batch[sen_position][0][0][w_pos]):
                    ch[offset] = model.c2int.get(batch[sen_position][0][0][w_pos][c_pos], 0)
                offset += 1
        chars[c_pos] = np.array(ch)
    chars = np.array(chars)
    sen_lens = [len(batch[i][0][0]) for i in range(len(batch))]
    masks = np.array([np.array([1 if 0 <= j < len(batch[i][0][0]) else 0 for i in range(len(batch))]) for j in range(cur_len)])
    mini_batches.append((words, pwords, pos, output_words, positions, chars, sen_lens, foreign_words, masks))

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

def eval_trigram(gold_data, out_file, is_str = False):
    r2 = open(out_file, 'r')

    ac_c, all_c = 0.0, 0
    index = 2 if is_str else 1
    for i in range(len(gold_data)):
        l2 = r2.readline()
        spl1 = ['<s>', '<s>'] + [str(gold_data[i][index][j]) for j in range(1, len(gold_data[i][index])-1)] + ['</s>', '</s>']
        spl2 = ['<s>', '<s>'] + l2.strip().split() + ['</s>', '</s>']
        assert len(spl1) == len(spl2)
        gc, oc = set(), set()
        for i in range(2, len(spl1)-2):
            s1, s2 = ' '.join(spl1[i:i+3]), ' '.join(spl2[i:i+3])
            gc.add(s1)
            oc.add(s2)
        for s in oc:
            if s in gc:
                ac_c += 1
        all_c += len(gc)

    return round(ac_c * 100.0/all_c, 2)

