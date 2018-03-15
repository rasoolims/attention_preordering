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


def vocab(train_data, min_count=2):
    word_counts = defaultdict(int)
    tags, chars, relations, langs = set(), set(), set(), set()
    for data in train_data:
        ws, ts, rels, _, _, _, lang_id = data[0]
        for w in ws:
            word_counts[w] += 1
            for c in list(w):
                chars.add(c)
        for t in ts:
            tags.add(t)
        for r in rels:
            relations.add(r)
        langs.add(lang_id)
    return [w for w in word_counts.keys() if word_counts[w]>min_count], list(tags), list(relations), list(langs), list(chars)


def read_tree_as_data(tree_path):
    trees = DepTree.load_trees_from_conll_file(tree_path)
    data = []
    for tree in trees:
        words, tags,rels, heads = tree.lemmas, tree.tags, tree.labels, tree.heads
        ws = ['<EOS>'] + [normalize(words[i]) for i in range(len(words))] + ['<EOS>']
        tags = ['<EOS>'] + tags + ['<EOS>']
        relations = ['<EOS>'] + [rels[i]+ ('-left' if heads[i] >=i else '-right') for i in range(len(words))]+ ['<EOS>']
        heads = [0] + [heads[i] for i in range(len(words))] + [0]
        deps = ['<EOS>'] + [relations[i] for i in range(len(words))] + ['<EOS>']
        data = (ws, tags, relations, ws, tags, relations, tree.lang_id)
        data.append((data, [int(l) for l in range(len(ws))], heads, deps))
    assert len(data) == len(trees)
    return trees, data


def split_data(train_path, output_path, coutput_path, dev_percent):
    trees = DepTree.load_trees_from_conll_file(train_path)
    orders = codecs.open(output_path, 'r').read().strip().split('\n')
    ctrees = DepTree.load_trees_from_conll_file(coutput_path)
    tdata, ddata = [], []
    assert len(orders) == len(trees) == len(ctrees)
    for i in range(len(trees)):
        tree, ctree = trees[i], ctrees[i]
        order = [0] + [int(l) for l in orders[i].split()] + [len(orders[i].split()) + 1]
        words, tags, rels, heads = tree.lemmas, tree.tags, tree.labels, tree.heads
        cwords, ctags, crels, cheads = ctree.lemmas, ctree.tags, ctree.labels, ctree.heads
        ws = ['<EOS>'] + [normalize(words[i]) for i in range(len(words))] + ['<EOS>']
        cws = ['<EOS>'] + [normalize(cwords[i]) for i in range(len(cwords))] + ['<EOS>']
        tags = ['<EOS>'] + tags + ['<EOS>']
        ctags = ['<EOS>'] + ctags + ['<EOS>']
        relations = ['<EOS>'] + [rels[i] + ('-left' if heads[i] >= i else '-right') for i in range(len(words))] + ['<EOS>']
        crelations = ['<EOS>'] + [crels[i] + ('-left' if cheads[i] >= i else '-right') for i in range(len(cwords))] + ['<EOS>']
        heads = [0] + [heads[i] for i in range(len(words))] + [0]
        deps = ['<EOS>'] + [rels[i] for i in range(len(words))] + ['<EOS>']
        data = (ws, tags, relations, cws, ctags, crelations, tree.lang_id)
        if random.randint(0, 100) == dev_percent:
            ddata.append((data, order, heads, deps))
        else:
            tdata.append((data, order, heads, deps))
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
        [model.evocab.get(batch[i][0][0][j], 0) if j < len(batch[i][0][0]) else 0 for i in
         range(len(batch))]) for j in range(cur_len)])
    pos = np.array([np.array(
        [model.t2int.get(batch[i][0][1][j], 0) if j < len(batch[i][0][1]) else model.t2int[model.EOS] for i in
         range(len(batch))]) for j in range(cur_len)])
    rels = np.array([np.array(
        [model.rel2int.get(batch[i][0][2][j], 0) if j < len(batch[i][0][2]) else model.rel2int[model.EOS] for i in
         range(len(batch))]) for j in range(cur_len)])
    positions = np.array([np.array([batch[i][1][j] if j < len(batch[i][1]) else model.PAD for i in range(len(batch))]) for j in range(cur_len)])
    output_words = np.array([np.array(
        [model.w2int.get(batch[i][0][3][j], 0) if j < len(batch[i][0][3]) else model.w2int[model.EOS] for i in
         range(len(batch))]) for j in range(cur_len)])
    output_tags = np.array([np.array(
        [model.t2int.get(batch[i][0][4][j], 0) if j < len(batch[i][0][4]) else model.t2int[model.EOS] for i in
         range(len(batch))]) for j in range(cur_len)])
    output_rels = np.array([np.array(
        [model.rel2int.get(batch[i][0][5][j], 0) if j < len(batch[i][0][5]) else model.rel2int[model.EOS] for i in
         range(len(batch))]) for j in range(cur_len)])
    langs = np.array([np.array(
        [model.lang2int.get(batch[i][0][6], 0) for i in range(len(batch))]) for j in range(cur_len)])
    heads = np.array([np.array(
        [batch[i][2][j] if j < len(batch[i][2]) else 0 for i in range(len(batch))]) for j in range(cur_len)])
    dependencies = np.array([np.array(
        [batch[i][3][j] if j < len(batch[i][3]) else 'ROOT' for i in range(len(batch))]) for j in range(cur_len)])

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
    mini_batches.append((words, pwords, pos, rels, langs, output_words, output_tags, output_rels, heads, dependencies, positions, chars, sen_lens, masks))

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

def eval_trigram(gold_data, out_file):
    r2 = open(out_file, 'r')

    ac_c, all_c = 0.0, 0

    for i in range(len(gold_data)):
        l2 = r2.readline()
        spl1 = ['<s>', '<s>'] + [str(gold_data[i][1][j]) for j in range(1, len(gold_data[i][1])-1)] + ['</s>', '</s>']
        spl2 = ['<s>', '<s>'] + l2.strip().split() + ['</s>', '</s>']
        assert len(spl1)==len(spl2)
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