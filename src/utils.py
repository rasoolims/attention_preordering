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
    tags, relations, langs = set(), set(), set()
    for data in train_data:
        ws, ows, ts, lang_id = data[0]
        for w in ws:
            word_counts[w] += 1
        for w in ows:
            word_counts[w] += 1
        for t in ts:
            tags.add(t)
        langs.add(lang_id)
    return [w for w in word_counts.keys() if word_counts[w] > min_count], list(tags), list(langs)


def get_order_data(tree, h):
    data = []
    if h > 0 and len(tree.reverse_tree[h]) > 0:
        all_deps = sorted(list(tree.reverse_tree[h]) + [h])
        all_relations = []
        for d in all_deps:
            if d == h:
                all_relations.append('HEAD')
            else:
                all_relations.append(tree.labels[d - 1])

        data.append((h, all_deps, all_relations))

    for dep in tree.reverse_tree[h]:
        data += get_order_data(tree, dep)

    return data

def read_tree_as_data(tree_path):
    trees = DepTree.load_trees_from_conll_file(tree_path)
    data = []
    for tree in trees:
        words, orig_words, tags = tree.words, tree.lemmas, tree.tags
        ws = ['<EOS>'] + [normalize(words[i]) for i in range(len(words))] + ['<EOS>']
        ows = ['<EOS>'] + [normalize(orig_words[i]) for i in range(len(orig_words))] + ['<EOS>']
        tags = ['<EOS>'] + tags + ['<EOS>']

        d = (ws, ows, tags, tree.lang_id)
        order_data = get_order_data(tree, 0)
        order = dict()
        for ord in order_data:
            head = ord[0]
            deps, r_deps = ord[1], ord[1]
            rels, r_rels = ord[2], ord[2]
            num_order = [i for i in range(len(deps))]
            order[head] = [deps, rels, r_deps, r_rels, num_order]

        data.append((d, order))
    assert len(data) == len(trees)
    return trees, data


def split_data(train_path, output_path, dev_percent):
    trees = DepTree.load_trees_from_conll_file(train_path)
    orders = codecs.open(output_path, 'r').read().strip().split('\n\n')
    tdata, ddata = [], []
    relations = set()
    assert len(orders) == len(trees)
    for i in range(len(trees)):
        order = dict()
        for ord in orders[i].strip().split('\n'):
            fields = ord.strip().split('\t')
            head = int(fields[0])
            deps = [int(f) for f in fields[1].split(' ')]
            rels = [f for f in fields[2].split(' ')]
            r_deps = [int(f) for f in fields[3].split(' ')]
            r_rels = [f for f in fields[4].split(' ')]
            num_order = [int(f) for f in fields[5].split(' ')]
            for rel in rels:
                relations.add(rel)
            order[head] = [deps, rels, r_deps, r_rels, num_order]

        words, orig_words, tags= trees[i].words, trees[i].lemmas, trees[i].tags
        ws = ['<EOS>'] + [normalize(words[j]) for j in range(len(words))] + ['<EOS>']
        ows = ['<EOS>'] + [normalize(orig_words[j]) for j in range(len(orig_words))] + ['<EOS>']
        tags = ['<EOS>'] + tags + ['<EOS>']
        data = (ws, ows, tags, trees[i].lang_id)
        if random.randint(0, 100) == dev_percent:
            ddata.append((data, order))
        else:
            tdata.append((data, order))
    return tdata, ddata, sorted(list(relations))


def get_batches(buckets, model, is_train):
    d_copy = [buckets[i][:] for i in range(len(buckets))]
    if is_train:
        for dc in d_copy:
            random.shuffle(dc)
    mini_batches, dep_mini_batches = [], []
    batch, cur_len = [], 0
    for dc in d_copy:
        for d in dc:
            if (is_train and len(d[1])<=100) or not is_train:
                batch.append(d)
                cur_len = max(cur_len, len(d[0][0]))
            if cur_len * len(batch) >= model.options.batch:
                add_to_minibatch(batch, cur_len, mini_batches, model)
                add_to_dep_minibatch(batch, dep_mini_batches, model)
                batch, cur_len = [], 0

    if len(batch)>0:
        add_to_minibatch(batch, cur_len, mini_batches, model)
        add_to_dep_minibatch(batch, dep_mini_batches, model)
    if is_train:
        c = list(zip(mini_batches, dep_mini_batches))
        random.shuffle(c)
        mini_batches, dep_mini_batches = zip(*c)
    return mini_batches, dep_mini_batches


def add_to_minibatch(batch, cur_len, mini_batches, model):
    words = np.array([np.array(
        [model.w2int.get(batch[i][0][0][j], 0) if j < len(batch[i][0][0]) else model.w2int[model.EOS] for i in
         range(len(batch))]) for j in range(cur_len)])
    pwords = np.array([np.array(
        [model.evocab.get(batch[i][0][0][j], 0) if j < len(batch[i][0][0]) else 0 for i in
         range(len(batch))]) for j in range(cur_len)])
    orig_words = np.array([np.array(
        [model.w2int.get(batch[i][0][1][j], 0) if j < len(batch[i][0][1]) else model.w2int[model.EOS] for i in
         range(len(batch))]) for j in range(cur_len)])
    orig_pwords = np.array([np.array(
        [model.evocab.get(batch[i][0][1][j], 0) if j < len(batch[i][0][1]) else 0 for i in
         range(len(batch))]) for j in range(cur_len)])
    pos = np.array([np.array(
        [model.t2int.get(batch[i][0][2][j], 0) if j < len(batch[i][0][2]) else model.t2int[model.EOS] for i in
         range(len(batch))]) for j in range(cur_len)])
    langs = np.array([np.array(
        [model.lang2int.get(batch[i][0][3], 0) for i in range(len(batch))]) for j in range(cur_len)])

    sen_lens = [len(batch[i][0][0]) for i in range(len(batch))]
    masks = np.array([np.array([1 if 0 <= j < len(batch[i][0][0]) else 0 for i in range(len(batch))]) for j in range(cur_len)])
    mini_batches.append((words, pwords, orig_words, orig_pwords, pos, langs, sen_lens, masks))

def add_to_dep_minibatch(batch, mini_batches, model):
    dep_mini_batch_length = defaultdict(list)
    for i in range(len(batch)):
        order_dic = batch[i][1]
        for head in order_dic.keys():
            od, ol, ord, orl, numo = order_dic[head]
            l = len(od)
            dep_mini_batch_length[l].append((i, head, order_dic[head]))

    dep_mini_batches = dict()
    for l in dep_mini_batch_length.keys():
        sen_ids = np.array([e[0] for e in dep_mini_batch_length[l]])
        heads = np.array([e[1] for e in dep_mini_batch_length[l]])
        deps = np.array([np.array([e[2][0][j] for e in dep_mini_batch_length[l]]) for j in range(l)])
        labels = np.array([np.array([model.rel2int.get(e[2][1][j], 0) for e in dep_mini_batch_length[l]]) for j in range(l)])
        o_deps = np.array([np.array([e[2][2][j] for e in dep_mini_batch_length[l]]) for j in range(l)])
        o_labels = np.array([np.array([model.rel2int.get(e[2][3][j], 0) for e in dep_mini_batch_length[l]]) for j in range(l)])
        num_orders = np.array([np.array([e[2][4][j] for e in dep_mini_batch_length[l]]) for j in range(l)])

        dep_mini_batches[l] = sen_ids, heads, deps, labels, o_deps, o_labels, num_orders
    mini_batches.append(dep_mini_batches)

    #   heads, deps, labels, r_deps, r_labels, sen_ids, num_order = [], [], [], [], [], [], []
    #   for i in range(len(batch)):
    #     order_dic = batch[i][1]
    #     for head in order_dic.keys():
    #         heads.append(head)
    #         sen_ids.append(i)
    #         od, ol, ord, orl, numo = order_dic[head]
    #         deps.append(od)
    #         labels.append([model.rel2int.get(rel, 0) for rel in ol])
    #         r_deps.append(ord)
    #         r_labels.append([model.rel2int.get(rel, 0) for rel in orl])
    #         num_order.append(numo)
    # mini_batches.append((heads, deps, labels, r_deps, r_labels, sen_ids, num_order))


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