import codecs, os, sys
from collections import defaultdict


class DepTree:
    def __init__(self, words, lemmas, tags, heads, labels):
        self.words = words
        self.lemmas = lemmas
        self.tags = tags
        self.ftags = tags
        self.heads = heads
        self.labels = labels
        self.reverse_tree = defaultdict(set)
        self.lang_id = ''
        self.weight = 1.0

        self.index = dict()
        self.reverse_index = dict()
        for i in range(0, len(words)):
            self.index[i + 1] = i + 1
            self.reverse_index[i + 1] = i + 1

        # We need to increment index by one, because of the root.
        for i in range(0, len(heads)):
            self.reverse_tree[heads[i]].add(i + 1)

    @staticmethod
    def load_tree_from_string(tree_str):
        spl = tree_str.strip().split('\n')
        words = spl[0].split()
        tags = spl[1].split()
        labels = spl[2].split()
        heads = [int(x) for x in spl[3].split()]
        return DepTree(words, words, tags, heads, labels)

    @staticmethod
    def load_tree_from_conll_string(tree_str):
        lines = tree_str.strip().split('\n')
        words = list()
        tags = list()
        heads = list()
        labels = list()
        lemmas = list()
        ftags = list()

        l_id = ''
        w = 1
        for line in lines:
            spl = line.split('\t')
            words.append(spl[1])
            lemmas.append(spl[2])
            tags.append(spl[3])
            ftags.append(spl[4])
            heads.append(int(spl[6]))
            l_id = spl[5]
            labels.append(spl[7])
            try:
                w = float(spl[8])
            except:
                w = 1

        tree = DepTree(words, lemmas, tags, heads, labels)
        tree.lang_id = l_id
        tree.ftags = ftags
        tree.weight = w
        return tree

    @staticmethod
    def load_trees_from_file(file_str):
        tree_list = list()
        [tree_list.append(DepTree.load_tree_from_string(tree_str)) for tree_str in
         codecs.open(file_str, 'r').read().strip().split('\n\n')]
        return tree_list

    @staticmethod
    def load_trees_from_conll_file(file_str):
        tree_list = list()
        [tree_list.append(DepTree.load_tree_from_conll_string(tree_str)) for tree_str in
         codecs.open(file_str, 'r').read().strip().split('\n\n')]
        return tree_list

    def get_span_list(self, head, span_set):
        span_set.add(head)
        if self.reverse_tree.has_key(head):
            for child in self.reverse_tree[head]:
                self.get_span_list(child, span_set)

    def reorder(self, new_order):
        new_words, new_lemmas, new_tags, new_heads, new_labels = [], [], [], [], []
        rev_order = {0:0, -1:-1}
        for i, o in enumerate(new_order):
            new_words.append(self.words[o-1])
            new_lemmas.append(self.lemmas[o - 1])
            new_tags.append(self.tags[o - 1])
            new_labels.append(self.labels[o - 1])
            rev_order[o] = i + 1

        for o in new_order:
            try:
                new_head = rev_order[self.heads[o-1]]
                new_heads.append(new_head)
            except:
                print o-1
                print self.heads[o-1]
                print rev_order[self.heads[o-1]]
                new_head = rev_order[self.heads[o - 1]]
                new_heads.append(new_head)

        tree = DepTree(new_words, new_lemmas, new_tags, new_heads, new_labels)
        tree.lang_id = self.lang_id
        return tree

    def tree_str(self):
        lst = list()
        lst.append('\t'.join(self.words))
        lst.append('\t'.join(self.tags))
        lst.append('\t'.join(self.labels))
        lst.append('\t'.join(str(x) for x in self.heads))
        return '\n'.join(lst)

    def conll_str(self):
        lst = list()

        for i in range(0, len(self.words)):
            ln = str(i + 1) + '\t' + self.words[i] + '\t' + self.lemmas[i] + '\t' + self.tags[i] + '\t' + self.ftags[
                i] + '\t' + self.lang_id + '\t' + str(self.heads[i]) + '\t' + self.labels[i] + '\t' + str(
                self.weight) + '\t_'
            lst.append(ln)
        return '\n'.join(lst)
