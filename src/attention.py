import dynet as dy
import codecs, sys, time, gzip
import numpy as np
from collections import defaultdict

class MT:
    def __init__(self, options, words, tags, relations, langs):
        self.ignore_deps = set(['conj','cc','fixed','flat','compound','list','parataxis','orphan','goeswith','reparandum','punct','root','discourse','dep', '_','case','clf','det','mark'])
        self.EOS = "<EOS>"
        self.PAD = 1
        self.options = options
        words.append(self.EOS)
        tags.append(self.EOS)
        self.int2w = ['<unk>', '<pad>'] + words
        self.int2t = ['<unk>', '<pad>'] + tags
        self.w2int = {w: i + 2 for i, w in enumerate(words)}
        self.t2int = {t: i + 2 for i, t in enumerate(tags)}
        self.rel2int = {r: i + 2 for i, r in enumerate(relations)}
        self.lang2int = {l: i + 2 for i, l in enumerate(langs)}
        self.WVOCAB_SIZE = len(self.int2w)
        self.TVOCAB_SIZE = len(self.int2t)

        self.ignore_deps_ids = set()
        for id in self.ignore_deps:
            self.ignore_deps_ids.add(self.rel2int[id])

        self.LSTM_NUM_OF_LAYERS = options.layer
        self.DEP_LSTM_NUM_OF_LAYERS = options.dep_layer
        self.ATTENTION_SIZE = options.attention
        self.model = dy.Model()

        external_embedding_fp = gzip.open(options.external_embedding, 'r')
        external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in
                              external_embedding_fp if len(line.split(' ')) > 2}
        external_embedding_fp.close()
        self.evocab = {word: i + 2 for i, word in enumerate(external_embedding)}

        edim = len(external_embedding.values()[0])
        assert edim == options.we
        self.elookup = self.model.add_lookup_parameters((len(external_embedding) + 2, edim))
        self.elookup.set_updated(False)
        self.elookup.init_row(0, [0] * edim)
        for word in external_embedding.keys():
            self.elookup.init_row(self.evocab[word], external_embedding[word])
            if word == '_UNK_':
                self.elookup.init_row(0, external_embedding[word])

        print 'Initialized with pre-trained embedding. Vector dimensions', edim, 'and', len(external_embedding), \
            'words, number of training words', len(self.w2int) + 2

        input_dim = 2 * options.we + options.pe
        self.encoder_bilstm = dy.BiRNNBuilder(self.LSTM_NUM_OF_LAYERS, input_dim, options.hdim * 2, self.model, dy.VanillaLSTMBuilder)

        self.wlookup = self.model.add_lookup_parameters((self.WVOCAB_SIZE, options.we))
        self.tlookup = self.model.add_lookup_parameters((self.TVOCAB_SIZE, options.pe))
        self.rlookup = self.model.add_lookup_parameters((len(self.rel2int)+2, options.re))
        self.dir_lookup = self.model.add_lookup_parameters((2, 2))
        self.llookup = self.model.add_lookup_parameters((len(self.lang2int)+2, options.le))

        self.H = self.model.add_parameters((options.hdim, options.hdim *  4 + options.re + options.le + 2))
        self.HB = self.model.add_parameters((options.hdim, ), init = dy.ConstInitializer(0.2))
        self.O = self.model.add_parameters((2,  options.hdim))
        self.trainer = dy.AdamTrainer(self.model, options.lr, options.beta1, options.beta2)

        def _emb_mask_generator(seq_len, batch_size):
            ret = []
            for _ in xrange(seq_len):
                word_mask = np.random.binomial(1, 1. - self.options.dropout, batch_size).astype(np.float32)
                tag_mask = np.random.binomial(1, 1. - self.options.dropout, batch_size).astype(np.float32)
                rel_mask = np.random.binomial(1, 1. - self.options.dropout, batch_size).astype(np.float32)
                lang_mask = np.random.binomial(1, 1. - self.options.dropout, batch_size).astype(np.float32)
                scale = 6. / (3. * word_mask + tag_mask + rel_mask + lang_mask + 1e-12)
                word_mask *= scale
                tag_mask *= scale
                rel_mask *= scale
                lang_mask *= scale
                word_mask = dy.inputTensor(word_mask, batched=True)
                tag_mask = dy.inputTensor(tag_mask, batched=True)
                rel_mask = dy.inputTensor(rel_mask, batched=True)
                lang_mask = dy.inputTensor(lang_mask, batched=True)
                ret.append((word_mask, tag_mask, rel_mask, lang_mask))

            return ret

        self.generate_emb_mask = _emb_mask_generator

    def embed_sentence(self, ws, pwords, ows, o_pwords, ts):
        wembed = [dy.lookup_batch(self.wlookup, ws[i]) + dy.lookup_batch(self.elookup, pwords[i]) for i in range(len(ws))]
        owembed = [dy.lookup_batch(self.wlookup, ows[i]) + dy.lookup_batch(self.elookup, o_pwords[i]) for i in range(len(ows))]
        posembed = [dy.lookup_batch(self.tlookup, ts[i]) for i in range(len(ts))]
        return [dy.concatenate([wembed[i], posembed[i], owembed[i]]) for i in range(len(ts))]

    def encode_sentence(self, input_embeds):
        for fb, bb in self.encoder_bilstm.builder_layers:
            f, b = fb.initial_state(), bb.initial_state()
            fs, bs = f.transduce(input_embeds), b.transduce(reversed(input_embeds))
            input_embeds = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
        return input_embeds



    def generate(self, minibatch, dep_minibatch):
        output_values = self.decode(minibatch, dep_minibatch).npvalue()
        outputs = np.argmax(output_values, axis=0)
        dy.renew_cg()
        return outputs

    def get_loss(self, minibatch, dep_minibatch):
        output_layer = self.decode(minibatch, dep_minibatch)
        loss = dy.pickneglogsoftmax_batch(output_layer, dep_minibatch[-1])
        return loss

    def decode(self, minibatch, dep_minibatch):
        words, pwords, orig_words, orig_pwords, tags, sen_lens, masks = minibatch
        embedded = self.embed_sentence(words, pwords, orig_words, orig_pwords, tags)
        encoded = self.encoder_bilstm.transduce(embedded)
        sentence_embeddings = [dy.transpose(dy.reshape(e, (e.dim()[0][0], e.dim()[1]))) for e in encoded]
        sen_ids, heads, deps, labels, directions, langs, _ = dep_minibatch
        rel_embedding = dy.lookup_batch(self.rlookup, labels)
        lang_embedding = dy.lookup_batch(self.llookup, langs)
        dir_embedding = dy.lookup_batch(self.dir_lookup, directions)
        head_se = dy.concatenate_to_batch([sentence_embeddings[h][sen_ids[i]] for i, h in enumerate(heads)])
        dep_se = dy.concatenate_to_batch([sentence_embeddings[d][sen_ids[i]] for i, d in enumerate(deps)])
        input_layer = dy.concatenate([rel_embedding, dir_embedding, lang_embedding, head_se, dep_se])
        h = dy.rectify(dy.affine_transform([self.HB.expr(), self.H.expr(), input_layer]))
        output_layer = self.O.expr() * h
        return output_layer

    def train(self, train_batches, train_dep_batches, dev_batches, dev_dep_batches, dev_out, t):
        start = time.time()
        loss_sum, b = 0, 0
        for d_i, minibatch in enumerate(train_batches):
            loss = self.get_loss(minibatch, train_dep_batches[d_i])
            loss_sum += self.backpropagate(loss)
            b += 1
            dy.renew_cg()

            if b % 100 == 0:
                progress = round((d_i + 1) * 100.0 / len(train_batches), 2)
                print 'progress', str(progress), '%', 'loss', loss_sum / b, 'time', time.time() - start
                start = time.time()
                loss_sum, b = 0, 0
            if self.options.anneal:
                decay_steps = min(1.0, float(t) / 50000)
                lr = self.options.lr * 0.75 ** decay_steps
                self.trainer.learning_rate = lr
            t += 1

        all_outputs, correct = 0, 0
        for di in range(len(dev_batches)):
            outputs = self.generate(dev_batches[di], dev_dep_batches[di])
            gold_orders = dev_dep_batches[di][-1]
            correct += np.sum(outputs == gold_orders)
            all_outputs += len(outputs)
            dy.renew_cg()
        dev_acc = round(float(100 * correct) / all_outputs, 2)
        print 'dev accuracy', dev_acc
        return t, dev_acc

    def reorder(self, batches, dep_minibatch,  out_file):
        writer = codecs.open(out_file, 'w')
        for d, minibatch in enumerate(batches):
            writer.write('\n'.join(self.get_output(minibatch, dep_minibatch)) + '\n')
            if (d + 1) % 100 == 0:
                sys.stdout.write(str(d + 1) + '...')
        sys.stdout.write(str(d) + '\n')
        writer.close()

    def reorder_tree(self, batches, dep_batches, trees, out_file):
        writer = codecs.open(out_file, 'w')
        print 'get new order'
        new_trees, t_num = [], 0
        offset = 0
        num_ignored = 0
        left_output_orders = defaultdict(dict)
        right_output_orders = defaultdict(dict)
        for d in range(len(batches)):
            output = self.generate(batches[d], dep_batches[d])
            sen_ids, heads, deps, labels, directions, langs, _ = dep_batches[d]
            max_id = max(sen_ids)
            predicted, orig_deps = output.T, deps.T
            for p in range(len(predicted)):
                sen_id = sen_ids[p] + offset
                if not heads[p] in left_output_orders[sen_id]:
                    left_output_orders[sen_id][heads[p]] = list()
                    right_output_orders[sen_id][heads[p]] = list()
                if labels[p] in self.ignore_deps_ids:
                    predicted[p] = directions[p]
                    num_ignored+= 1
                if predicted[p] == 0:
                    left_output_orders[sen_id][heads[p]].append(deps[p])
                else:
                    right_output_orders[sen_id][heads[p]].append(deps[p])

            offset += max_id + 1
            if (d + 1) % 100 == 0:
                sys.stdout.write(str(d + 1) + '...')
        sys.stdout.write(str(d) + '\n')
        for t, tree in enumerate(trees):
            output_order = dict()
            for head in left_output_orders[t].keys():
                output_order[head] = left_output_orders[t][head] + [head] + right_output_orders[t][head]
            new_linear_order = tree.get_linear_order(0, output_order)
            new_trees.append(tree.reorder_with_order(new_linear_order))
        for tree in new_trees:
            writer.write(tree.conll_str())
            writer.write('\n\n')
        writer.close()
        print 'num_ignored', num_ignored

    def backpropagate(self, loss):
        loss = dy.sum_batches(loss) / loss.dim()[1]
        loss_value = loss.value()
        loss.backward()
        self.trainer.update()
        return loss_value

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.populate(filename)

