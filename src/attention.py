import dynet as dy
import codecs, sys, time, gzip
import numpy as np

class MT:
    def __init__(self, options, words, tags, relations, langs):
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

        input_dim = 2 * options.we + options.pe + options.le
        self.encoder_bilstm = dy.BiRNNBuilder(self.LSTM_NUM_OF_LAYERS, input_dim, options.hdim * 2, self.model, dy.VanillaLSTMBuilder)
        dep_inp_dim = options.hdim * 2 + options.re
        self.dep_encoder_bilstm = dy.BiRNNBuilder(self.DEP_LSTM_NUM_OF_LAYERS, dep_inp_dim, options.hdim * 2, self.model, dy.VanillaLSTMBuilder)

        self.dec_lstm = dy.LSTMBuilder(self.DEP_LSTM_NUM_OF_LAYERS, dep_inp_dim + options.hdim * 2, options.phdim, self.model)

        self.wlookup = self.model.add_lookup_parameters((self.WVOCAB_SIZE, options.we))
        self.tlookup = self.model.add_lookup_parameters((self.TVOCAB_SIZE, options.pe))
        self.rlookup = self.model.add_lookup_parameters((len(self.rel2int)+2, options.re))
        self.llookup = self.model.add_lookup_parameters((len(self.lang2int)+2, options.le))
        self.attention_w1 = self.model.add_parameters((self.ATTENTION_SIZE, options.hdim * 2))
        self.attention_w2 = self.model.add_parameters((self.ATTENTION_SIZE, options.phdim * self.DEP_LSTM_NUM_OF_LAYERS * 2))
        self.attention_v = self.model.add_parameters((1, self.ATTENTION_SIZE))
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

    def embed_sentence(self, ws, pwords, ows, o_pwords, ts, langs):
        wembed = [dy.lookup_batch(self.wlookup, ws[i]) + dy.lookup_batch(self.elookup, pwords[i]) for i in range(len(ws))]
        owembed = [dy.lookup_batch(self.wlookup, ows[i]) + dy.lookup_batch(self.elookup, o_pwords[i]) for i in range(len(ows))]
        posembed = [dy.lookup_batch(self.tlookup, ts[i]) for i in range(len(ts))]
        langembed = [dy.lookup_batch(self.llookup, langs[i]) for i in range(len(langs))]
        return [dy.concatenate([wembed[i], posembed[i], owembed[i], langembed[i]]) for i in range(len(ts))]

    def encode_sentence(self, input_embeds):
        for fb, bb in self.encoder_bilstm.builder_layers:
            f, b = fb.initial_state(), bb.initial_state()
            fs, bs = f.transduce(input_embeds), b.transduce(reversed(input_embeds))
            input_embeds = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
        return input_embeds

    def attend(self, state, w1dt, is_train):
        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = self.attention_w2.expr() * dy.concatenate(list(state.s()))
        # if is_train:
        #     w2dt = dy.dropout(w2dt, self.options.dropout)
        # att_weights: (seqlen,) row vector
        unnormalized = dy.transpose(self.attention_v.expr() * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized) + dy.scalarInput(1e-12)
        return att_weights

    def decode(self, encoded, out_e, out_idx, length):
        input_mat = dy.concatenate_cols(encoded)
        w1dt = None

        empty_tensor = dy.reshape(dy.inputTensor(np.zeros((self.options.hdim * 4 + self.options.re, len(out_idx[0])), dtype=float)),
                                  (self.options.hdim * 4 + self.options.re,), len(out_idx[0]))
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([empty_tensor]))
        loss = []
        for p in range(length):
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or self.attention_w1.expr() * input_mat
            att_weights = self.attend(s, w1dt, True)
            vector = dy.concatenate([input_mat * att_weights, out_e[p]])
            # vector = dy.dropout(vector, self.options.dropout)
            s = s.add_input(vector)
            loss_p = dy.pick_batch(-dy.log(att_weights), out_idx[p])
            loss.append(dy.sum_batches(loss_p)/loss_p.dim()[1])
        return loss

    def generate(self, minibatch, dep_minibatch):
        words, pwords, orig_words, orig_pwords, tags, langs, sen_lens, masks = minibatch
        embedded = self.embed_sentence(words, pwords, orig_words, orig_pwords, tags, langs)
        encoded = self.encoder_bilstm.transduce(embedded)
        sentence_embeddings = [dy.transpose(dy.reshape(e, (e.dim()[0][0], e.dim()[1]))) for e in encoded]

        outputs = []
        for length in dep_minibatch.keys():
            sen_ids, heads, deps, labels, _, _, _ = dep_minibatch[length]

            in_embedding = [dy.lookup_batch(self.rlookup, labels[j]) for j in range(length)]
            in_se = [dy.concatenate_to_batch([sentence_embeddings[d][sen_ids[i]] for i, d in enumerate(deps[j])]) for j
                     in range(length)]
            input_embeddings = [dy.concatenate([l, s]) for l, s in zip(in_embedding, in_se)]
            e = self.dep_encoder_bilstm.transduce(input_embeddings)
            input_mat = dy.concatenate_cols(e)
            w1dt = None

            empty_tensor = dy.reshape(
                dy.inputTensor(np.zeros((self.options.hdim * 4 + self.options.re, len(deps[0])), dtype=float)),
                (self.options.hdim * 4 + self.options.re,), len(deps[0]))
            s = self.dec_lstm.initial_state().add_input(dy.concatenate([empty_tensor]))
            mask = np.zeros((deps.shape[0], deps.shape[1]), dtype=float)
            out = np.zeros((deps.shape[0], deps.shape[1]), dtype=int)
            for p in range(length):
                # w1dt can be computed and cached once for the entire decoding phase
                w1dt = w1dt or self.attention_w1.expr() * input_mat
                att_weights = self.attend(s, w1dt, True)
                scores = (att_weights).npvalue().reshape((mask.shape[0], mask.shape[1]))
                scores = np.sum([scores, mask], axis=0)
                next_positions = np.argmax(scores, axis=0)
                next_labels = []
                for i, position in enumerate(next_positions):
                    mask[position][i] = -float('inf')
                    out[p][i] = position
                    next_labels.append(labels[position][i])

                out_embedding = dy.lookup_batch(self.rlookup, next_labels)
                out_se = dy.concatenate_to_batch([sentence_embeddings[d][sen_ids[i]] for i, d in enumerate(next_positions)])
                output_embeddings = dy.concatenate([out_embedding, out_se])
                vector = dy.concatenate([input_mat * att_weights, output_embeddings])
                # vector = dy.dropout(vector, self.options.dropout)
                s = s.add_input(vector)
            outputs.append(out)
        dy.renew_cg()
        return outputs

    def get_loss(self, minibatch, dep_minibatch):
        words, pwords, orig_words, orig_pwords, tags, langs, sen_lens, masks = minibatch
        embedded = self.embed_sentence(words, pwords, orig_words, orig_pwords, tags, langs)
        encoded = self.encoder_bilstm.transduce(embedded)
        sentence_embeddings = [dy.transpose(dy.reshape(e, (e.dim()[0][0], e.dim()[1]))) for e in encoded]

        losses = []
        for length in dep_minibatch.keys():
            sen_ids, heads, deps, labels, o_deps, o_labels, num_orders = dep_minibatch[length]

            in_embedding = [dy.lookup_batch(self.rlookup, labels[j]) for j in range(length)]
            out_embedding = [dy.lookup_batch(self.rlookup, o_labels[j]) for j in range(length)]
            in_se = [dy.concatenate_to_batch([sentence_embeddings[d][sen_ids[i]] for i,d in enumerate(deps[j])]) for j in range(length)]
            out_se = [dy.concatenate_to_batch([sentence_embeddings[d][sen_ids[i]] for i,d in enumerate(o_deps[j])]) for j in range(length)]

            input_embeddings = [dy.concatenate([l, s]) for l, s in zip(in_embedding, in_se)]
            output_embeddings = [dy.concatenate([l, s]) for l, s in zip(out_embedding, out_se)]

            e = self.dep_encoder_bilstm.transduce(input_embeddings)
            losses += self.decode(e, output_embeddings, num_orders, length)

        return losses

    def get_output_int(self, minibatch):
        gen_out, masks = self.generate(minibatch), minibatch[-1]
        out = [[gen_out[i][j] for j in range(1, len(gen_out[i])) if masks[j][i] == 1] for i in range(len(gen_out))]
        out = [o[:-1] for o in out]
        return out

    def train(self, train_batches, train_dep_batches, dev_batches, dev_dep_batches, dev_out, t):
        start = time.time()
        loss_sum, b = 0, 0
        for d_i, minibatch in enumerate(train_batches):
            loss = self.get_loss(minibatch, train_dep_batches[d_i])
            loss_sum += self.backpropagate(loss)
            b += 1
            dy.renew_cg()

            loss = []
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
            output = self.generate(dev_batches[di], dev_dep_batches[di])
            for i, length in enumerate(dev_dep_batches[di].keys()):
                gold_orders = dev_dep_batches[di][length][-1].T
                predicted = output[i].T
                correct += np.sum(predicted == gold_orders)
                all_outputs += predicted.shape[0] * predicted.shape[1]
            dy.renew_cg()
        print 'dev accuracy', round(float(100 * correct) / all_outputs, 2)
        
        return t

    def reorder(self, batches, dep_minibatch,  out_file):
        writer = codecs.open(out_file, 'w')
        for d, minibatch in enumerate(batches):
            writer.write('\n'.join(self.get_output(minibatch, dep_minibatch)) + '\n')
            if (d + 1) % 100 == 0:
                sys.stdout.write(str(d + 1) + '...')
        sys.stdout.write(str(d) + '\n')
        writer.close()

    def reorder_tree(self, batches, trees, out_file):
        writer = codecs.open(out_file, 'w')
        print 'get new order'
        new_trees, t_num = [], 0
        for d, minibatch in enumerate(batches):
            for order in self.get_output_int(minibatch):
                new_trees.append(trees[t_num].reorder(order))
                t_num += 1
            if (d + 1) % 100 == 0:
                sys.stdout.write(str(d + 1) + '...')
        sys.stdout.write(str(d) + '\n')
        for tree in new_trees:
            writer.write(tree.conll_str())
            writer.write('\n\n')
        writer.close()

    def backpropagate(self, loss):
        loss = dy.esum(loss) / len(loss)
        loss_value = loss.value()
        loss.backward()
        self.trainer.update()
        return loss_value

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.populate(filename)

