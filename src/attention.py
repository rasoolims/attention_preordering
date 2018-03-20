import dynet as dy
import codecs, sys, time, gzip
import numpy as np


class MT:
    def __init__(self, options, words, tags, chars):
        self.EOS = "<EOS>"
        self.PAD = 1
        self.options = options
        words.append(self.EOS)
        tags.append(self.EOS)
        self.int2w = ['<unk>', '<pad>'] + words
        self.int2t = ['<unk>', '<pad>'] + tags
        self.w2int = {w: i + 2 for i, w in enumerate(words)}
        self.t2int = {t: i + 2 for i, t in enumerate(tags)}
        self.c2int = {c: i + 2 for i, c in enumerate(chars)}
        self.WVOCAB_SIZE = len(self.int2w)
        self.TVOCAB_SIZE = len(self.int2t)

        self.LSTM_NUM_OF_LAYERS = options.layer
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

        input_dim = options.we + options.pe
        self.encoder_bilstm = dy.BiRNNBuilder(self.LSTM_NUM_OF_LAYERS, input_dim, options.hdim * 2, self.model, dy.VanillaLSTMBuilder)

        self.clookup = self.model.add_lookup_parameters((len(chars) + 2, options.ce))
        self.char_lstm = dy.BiRNNBuilder(1, options.ce, options.we, self.model, dy.VanillaLSTMBuilder)

        self.dec_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, options.hdim * 2 + options.we + options.pe, options.phdim, self.model)

        self.wlookup = self.model.add_lookup_parameters((self.WVOCAB_SIZE, options.we))
        self.tlookup = self.model.add_lookup_parameters((self.TVOCAB_SIZE, options.pe))
        self.attention_w1 = self.model.add_parameters((self.ATTENTION_SIZE, options.hdim * 2))
        self.attention_w2 = self.model.add_parameters((self.ATTENTION_SIZE, options.phdim * self.LSTM_NUM_OF_LAYERS * 2))
        self.attention_v = self.model.add_parameters((1, self.ATTENTION_SIZE))
        self.trainer = dy.AdamTrainer(self.model, options.lr, options.beta1, options.beta2)

        def _emb_mask_generator(seq_len, batch_size):
            ret = []
            for _ in xrange(seq_len):
                word_mask = np.random.binomial(1, 1. - self.options.dropout, batch_size).astype(np.float32)
                tag_mask = np.random.binomial(1, 1. - self.options.dropout, batch_size).astype(np.float32)
                scale = 5. / (4. * word_mask + tag_mask + 1e-12)
                word_mask *= scale
                tag_mask *= scale
                word_mask = dy.inputTensor(word_mask, batched=True)
                tag_mask = dy.inputTensor(tag_mask, batched=True)
                ret.append((word_mask, tag_mask))

            return ret

        self.generate_emb_mask = _emb_mask_generator

    def embed_sentence(self, ws, pwords, ts, chars, is_train):
        cembed = [dy.lookup_batch(self.clookup, c) for c in chars]
        char_fwd, char_bckd = self.char_lstm.builder_layers[0][0].initial_state().transduce(cembed)[-1], \
                              self.char_lstm.builder_layers[0][1].initial_state().transduce(reversed(cembed))[-1]
        crnn = dy.reshape(dy.concatenate_cols([char_fwd, char_bckd]), (self.options.we, ws.shape[0] * ws.shape[1]))
        cnn_reps = [list() for _ in range(len(ws))]
        for i in range(ws.shape[0]):
            cnn_reps[i] = dy.pick_batch(crnn, [i * ws.shape[1] + j for j in range(ws.shape[1])], 1)

        wembed = [dy.lookup_batch(self.wlookup, ws[i]) + dy.lookup_batch(self.elookup, pwords[i]) + cnn_reps[i] for i in range(len(ws))]
        posembed = [dy.lookup_batch(self.tlookup, ts[i]) for i in range(len(ts))]
        if not is_train:
            return [dy.concatenate([wembed[i], posembed[i]]) for i in range(len(ts))]
        else:
            emb_masks = self.generate_emb_mask(ws.shape[0], ws.shape[1])
            return [dy.concatenate([dy.cmult(w, wm), dy.cmult(pos, posm)]) for w, pos, (wm, posm) in
                      zip(wembed, posembed, emb_masks)]



    def encode_sentence(self, input_embeds, batch_size=None, dropout_x=0., dropout_h=0.):
        for fb, bb in self.encoder_bilstm.builder_layers:
            f, b = fb.initial_state(), bb.initial_state()
            fb.set_dropouts(dropout_x, dropout_h)
            bb.set_dropouts(dropout_x, dropout_h)
            if batch_size is not None:
                fb.set_dropout_masks(batch_size)
                bb.set_dropout_masks(batch_size)
            fs, bs = f.transduce(input_embeds), b.transduce(reversed(input_embeds))
            input_embeds = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
        return input_embeds

    def attend(self, state, w1dt, is_train):
        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = self.attention_w2.expr() * dy.concatenate(list(state.s()))
        if is_train:
            w2dt = dy.dropout(w2dt, self.options.dropout)
        # att_weights: (seqlen,) row vector
        unnormalized = dy.transpose(self.attention_v.expr() * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized) + dy.scalarInput(1e-12)
        return att_weights

    def decode(self, encoded, output_words, output_tags, output_index, masks):
        input_mat = dy.concatenate_cols(encoded)
        w1dt = None

        last_output_embeddings = dy.lookup_batch(self.wlookup, output_words[0])
        last_tag_embeddings = dy.lookup_batch(self.tlookup, output_tags[0])
        empty_tensor = dy.reshape(dy.inputTensor(np.zeros((self.options.hdim * 2, len(output_words[0])), dtype=float)),
                                  (self.options.hdim * 2,), len(output_words[0]))
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([empty_tensor, last_output_embeddings, last_tag_embeddings]))
        loss = []
        for p, word in enumerate(output_words):
            # w1dt can be computed and cached once for the entire decoding phase
            mask_tensor = dy.reshape(dy.inputTensor(masks[p]), (1,), len(masks[p]))
            w1dt = w1dt or self.attention_w1.expr() * input_mat
            att_weights = self.attend(s, w1dt, True)
            vector = dy.concatenate([input_mat * att_weights, last_output_embeddings, last_tag_embeddings])
            vector = dy.dropout(vector, self.options.dropout)
            s = s.add_input(vector)
            last_output_embeddings = dy.lookup_batch(self.wlookup, word)
            last_tag_embeddings = dy.lookup_batch(self.tlookup, output_tags[p])
            loss_p = dy.cmult(dy.pick_batch(-dy.log(att_weights), output_index[p]), mask_tensor)
            loss.append(dy.sum_batches(loss_p)/loss_p.dim()[1])
        return loss

    def generate(self, minibatch):
        words, pwords, tags, _, _, _, chars, sen_lens, masks = minibatch
        embedded = self.embed_sentence(words, pwords, tags, chars, False)
        encoded = self.encode_sentence(embedded)
        input_mat = dy.concatenate_cols(encoded)
        w1dt = None

        last_output_embeddings = dy.lookup_batch(self.wlookup, words[0])
        last_tag_embeddings = dy.lookup_batch(self.tlookup, tags[0])
        empty_tensor = dy.reshape(dy.inputTensor(np.zeros((self.options.hdim * 2, len(words[0])), dtype=float)),
                                  (self.options.hdim * 2,), len(words[0]))
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([empty_tensor, last_output_embeddings, last_tag_embeddings]))

        out = np.zeros((words.shape[1], words.shape[0]), dtype=int)
        first_mask = np.full((words.shape[0], words.shape[1]), -float('inf'), dtype=float)
        mask = np.zeros((words.shape[0], words.shape[1]), dtype=float)
        first_mask[0] = np.array([0] * words.shape[1])
        mask[0] = np.array([-float('inf')] * words.shape[1])
        for m1 in range(masks.shape[0]):
            for m2 in range(masks.shape[1]):
                if masks[m1][m2] == 0:
                    mask[m1][m2] = -float('inf')
                if sen_lens[m2] - 1 <= m1:
                    mask[m1][m2] = -float('inf')

        for p in range(len(words)):
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or self.attention_w1.expr() * input_mat
            att_weights = self.attend(s, w1dt, False)
            vector = dy.concatenate([input_mat * att_weights, last_output_embeddings, last_tag_embeddings])
            s = s.add_input(vector)

            scores = (att_weights).npvalue().reshape((mask.shape[0], mask.shape[1]))
            cur_mask = first_mask if p == 0 else mask
            scores = np.sum([scores, cur_mask], axis=0)
            if p==1:
                for i in range(len(scores)):
                    if np.isinf(scores[i]).all():
                        print 'all_inf', i
                        print scores[i]

            next_positions = np.argmax(scores, axis=0)
            next_words = [words[position][i] for i, position in enumerate(next_positions)]
            next_tags = [tags[position][i] for i, position in enumerate(next_positions)]
            for i, position in enumerate(next_positions):
                mask[position][i] = -float('inf')
                out[i][p] = position
            last_output_embeddings = dy.lookup_batch(self.wlookup, next_words)
            last_tag_embeddings = dy.lookup_batch(self.tlookup, next_tags)
        dy.renew_cg()
        return out

    def get_loss(self, minibatch):
        words, pwords, tags, output_words, output_tags, positions, chars, _, masks = minibatch
        embedded = self.embed_sentence(words, pwords, tags, chars, True)
        encoded = self.encode_sentence(embedded,  words.shape[1],  self.options.dropout,  self.options.dropout)
        return self.decode(encoded, output_words, output_tags, positions, masks)

    def get_output_int(self, minibatch):
        gen_out, masks = self.generate(minibatch), minibatch[-1]
        out = [[gen_out[i][j] for j in range(1, len(gen_out[i])) if masks[j][i] == 1] for i in range(len(gen_out))]
        out = [o[:-1] for o in out]
        return out

    def get_output(self, minibatch):
        gen_out, masks = self.generate(minibatch), minibatch[-1]
        out = [[str(gen_out[i][j]) for j in range(1, len(gen_out[i])) if masks[j][i] == 1] for i in range(len(gen_out))]
        out = [' '.join(o[:-1]) for o in out]
        return out

    def train(self, train_batches, dev_batches, dev_out, t, batch_size=1):
        start = time.time()
        loss_sum, b = 0, 0
        for d_i, minibatch in enumerate(train_batches):
            loss = self.get_loss(minibatch)
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
        self.reorder(dev_batches, dev_out)
        return t

    def reorder(self, batches, out_file):
        writer = codecs.open(out_file, 'w')
        for d, minibatch in enumerate(batches):
            writer.write('\n'.join(self.get_output(minibatch)) + '\n')
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
