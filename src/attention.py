import dynet as dy
import codecs, sys, time
import numpy as np


class MT:
    def __init__(self, options, words, tags):
        self.EOS = "<EOS>"
        self.PAD = 1
        self.options = options
        words.append(self.EOS)
        tags.append(self.EOS)
        self.int2w = ['<unk>', '<pad>'] + words
        self.int2t = ['<unk>', '<pad>'] + tags
        self.w2int = {w: i + 2 for i, w in enumerate(words)}
        self.t2int = {t: i + 2 for i, t in enumerate(tags)}
        self.WVOCAB_SIZE = len(self.int2w)
        self.TVOCAB_SIZE = len(self.int2t)

        self.LSTM_NUM_OF_LAYERS = options.layer
        self.ATTENTION_SIZE = options.attention
        self.hdim = options.hdim

        self.model = dy.Model()

        input_dim = options.we + options.pe
        self.encoder_bilstm = dy.BiRNNBuilder(self.LSTM_NUM_OF_LAYERS, input_dim, options.hdim * 2, self.model,
                                              dy.VanillaLSTMBuilder)

        self.dec_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, options.hdim * 2 + options.we, options.hdim, self.model)

        self.wlookup = self.model.add_lookup_parameters((self.WVOCAB_SIZE, options.we))
        self.tlookup = self.model.add_lookup_parameters((self.TVOCAB_SIZE, options.pe))
        self.attention_w1 = self.model.add_parameters((self.ATTENTION_SIZE, self.hdim * 2))
        self.attention_w2 = self.model.add_parameters((self.ATTENTION_SIZE, self.hdim * self.LSTM_NUM_OF_LAYERS * 2))
        self.attention_v = self.model.add_parameters((1, self.ATTENTION_SIZE))
        self.decoder_w = self.model.add_parameters((self.WVOCAB_SIZE, options.hdim))
        self.decoder_b = self.model.add_parameters((self.WVOCAB_SIZE))
        self.position_h = self.model.add_parameters((options.phdim, options.hdim * 3))
        self.position_hb = self.model.add_parameters((options.phdim))
        self.position_decoder = self.model.add_parameters((1, options.phdim))
        self.output_lookup = self.model.add_lookup_parameters((self.WVOCAB_SIZE, options.we))

        self.trainer = dy.AdamTrainer(self.model, options.lr, options.beta1, options.beta2)

    def embed_sentence(self, ws, ts):
        return [dy.concatenate([dy.lookup_batch(self.wlookup, ws[i]), dy.lookup_batch(self.tlookup, ts[i])]) for i in
                range(len(ts))]

    def run_lstm(self, init_state, input_vecs):
        s = init_state

        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors

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

    def attend(self, input_mat, state, w1dt):
        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = self.attention_w2.expr() * dy.concatenate(list(state.s()))
        # att_weights: (seqlen,) row vector
        unnormalized = dy.transpose(self.attention_v.expr() * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized)
        # context: (encoder_state)
        context = input_mat * att_weights
        return context

    def decode(self, encoded, output_words, output_index, masks):
        input_mat = dy.concatenate_cols(encoded)
        w1dt = None

        last_output_embeddings = dy.lookup_batch(self.output_lookup, output_words[0])
        empty_tensor = dy.reshape(dy.inputTensor(np.zeros((self.hdim * 2, len(output_words[0])), dtype=float)),
                                  (self.hdim * 2,), len(output_words[0]))
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([empty_tensor, last_output_embeddings]))
        loss = []

        for p, word in enumerate(output_words):
            # w1dt can be computed and cached once for the entire decoding phase
            mask_tensor = dy.reshape(dy.inputTensor(masks[p]), (1,), len(masks[p]))
            w1dt = w1dt or self.attention_w1.expr() * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            out_vector = self.decoder_w.expr() * s.output() + self.decoder_b.expr()
            position_hidden = dy.concatenate_cols([dy.tanh(dy.affine_transform(
                [self.position_hb.expr(), self.position_h.expr(), dy.concatenate([s.output(), encoded[i]])])) for i in
                                                   range(len(output_index))])
            position_score = dy.transpose(self.position_decoder.expr() * position_hidden)
            last_output_embeddings = dy.lookup_batch(self.output_lookup, word)
            loss1 = dy.cmult(dy.pickneglogsoftmax_batch(out_vector, word), mask_tensor)
            loss2 = dy.cmult(dy.pickneglogsoftmax_batch(position_score, output_index[p]), mask_tensor)
            loss.append(dy.sum_batches(loss1)/loss1.dim()[1])
            loss.append(dy.sum_batches(loss2)/loss2.dim()[1])
        return loss

    def generate(self, minibatch):
        words, tags, _, _, sen_lens, masks = minibatch
        embedded = self.embed_sentence(words, tags)
        encoded = self.encode_sentence(embedded)
        input_mat = dy.concatenate_cols(encoded)
        w1dt = None

        last_output_embeddings = dy.lookup_batch(self.output_lookup, words[0])
        empty_tensor = dy.reshape(dy.inputTensor(np.zeros((self.hdim * 2, len(words[0])), dtype=float)),
                                  (self.hdim * 2,), len(words[0]))
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([empty_tensor, last_output_embeddings]))

        out = np.zeros((words.shape[1], words.shape[0]), dtype=int)
        first_mask = np.full((words.shape[0], words.shape[1]), -float('inf'), dtype=float)
        mask = np.zeros((words.shape[0], words.shape[1]), dtype=float)
        mask[0] = np.array([-float('inf')] * words.shape[1])
        for m1 in range(masks.shape[0]):
            for m2 in range(masks.shape[1]):
                if masks[m1][m2] == 0:
                    mask[m1][m2] = -float('inf')
                if sen_lens[m2] - 1 <= m1:
                    mask[m1][m2] = -float('inf')

        decoder_w = dy.transpose(dy.concatenate_cols([dy.pick_batch(self.decoder_w.expr(), w) for w in words]))
        decoder_b = dy.transpose(dy.concatenate_cols([dy.pick_batch(self.decoder_b.expr(), w) for w in words]))

        for p in range(len(words)):
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or self.attention_w1.expr() * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            out_vector = decoder_w * s.output() + decoder_b

            position_hidden = dy.concatenate_cols([dy.tanh(dy.affine_transform(
                [self.position_hb.expr(), self.position_h.expr(), dy.concatenate([s.output(), encoded[i]])])) for i in
                                                   range(len(words))])
            position_scores = dy.transpose(self.position_decoder.expr() * position_hidden)
            scores = (out_vector + position_scores).npvalue().reshape((mask.shape[0], mask.shape[1]))
            scores = np.sum([scores, first_mask if p == 0 else mask], axis=0)
            next_positions = np.argmax(scores, axis=0)
            next_words = [words[position][i] for i, position in enumerate(next_positions)]
            for i, position in enumerate(next_positions):
                mask[position][i] = -float('inf')
                out[i][p] = position
            last_output_embeddings = dy.lookup_batch(self.output_lookup, next_words)
        dy.renew_cg()
        return out

    def get_loss(self, minibatch):
        words, tags, output_words, positions, _, masks = minibatch
        embedded = self.embed_sentence(words, tags)
        encoded = self.encode_sentence(embedded)
        return self.decode(encoded, output_words, positions, masks)

    def get_output(self, minibatch):
        gen_out, masks = self.generate(minibatch), minibatch[-1]
        out = [[str(gen_out[i][j]) for j in range(1, len(gen_out[i])) if masks[j][i] == 1] for i in range(len(gen_out))]
        out = [' '.join(o[:-1]) for o in out]
        return out

    def train(self, train_batches, dev_batches, dev_out, batch_size=1):
        start = time.time()
        loss = []
        loss_sum, b = 0, 0
        for d_i, minibatch in enumerate(train_batches):
            loss += self.get_loss(minibatch)
            if len(loss) >= batch_size:
                loss_sum += self.backpropagate(loss)
                b += 1
                dy.renew_cg()
                loss = []
                if b % 10 == 0:
                    progress = round((d_i + 1) * 100.0 / len(train_batches), 2)
                    print 'progress', str(progress), '%', 'loss', loss_sum / b, 'time', time.time() - start
                    start = time.time()
                    loss_sum, b = 0, 0
        if len(loss) > 0:
            loss_sum += self.backpropagate(loss)
            b += 1
            dy.renew_cg()
            loss = []
        writer = codecs.open(dev_out, 'w')
        for d, minibatch in enumerate(dev_batches):
            writer.write('\n'.join(self.get_output(minibatch))+'\n')
            if (d + 1) % 10 == 0:
                sys.stdout.write(str(d + 1) + '...')
        sys.stdout.write(str(d) + '\n')
        writer.close()

    def backpropagate(self, loss):
        loss = dy.esum(loss) / len(loss)
        loss_value = loss.value()
        loss.backward()
        self.trainer.update()
        return loss_value
