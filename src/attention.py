import dynet as dy
import random, time, codecs
import numpy as np

class MT:
    def __init__(self, options, words, tags):
        self.EOS = "<EOS>"
        words.append(self.EOS)
        tags.append(self.EOS)
        self.int2w = ['<unk>', '<pad>'] + words
        self.int2t = ['<unk>', '<pad>'] + tags
        self.w2int = {w:i+2 for i, w in enumerate(words)}
        self.t2int = {t:i+2 for i, t in enumerate(tags)}

        self.WVOCAB_SIZE = len(self.int2w)
        self.TVOCAB_SIZE = len(self.int2t)

        self.LSTM_NUM_OF_LAYERS = options.layer
        self.ATTENTION_SIZE = options.attention
        self.hdim = options.hdim

        self.model = dy.Model()

        input_dim = options.we + options.pe
        self.encoder_bilstm = dy.BiRNNBuilder(self.LSTM_NUM_OF_LAYERS, input_dim, options.hdim * 2, self.model, dy.VanillaLSTMBuilder)

        self.dec_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, options.hdim * 2 + options.we, options.hdim, self.model)

        self.wlookup = self.model.add_lookup_parameters((self.WVOCAB_SIZE, options.we))
        self.tlookup = self.model.add_lookup_parameters((self.TVOCAB_SIZE, options.pe))
        self.attention_w1 = self.model.add_parameters( (self.ATTENTION_SIZE, self.hdim*2))
        self.attention_w2 = self.model.add_parameters( (self.ATTENTION_SIZE, self.hdim*self.LSTM_NUM_OF_LAYERS*2))
        self.attention_v = self.model.add_parameters( (1, self.ATTENTION_SIZE))
        self.decoder_w = self.model.add_parameters((self.WVOCAB_SIZE, options.hdim))
        self.decoder_b = self.model.add_parameters((self.WVOCAB_SIZE))
        self.output_lookup = self.model.add_lookup_parameters((self.WVOCAB_SIZE, options.we))

        self.trainer = dy.AdamTrainer(self.model, options.lr, options.beta1, options.beta2)

    def embed_sentence(self, ws, ts):
        words = [self.EOS] + ws + [self.EOS]
        tags = [self.EOS] + ts + [self.EOS]
        words = [self.w2int.get(w, 0) for w in words]
        tags = [self.t2int.get(t, 0) for t in tags]
        return [dy.concatenate([self.wlookup[w], self.tlookup[t]]) for w,t in zip(words, tags)], words


    def run_lstm(self, init_state, input_vecs):
        s = init_state

        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors


    def encode_sentence(self, input_embeds):
        return self.encoder_bilstm.transduce(input_embeds)


    def attend(self, input_mat, state, w1dt):
        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = self.attention_w2.expr()*dy.concatenate(list(state.s()))
        # att_weights: (seqlen,) row vector
        unnormalized = dy.transpose(self.attention_v.expr() * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized)
        # context: (encoder_state)
        context = input_mat * att_weights
        return context


    def decode(self, vectors, input, output_index):
        output = [self.EOS] + [input[o] for o in output_index] + [self.EOS]
        output = [self.w2int.get(w, 0) for w in output]

        input_mat = dy.concatenate_cols(vectors)
        w1dt = None

        last_output_embeddings = self.output_lookup[self.w2int[self.EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.hdim*2), last_output_embeddings]))
        loss = []

        for word in output:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or self.attention_w1.expr() * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            out_vector = self.decoder_w.expr() * s.output() + self.decoder_b.expr()
            probs = dy.softmax(out_vector)
            last_output_embeddings = self.output_lookup[word]
            loss.append(-dy.log(dy.pick(probs, word)))
        loss = dy.esum(loss)
        return loss

    def generate(self, words, tags):
        embedded, word_ids = self.embed_sentence(words, tags)
        encoded = self.encode_sentence(embedded)
        input_mat = dy.concatenate_cols(encoded)
        w1dt = None

        last_output_embeddings = self.output_lookup[self.w2int[self.EOS]]
        s = self.dec_lstm.initial_state().add_input(
            dy.concatenate([dy.vecInput(self.hdim * 2), last_output_embeddings]))

        out = []
        count_EOS = 0
        mask = np.zeros(len(word_ids))
        mask[-1] = -float('inf')
        decoder_w = dy.transpose(dy.concatenate_cols([self.decoder_w.expr()[w] for w in word_ids]))
        decoder_b = dy.transpose(dy.concatenate_cols([self.decoder_b.expr()[w] for w in word_ids]))

        while len(out) < len(word_ids)-2:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or self.attention_w1.expr() * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            out_vector = decoder_w * s.output() + decoder_b
            scores = out_vector.npvalue()
            scores = np.sum([scores, mask], axis=0)
            next_pos = np.argmax(scores)
            next_word = word_ids[next_pos]
            last_output_embeddings = self.output_lookup[next_word]
            mask[next_pos] = -float('inf')
            if self.int2w[next_word] == self.EOS:
                count_EOS += 1
                continue

            out.append(str(next_pos - 1))
        return ' '.join(out)

    def get_loss(self, input_words, input_tags, output_index):
        embedded, _ = self.embed_sentence(input_words, input_tags)
        encoded = self.encode_sentence(embedded)
        return self.decode(encoded, input_words, output_index)


    def train(self, train_data, batch_size):
        random.shuffle(train_data)
        loss_value, b, start = 0, 0, time.time()
        loss_vec = []
        for i, data in enumerate(train_data):
            loss = self.get_loss(data[0][0], data[0][1], data[1])
            loss_vec.append(loss)
            loss_value += loss.value()
            b += 1

            if len(loss_vec) >= batch_size:
                loss = dy.esum(loss_vec)/len(loss_vec)
                loss.backward()
                self.trainer.update()
                loss_vec = []
                dy.renew_cg()

            if (i+1) % 100 == 0:
                progress = round((i+1)*100.0/len(train_data), 2)
                print 'progress', progress, '%', 'loss', loss_value/b, 'time', time.time()-start
                loss_value, b, start = 0, 0, time.time()

        if len(loss_vec) > 0:
            loss = dy.esum(loss_vec)/len(loss_vec)
            loss.backward()
            self.trainer.update()
            loss_vec = []
            dy.renew_cg()

    def reorder(self, dev_data, outfile):
        writer = codecs.open(outfile, 'w')
        lwriter = codecs.open(outfile+'.log', 'w')
        for data in dev_data:
            output = self.generate(data[0][0], data[0][1])
            writer.write(output)
            writer.write('\n')
            lwriter.write('<<<<\n')
            lwriter.write(' '.join(data[0][0])+'\n')
            lwriter.write(output+'\n')
        writer.close()
        lwriter.close()