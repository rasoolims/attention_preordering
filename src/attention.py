import dynet as dy
import random

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
        words = [self.w2int[w] for w in words]
        tags = [self.t2int[t] for t in tags]
        return [dy.concatenate([self.wlookup[w], self.tlookup[t]]) for w,t in zip(words, tags)]


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


    def decode(self, vectors, output):
        output = [self.EOS] + output + [self.EOS]
        output = [self.w2int[w] for w in output]

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
        embedded = self.embed_sentence(words, tags)
        encoded = self.encode_sentence(embedded)
        input_mat = dy.concatenate_cols(encoded)
        w1dt = None

        last_output_embeddings = self.output_lookup[self.w2int[self.EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.hdim * 2), last_output_embeddings]))

        out = []
        count_EOS = 0
        for i in range(len(words)*2):
            if count_EOS == 2:
                break
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or self.attention_w1.expr() * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            out_vector = self.decoder_w.expr() * s.output() + self.decoder_b.expr()
            probs = dy.softmax(out_vector).vec_value()
            next_word = probs.index(max(probs))
            last_output_embeddings = self.output_lookup[next_word]
            if self.int2w[next_word] == self.EOS:
                count_EOS += 1
                continue

            out.append(self.int2w[next_word])
        return ' '.join(out)


    def get_loss(self, input_words, input_tags, output_words):
        dy.renew_cg()
        embedded = self.embed_sentence(input_words, input_tags)
        encoded = self.encode_sentence(embedded)
        return self.decode(encoded, output_words)


    def train(self, train_data, epochs):
        for i in range(epochs):
            random.shuffle(train_data)
            for data in train_data:
                loss = self.get_loss(data[0][0], data[0][1], [data[0][0][d] for d in data[1]])
                loss_value = loss.value()
                loss.backward()
                self.trainer.update()

            if (i+1)%100==0:
                print(loss_value)
                print ' '.join(data[0][0])
                #print ' '.join(data[1])
                print(self.generate(data[0][0], data[0][1]))
                print '---'


