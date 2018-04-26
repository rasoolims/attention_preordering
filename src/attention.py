import dynet as dy
import codecs, sys, time, gzip
import numpy as np
from collections import defaultdict

class MT:
    def __init__(self, options, tags, relations):
        self.EOS = "<EOS>"
        self.PAD = 1
        self.options = options
        tags.append(self.EOS)
        self.int2t = ['<unk>', '<pad>'] + tags
        self.t2int = {t: i + 2 for i, t in enumerate(tags)}
        self.rel2int = {r: i + 2 for i, r in enumerate(relations)}
        self.TVOCAB_SIZE = len(self.int2t)

        self.DEP_LSTM_NUM_OF_LAYERS = options.dep_layer
        self.ATTENTION_SIZE = options.attention
        self.model = dy.Model()

        dep_inp_dim = options.hdim * 2 + options.re
        self.dep_encoder_bilstm = dy.BiRNNBuilder(self.DEP_LSTM_NUM_OF_LAYERS, options.re, options.hdim * 2, self.model, dy.VanillaLSTMBuilder)

        self.dec_lstm = dy.LSTMBuilder(self.DEP_LSTM_NUM_OF_LAYERS, dep_inp_dim, options.phdim, self.model)

        self.tlookup = self.model.add_lookup_parameters((self.TVOCAB_SIZE, options.pe))
        self.rlookup = self.model.add_lookup_parameters((len(self.rel2int)+2, options.re))
        self.attention_w1 = self.model.add_parameters((self.ATTENTION_SIZE, options.hdim * 2))
        self.attention_w2 = self.model.add_parameters((self.ATTENTION_SIZE, options.phdim * self.DEP_LSTM_NUM_OF_LAYERS * 2))
        self.attention_v = self.model.add_parameters((1, self.ATTENTION_SIZE))
        self.trainer = dy.AdamTrainer(self.model, options.lr, options.beta1, options.beta2)

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

        empty_tensor = dy.reshape(dy.inputTensor(np.zeros((self.options.hdim * 2 + self.options.re, len(out_idx[0])), dtype=float)),
                                  (self.options.hdim * 2 + self.options.re,), len(out_idx[0]))
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

    def generate(self, dep_minibatch):
        outputs = []
        for length in dep_minibatch.keys():
            sen_ids, heads, deps, labels, _, _, _ = dep_minibatch[length]

            input_embeddings = [dy.lookup_batch(self.rlookup, labels[j]) for j in range(length)]
            e = self.dep_encoder_bilstm.transduce(input_embeddings)
            input_mat = dy.concatenate_cols(e)
            w1dt = None

            empty_tensor = dy.reshape(
                dy.inputTensor(np.zeros((self.options.hdim * 2 + self.options.re, len(deps[0])), dtype=float)),
                (self.options.hdim * 2 + self.options.re,), len(deps[0]))
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

                output_embeddings = dy.lookup_batch(self.rlookup, next_labels)
                vector = dy.concatenate([input_mat * att_weights, output_embeddings])
                # vector = dy.dropout(vector, self.options.dropout)
                s = s.add_input(vector)
            outputs.append(out)
        dy.renew_cg()
        return outputs

    def get_loss(self, dep_minibatch):
        losses = []
        for length in dep_minibatch.keys():
            sen_ids, heads, deps, labels, o_deps, o_labels, num_orders = dep_minibatch[length]
            in_embedding = [dy.lookup_batch(self.rlookup, labels[j]) for j in range(length)]
            out_embedding = [dy.lookup_batch(self.rlookup, o_labels[j]) for j in range(length)]
            e = self.dep_encoder_bilstm.transduce(in_embedding)
            losses += self.decode(e, out_embedding, num_orders, length)

        return losses

    def get_output_int(self, minibatch):
        gen_out, masks = self.generate(minibatch), minibatch[-1]
        out = [[gen_out[i][j] for j in range(1, len(gen_out[i])) if masks[j][i] == 1] for i in range(len(gen_out))]
        out = [o[:-1] for o in out]
        return out

    def train(self, train_dep_batches, dev_dep_batches, dev_out, t):
        start = time.time()
        loss_sum, b = 0, 0
        for d_i, train_dep_batch in enumerate(train_dep_batches):
            loss = self.get_loss(train_dep_batch)
            loss_sum += self.backpropagate(loss)
            b += 1
            dy.renew_cg()

            if b % 100 == 0:
                progress = round((d_i + 1) * 100.0 / len(train_dep_batches), 2)
                print 'progress', str(progress), '%', 'loss', loss_sum / b, 'time', time.time() - start
                start = time.time()
                loss_sum, b = 0, 0
            if self.options.anneal:
                decay_steps = min(1.0, float(t) / 50000)
                lr = self.options.lr * 0.75 ** decay_steps
                self.trainer.learning_rate = lr
            t += 1

        all_outputs, correct = 0, 0
        for di in range(len(dev_dep_batches)):
            output = self.generate(dev_dep_batches[di])
            for i, length in enumerate(dev_dep_batches[di].keys()):
                gold_orders = dev_dep_batches[di][length][-1].T
                predicted = output[i].T
                correct += np.sum(predicted == gold_orders)
                all_outputs += predicted.shape[0] * predicted.shape[1]
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

    def reorder_tree(self, dep_batches, trees, out_file):
        writer = codecs.open(out_file, 'w')
        print 'get new order'
        new_trees, t_num = [], 0
        offset = 0
        output_orders = defaultdict(dict)
        for d in range(len(dep_batches)):
            output = self.generate(dep_batches[d])
            max_id = 0
            for i, length in enumerate(dep_batches[d].keys()):
                sen_ids, heads, deps, labels, _, _, _ = dep_batches[d][length]
                max_id = max(max_id, max(sen_ids))
                predicted, orig_deps = output[i].T, deps.T
                for p in range(len(predicted)):
                    sen_id = sen_ids[p] + offset
                    output_orders[sen_id][heads[p]] = [orig_deps[p][f] for f in predicted[p]]
            offset += max_id + 1
            if (d + 1) % 100 == 0:
                sys.stdout.write(str(d + 1) + '...')
        sys.stdout.write(str(d) + '\n')
        for t, tree in enumerate(trees):
            new_linear_order = tree.get_linear_order(0, output_orders[t])
            new_trees.append(tree.reorder_with_order(new_linear_order))
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

