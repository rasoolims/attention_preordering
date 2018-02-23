from optparse import OptionParser
import utils, os, pickle
from attention import MT

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="train_file", metavar="FILE", default=None)
    parser.add_option("--train_t", dest="train_t", metavar="FILE", default=None)
    parser.add_option("--test", dest="test_file", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_file",  metavar="FILE", default=None)
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="model")
    parser.add_option("--we", type="int", dest="we", default=100)
    parser.add_option("--batch", type="int", dest="batch", default=5000)
    parser.add_option("--pe", type="int", dest="pe", default=100)
    parser.add_option("--ce", type="int", dest="ce", default=100)
    parser.add_option("--re", type="int", dest="re", default=25)
    parser.add_option("--t", type="int", dest="t", default=50000)
    parser.add_option("--epoch", type="int", dest="epoch", default=1000)
    parser.add_option("--lr", type="float", dest="lr", default=0.002)
    parser.add_option("--beta1", type="float", dest="beta1", default=0.9)
    parser.add_option("--beta2", type="float", dest="beta2", default=0.9)
    parser.add_option("--dropout", type="float", dest="dropout", default=0.33)
    parser.add_option("--outdir", type="string", dest="outdir", default="results")
    parser.add_option("--layer", type="int", dest="layer", default=3)
    parser.add_option("--hdim", type="int", dest="hdim", default=200)
    parser.add_option("--phdim", type="int", dest="phdim", default=200)
    parser.add_option("--attention", type="int", dest="attention", default=200)
    parser.add_option("--min_freq", type="int", dest="min_freq", default=1)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--eval_non_avg", action="store_true", dest="eval_non_avg", default=False)
    parser.add_option("--no_anneal", action="store_false", dest="anneal", default=True)
    parser.add_option("--no_char", action="store_false", dest="use_char", default=True)
    parser.add_option("--no_pos", action="store_false", dest="use_pos", default=True)
    parser.add_option("--stop", type="int", dest="stop", default=50)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)
    parser.add_option("--dynet-autobatch", type="int", dest="dynet-autobatch", default=0)
    parser.add_option("--dynet-gpus", action="store_true", dest="dynet-gpus", default=False, help='Use GPU instead of cpu.')

(options, args) = parser.parse_args()
if options.train_file:
    train_data, dev_data = utils.split_data(options.train_file, options.train_t)
    print len(train_data), len(dev_data)
    utils.write_data(dev_data, options.outdir+'/dev.gold')
    words, tags, chars, output_words = utils.vocab(train_data, options.min_freq)
    max_len = max([len(d[1]) for d in train_data])
    min_len = min([len(d[1]) for d in train_data])
    buckets = [list() for i in range(min_len, max(max_len, min_len+1))]
    for d in train_data:
        buckets[max(0, len(d[1]) - min_len - 1)].append(d)
    dev_buckets = [list()]
    for d in dev_data:
        dev_buckets[0].append(d)

    with open(os.path.join(options.outdir, options.params), 'w') as paramsfp:
        pickle.dump((words, tags, chars, output_words, options), paramsfp)
    t = MT(options, words, tags, chars, output_words)

    dev_batches = utils.get_batches(dev_buckets, t, False)
    best_dev = 0
    for i in range(options.epoch):
        train_batches = utils.get_batches(buckets, t, True)
        t.train(train_batches, dev_batches, options.outdir+'/dev.out', options.batch)
        dev_ac = utils.eval_trigram(dev_data, options.outdir + '/dev.out')
        print 'dev accuracy', dev_ac
        if dev_ac > best_dev:
            best_dev = dev_ac
            print 'saving', best_dev
            t.save(os.path.join(options.outdir, options.model))

if options.test_file and options.output_file:
    with open(os.path.join(options.outdir, options.params), 'r') as paramsfp:
        words, tags, chars, output_words, stored_options = pickle.load(paramsfp)
    stored_options.external_embedding = options.external_embedding
    t = MT(stored_options, words, tags, chars, output_words)
    t.load(os.path.join(options.outdir, options.model))
    test_buckets = [list()]
    trees, test_data = utils.read_tree_as_data(options.test_file)
    for d in test_data:
        test_buckets[0].append(d)
    t.options.batch = options.batch
    test_batches = utils.get_batches(test_buckets, t, False)
    t.reorder_tree(test_batches, trees, options.output_file)
