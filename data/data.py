from collections import OrderedDict, Counter
from itertools import chain

import six
import torch
import os

from torchtext import data, datasets
from torchtext.data import Example, Batch, Pipeline
from contextlib import ExitStack

import numpy as np

from random import shuffle


class DocField(data.Field):
    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.

        If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        if (six.PY2 and isinstance(x, six.string_types) and
                not isinstance(x, six.text_type)):
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)

        if self.lower:
            x = Pipeline(six.text_type.lower)(x)

        if self.sequential and isinstance(x, six.text_type):
            doc = []
            sents = x.strip().split(' <eos> ')
            for sent in sents:
                # doc.append(sent.strip().split() + ['<eos>'])
                doc.append(sent.strip().split())
            return doc
        else:
            raise RuntimeError('text_type')


class GraphField(data.Field):
    def preprocess(self, x):
        return x.strip()

class DocDataset(data.Dataset):
    def __init__(self, path, text_field, order_field, graph_field,
                 encoding='utf-8', **kwargs):
        fields = [('doc', text_field), ('order', order_field), ('entity', graph_field)]
        examples = []
        path, einspath = path
        with open(path, 'r', encoding=encoding) as f, open(einspath, 'r') as fe:
            for line, lineeins in zip(f, fe):
                line = line.strip()
                order = list(range(line.count('<eos>') + 1))
                if 'train' in path:
                    if len(order) > 1:
                        examples.append(data.Example.fromlist([line, order, lineeins], fields))
                else:
                    examples.append(data.Example.fromlist([line, order, lineeins], fields))

        super(DocDataset, self).__init__(examples, fields, **kwargs)


class MyBatch(Batch):
    def __init__(self, allsentences, orders, doc_len, ewords, elocs, dataset=None,
                 device=None):
        """Create a Batch from a list of examples."""
        if data is not None:
            setattr(self, 'doc_len', doc_len)
            setattr(self, 'elocs', elocs)

            self.batch_size = len(doc_len)
            self.dataset = dataset
            self.fields = dataset.fields.keys()  # copy field names
            self.input_fields = [k for k, v in dataset.fields.items() if
                                 v is not None and not v.is_target]
            self.target_fields = [k for k, v in dataset.fields.items() if
                                  v is not None and v.is_target]

            setattr(self, 'order', dataset.fields['order'].process(orders, device=device))
            setattr(self, 'doc', dataset.fields['doc'].process(allsentences, device=device))

            setattr(self, 'e_words', dataset.fields['doc'].process(ewords, device=device))

            # setattr(self, 'docwords', dataset.fields['doc'].process(doc_words, device=device))
            # setattr(self, 'graph', dataset.fields['e2e'].process_graph(e2ebatch, e2sbatch, orders,
            #                                                            doc_sent_word_len, device=device))
            # setattr(self, 'alllen', doc_sent_word_len)

            # for (name, field) in dataset.fields.items():
            #     if field is not None:
            #         batch = [getattr(x, name) for x in data]
            #         setattr(self, name, field.process(batch, device=device))


class DocIter(data.BucketIterator):
    def data(self):
        if self.shuffle:
            xs = [self.dataset[i] for i in self.random_shuffler(range(len(self.dataset)))]
        else:
            xs = self.dataset
        return xs

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # print(idx+1)
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1

                doc_len = []
                allsentences = []
                orders = []

                for ex in minibatch:
                    doc_len.append(len(ex.order))
                maxdoclen = max(doc_len)

                ewords = []
                elocs = []
                for ex in minibatch:
                    doc, order = ex.doc, ex.order

                    randid = list(range(len(order)))
                    shuffle(randid)

                    sfdoc = [doc[ri] for ri in randid]

                    sforder = [order[ri] for ri in randid]
                    sforder = list(np.argsort(sforder))

                    orders.append(sforder)

                    padnum = maxdoclen - len(sforder)
                    padded = sfdoc
                    for _ in range(padnum):
                        padded.append(['<pad>'])
                    allsentences.extend(padded)

                    eg = ex.entity

                    ew = []
                    newlocs = []
                    target = sforder

                    for eandloc in eg.split():
                        e, loc_role = eandloc.split(':')
                        ew.append(e)

                        word_newlocs = []
                        # print(loc_role)
                        for lr in loc_role.split('|'):
                            oriloc, r = lr.split('-')
                            word_newlocs.append([target[int(oriloc)], int(r)])

                        newlocs.append(word_newlocs)

                    elocs.append(newlocs)
                    ewords.append(ew)

                yield MyBatch(allsentences, orders, doc_len, ewords, elocs,
                              self.dataset, self.device)
            if not self.repeat:
                return


'''
def batch(data, batch_size, batch_size_fn=None):
    """Yield elements from data in chunks of batch_size."""
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
    if minibatch:
        yield minibatch


def pool(data, batch_size, key, batch_size_fn=lambda new, count, sofar: count,
         random_shuffler=None, shuffle=False, sort_within_batch=False):
    """Sort within buckets, then batch, then shuffle batches.

    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    if random_shuffler is None:
        random_shuffler = random.shuffle
    for p in batch(data, batch_size * 100, batch_size_fn):
        p_batch = batch(sorted(p, key=key), batch_size, batch_size_fn) \
            if sort_within_batch \
            else batch(p, batch_size, batch_size_fn)
        if shuffle:
            for b in random_shuffler(list(p_batch)):
                yield b
        else:
            for b in list(p_batch):
                yield b
'''


# load the dataset + reversible tokenization
class NormalField(data.Field):
    def _getattr(self, dataset, attr):
        for x in dataset:
            yield getattr(x, attr)

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            sources += [self._getattr(arg, name) for name, field in
                        arg.fields if field is self]
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def reverse(self, batch, unbpe=True, returen_token=False):
        if not self.batch_first:
            batch.t_()

        with torch.cuda.device_of(batch):
            batch = batch.tolist()

        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        if unbpe:
            batch = [" ".join(filter(filter_special, ex)).replace("@@ ", "") for ex in batch]
        else:
            batch = [" ".join(filter(filter_special, ex)) for ex in batch]

        if returen_token:
            batch = [ex.split() for ex in batch]
        return batch


class ParallelDataset(object):
    """ Define a N-parallel dataset: supports abitriry numbers of input streams"""

    def __init__(self, path=None, exts=None, fields=None, max_len=None):
        assert len(exts) == len(fields), 'N parallel dataset must match'
        self.N = len(fields)

        if not isinstance(fields[0], (tuple, list)):
            newfields = [('src', fields[0]), ('trg', fields[1])]
            for i in range(len(exts) - 2):
                newfields.append(('extra_{}'.format(i), fields[2 + i]))
            self.fields = newfields
        self.paths = tuple(os.path.expanduser(path + '.' + x) for x in exts)
        self.max_len = max_len

    def __iter__(self):
        with ExitStack() as stack:
            files = [stack.enter_context(open(fname, 'r', errors='ignore')) for fname in self.paths]
            for i, lines in enumerate(zip(*files)):
                lines = [line.strip() for line in lines]
                if not any(line == '' for line in lines):
                    example = Example.fromlist(lines, self.fields)
                    if self.max_len is None:
                        yield example
                    elif len(example.src) <= self.max_len and len(example.trg) <= self.max_len:
                        yield example
