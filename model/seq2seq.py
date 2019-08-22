import re
import torch
import numpy as np
import math
import torch.nn as nn
import time
import subprocess
import torch.nn.functional as F
from model.generator import Beam
from data.data import DocField, DocDataset, DocIter


class GRUCell(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(GRUCell, self).__init__()
        self.r = nn.Linear(x_dim + h_dim, h_dim, True)
        self.z = nn.Linear(x_dim + h_dim, h_dim, True)

        self.c = nn.Linear(x_dim, h_dim, True)
        self.u = nn.Linear(h_dim, h_dim, True)

    def forward(self, x, h):
        rz_input = torch.cat((x, h), -1)
        r = torch.sigmoid(self.r(rz_input))
        z = torch.sigmoid(self.z(rz_input))

        u = torch.tanh(self.c(x) + r * self.u(h))

        new_h = z * h + (1 - z) * u
        return new_h


class SGRU(nn.Module):
    def __init__(self, s_emb, e_emb, sh_dim, eh_dim, label_dim):
        super(SGRU, self).__init__()

        g_dim = sh_dim
        self.s_gru = GRUCell(s_emb + sh_dim + label_dim + eh_dim + g_dim, sh_dim)

        self.e_gru = GRUCell(e_emb + sh_dim + label_dim + g_dim, eh_dim)

        self.g_gru = GRUCell(sh_dim + eh_dim, g_dim)

    def forward(self, it, h, g, mask):
        '''
        :param it: B T 2H
        :param h: B T H
        :param g: B H
        :return:
        '''

        si, ei = it
        sh, eh = h
        smask, wmask = mask

        # update sentence node
        g_expand_s = g.unsqueeze(1).expand_as(sh)
        x = torch.cat((si, g_expand_s), -1)
        new_sh = self.s_gru(x, sh)

        # update entity node
        g_expand_e = g.unsqueeze(1).expand(eh.size(0), eh.size(1), g.size(-1))
        x = torch.cat((ei, g_expand_e), -1)
        new_eh = self.e_gru(x, eh)

        new_sh.masked_fill_((smask == 0).unsqueeze(2), 0)
        new_eh.masked_fill_((wmask == 0).unsqueeze(2), 0)

        # update global
        sh_mean = new_sh.sum(1) / smask.float().sum(1, True)
        eh_mean = new_eh.sum(1) / (wmask.float().sum(1, True) + 1)

        mean = torch.cat((sh_mean, eh_mean), -1)
        new_g = self.g_gru(mean, g)

        return new_sh, new_eh, new_g


class GRNGOB(nn.Module):
    def __init__(self, s_emb, e_emb, s_hidden, e_hidden, label_dim, dp=0.1, layer=2, agg='sum'):
        super(GRNGOB, self).__init__()
        self.layer = layer
        self.dp = dp

        self.slstm = SGRU(s_emb, e_emb, s_hidden, e_hidden, label_dim)

        self.s_hid = s_hidden
        self.e_hid = e_hidden

        self.agg = agg

        self.edgeemb = nn.Embedding(4, label_dim)

        self.gate1 = nn.Linear(s_hidden + e_hidden + label_dim, e_hidden + label_dim)
        self.gate2 = nn.Linear(s_hidden + e_hidden + label_dim, s_hidden + label_dim)

    def mean(self, x, m, smooth=0):
        mean = torch.matmul(m, x)
        return mean / (m.sum(2, True) + smooth)

    def sum(self, x, m):
        return torch.matmul(m, x)

    def forward(self, sent, smask, word, wmask, elocs):
        '''
        :param wmask: B E
        :param smask: B S
        :param word: B E H
        :param sent: B S H
        :return:
        '''
        batch = sent.size(0)
        snum = smask.size(1)
        wnum = wmask.size(1)

        # batch sent_num word_num
        # B S E
        matrix = sent.new_zeros(batch, snum, wnum).long()
        for ib, eloc in enumerate(elocs):
            for ixw, loc in enumerate(eloc):
                for aftersf_ixs, r in loc:
                    matrix[ib, aftersf_ixs, ixw] = r

        mask_se = (matrix != 0).float()
        mask_se_t = mask_se.transpose(1, 2)

        # B S E H
        label_emb = self.edgeemb(matrix)
        label_emb_t = label_emb.transpose(1, 2)

        # B S S
        # connect two sentence if thay have at least one same entity
        s2smatrix = torch.matmul(mask_se, mask_se_t)
        s2smatrix = s2smatrix != 0

        eye = torch.eye(snum).byte().cuda()
        s2s = smask.new_ones(snum, snum)
        eyemask = (s2s - eye).unsqueeze(0)

        s2smatrix = s2smatrix * eyemask
        s2smatrix = s2smatrix & smask.unsqueeze(1)
        s2smatrix = s2smatrix.float()

        s_h = torch.zeros_like(sent)
        g_h = sent.new_zeros(batch, self.s_hid)
        e_h = sent.new_zeros(batch, wnum, self.e_hid)

        for i in range(self.layer):
            # 1.aggregation
            # s_neigh_s_h = self.mean(s_h, s2smatrix)
            s_neigh_s_h = self.sum(s_h, s2smatrix)

            # B S E H
            if self.agg == 'gate':
                s_h_expand = s_h.unsqueeze(2).expand(batch, snum, wnum, self.s_hid)

                e_h_expand = e_h.unsqueeze(1).expand(batch, snum, wnum, self.e_hid)
                e_h_expand_edge = torch.cat((e_h_expand, label_emb), -1)

                s_e_l = torch.cat((s_h_expand, e_h_expand_edge), -1)
                g = torch.sigmoid(self.gate1(s_e_l))

                s_neigh_e_h = e_h_expand_edge * g * mask_se.unsqueeze(3)
                s_neigh_e_h = s_neigh_e_h.sum(2)
                ####
                s_h_expand = s_h.unsqueeze(1).expand(batch, wnum, snum, self.s_hid)
                s_h_expand_edge = torch.cat((s_h_expand, label_emb_t), -1)

                e_h_expand = e_h.unsqueeze(2).expand(batch, wnum, snum, self.e_hid)

                e_s_l = torch.cat((e_h_expand, s_h_expand_edge), -1)
                g2 = torch.sigmoid(self.gate2(e_s_l))

                e_neigh_s_h = s_h_expand_edge * g2 * mask_se_t.unsqueeze(3)
                e_neigh_s_h = e_neigh_s_h.sum(2)

            s_input = torch.cat((sent, s_neigh_s_h, s_neigh_e_h), -1)
            e_input = torch.cat((word, e_neigh_s_h), -1)

            # 2.update
            s_h, e_h, g_h = self.slstm((s_input, e_input), (s_h, e_h), g_h, (smask, wmask))

        if self.dp > 0:
            s_h = F.dropout(s_h, self.dp, self.training)

        return s_h, g_h


class PointerNet(nn.Module):
    def __init__(self, args):
        super(PointerNet, self).__init__()

        self.emb_dp = args.input_drop_ratio
        self.model_dp = args.drop_ratio
        self.d_emb = args.d_emb
        self.sen_enc_type = args.senenc

        self.src_embed = nn.Embedding(args.doc_vocab, self.d_emb)

        self.sen_enc = nn.LSTM(self.d_emb, args.d_rnn // 2, bidirectional=True, batch_first=True)

        self.entityemb = args.entityemb

        self.encoder = GRNGOB(s_emb=args.d_rnn,
                              e_emb=args.d_emb if self.entityemb == 'glove' else args.d_rnn,
                              s_hidden=args.d_rnn,
                              e_hidden=args.ehid, label_dim=args.labeldim,
                              layer=args.gnnl, dp=args.gnndp, agg=args.agg)

        d_mlp = args.d_mlp
        self.linears = nn.ModuleList([nn.Linear(args.d_rnn, d_mlp),
                                      nn.Linear(args.d_rnn * 2, d_mlp),
                                      nn.Linear(d_mlp, 1)])
        self.decoder = nn.LSTM(args.d_rnn, args.d_rnn, batch_first=True)
        self.critic = None

    def equip(self, critic):
        self.critic = critic

    def forward(self, src_and_len, tgt_and_len, doc_num, ewords_and_len, elocs):
        document_matrix, _, hcn, key = self.encode(src_and_len, doc_num, ewords_and_len, elocs)

        start = document_matrix.new_zeros(document_matrix.size(0), 1, document_matrix.size(2))
        target, tgt_len = tgt_and_len

        # B N-1 H
        dec_inputs = document_matrix[torch.arange(document_matrix.size(0)).unsqueeze(1), target[:, :-1]]
        # B N H
        dec_inputs = torch.cat((start, dec_inputs), 1)

        sorted_len, ix = torch.sort(tgt_len, descending=True)
        sorted_dec_inputs = dec_inputs[ix]

        packed_dec_inputs = nn.utils.rnn.pack_padded_sequence(sorted_dec_inputs, sorted_len, True)
        hn, cn = hcn
        sorted_hn = hn[:, ix]
        sorted_cn = cn[:, ix]
        packed_dec_outputs, _ = self.decoder(packed_dec_inputs, (sorted_hn, sorted_cn))

        dec_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_dec_outputs, True)

        _, recovered_ix = torch.sort(ix, descending=False)
        dec_outputs = dec_outputs[recovered_ix]

        # B qN 1 H
        query = self.linears[0](dec_outputs).unsqueeze(2)
        # B 1 kN H
        key = key.unsqueeze(1)
        # B qN kN H
        e = torch.tanh(query + key)
        # B qN kN
        e = self.linears[2](e).squeeze(-1)

        # mask already pointed nodes
        pointed_mask = [e.new_zeros(e.size(0), 1, e.size(2)).byte()]

        for t in range(1, e.size(1)):
            # B
            tar = target[:, t - 1]
            # B kN
            pm = pointed_mask[-1].clone().detach()
            pm[torch.arange(e.size(0)), :, tar] = 1
            pointed_mask.append(pm)
        # B qN kN
        pointed_mask = torch.cat(pointed_mask, 1)

        pointed_mask_by_target = pointed_mask.new_zeros(pointed_mask.size(0), pointed_mask.size(2))
        target_mask = pointed_mask.new_zeros(pointed_mask.size(0), pointed_mask.size(1))

        for b in range(target_mask.size(0)):
            pointed_mask_by_target[b, :tgt_len[b]] = 1
            target_mask[b, :tgt_len[b]] = 1

        pointed_mask_by_target = pointed_mask_by_target.unsqueeze(1).expand_as(pointed_mask)

        e.masked_fill_(pointed_mask == 1, -1e9)

        e.masked_fill_(pointed_mask_by_target == 0, -1e9)

        logp = F.log_softmax(e, dim=-1)

        logp = logp.view(-1, logp.size(-1))
        loss = self.critic(logp, target.contiguous().view(-1))

        target_mask = target_mask.view(-1)
        loss.masked_fill_(target_mask == 0, 0)

        loss = loss.sum()/target.size(0)

        return loss

    def rnn_enc(self, src_and_len, doc_num):
        '''
        :param src_and_len:
        :param doc_num: B, each doc has sentences number
        :return: document matirx
        '''
        src, length = src_and_len

        sorted_len, ix = torch.sort(length, descending=True)
        sorted_src = src[ix]

        # bi-rnn must uses pack, else needs mask
        packed_x = nn.utils.rnn.pack_padded_sequence(sorted_src, sorted_len, batch_first=True)
        x = packed_x.data

        x = self.src_embed(x)

        if self.emb_dp > 0:
            x = F.dropout(x, self.emb_dp, self.training)

        packed_x = nn.utils.rnn.PackedSequence(x, packed_x.batch_sizes)

        # 2 TN H
        states, (hn, _) = self.sen_enc(packed_x)

        # TN T 2H
        allwordstates, _ = nn.utils.rnn.pad_packed_sequence(states, True)

        # TN 2H
        hn = hn.transpose(0, 1).contiguous().view(src.size(0), -1)

        _, recovered_ix = torch.sort(ix, descending=False)
        hn = hn[recovered_ix]
        allwordstates = allwordstates[recovered_ix]

        batch_size = len(doc_num)
        maxdoclen = max(doc_num)
        output = hn.view(batch_size, maxdoclen, -1)

        allwordstates = allwordstates.view(batch_size, -1, hn.size(-1))

        return output, allwordstates

    def encode(self, src_and_len, doc_num, ewords_and_len, elocs):
        # get sentence emb and mask
        sentences, words_states = self.rnn_enc(src_and_len, doc_num)

        if self.model_dp > 0:
            sentences = F.dropout(sentences, self.model_dp, self.training)

        batch = sentences.size(0)
        sents_mask = sentences.new_zeros(batch, sentences.size(1)).byte()

        for i in range(batch):
            sents_mask[i, :doc_num[i]] = 1

        sentences.masked_fill_(sents_mask.unsqueeze(2) == 0, 0)

        # get entity emb and mask
        words, _ = ewords_and_len
        # <pad> 1
        words_mask = (words != 1)

        entity_emb = self.src_embed(words)
        if self.emb_dp > 0:
            entity_emb = F.dropout(entity_emb, self.emb_dp, self.training)

        para, hn = self.encoder(sentences, sents_mask, entity_emb, words_mask, elocs)

        hn = hn.unsqueeze(0)
        cn = torch.zeros_like(hn)
        hcn = (hn, cn)

        keyinput = torch.cat((sentences, para), -1)
        key = self.linears[1](keyinput)

        return sentences, para, hcn, key

    def step(self, prev_y, prev_handc, keys, mask):
        '''
        :param prev_y: (seq_len=B, 1, H)
        :param prev_handc: (1, B, H)
        :return:
        '''
        # 1 B H
        _, (h, c) = self.decoder(prev_y, prev_handc)
        # 1 B H-> B H-> B 1 H
        query = h.squeeze(0).unsqueeze(1)
        query = self.linears[0](query)
        # B N H
        e = torch.tanh(query + keys)
        # B N
        e = self.linears[2](e).squeeze(2)
        '''
        keys = keys.transpose(1, 2)
        e = torch.matmul(query, keys).squeeze(1)
        '''
        e.masked_fill_(mask, -1e9)
        logp = F.log_softmax(e, dim=-1)

        return h, c, logp

    def load_pretrained_emb(self, emb):
        self.src_embed = nn.Embedding.from_pretrained(emb, freeze=False).cuda()


def beam_search_pointer(args, model, src_and_len, doc_num, ewords_and_len, elocs):
    sentences, _, dec_init, keys = model.encode(src_and_len, doc_num,  ewords_and_len, elocs)

    document = sentences.squeeze(0)
    T, H = document.size()

    W = args.beam_size

    prev_beam = Beam(W)
    prev_beam.candidates = [[]]
    prev_beam.scores = [0]

    target_t = T - 1

    f_done = (lambda x: len(x) == target_t)

    valid_size = W
    hyp_list = []

    for t in range(target_t):
        candidates = prev_beam.candidates
        if t == 0:
            # start
            dec_input = sentences.new_zeros(1, 1, H)
            pointed_mask = sentences.new_zeros(1, T).byte()
        else:
            index = sentences.new_tensor(list(map(lambda cand: cand[-1], candidates))).long()
            # beam 1 H
            dec_input = document[index].unsqueeze(1)

            pointed_mask[torch.arange(index.size(0)), index] = 1

        dec_h, dec_c, log_prob = model.step(dec_input, dec_init, keys, pointed_mask)

        next_beam = Beam(valid_size)
        done_list, remain_list = next_beam.step(-log_prob, prev_beam, f_done)
        hyp_list.extend(done_list)
        valid_size -= len(done_list)

        if valid_size == 0:
            break

        beam_remain_ix = src_and_len[0].new_tensor(remain_list)
        dec_h = dec_h.index_select(1, beam_remain_ix)
        dec_c = dec_c.index_select(1, beam_remain_ix)
        dec_init = (dec_h, dec_c)

        pointed_mask = pointed_mask.index_select(0, beam_remain_ix)

        prev_beam = next_beam

    score = dec_h.new_tensor([hyp[1] for hyp in hyp_list])
    sort_score, sort_ix = torch.sort(score)
    output = []
    for ix in sort_ix.tolist():
        output.append((hyp_list[ix][0], score[ix].item()))
    best_output = output[0][0]

    the_last = list(set(list(range(T))).difference(set(best_output)))
    best_output.append(the_last[0])

    return best_output


def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters()]))


def train(args, train_iter, dev, fields, checkpoint):
    model = PointerNet(args)
    model.cuda()

    DOC, ORDER, GRAPH = fields
    print('1:', DOC.vocab.itos[1])
    model.load_pretrained_emb(DOC.vocab.vectors)

    print_params(model)
    print(model)

    wd = 1e-5
    opt = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95, weight_decay=wd)

    best_score = -np.inf
    best_iter = 0
    offset = 0

    criterion = nn.NLLLoss(reduction='none')
    model.equip(criterion)

    start = time.time()

    early_stop = args.early_stop

    test_data = DocDataset(path=args.test, text_field=DOC, order_field=ORDER, graph_field=GRAPH)
    test_real = DocIter(test_data, 1, device='cuda', batch_size_fn=None,
                        train=False, repeat=False, shuffle=False, sort=False)

    for epc in range(args.maximum_steps):
        for iters, batch in enumerate(train_iter):
            model.train()

            model.zero_grad()

            t1 = time.time()

            loss = model(batch.doc, batch.order, batch.doc_len, batch.e_words, batch.elocs)

            loss.backward()
            opt.step()

            t2 = time.time()
            print('epc:{} iter:{} loss:{:.2f} t:{:.2f} lr:{:.1e}'.format(epc, iters + 1, loss, t2 - t1,
                                                                         opt.param_groups[0]['lr']))

        if epc < 5:
            continue

        with torch.no_grad():
            print('valid..............')
            if args.loss:
                score = valid_model(args, model, dev, DOC, 'loss')
                print('epc:{}, loss:{:.2f} best:{:.2f}\n'.format(epc, score, best_score))
            else:
                score, pmr, ktau, _ = valid_model(args, model, dev, DOC)
                print('epc:{}, val acc:{:.4f} best:{:.4f} pmr:{:.2f} ktau:{:.4f}'.format(epc, score, best_score,
                                                                                              pmr, ktau))

            if score > best_score:
                best_score = score
                best_iter = epc

                print('save best model at epc={}'.format(epc))
                checkpoint = {'model': model.state_dict(),
                              'args': args,
                              'loss': best_score}
                torch.save(checkpoint, '{}/{}.best.pt'.format(args.model_path, args.model))

            if early_stop and (epc - best_iter) >= early_stop:
                print('early stop at epc {}'.format(epc))
                break

    print('\n*******Train Done********{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    minutes = (time.time() - start) // 60
    if minutes < 60:
        print('best:{:.2f}, iter:{}, time:{} mins, lr:{:.1e}, '.format(best_score, best_iter, minutes,
                                                                       opt.param_groups[0]['lr']))
    else:
        hours = minutes / 60
        print('best:{:.2f}, iter:{}, time:{:.1f} hours, lr:{:.1e}, '.format(best_score, best_iter, hours,
                                                                            opt.param_groups[0]['lr']))

    checkpoint = torch.load('{}/{}.best.pt'.format(args.model_path, args.model), map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        acc, pmr, ktau, pm = valid_model(args, model, test_real, DOC, shuflle_times=1)
        print('test acc:{:.4%} pmr:{:.2%} ktau:{:.4f} pm:{:.2%}'.format(acc, pmr, ktau, pm))


def valid_model(args, model, dev, field, dev_metrics=None, shuflle_times=1):
    model.eval()

    if dev_metrics == 'loss':
        total_score = []
        number = 0

        for iters, dev_batch in enumerate(dev):
            loss = model(dev_batch.doc, dev_batch.order, dev_batch.doc_len, dev_batch.e_words, dev_batch.elocs)
            n = dev_batch.order[0].size(0)
            batch_loss = -loss.item() * n
            total_score.append(batch_loss)
            number += n

        return sum(total_score) / number
    else:
        f = open(args.writetrans, 'w')

        if args.beam_size != 1:
            print('beam search with beam', args.beam_size)

        best_acc = []
        for epc in range(shuflle_times):
            truth = []
            predicted = []

            for j, dev_batch in enumerate(dev):
                tru = dev_batch.order[0].view(-1).tolist()
                truth.append(tru)

                if len(tru) == 1:
                    pred = tru
                else:
                    pred = beam_search_pointer(args, model, dev_batch.doc, dev_batch.doc_len, dev_batch.e_words, dev_batch.elocs)

                predicted.append(pred)
                print('{}|||{}'.format(' '.join(map(str, pred)), ' '.join(map(str, truth[-1]))),
                      file=f)

            right, total = 0, 0
            pmr_right = 0
            taus = []
            # pm
            pm_p, pm_r = [], []
            import itertools

            from sklearn.metrics import accuracy_score

            for t, p in zip(truth, predicted):
                if len(p) == 1:
                    right += 1
                    total += 1
                    pmr_right += 1
                    taus.append(1)
                    continue

                eq = np.equal(t, p)
                right += eq.sum()
                total += len(t)

                pmr_right += eq.all()

                # pm
                s_t = set([i for i in itertools.combinations(t, 2)])
                s_p = set([i for i in itertools.combinations(p, 2)])
                pm_p.append(len(s_t.intersection(s_p)) / len(s_p))
                pm_r.append(len(s_t.intersection(s_p)) / len(s_t))

                cn_2 = len(p) * (len(p) - 1) / 2
                pairs = len(s_p) - len(s_p.intersection(s_t))
                tau = 1 - 2 * pairs / cn_2
                taus.append(tau)

            # acc = right / total

            acc = accuracy_score(list(itertools.chain.from_iterable(truth)),
                                 list(itertools.chain.from_iterable(predicted)))

            best_acc.append(acc)

            pmr = pmr_right / len(truth)
            taus = np.mean(taus)

            pm_p = np.mean(pm_p)
            pm_r = np.mean(pm_r)
            pm = 2 * pm_p * pm_r / (pm_p + pm_r)

            print('acc:', acc)

        f.close()
        acc = max(best_acc)
        return acc, pmr, taus, pm


def decode(args, test_real, fields, checkpoint):
    with torch.no_grad():
        model = PointerNet(args)
        model.cuda()

        print('load parameters')
        model.load_state_dict(checkpoint['model'])
        DOC, ORDER = fields
        acc, pmr, ktau, _ = valid_model(args, model, test_real, DOC)
        print('test acc:{:.2%} pmr:{:.2%} ktau:{:.2f}'.format(acc, pmr, ktau))


