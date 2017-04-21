import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import reporter


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0, force_tuple=True)
    return exs


class Seq2seqAttention(chainer.Chain):

    """Implementaion of Luong's attentional NMT model."""

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units):
        super(Seq2seqAttention, self).__init__(
            embed_x=L.EmbedID(n_source_vocab, n_units),
            embed_y=L.EmbedID(n_target_vocab, n_units),
            encoder=L.NStepLSTM(n_layers, n_units, n_units, 0.1),
            decoder=L.NStepLSTM(n_layers, n_units, n_units, 0.1),
            W=L.Linear(n_units, n_target_vocab),
            Wc=L.Linear(n_units * 2, n_units),
        )

        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, *inputs):
        xs = inputs[:len(inputs) // 2]
        ys = inputs[len(inputs) // 2:]

        xs = [x[::-1] for x in xs]

        eos = self.xp.zeros(1, 'i')
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # Initial hidden variable and cell variable
        zero = self.xp.zeros((self.n_layers, batch, self.n_units), 'f')
        hx, cx, hxs = self.encoder(zero, zero, exs)
        _, _, os = self.decoder(hx, cx, eys)

        indices = numpy.argsort([-y.shape[0] for y in ys]).astype('i')

        hxs = [hxs[i] for i in indices]
        hxs_zero = F.pad_sequence(hxs, padding=0)

        inf = self.xp.zeros(hxs_zero.shape[0:2], 'f')
        for i, hx in enumerate(hxs):
            inf[i, hx.shape[0]:] = -1000

        os = [os[i] for i in indices]
        os_t = F.transpose_sequence(os)
        ys_out = [ys_out[i] for i in indices]
        os_new_t = []
        for h in os_t:
            batch = h.shape[0]
            hxs_z = hxs_zero[0:batch]

            s = F.batch_matmul(hxs_z, h)
            s += inf[0:batch, :, None]

            a = F.softmax(s)
            a = F.broadcast_to(a, hxs_z.shape)
            wh = hxs_z * a
            c = F.average(wh, axis=1)
            h = self.Wc(F.concat([c, h]))
            os_new_t.append(h)
        os = F.transpose_sequence(os_new_t)

        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, normalize=False) \
            * concat_ys_out.shape[0] / batch

        reporter.report({'loss': loss.data}, self)
        perp = self.xp.exp(loss.data / concat_ys_out.shape[0] * batch)
        reporter.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=50):
        batch = len(xs)
        with chainer.no_backprop_mode():
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            # Initial hidden variable and cell variable
            zero = self.xp.zeros((self.n_layers, batch, self.n_units), 'f')
            h, c, hxs = self.encoder(zero, zero, exs, train=False)

            hxs_zero = F.pad_sequence(hxs, padding=0)

            inf = self.xp.zeros(hxs_zero.shape[0:2], 'f')
            for i, hx in enumerate(hxs):
                inf[i, 0:hx.shape[0]] = -1000

            ys = self.xp.zeros(batch, 'i')
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = chainer.functions.split_axis(
                    eys, batch, 0, force_tuple=True)
                h, c, ys = self.decoder(h, c, eys, train=False)
                cys = chainer.functions.concat(ys, axis=0)

                s = F.batch_matmul(hxs_zero, cys)
                s += inf[0:batch, :, None]

                a = F.softmax(s)
                a = F.broadcast_to(a, hxs_zero.shape)
                wh = hxs_zero * a
                cy = F.average(wh, axis=1)
                cys = self.Wc(F.concat([cy, cys]))

                wy = self.W(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype('i')
                result.append(ys)

        result = cuda.to_cpu(self.xp.stack(result).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == 0)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs
