import torch
import torch.nn as nn


class Encoder(nn.Module):
    
    
    def __init__(self,input_dim,embed_dim,hidden_dim,rnn_type='lstm',layers=1,bidirectional=True,dropout=0,device='cpu'):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.enc_embed_dim = embed_dim
        self.enc_hidden_dim = hidden_dim
        self.enc_rnn_type = rnn_type
        self.enc_layers = layers
        self.enc_directions = 2 if bidirectional else 1
        self.device = device
        self.embedding = nn.Embedding(self.input_dim, self.enc_embed_dim)

        if self.enc_rnn_type == 'gru':
            self.enc_rnn = nn.GRU(input_size=self.enc_embed_dim,hidden_size=self.enc_hidden_dim,num_layers=self.enc_layers,bidirectional=bidirectional)
        elif self.enc_rnn_type == 'lstm':
            self.enc_rnn = nn.LSTM(input_size=self.enc_embed_dim,hidden_size=self.enc_hidden_dim,num_layers=self.enc_layers,bidirectional=bidirectional)
        else:
            raise Exception('Unknown RNN type mentioned')

    def forward(self, x, x_sz, hidden=None):
        batch_sz = x.shape[0]
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = nn.utils.rnn.pack_padded_sequence(x, x_sz, enforce_sorted=False)
        output, hidden = self.enc_rnn(x)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = output.permute(1, 0, 2)

        return output, hidden


class Decoder(nn.Module):
    def __init__(self,output_dim,embed_dim,hidden_dim,rnn_type='lstm',layers=1,use_attention=True,enc_outstate_dim=None,dropout=0,device='cpu'):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.dec_hidden_dim = hidden_dim
        self.dec_embed_dim = embed_dim
        self.dec_rnn_type = rnn_type
        self.dec_layers = layers
        self.use_attention = use_attention
        self.device = device
        if self.use_attention:
            self.enc_outstate_dim = enc_outstate_dim if enc_outstate_dim else hidden_dim
        else:
            self.enc_outstate_dim = 0

        self.embedding = nn.Embedding(self.output_dim, self.dec_embed_dim)

        if self.dec_rnn_type == 'gru':
            self.dec_rnn = nn.GRU(input_size=self.dec_embed_dim + self.enc_outstate_dim,hidden_size=self.dec_hidden_dim,num_layers=self.dec_layers,batch_first=True)
        elif self.dec_rnn_type == 'lstm':
            self.dec_rnn = nn.LSTM(input_size=self.dec_embed_dim + self.enc_outstate_dim,hidden_size=self.dec_hidden_dim,num_layers=self.dec_layers,batch_first=True)
        else:
            raise Exception('Unknown RNN type mentioned')

        self.fc = nn.Sequential(nn.Linear(self.dec_hidden_dim, self.dec_embed_dim),nn.LeakyReLU(),nn.Linear(self.dec_embed_dim, self.output_dim),)

        if self.use_attention:
            self.W1 = nn.Linear(self.enc_outstate_dim , self.dec_hidden_dim)
            self.W2 = nn.Linear(self.dec_hidden_dim, self.dec_hidden_dim)
            self.V = nn.Linear(self.dec_hidden_dim, 1)

    def attention(self, x, hidden, enc_output):
        hidden_with_time_axis = torch.sum(hidden, dim=0)
        hidden_with_time_axis = hidden_with_time_axis.unsqueeze(1)

        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))

        attention_weights = torch.softmax(self.V(score), dim=1)

        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)
        context_vector = context_vector.unsqueeze(1)

        attend_out = torch.cat((context_vector, x), -1)

        return attend_out, attention_weights

    def forward(self, x, hidden, enc_output):
        if hidden is None and self.use_attention is False:
            raise Exception("Decoder with no attention and no hidden state is not supported.")

        batch_size = x.shape[0]

        if hidden is None:
            hid_for_att = torch.zeros((self.dec_layers, batch_size, self.dec_hidden_dim)).to(self.device)
        elif self.dec_rnn_type == 'lstm':
            hid_for_att = hidden[0]
        else:
            hid_for_att = hidden

        x = self.embedding(x)

        if self.use_attention:
            x, attention_weights = self.attention(x, hid_for_att, enc_output)
        else:
            attention_weights = 0

        output, hidden = self.dec_rnn(x, hidden) if hidden is not None else self.dec_rnn(x)

        output = output.view(-1, output.size(2))
        output = self.fc(output)

        return output, hidden, attention_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pass_enc2dec_hid=False, dropout=0, device="cpu"):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pass_enc2dec_hid = pass_enc2dec_hid

        if self.pass_enc2dec_hid:
            assert decoder.dec_hidden_dim == encoder.enc_hidden_dim, "Hidden Dimension of encoder and decoder must be the same, or unset `pass_enc2dec_hid`"
        if decoder.use_attention:
            assert decoder.enc_outstate_dim == encoder.enc_directions * encoder.enc_hidden_dim, "Set `enc_out_dim` correctly in decoder"
        assert self.pass_enc2dec_hid or decoder.use_attention, "No use of a decoder with No attention and No Hidden from Encoder"

    def forward(self, src, tgt, src_sz, teacher_forcing_ratio=0):
        batch_size = tgt.shape[0]

        enc_output, enc_hidden = self.encoder(src, src_sz)

        if self.pass_enc2dec_hid:
            dec_hidden = enc_hidden
        else:
            dec_hidden = None

        pred_vecs = torch.zeros(batch_size, self.decoder.output_dim, tgt.size(1)).to(self.device)

        dec_input = tgt[:, 0].unsqueeze(1)
        pred_vecs[:, 1, 0] = 1
        for t in range(1, tgt.size(1)):
            dec_output, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            pred_vecs[:, :, t] = dec_output

            prediction = torch.argmax(dec_output, dim=1)

            if torch.rand(1) < teacher_forcing_ratio:
                dec_input = tgt[:, t].unsqueeze(1)
            else:
                dec_input = prediction.unsqueeze(1)

        return pred_vecs

    def inference(self, src, max_tgt_sz=50, debug=0):
        batch_size = 1
        start_tok = src[0]
        end_tok = src[-1]
        src_sz = torch.tensor([len(src)])
        src_ = src.unsqueeze(0)

        enc_output, enc_hidden = self.encoder(src_, src_sz)

        if self.pass_enc2dec_hid:
            dec_hidden = enc_hidden
        else:
            dec_hidden = None

        pred_arr = torch.zeros(max_tgt_sz, 1).to(self.device)
        if debug:
            attend_weight_arr = torch.zeros(max_tgt_sz, len(src)).to(self.device)

        dec_input = start_tok.view(1, 1)
        pred_arr[0] = start_tok.view(1, 1)
        for t in range(max_tgt_sz):
            dec_output, dec_hidden, aw = self.decoder(dec_input, dec_hidden, enc_output)

            prediction = torch.argmax(dec_output, dim=1)
            dec_input = prediction.unsqueeze(1)
            pred_arr[t] = prediction
            if debug:
                attend_weight_arr[t] = aw.squeeze(-1)

            if torch.eq(prediction, end_tok):
                break

        if debug:
            return pred_arr.squeeze(), attend_weight_arr

        return pred_arr.squeeze().to(dtype=torch.long)

    def active_beam_inference(self, src, beam_width=3, max_tgt_sz=50):
        def _avg_score(p_tup):
            return p_tup[0]

        batch_size = 1
        start_tok = src[0]
        end_tok = src[-1]
        src_sz = torch.tensor([len(src)])
        src_ = src.unsqueeze(0)

        enc_output, enc_hidden = self.encoder(src_, src_sz)

        if self.pass_enc2dec_hid:
            init_dec_hidden = enc_hidden
        else:
            init_dec_hidden = None

        top_pred_list = [(0, start_tok.unsqueeze(0), init_dec_hidden)]

        for t in range(max_tgt_sz):
            cur_pred_list = []

            for p_tup in top_pred_list:
                if p_tup[1][-1] == end_tok:
                    cur_pred_list.append(p_tup)
                    continue

                dec_output, dec_hidden, _ = self.decoder(p_tup[1][-1].view(1, 1), p_tup[2], enc_output)

                dec_output = nn.functional.log_softmax(dec_output, dim=1)

                pred_topk = torch.topk(dec_output, k=beam_width, dim=1)

                for i in range(beam_width):
                    sig_logsmx_ = p_tup[0] + pred_topk.values[0][i]
                    seq_tensor_ = torch.cat((p_tup[1], pred_topk.indices[0][i].view(1)))

                    cur_pred_list.append((sig_logsmx_, seq_tensor_, dec_hidden))

            cur_pred_list.sort(key=_avg_score, reverse=True)
            top_pred_list = cur_pred_list[:beam_width]

            end_flags_ = [1 if t[1][-1] == end_tok else 0 for t in top_pred_list]
            if beam_width == sum(end_flags_):
                break

        pred_tnsr_list = [t[1] for t in top_pred_list]

        return pred_tnsr_list

    def passive_beam_inference(self, src, beam_width=7, max_tgt_sz=50):
        def _avg_score(p_tup):
            return p_tup[0]

        def _beam_search_topk(topk_obj, start_tok, beam_width):
            top_pred_list = [(0, start_tok.unsqueeze(0)), ]

            for obj in topk_obj:
                new_lst_ = list()
                for itm in top_pred_list:
                    for i in range(beam_width):
                        sig_logsmx_ = itm[0] + obj.values[0][i]
                        seq_tensor_ = torch.cat((itm[1], obj.indices[0][i].view(1)))
                        new_lst_.append((sig_logsmx_, seq_tensor_))

                new_lst_.sort(key=_avg_score, reverse=True)
                top_pred_list = new_lst_[:beam_width]
            return top_pred_list

        batch_size = 1
        start_tok = src[0]
        end_tok = src[-1]
        src_sz = torch.tensor([len(src)])
        src_ = src.unsqueeze(0)

        enc_output, enc_hidden = self.encoder(src_, src_sz)

        if self.pass_enc2dec_hid:
            dec_hidden = enc_hidden
        else:
            dec_hidden = None

        dec_input = start_tok.view(1, 1)

        topk_obj = []
        for t in range(max_tgt_sz):
            dec_output, dec_hidden, aw = self.decoder(dec_input, dec_hidden, enc_output)

            dec_output = nn.functional.log_softmax(dec_output, dim=1)

            pred_topk = torch.topk(dec_output, k=beam_width, dim=1)

            topk_obj.append(pred_topk)
            dec_input = pred_topk.indices[0][0].view(1, 1)
            if torch.eq(dec_input, end_tok):
                break

        top_pred_list = _beam_search_topk(topk_obj, start_tok, beam_width)
        pred_tnsr_list = [t[1] for t in top_pred_list]

        return pred_tnsr_list

def load_pretrained_model(model, weight_path, flexible=False):
    if not weight_path:
        return model

    pretrain_dict = torch.load(weight_path, map_location=torch.device('cpu'))

    model_dict = model.state_dict()
    
    if flexible:
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}

    print("Pretrained layers:", pretrain_dict.keys())
    
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    return model
   


