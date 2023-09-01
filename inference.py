import torch
from model import Encoder, Decoder, Seq2Seq, load_pretrained_model
from data_loading import Vectorization  
from data_loading import english_lower_script, devanagari_script

device = 'cuda' if torch.cuda.is_available() else 'cpu'
INST_NAME = "Training_1"
WGT_PATH = INST_NAME + "/weights/" + INST_NAME + "_model.pth"

src_glyph = Vectorization(english_lower_script)  
tgt_glyph = Vectorization(devanagari_script)    



input_dim = src_glyph.size()

output_dim = tgt_glyph.size()

enc_emb_dim = 300
dec_emb_dim = 300
enc_hidden_dim = 512
dec_hidden_dim = 512
rnn_type = "lstm"
enc2dec_hid = True
attention = True
enc_layers = 1
dec_layers = 2
m_dropout = 0
enc_bidirect = True
enc_outstate_dim = enc_hidden_dim * (2 if enc_bidirect else 1)

enc = Encoder(  input_dim= input_dim, embed_dim = enc_emb_dim,
                hidden_dim= enc_hidden_dim,
                rnn_type = rnn_type, layers= enc_layers,
                dropout= m_dropout, device = device,
                bidirectional= enc_bidirect)
dec = Decoder(  output_dim= output_dim, embed_dim = dec_emb_dim,
                hidden_dim= dec_hidden_dim,
                rnn_type = rnn_type, layers= dec_layers,
                dropout= m_dropout,
                use_attention = attention,
                enc_outstate_dim= enc_outstate_dim,
                device = device,)
model = Seq2Seq(enc, dec, pass_enc2dec_hid=enc2dec_hid,
                device=device)


model = load_pretrained_model(model, WGT_PATH)
model.to(device)
model.eval()

def perform_inference(input_sentence):
    in_vec = torch.from_numpy(src_glyph.word_to_vec(input_sentence)).to(device)
    
    p_out_list = model.active_beam_inference(in_vec, beam_width=10)
    p_result = [tgt_glyph.vec_to_word(out.cpu().numpy()) for out in p_out_list]

    result = p_result
    
    return result

def translit_sentence(sentence):
    sentence_eng = sentence.replace(".", "") 
    word_list_eng = sentence_eng.split()  
    length1 = len(word_list_eng)
    word_list_hin = []
    for i in word_list_eng:
        output = perform_inference(i)
        word_list_hin.append(output[0])

    sentence_hin = " ".join(word_list_hin)
    if sentence.endswith("."):
        sentence_hin += "|"  
    return sentence_hin







