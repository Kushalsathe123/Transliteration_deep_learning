
from data_loading import english_lower_script, devanagari_script, Vectorization

src_glyph = Vectorization(english_lower_script)  
tgt_glyph = Vectorization(devanagari_script)    



input_dim = src_glyph.size()
print(input_dim,src_glyph)
output_dim = tgt_glyph.size()
print(output_dim,tgt_glyph)
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