from data_loading import TranslitDataset, DataLoader, Vectorization
from model import Encoder, Decoder, Seq2Seq, load_pretrained_model
from training import train_model
from inference import perform_inference
import torch
from data_loading import english_lower_script, devanagari_script
from training import TRAIN_FILE, VALID_FILE, batch_size



device = 'cuda' if torch.cuda.is_available() else 'cpu'
INST_NAME = "Training_1"
WGT_PATH = INST_NAME + "/weights/" + INST_NAME + "_model.pth"


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


train_dataset = TranslitDataset(src_glyph, tgt_glyph, TRAIN_FILE)
val_dataset = TranslitDataset(src_glyph, tgt_glyph, VALID_FILE)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


enc = Encoder(input_dim=src_glyph.size(), embed_dim=300, hidden_dim=512, device=device)
dec = Decoder(output_dim=tgt_glyph.size(), embed_dim=300, hidden_dim=512, device=device)
model = Seq2Seq(enc, dec, pass_enc2dec_hid=True, device=device)


model = load_pretrained_model(model, WGT_PATH)


num_epochs = 10
learning_rate = 1e-3


trained_model = train_model(model, train_dataloader, val_dataloader, num_epochs, learning_rate)


torch.save(trained_model, "trained_model.pth")


input_sentence = "Hello, how are you?"
output_sentence = perform_inference(trained_model, src_glyph, tgt_glyph, input_sentence)
print("Input:", input_sentence)
print("Output:", output_sentence)
