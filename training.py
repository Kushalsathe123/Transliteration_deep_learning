import torch
import os
from tqdm import tqdm
from model import Encoder, Decoder, Seq2Seq, load_pretrained_model
from data_loading import create_dataloader
import csv
from data_loading import english_lower_script, devanagari_script
import torch.nn as nn 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
INST_NAME = "Training_1"
LOG_PATH = INST_NAME + "/"
WGT_PREFIX = LOG_PATH + "weights/" + INST_NAME
if not os.path.exists(LOG_PATH + "weights"):
    os.makedirs(LOG_PATH + "weights")

num_epochs = 5
batch_size = 1000
acc_grad = 1
learning_rate = 1e-3
teacher_forcing, teach_force_till, teach_decay_pereph = 1, 20, 0
pretrain_wgt_path = None  

from data_loading import Vectorization  

src_glyph = Vectorization(english_lower_script)  
tgt_glyph = Vectorization(devanagari_script)    

TRAIN_FILE = "HiEn_ann1_train.json"
VALID_FILE = "HiEn_ann1_test.json"

train_dataloader = create_dataloader(TRAIN_FILE, src_glyph, tgt_glyph, batch_size, shuffle=True)
val_dataloader = create_dataloader(VALID_FILE, src_glyph, tgt_glyph, batch_size, shuffle=False)

enc = Encoder(input_dim=src_glyph.size(), embed_dim=300, hidden_dim=512, device=device)
dec = Decoder(output_dim=tgt_glyph.size(), embed_dim=300, hidden_dim=512, device=device)

model = Seq2Seq(enc, dec, pass_enc2dec_hid=True, device=device)

model = load_pretrained_model(model, pretrain_wgt_path)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)

best_loss = float("inf")
for epoch in range(num_epochs):
    model.train()
    acc_loss = 0
    running_loss = []

    for ith, (src, tgt) in enumerate(train_dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        output = model(src=src, tgt=tgt, teacher_forcing_ratio=teacher_forcing)
        loss = criterion(output.view(-1, output.shape[-1]), tgt.view(-1))
        acc_loss += loss

        loss.backward()
        if (ith + 1) % acc_grad == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch [{epoch+1}/{num_epochs}], MiniBatch [{ith+1}/{len(train_dataloader)}], Loss: {acc_loss.item():.4f}')
            running_loss.append(acc_loss.item())
            acc_loss = 0

    with open(LOG_PATH + "trainLoss.csv", 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(running_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for jth, (v_src, v_tgt) in enumerate(val_dataloader):
            v_src = v_src.to(device)
            v_tgt = v_tgt.to(device)
            v_output = model(src=v_src, tgt=v_tgt)
            v_loss = criterion(v_output.view(-1, v_output.shape[-1]), v_tgt.view(-1))
            val_loss += v_loss

    val_loss /= len(val_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss.item():.4f}')

    with open(LOG_PATH + "valLoss.csv", 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([val_loss.item()])

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model, WGT_PREFIX + "_model.pth")
        print("*** Best model saved ***")

print("Training finished.")
