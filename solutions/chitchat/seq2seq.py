import jieba

lan1 = Vocabulary()
lan2 = Vocabulary()

with open('../../../resources/corpus/chitchat/format_v3.txt', mode='r', encoding='utf-8') as f:
    lines = f.readlines()
data = []
for item in lines[:10000]:
    data.append( [' '.join(jieba.cut(item.split('\t')[0])), ' '.join(jieba.cut(item.split('\t')[1]))] )

print(data[:10])

for i,j in data:
    lan1.add_sentence(i)
    lan2.add_sentence(j)
learning_rate = 0.001
hidden_size = 256

encoder = EncoderRNN(len(lan1),hidden_size).to(device)
decoder = DecoderRNN(hidden_size,len(lan2)).to(device)
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=learning_rate)

loss = 0
criterion = nn.NLLLoss()
turns = 200
print_every = 20
print_loss_total = 0
training_pairs = [pair2tensor(random.choice(data)) for pair in range(turns)]

for turn in range(turns):
    optimizer.zero_grad()
    loss = 0
    x,y = training_pairs[turn]
    input_length = x.size(0)
    target_length = y.size(0)
    h = encoder.initHidden()
    for i in range(input_length):
        h = encoder(x[i],h)
    decoder_input = torch.LongTensor([SOS_token]).to(device)
    for i in range(target_length):
        decoder_output,h = decoder(decoder_input,h)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        loss += criterion(decoder_output, y[i])
        if decoder_input.item() == EOS_token:break   
    print_loss_total += loss.item()/target_length
    if (turn+1) % print_every == 0 :
        print("loss:{loss:,.4f}".format(loss=print_loss_total/print_every))
        print_loss_total = 0    
    loss.backward()
    optimizer.step()

for pr in data[:10]:
    print('>>',pr[0])
    print('==',pr[1])
    print('result:',translate(pr[0]))
    print()