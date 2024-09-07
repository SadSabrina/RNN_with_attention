from tqdm import tqdm
import time
import torch.nn as nn
from torch import optim
import random
import torch

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] # предложение в вектор индексов

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence) 
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(second_lang, pair[1])
    return (input_tensor, target_tensor)

EOS_token = 1

#TRAIN
def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100):
    start = time.time()
    losses_of_epoch = []

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in tqdm(range(1, n_epochs + 1)):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        losses_of_epoch.append(loss)

        if epoch % print_every == 0:
            print(f'Loss of current epoch is {loss}')  
    total_time = time.time() - start

    return losses_of_epoch, total_time

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for dt in tqdm(dataloader):
        input_tensor, target_tensor = dt

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# EVALUATE


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('EOS')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

def evaluateRandomly(encoder, decoder, pairs, n=3):
    for i in range(n):
        pair = random.choice(pairs)
        print('Input:', pair[0])
        #print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, second_lang)
        output_sentence = ' '.join(output_words)
        print('Output:', output_sentence)