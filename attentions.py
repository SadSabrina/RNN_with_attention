import torch
import torch.nn as nn
import torch.nn.functional as F

# LUONG ATTENTION

class LuongAttention(nn.Module):                 # при инициализации, будем выбирать метод и задавать размерность скрытого состояния декодера
    def __init__(self, method, hidden_size): 
        super(LuongAttention, self).__init__()
        self.hidden_size = hidden_size
        self.method = method
        
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
      
        if self.method == 'general':
            self.Wa = nn.Linear(self.hidden_size, hidden_size)          # Обучаемая матрица весовдля обобщенного ск. произведения
        
        elif self.method == 'concat':
            self.Wa = nn.Linear(self.hidden_size * 2, hidden_size)      # Обучаемая матрица весовдля обобщенного ск. произведения
            self.va = nn.Parameter(torch.FloatTensor(hidden_size))      # И вспомогательный вектор


    def forward(self, hidden, encoder_outputs): # forward по скрытым состояним декодра и выходам энкодера

        if self.method == 'general':
            scores = self.Wa(encoder_outputs)
            attn_weights = torch.sum(hidden*scores, dim=2)

        elif self.method == 'concat':
            scores = self.Wa(torch.cat((hidden.expand(-1, encoder_outputs.size(1), -1), encoder_outputs), 2)).tanh()
            attn_weights = torch.sum(self.va * scores, dim=2)

        elif self.method == 'dot':
            attn_weights = torch.sum(hidden * encoder_outputs, dim=2)

        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)
        context = attn_weights.bmm(encoder_outputs)

        return context, attn_weights
    

# BAHDANAU ATTENTION

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size) # обучаемая матрица весов для векторов скрытого состояния декодера
        self.Ua = nn.Linear(hidden_size, hidden_size) # обучаемая матрица весов для скрытого состояния энкодера
        self.Va = nn.Linear(hidden_size, 1) # обучаемый вектор весов для вычисленного внимания

    def forward(self, decoder_hidden, encoder_hidden):
        scores = self.Va(torch.tanh(self.Wa(decoder_hidden) + self.Ua(encoder_hidden))) 
        scores = scores.squeeze(2).unsqueeze(1) 

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, encoder_hidden)


        return context, attn_weights