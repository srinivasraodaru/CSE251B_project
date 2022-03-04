################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

# Build and return the model here based on the configuration.

import torch
import torch.nn as nn
from torchvision import models

def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    vocab_len=len(vocab)
    model_lstm = Img_Cap(config_data, vocab_len)
    
    return model_lstm
    
#     raise NotImplementedError("Model Factory Not Implemented")

class Img_Cap(nn.Module):
    
    def __init__(self, config_data , vocab_len,train=True):
        super().__init__()
        self.model_type = config_data["model"]["model_type"]
        self.embedding_size = config_data["model"]["embedding_size"]
        self.hidden_size = config_data["model"]["hidden_size"]
        self.vocab_len=vocab_len
        self.encoder = ResNet50CNN(self.embedding_size)
        self.decoder = Decoder( config_data ,self.vocab_len)

        for params in self.decoder.parameters():
            params.requires_grad = True
            
    def forward(self, images,captions,train=True):
        encoded_images = self.encoder(images)
        output_captions , output_captions_idx= self.decoder(encoded_images,captions,train)
        
        return output_captions, output_captions_idx
            
        

class ResNet50CNN(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.embedding_size = embedding_size
        num_input_ftrs = models.resnet50(pretrained = True).fc.in_features
        self.pretrained_model = nn.Sequential(*(list(models.resnet50(pretrained = True).children())[:-1]))
        self.linear = nn.Linear(num_input_ftrs , self.embedding_size)
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False        
        
        for param in self.linear.parameters():
            param.requires_grad = True
    def forward(self,images):
        
        features = self.pretrained_model(images)
        features = features.reshape(features.size(0),-1)
        encoded_image = self.linear(features)
        
        return encoded_image
        
    
    
class Decoder(nn.Module):
    def __init__(self, config_data, vocab_len):
        super().__init__()

        self.num_layers     = config_data['model']['num_layers']
        self.embedding_size = config_data['model']['embedding_size']
        self.hidden_size    = config_data['model']['hidden_size']
        self.model_type     = config_data['model']['model_type']
        self.max_length     = config_data['generation']['max_length']
        self.temp           = config_data['generation']['temperature']
        self.deterministic  = config_data['generation']['deterministic']
        self.vocab_len      = vocab_len

        self.embedding      = nn.Embedding(self.vocab_len, self.embedding_size)
        
        #in baseline put no_layers=2
        if self.model_type == 'LSTM':
            self.layer = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)
        elif self.model_type == 'RNN':
            self.layer = nn.RNN(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)
            
        self.linear=nn.Linear(self.hidden_size, self.vocab_len)

        
    def forward(self, encoded_images, captions, train=True):
                       
        if train == True :
            word_embeddings = self.embedding(captions)
            embeddings      = torch.cat((encoded_images.unsqueeze(1), word_embeddings),1)
            hidden,_        = self.layer(embeddings)
            vocab_output    = self.linear(hidden) 
            vocab_output_prob = nn.functional.softmax(vocab_output,dim=1)
            _, a = torch.max(vocab_output_prob,2)
            sampled_index = a
            return vocab_output, sampled_index
                            
        else :
            input_embedding = encoded_images.unsqueeze(1)
            output_captions = []
            output_captions_idx=[]
            for i in range(self.max_length):
                hidden,_         = self.layer(input_embedding)
                output_embedding = self.linear(hidden[0])
                output_prob      = nn.functional.softmax((output_embedding)/(self.temp),dim=1)
                output_prob.to("cuda")
                
                if self.deterministic == True :
                    _, a = torch.max(output_prob,1)
                    sampled_index = a
                    input_embedding = self.embedding(a).unsqueeze(1)
                    
                else :
                    sampled_index = list(torch.utils.data.WeightedRandomSampler(output_prob,1))
                    sampled_index=torch.Tensor(sampled_index[0])
                    sampled_index=sampled_index.int()
                    sampled_index=sampled_index.cuda()
                    input_embedding = self.embedding(sampled_index).unsqueeze(1)                    
                    
                output_captions.append(output_embedding.tolist())
                output_captions_idx.append(sampled_index)

                
            return output_captions, output_captions_idx
       
        