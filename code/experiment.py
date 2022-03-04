################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch.nn as nn

from constants import ROOT_STATS_DIR
import torch
import torchvision
from file_utils import *
from model_factory import get_model
import warnings
from torch.nn.utils.rnn import pack_padded_sequence

warnings.filterwarnings("ignore")

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
#         self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
#             config_data)
        self.__trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        self.__train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

        self.__testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
        self.__test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
        self.__classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
#         self.__model = get_model(config_data, self.__vocab)

        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.AdamW(self.__model.parameters(), lr=config_data["experiment"]["learning_rate"])

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)
            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)
            

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = 0
      
        device = torch.device('cuda') # determine which device to use (gpu or cpu)
        use_gpu = torch.cuda.is_available()
        print("gpu availability ----------------->" , use_gpu)


        for i, (images, captions, lengths, _) in enumerate(self.__train_loader):
#             reset optimizer gradients
            self.__optimizer.zero_grad()
            pred_captions  = []
            label_captions = []
       
            # # both inputs and labels have to reside in the same device as the model's
            images   = images.to(device) #transfer the input to the same device as the model's
            captions = captions.to(device) #transfer the labels to the same device as the model's
            self.__model.to(device)
            output_captions, output_captions_idx = self.__model(images,captions,train=True)
            
            packed_output_captions = pack_padded_sequence(output_captions, lengths,batch_first = True)
            packed_captions=pack_padded_sequence(captions, lengths,batch_first = True)
            

            loss = self.__criterion(packed_output_captions.data, packed_captions.data)#calculate loss
            loss.backward()
            
            
            label_captions = []
            for c in captions:
                sent = []
                for w in c:
                    word_value = self.__vocab.idx2word[w.item()]
                    sent.append(word_value)
                label_captions.append(sent)

            pred_captions = []
            for c in output_captions_idx:
                sent = []
                for w in c:
                    word_value = self.__vocab.idx2word[w.item()]
                    sent.append(word_value)
                pred_captions.append(sent)
                
            if i%500 == 0 :
                print("label ---->", label_captions[0])
                print("out ---->",pred_captions[0])
                
            if i%500 == 0 :            
                print("i : {}, loss: {}".format(i, loss.item()))
                print("******************************")
            # update the weights
            self.__optimizer.step()
            
            #raise NotImplementedError()
        return loss.item()

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = 0
        device = torch.device('cuda') # determine which device to use (gpu or cpu)
        use_gpu = torch.cuda.is_available()
        self.__model.to(device)
        
        val_loss_list = []
        bleu1_list = []
        bleu4_list = []
        bleu1_value = 0
        bleu4_value = 0

        with torch.no_grad():
            for i, (images, captions, lengths, _) in enumerate(self.__val_loader):
                pred_captions  = []
                label_captions = []      
                images   = images.to(device)
                captions = captions.to(device)
                
                output_captions, output_captions_idx = self.__model(images,captions,train=False)     
                output_captions_for_loss, output_captions_idx_for_loss = self.__model(images,captions,train=True)
                
                for word in output_captions_idx:
                    a = word.item()
                    word_value = self.__vocab.idx2word[a]
                    if (word_value != "<start>") and (word_value != "<end>" ):
                        pred_captions.append(word_value)
                    elif word_value == "<end>" :
                        break
                for sent in captions:
                    temp_sent = []
                    for word in sent:
                        a = word.item()
                        word_value = self.__vocab.idx2word[a]
                        if (word_value != "<start>") and (word_value != "<end>" ):
                            temp_sent.append(word_value)
                    label_captions.append(temp_sent)
                
                output_captions = (torch.Tensor(output_captions)).to(device)
                output_captions = output_captions.permute([1,0,2])[0]
               
                 

                captions_new = []
                captions_new.extend(captions[0])
                captions_new = torch.Tensor(captions_new)
                captions_new = captions_new.to(device)
                output_captions_for_loss=output_captions_for_loss.to(device)
                
                val_loss       = self.__criterion(output_captions_for_loss, captions_new.long()) #calculate loss   
                bleu1_value     = bleu1(label_captions, pred_captions)
                bleu4_value     = bleu4(label_captions, pred_captions)
                val_loss_list.append(val_loss.item())
                bleu1_list.append(bleu1_value)
                bleu4_list.append(bleu4_value)
        result_str = "Val Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(np.mean(val_loss_list), np.mean(bleu1_list),np.mean(bleu4_list))
        self.__log(result_str)

        return np.mean(val_loss_list)
    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__model.eval()
        test_loss = 0
        device = torch.device('cuda') # determine which device to use (gpu or cpu)
        use_gpu = torch.cuda.is_available()
        self.__model.to(device)
        
        test_loss_list = []
        bleu1_list = []
        bleu4_list = []
        bleu1_value = 0
        bleu4_value = 0

  
        with torch.no_grad():
            for i, (images, captions, img_ids) in enumerate(self.__test_loader):
                pred_captions  = []
                label_captions = []      
                images   = images.to(device)
                captions = captions.to(device)
                
                output_captions, output_captions_idx = self.__model(images,captions,train=False)     
                output_captions_for_loss, output_captions_idx_for_loss = self.__model(images,captions,train=True)
                
                for word in output_captions_idx:
                    a = word.item()
                    word_value = self.__vocab.idx2word[a]
                    if (word_value != "<start>") and (word_value != "<end>" ):
                        pred_captions.append(word_value)
                    elif word_value == "<end>" :
                        break
                for sent in captions:
                    temp_sent = []
                    for word in sent:
                        a = word.item()
                        word_value = self.__vocab.idx2word[a]
                        if (word_value != "<start>") and (word_value != "<end>" ):
                            temp_sent.append(word_value)
                    label_captions.append(temp_sent)
                
                output_captions = (torch.Tensor(output_captions)).to(device)
                output_captions = output_captions.permute([1,0,2])[0]
               
                 

                captions_new = []
                captions_new.extend(captions[0])
                captions_new = torch.Tensor(captions_new)
                captions_new = captions_new.to(device)
                output_captions_for_loss=output_captions_for_loss.to(device)
                
                test_loss       = self.__criterion(output_captions_for_loss, captions_new.long()) #calculate loss   
                bleu1_value     = bleu1(label_captions, pred_captions)
                bleu4_value     = bleu4(label_captions, pred_captions)
                test_loss_list.append(test_loss.item())
                bleu1_list.append(bleu1_value)
                bleu4_list.append(bleu4_value)
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(np.mean(test_loss_list), np.mean(bleu1_list),np.mean(bleu4_list))
        self.__log(result_str)

        return np.mean(test_loss_list)

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        training_losses = torch.tensor(self.__training_losses, device="cpu")
        val_losses = torch.tensor(self.__val_losses, device="cpu")
        plt.plot(x_axis, training_losses, label="Training Loss")
        plt.plot(x_axis, val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
