#hpc
import torch
import sys
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from utils.utils import *
from model.Detector import Detector
from data_factory.data_loader import get_loader_segment
from data_factory.data_loader import SMDSegLoader
from einops import rearrange
from metrics.metrics import *
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from torchviz import make_dot
from datetime import datetime
#from main import config
warnings.filterwarnings('ignore')
 
writer = SummaryWriter()  #tensorboard 
#
#os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
  

def tsallis_divergence(p, q, q_param=1.5, eps=1e-6):
    """Tsallis Divergence: D_q(P||Q) = 1/(q-1) * (sum_x[P(x)^q * Q(x)^(1-q)] - 1)"""
    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)
    inner = p_safe.pow(q_param) * q_safe.pow(1 - q_param)
    summed = torch.sum(inner, dim=-1) - 1
    return torch.mean(summed / (q_param - 1), dim=1)

def renyi_divergence(p, q, q_param=1.5, eps=1e-6):
    """Renyi Divergence: D_q(P||Q) = 1/(q-1) * log(sum_x[P(x)^q * Q(x)^(1-q)])"""
    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)
    inner = p_safe.pow(q_param) * q_safe.pow(1 - q_param)
    summed = torch.sum(inner, dim=-1)
    return torch.mean(torch.log(summed) / (q_param - 1), dim=1)

def kl_divergence(p, q, eps=1e-6):
    """Standard KL divergence: D_KL(P||Q) = sum_x[P(x) * log(P(x)/Q(x))]"""
    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)
    kl = p_safe * (torch.log(p_safe) - torch.log(q_safe))
    return torch.mean(torch.sum(kl, dim=-1), dim=1)

def get_divergence_fn(divergence_type, q_param=1.5):
    """Factory function to select the right divergence measure"""
    if divergence_type == 'tsallis':
        return lambda p, q: tsallis_divergence(p, q, q_param)
    elif divergence_type == 'renyi':
        return lambda p, q: renyi_divergence(p, q, q_param)
    elif divergence_type == 'kl':
        return lambda p, q: kl_divergence(p, q)
    else:
        raise ValueError(f"Unknown divergence type: {divergence_type}")


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

        
class Solver(object):
    DEFAULTS = {}
    
    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)
    
        # Initialize divergence function based on command line arguments
        self.divergence_fn = get_divergence_fn(
            self.divergence, 
            q_param=self.q_param
        )
        
        self.train_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset, shuffle=True) 
        self.vali_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='thre', dataset=self.dataset)

        self.build_model()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
        

    def build_model(self):
    
        self.model = Detector(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, n_heads=self.n_heads, d_model=self.d_model, e_layers=self.e_layers, patch_size=self.patch_size, channel=self.input_c)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if torch.cuda.is_available():
            self.model.cuda()

            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # # Add model to TensorBoard
        # from torch import randn
        # dummy = randn(1, self.win_size, self.input_c, device=self.device)
        # writer.add_graph(self.model, dummy)
        # writer.flush()
        
        # Visualize with torchviz - fixed to handle list outputs
        # try:
        #     dummy_input = torch.randn(1, self.win_size, self.input_c).to(self.device)
        #     y = self.model(dummy_input)
            
        #     # Either pick first tensor from the list
        #     if isinstance(y[0], list) and len(y[0]) > 0:
        #         # Create a scalar output from all tensors for visualization
        #         series, prior = y
        #         dummy_output = sum(tensor.sum() for tensor in series + prior)
        #         dot = make_dot(dummy_output, params=dict(self.model.named_parameters()))


        #     # Add random number to filename
        #     random_id = np.random.randint(1, 9999)
        #     filename = f'model_visualization_{random_id}'
            
        #     dot.format = 'png'
        #     dot.render(filename)
        #     print(f"Model visualization saved to {filename}.png")
        # except Exception as e:
        #     print(f"Warning: Could not create torchviz visualization: {e}")
        
    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(self.divergence_fn(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    self.divergence_fn(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    self.divergence_fn((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    self.divergence_fn(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            loss_1.append((prior_loss - series_loss).item())

        return np.average(loss_1), np.average(loss_2)

        ####################################################################################################
        #                                       T R A I N                                                  #
        ####################################################################################################   
    def train(self):
        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.data_path)
        
        # Create epoch progress bar
        epoch_pbar = tqdm(range(self.num_epochs), desc="Epochs")
        
        for epoch in epoch_pbar:
            # re-shuffle train set every epoch
            self.train_loader = get_loader_segment(
                self.index,
                'dataset/' + self.data_path,
                batch_size=self.batch_size,
                win_size=self.win_size,
                mode='train',
                dataset=self.dataset,
                shuffle=True
            )
            train_steps = len(self.train_loader)
            running_loss = 0.0
            iter_count = 0
            epoch_time = time.time()
            self.model.train()

            # Create iteration progress bar
            iter_pbar = tqdm(enumerate(self.train_loader), 
                            total=train_steps, 
                            desc=f"Epoch {epoch+1}/{self.num_epochs}",
                            leave=False)  # leave=False prevents multiple progress bars

            for i, (input_data, labels) in iter_pbar:
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                series, prior = self.model(input)
              
                series_loss = 0.0
                prior_loss = 0.0

                for u in range(len(prior)):
                    series_loss += (torch.mean(self.divergence_fn(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        self.divergence_fn((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(self.divergence_fn(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        self.divergence_fn(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                loss = prior_loss - series_loss
                running_loss += loss.item()
                
                # Update iteration progress bar with current loss
                iter_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    
                    # Update progress bar with speed and ETA
                    iter_pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "speed": f"{speed:.4f}s/iter", 
                        "ETA": f"{left_time:.1f}s"
                    })
                    
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()
                
            # Calculate epoch metrics
            avg_epoch_loss = running_loss / train_steps
            
            # Update the epoch progress bar with epoch loss
            epoch_pbar.set_postfix({"loss": f"{avg_epoch_loss:.4f}"})
            
            # Log to tensorboard
            writer.add_scalar('Train/Loss', avg_epoch_loss, epoch)
            
            # Validation and early stopping
            vali_loss1, vali_loss2 = self.vali(self.test_loader)
            writer.add_scalar('Validation/Loss1', vali_loss1, epoch)
            writer.add_scalar('Validation/Loss2', vali_loss2, epoch)
            
            print(
                "Epoch: {0}, Cost time: {1:.3f}s, Loss: {2:.6f}".format(
                    epoch + 1, time.time() - epoch_time, avg_epoch_loss))
            
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
            
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
 
        ####################################################################################################
        #                                          T E S T                                                 #
        ####################################################################################################           
    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.data_path) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = self.divergence_fn(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = self.divergence_fn(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += self.divergence_fn(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += self.divergence_fn(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            # NEW CODE : Log anomaly scores to TensorBoard
            #writer.add_scalar('Anomaly Score', np.mean(cri), i)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshol
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = self.divergence_fn(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = self.divergence_fn(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += self.divergence_fn(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += self.divergence_fn(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print('====================  Threshhold  ===================\n')
        #print(f"Threshold : {thresh}\n")
        print(f"\033[94mThreshold : {thresh}\033[0m\n")
        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = self.divergence_fn(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = self.divergence_fn(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += self.divergence_fn(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += self.divergence_fn(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # for i, val in enumerate(test_energy):
        #     if val > thresh:
        #         print(f"Index: {i}, Value: {val}")
        
        # #sys.exit()





        matrix = [self.index]
        print('==================== EVALUATION Metrics ===================\n')
        with tqdm(total=1, desc="Processing") as pbar:
            # Run the heavy process
            scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
            
            # Manually update the progress bar to 100% when done
            pbar.update(1)

        #scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        
        for key, value in scores_simple.items():
            matrix.append(value)
            print('{0:21} : {1:0.4f}'.format(key, value))


        # anomaly_state = False
        # for i in range(len(gt)):
        #     if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
        #         anomaly_state = True
        #         for j in range(i, 0, -1):
        #             if gt[j] == 0:
        #                 break
        #             else:
        #                 if pred[j] == 0:
        #                     pred[j] = 1
        #         for j in range(i, len(gt)):
        #             if gt[j] == 0:
        #                 break
        #             else:
        #                 if pred[j] == 0:
        #                     pred[j] = 1
        #     elif gt[i] == 0:
        #         anomaly_state = False
        #     if anomaly_state:
        #         pred[i] = 1



        gt = np.array(gt)
        pred = np.array(pred)
        print('====================  MODEL DETECTION  ===================')

        # Find all ground truth anomaly starts (regardless of predictions)
        gt_anomaly_starts = np.where((gt[:-1] == 0) & (gt[1:] == 1))[0] + 1

        if gt_anomaly_starts.size == 0:
            print('No ground truth anomalies found in the dataset.')
            return accuracy, precision, recall, f_score

        # Find detected anomaly starts (where both gt and pred transition)
        detected_anomaly_starts = np.where((gt[:-1] == 0) & (gt[1:] == 1) & (pred[:-1] == 0) & (pred[1:] == 1))[0] + 1

        print(f"Ground truth anomaly starts: {len(gt_anomaly_starts)}")
        print(f"Detected anomaly starts: {len(detected_anomaly_starts)}")

        if detected_anomaly_starts.size == 0:
            print('No anomalies were correctly detected by the model.')
        else:
            print("Detected anomaly starts at indices:", ", ".join(map(str, detected_anomaly_starts)))
            
            # Point adjustment: for each detected anomaly, fill the entire anomaly segment
            for start in detected_anomaly_starts:
                # Fill backwards to the beginning of the anomaly
                for j in range(start - 1, -1, -1):  # Fixed: now goes to index 0
                    if gt[j] == 0:
                        break
                    elif pred[j] == 0:
                        pred[j] = 1
                
                # Fill forwards to the end of the anomaly  
                for j in range(start, len(gt)):
                    if gt[j] == 0:
                        break
                    elif pred[j] == 0:
                        pred[j] = 1

        print(f"Total number of indices: {len(gt)}")
        pred = np.array(pred)
        gt = np.array(gt)

        ####################################################################################################
        #                                          FINAL METRICS                                           #
        ####################################################################################################
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')

        #print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | Divergence: {config.divergence}(q={config.q_param}) | Epochs: {config.num_epochs} ")
        print('====================  FINAL METRICS  ===================')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))
        print("\n\n")
        print('====================  CONFUSION MATRIX  ===================')
        if self.data_path == 'UCR' or 'UCR_AUG':
            import csv
            with open('result/'+self.data_path+'.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(matrix)

        # print('====================  GT values equal 1   ===================')
        # indices = np.where(gt == 1)[0]
        # print("Indices where gt is equal to 1:", ", ".join(map(str, indices)))
        ####################################################################################################
        #                                          Data Segment Extraction                                            #
        ####################################################################################################
        # Initialize the loader
        #data_path = "your/data/path"  # Replace with your actual data path
        loader = SMDSegLoader('dataset/'+self.data_path, win_size=100, step=10)

        # Access the TS variablefffff
        TS = loader.TS
       # print("Content of  TS:", TS[:100])

        #start_idx = np.random.choice(anomaly_starts)
        start_idx = 10000 # 43050
        def extract_random_segment(data, segment_length=200, start_idx=None):
            if len(data) <= segment_length:
                return data  # Return the entire data if it's shorter than the segment length
            
            if start_idx is None:
                #start_idx = np.random.randint(0, len(test_energy) - segment_length)
                start_idx = np.random.choice(anomaly_starts)
            
            #print(f"Extracting random segment from index {start_idx} to {start_idx + segment_length}")
            return data[start_idx:start_idx + segment_length]

        # Extract random segments of lengthfgdfg 150
        segment_length = 30000
        #start_idx = np.random.randint(0, len(anomaly_starts) - segment_length)
        
        print(f"start_idx: {start_idx}")
        as_segment = extract_random_segment(test_energy, segment_length, start_idx) #Anomaly Score
        as_segment = np.array(as_segment)
        #test_energy_segment = extract_random_segment(test_energy, segment_length, start_idx)
        gt_segment = extract_random_segment(gt, segment_length, start_idx) #ground truth
        TS_segment = extract_random_segment(TS, segment_length, start_idx) #Time Series Data
        pred_segment = extract_random_segment(pred, segment_length, start_idx)
        #pred_segment = (test_energy_segment > thresh).astype(int)
        #pred_segment[gt_segment == 1] = 1  # Force predictions to match ground truth anomalies
        #thresh_segment = np.percentile(test_energy_segment, 100 - self.anormly_ratio)
        

        print('as segment shape', as_segment.shape)
        print(f"Anomaly Score values\n {as_segment}")
        #gt_segment=np.array(gt_segment) 
        print('gt segment shap', gt_segment.shape)########
        #print(f"gt values\n {gt_segment}")
        print(f"gt values\n\033[94m{gt_segment}\033[0m")
        #max_value_rounded = math.ceil(max(test_energy_segment))
                
        # import matplotlib.pyplot as plt 
        # plt.figure(figsize=(10, 6))
        # plt.plot(as_segment, label='AS Data')
        # plt.title('Simple Plot of AS Array')
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig('AAA_plot.png')

        ####################################################################################################
        #                                          Mat PLOT                                                 #
        ####################################################################################################
        
        import seaborn as sns
        import matplotlib.pyplot as plt
        import statistics

        def smooth(y, box_pts=5):
            box = np.ones(box_pts)/box_pts
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth
        
        # plt.style.use(['science', 'ieee'])
        # plt.rcParams["text.usetex"] = False
        # plt.rcParams['figure.figsize'] = 6, 2
        
        
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
        plt.plot(smooth(TS_segment), label="Time Series Data", color='black', linewidth=0.2)
        plt.title("Time Series Plot")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax1 = plt.gca()  # Get the current axes (first subplot)
        ax1.tick_params(axis='both', direction='in')  # Set tick direction for both x and y axes
        ymin, ymax = plt.ylim()
        #plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
        plt.plot(smooth(as_segment), label='Anomaly Scores', color='blue', linewidth=0.2)
        plt.axhline(y=thresh, color='red', linestyle='--', label='Threshold', linewidth=0.2)
        plt.fill_between(range(len(as_segment)), 0, 1, where=(gt_segment == 1), color='blue', alpha=0.3, label='Ground Truth') # plt.ylim()[1]
        plt.xlabel('Time')
        plt.ylabel('Anomaly Score')
        plt.title(f'Anomaly Scores Over Time (Area{start_idx})')
        plt.legend()
        # Adjust tick direction for the secgtond subplot
        ax2 = plt.gca()  # Get the current axes (second subplot)
        ax2.tick_params(axis='both', direction='in')  # Set tick direction for both x and y axes
        #plt.ylim([ymin, ymax])
        # Save the plot to a file
        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the combined plot to a file
        timestamp = datetime.now().strftime('%d%m%y%H%M')
        plot_filename = f'NEWcombined_plot_idx_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {plot_filename}")
        #plt.show()
        return accuracy, precision, recall, f_score

        #########################################   3333333322222222222222
        ####################################################################################################
        #                                          SEABORNE PLOT                                           #
        ####################################################################################################
        
        # span = 10  # Adjust the span as needed
        # smoothed_TS_segment = pd.Series(TS_segment).ewm(span=span, adjust=False).mean()


        # def plot_with_seaborn(TS_segment, 
        #                     test_energy_segment, 
        #                     gt_segment, 
        #                     thresh, 
        #                     start_idx):
        #     # Optional: Set a Seaborn theme/style
        #     sns.set_theme(style="whitegrid")
            
        #     # Create figure and subplots
        #     fig, axes = plt.subplots(nrows=2, 
        #                             ncols=1, 
        #                             figsize=(6, 8), 
        #                             sharex=False)
            
        #     # -------------------------
        #     # First subplot: Time Series
        #     # -------------------------
        #     sns.lineplot(
        #         x=range(len(TS_segment)), 
        #         y=TS_segment,
        #         ax=axes[0],
        #         color='black',
        #         label='Time Series Data'
        #     )
        #     axes[0].set_title("Time Series Plot")
        #     axes[0].set_xlabel("Time")
        #     axes[0].set_ylabel("Value")
        #     axes[0].legend()
            
        #     # Adjust tick direction
        #     axes[0].tick_params(axis='both', direction='in')
            
        #     # Store y-limits from the first subplot
        #     ymin, ymax = axes[0].get_ylim()
            
        #     # --------------------------
        #     # Second subplot: Anomaly Scores
        #     # --------------------------
        #     sns.lineplot(
        #         x=range(len(test_energy_segment)),
        #         y=test_energy_segment,
        #         ax=axes[1],
        #         color='black',
        #         label='Anomaly Scores'
        #     )
            
        #     # Threshold line
        #     axes[1].axhline(y=thresh, color='red', linestyle='--', label='Threshold')
            
        #     # Fill region where ground truth is 1
        #     axes[1].fill_between(
        #         x=range(len(test_energy_segment)), 
        #         y1=ymin, 
        #         y2=axes[1].get_ylim()[1],
        #         where=(gt_segment == 1),
        #         color='green', 
        #         alpha=0.2, 
        #         label='Ground Truth'
        #     )
            
        #     axes[1].set_xlabel("Time")
        #     axes[1].set_ylabel("Anomaly Score")
        #     axes[1].set_title(f"Anomaly Scores Over Time (Area {start_idx})")
        #     axes[1].legend()
        #     axes[1].tick_params(axis='both', direction='in')
            
        #     # Tight layout to prevent overlap
        #     fig.tight_layout()

        #     # Save the combined plot to a file
        #     plot_filename = f'SEABORNcombined_plot_idx_{start_idx}.png'
        #     plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        #     print(f"Combined plot saved to {plot_filename}")

        #     # If you really want two separate saves for some reason
        #     plot_filename = f'SEABORNanomaly_scores_idx_{start_idx}.png'
        #     plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        #     print(f"Plot saved to {plot_filename}")

        #     # Close the figure if running in a script or notebook to manage resources
        #     plt.close(fig)



        # plot_with_seaborn(TS_segment, 
        #                 test_energy_segment, 
        #                 gt_segment, 
        #                 thresh, 
        #                 start_idx)
        # start_idx = np.random.choice(anomaly_starts)
        # test_energy_segment = extract_random_segment(attens_energy, segment_length, start_idx)
        # gt_segment = extract_random_segment(gt, segment_length, start_idx)
        # TS_segment = extract_random_segment(TS, segment_length, start_idx)

        


        # # OLD Plot for the random segment
        # def extract_random_segment(data, segment_length=200):
        #     if len(data) <= segment_length:
        #         return data  # Return the entire data if it's shorter than the segment length
        #     start_idx = np.random.randint(0, len(data) - segment_length)
        #     start_idx = 618200
        #     print(f"Extracting random segment from index {start_idx} to {start_idx + segment_length}")
        #     return data[start_idx:start_idx + segment_length]

        # # Extract random segments of lengthfgdfg 150
        # segment_length = 200
        # start_idx = 618200
        # #start_idx = np.random.randint(0, len(test_energy) - segment_length)

        # test_energy_segment = extract_random_segment(test_energy, segment_length, start_idx)
        
        # #thresh_segment = np.percentile(test_energy_segment, 100 - self.anormly_ratio)
        # gt_segment = extract_random_segment(gt, segment_length, start_idx)
        # #pred_segment = (test_energy_segment > thresh).astype(int)
        # #pred_segment[gt_segment == 1] = 1  # Force predictions to match ground truth anomalies

        # #pred_segment=np.array(pred_segment)
        # #gt_segment=np.array(gt_segment)


        # # Plot the random segment
        # plt.figure(figsize=(12, 6))
        # plt.plot(test_energy_segment, label='Anomaly Scores', color='blue')
        # plt.axhline(y=thresh, color='red', linestyle='--', label='Threshold')
        # plt.fill_between(range(len(test_energy_segment)), 0, 1, where=(gt_segment == 1), color='yellow', alpha=0.3, label='Ground Truth Anomalies')
        # plt.xlabel('Time')
        # plt.ylabel('Anomaly Score')
        # plt.title(f'Anomaly Scores Over Time (Random Segment of Length {segment_length})')
        # plt.legend()

        # # Save the plot to a file
        # plot_filename = f'AAAAAAAAAAAAAanomaly_idx{start_idx}.png'
        # # plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        # # print(f"Plot saved to {plot_filename}")
        # # plt.show()
        # return accuracy, precision, recall, f_score