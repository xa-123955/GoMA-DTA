import torch
import torch.nn as nn
import copy
import os
from prettytable import PrettyTable
from tqdm import tqdm
from utils import ci, mse, rmse, rm2,RandomLayer,get_pearson,get_spearman,r_squared_error,get_mae
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_scheduler
import numpy as np
import wandb
class Trainer(object):
    def __init__(self, model, optim, scheduler,device, train_dataloader, val_dataloader, test_dataloader, alpha=1, **config):
        self.criterion = nn.MSELoss()
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.is_da = config["DA"]["USE"]
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]
        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"]["LAMB_DA"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.use_da_entropy = config["DA"]["USE_ENTROPY"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.best_model = None
        self.best_epoch = None
        self.best_mse = 100
        # 早停参数
        self.patience = config["SOLVER"].get("PATIENCE", 50)  # 容忍的epoch数，默认30
        self.counter = 0  # 计数器，记录验证集性能没有提升的epoch数
        self.best_val_mse = float('inf')  # 最佳验证集MSE
        self.early_stop = False  # 是否提前停止训练

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch, self.val_mse_epoch, self.val_rmse_epoch = [], [], [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        valid_metric_header = ["# Epoch", "mse", "rmse", "loss"]
        test_metric_header = ["# Best Epoch", "mse", "rmse", "ci", "rm2","pearson","spearman","r2","Mae"]
        train_metric_header = ["# Epoch", "Train_loss", "mse"]

        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.scheduler = scheduler
    def train(self):
        float2str = lambda x: '%0.4f' % x
        # checkpoint = torch.load("/home/xiong123/L_tt/result/2020shiyu/best_2020shiyu_54.pth", self.device)
        # self.best_model.load_state_dict(checkpoint)
        if os.path.exists("./save_model/trimming/checkpoint.pth"):
        # 加载模型状态
            checkpoint = torch.load("./save_model/trimming/checkpoint.pth")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # 由于保存的是已经训练好的epoch，因此从下一个epoch开始
            epoch = checkpoint['epoch'] + 1
            print(f"Continuing from epoch {epoch}")
            torch.set_rng_state(checkpoint['rng_state'])  # 恢复CPU随机状态
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])  # 恢复所有GPU的随机状态
            del checkpoint
        else:
            epoch = 0
            self.counter = 0  # 初始化计数器
            self.early_stop = False  # 初始化早停标志
        for epoch in range(epoch,self.epochs):
            if self.early_stop:
                print("Early stopping")
                break
            self.current_epoch += 1

            # 执行训练和计算损失
            train_loss, mse = self.train_epoch()

            # 记录训练损失和MSE
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss, mse]))
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            # 评估模型
            mse_val, rmse_val, val_loss = self.test(dataloader="val")
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [mse_val, rmse_val, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_mse_epoch.append(mse_val)
            print(f"Epoch {epoch+1}, LR: {self.scheduler.optimizer.param_groups[0]['lr']:.20f}")
            self.scheduler.step()
            # wandb.log({'val_x0_mse': mse_val.item()})
            # 保存模型和更新最佳模型记录
            if mse_val < self.best_mse:
                self.best_mse = mse_val
                self.best_model = copy.deepcopy(self.model)
                # torch.save(self.model.state_dict(), "./save_model/model_weights.pth")
                self.best_epoch = self.current_epoch
                print('MSE improved at epoch ', self.best_epoch, ';\tbest_mse:', self.best_mse)
                self.counter = 0 
            else:
                self.counter += 1  # 验证集性能没有提升，计数器加1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True  
            #--------------------保存-------------------
            # if (epoch + 1) % 10 == 0:
            #     torch.save(self.model.state_dict(), "./save_model/trimming/denoise_model.pkl")
            #     state = {
            #         'model_state_dict': self.model.state_dict(),
            #         'optimizer_state_dict': self.optim.state_dict(),
            #         'scheduler_state_dict': self.scheduler.state_dict(),
            #         'epoch': epoch,
            #         'rng_state': torch.get_rng_state(),  # 保存CPU随机状态
            #         'cuda_rng_state': torch.cuda.get_rng_state_all()  # 保存所有GPU的随机状态（如果有）
            #     }
            #     torch.save(state, "./save_model/trimming/checkpoint.pth")
            #-----------------------------------------------------------
        # 使用最佳模型进行最终测试
        mse, rmse, ci, rm2,pearson,spearman,r2,Mae = self.test(dataloader="test")  # 调用回归测试函数，输出回归指标
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [mse, rmse, ci, rm2,pearson,spearman,r2,Mae]))
        self.test_table.add_row(test_lst)

        # 打印最终测试结果
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(rmse) +
              " MSE " + str(mse) + " CI " + str(ci) + " RM2 " + str(rm2)+ " pearson " + str(pearson)+ 
              " spearman " + str(spearman)+ " r2 " + str(r2)+ " Mae " + str(Mae))

        # 保存测试结果
        self.test_metrics["mse"] = mse
        self.test_metrics["rmse"] = rmse
        self.test_metrics["ci"] = ci
        self.test_metrics["rm2"] = rm2
        self.test_metrics["pearson"] = pearson
        self.test_metrics["spearman"] = spearman
        self.test_metrics["r2"] = r2
        self.test_metrics["Mae"] = Mae
        self.test_metrics["best_epoch"] = self.best_epoch
        self.save_result()

        return self.test_metrics

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        if self.is_da:
            state["train_model_loss"] = self.train_model_loss_epoch
            state["train_da_loss"] = self.train_da_loss_epoch
            state["da_init_epoch"] = self.da_init_epoch
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def mean_square_error(self,v_d, v_s):
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(v_d, v_s)
        return loss
    
    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        total_preds = torch.Tensor().to(self.device)
        total_labels = torch.Tensor().to(self.device)
        num_batches = len(self.train_dataloader)
        for i, (v_d, v_d_e,v_p,v_p_2,vp_mask, labels)  in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d, v_d_e, v_p,v_p_2, vp_mask,labels = v_d, v_d_e.to(self.device), v_p.to(self.device),v_p_2.to(self.device),vp_mask.to(self.device), labels.float().to(self.device)
            self.optim.zero_grad()
            v_d,v_dg, f, score = self.model(v_d, v_d_e,v_p,v_p_2,vp_mask,mode="train")
            loss = self.criterion(score, labels.view(-1, 1).float().to(self.device))
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            total_preds = torch.cat((total_preds, score.view(-1)), 0)
            total_labels = torch.cat((total_labels, labels.view(-1)), 0)

        loss_epoch = loss_epoch / num_batches

        # 将累积的预测值和标签值转换为 NumPy 数组
        total_labels = total_labels.detach().cpu().numpy().flatten()
        total_preds = total_preds.detach().cpu().numpy().flatten()

        # 计算 MSE
        MSE = mse(total_labels, total_preds)

        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch) +
              ' and MSE ' + str(MSE))

        return loss_epoch, MSE

    def test(self, dataloader="test"):
        test_loss = 0
        total_preds = torch.Tensor().to(self.device)
        total_labels = torch.Tensor().to(self.device)
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_d_e,v_p,v_p_2, vp_mask,labels) in  enumerate(data_loader):
                v_d, v_d_e, v_p,v_p_2, vp_mask,labels = v_d, v_d_e.to(self.device), v_p.to(self.device),v_p_2.to(self.device), vp_mask.to(self.device),labels.float().to(self.device)
                if dataloader == "val":
                    v_d, v_p, f, score = self.model(v_d, v_d_e,v_p,v_p_2,vp_mask,mode="val")
                elif dataloader == "test":
                    v_d, v_p, f, score = self.best_model(v_d, v_d_e,v_p,v_p_2,vp_mask,mode="val")
                loss = self.criterion(score, labels.view(-1, 1).float().to(self.device))
                test_loss += loss.item()
                total_preds = torch.cat((total_preds, score.view(-1)), 0)
                total_labels = torch.cat((total_labels, labels.view(-1)), 0)

        total_labels = total_labels.cpu().numpy().flatten()
        total_preds = total_preds.cpu().numpy().flatten()
        #-------------------------------------------
        # save_dir = "./results"
        # os.makedirs(save_dir, exist_ok=True)
        # # np.save(os.path.join(save_dir, "16_true_labels.npy"), total_labels)
        # np.save(os.path.join(save_dir, "16_pred_labels_noCross.npy"), total_preds)
        #------------------------------------------------------
        test_loss = test_loss / num_batches
        MSE = mse(total_labels, total_preds)
        RMSE = rmse(total_labels, total_preds)
        if dataloader == "test":
            CI = ci(total_labels, total_preds)
            RM2 = rm2(total_labels, total_preds)
            pearson = get_pearson(total_labels, total_preds)
            spearman =get_spearman(total_labels, total_preds)
            r2 = r_squared_error(total_labels, total_preds)
            Mae =get_mae(total_labels, total_preds)
            return MSE, RMSE, CI, RM2,pearson,spearman,r2,Mae
        else:
            return MSE, RMSE, test_loss
