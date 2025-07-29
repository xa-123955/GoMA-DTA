import torch.nn as nn
import torch.nn.functional as F
import torch
from mamba_ssm import Mamba2
from transformers import AutoTokenizer, AutoModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from Graph_encoder import Graph_Encoder

class GoMADTA(nn.Module):
    def __init__(self, config):
        super(GoMADTA, self).__init__()
        
        mamba_dim = config["DIM"]["MAMBA"]
        l_dim = config["DIM"]["LINEAR"]
        hidden1 = config["FUSION"]["HIDDEN1"]
        hidden2 = config["FUSION"]["HIDDEN2"]
        dropout = config["FUSION"]["DROPOUT"]
        attn_dim =config["ATTN"]["DIM"]
        attn_heads = config["ATTN"]["HEADS"]
        encoder_type = config["GRAPH_ENCODER"]["ENCODER_TYPE"]
        layer_num = config["GRAPH_ENCODER"]["LAYER_NUM"]
        in_channels = config["GRAPH_ENCODER"]["IN_CHANNELS"]
        basic_channels = config["GRAPH_ENCODER"]["BASIC_CHANNELS"]
        graph_dropout = config["GRAPH_ENCODER"]["DROPOUT"]
        conv_channels = config["CONV_CHANNELS"]

        class_hidden1 = config["CLASSIFIER"]["HIDDEN1"]
        class_hidden2 = config["CLASSIFIER"]["HIDDEN1"]
        class_hidden3 = config["CLASSIFIER"]["HIDDEN1"]
        class_dropout = config["CLASSIFIER"]["DROPOUT"]

        self.Mamba1 = Mamba2(d_model=mamba_dim)
        self.Mamba2 = Mamba2(d_model=mamba_dim)
        self.Mamba3 = Mamba2(d_model=mamba_dim)
        self.Mamba4 = Mamba2(d_model=mamba_dim)

        self.chem_tokenizer = AutoTokenizer.from_pretrained("/home/xiong123/L_tt/MoLFormer-XL-both-10pct", trust_remote_code=True)
        self.chem_model = AutoModel.from_pretrained("/home/xiong123/L_tt/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True).to(device)
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for in_c, out_c, k in conv_channels :
            self.conv_layers.append(nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=k, padding='same'))
            self.bn_layers.append(nn.BatchNorm1d(out_c))

        self.fusion = Fusion2(hidden1, hidden2 ,dropout)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.mix_attention_layer2 = nn.MultiheadAttention(attn_dim, attn_heads)
        self.mix_attention_layer3 = nn.MultiheadAttention(attn_dim, attn_heads)
        
        self.go_linear = nn.Linear(l_dim, l_dim)

        self.classifier = nn.Sequential(
        nn.Linear(class_hidden1, class_hidden2),
        nn.ReLU(),
        nn.Dropout(class_dropout),
        nn.Linear(class_hidden2, class_hidden3),
        nn.ReLU(),
        nn.Dropout(class_dropout),
        nn.Linear(class_hidden3, 1),
        )
        self.graph_encoder = Graph_Encoder(encoder_type, layer_num, in_channels, basic_channels,graph_dropout)
    
    def _conv_block(self, x, conv_layer, bn_layer):
        """Helper function for convolution + batch normalization"""
        x = F.relu(conv_layer(x))
        return bn_layer(x)

    def forward(self, v_d, bg_d,v_p,v_p_2, vp_mask,mode="train"):
        chem_input = self.chem_tokenizer(v_d, padding=True, return_tensors="pt").to(device)
        v_d1_attention_mask = chem_input['attention_mask']
        v_d1_mask = (v_d1_attention_mask == 0) 
        with torch.no_grad():
            chem_outputs = self.chem_model(**chem_input)
        v_d1 = chem_outputs.last_hidden_state 

        v_d1 = v_d1.permute(0, 2, 1)
        v_d1 = self._conv_block(v_d1, self.conv_layers[0], self.bn_layers[0])
        v_d1 = self._conv_block(v_d1, self.conv_layers[1], self.bn_layers[1]) 
        v_d1 = v_d1.permute(0, 2, 1) 

        v_d2,v_d2_mask = self.graph_encoder(bg_d)

        v_p = self._conv_block(v_p.permute(0, 2, 1), self.conv_layers[2], self.bn_layers[2])
        v_p = self._conv_block(v_p, self.conv_layers[3], self.bn_layers[3])
        v_p = v_p.permute(0, 2, 1)

        v_p2 = self._conv_block(v_p_2.permute(0, 2, 1), self.conv_layers[4], self.bn_layers[4])#
        v_p2 = self._conv_block(v_p2, self.conv_layers[5], self.bn_layers[5])#
        v_p2 = v_p2.permute(0, 2, 1)

        go_pooled = v_p2.mean(dim=1, keepdim=True)        
        gate = torch.sigmoid(self.go_linear(go_pooled))    
        v_p1 = v_p * gate                                  
        v_p1 = v_p1 + v_p
        v_p_mask = (vp_mask == 0) 

        protein_att = v_p1.permute(1, 0, 2)
        drug_att = v_d1.permute(1, 0, 2)
        p1, _ = self.mix_attention_layer2(protein_att, drug_att, drug_att,key_padding_mask = v_d1_mask)
        d1, _ = self.mix_attention_layer2(drug_att, protein_att, protein_att,key_padding_mask=v_p_mask)
        vp1 = p1.permute(1, 0, 2)
        vd1 = d1.permute(1, 0, 2)
        vp1 = self.Mamba1(vp1)
        vd1 = self.Mamba2(vd1)
        x_max1 = torch.cat([vd1,vp1], dim=1)
        x_max1 = self.avg_pool(x_max1.permute(0, 2, 1)).squeeze(-1)

        protein_att2 = v_p1.permute(1, 0, 2)
        drug_att2 = v_d2.permute(1, 0, 2)
        p2, _ = self.mix_attention_layer3(protein_att2, drug_att2, drug_att2, key_padding_mask = v_d2_mask)
        d2, _ = self.mix_attention_layer3(drug_att2, protein_att2, protein_att2,key_padding_mask=v_p_mask)
        vp2 = p2.permute(1, 0, 2)
        vd2 = d2.permute(1, 0, 2)
        vp2 = self.Mamba3(vp2)
        vd2 = self.Mamba4(vd2)
        x_max2 = torch.cat([vd2,vp2], dim=1)
        x_max2= self.avg_pool(x_max2.permute(0, 2, 1)).squeeze(-1)

        f = self.fusion(x_max1,x_max2)
        score = self.classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "val":
            return v_d, v_p, f, score
    
class Fusion2(nn.Module):
    def __init__(self, hidden1, hidden2, dropout=0.1):
        super(Fusion2, self).__init__()
        self.si_L = nn.Sigmoid()
        self.si_S = nn.Sigmoid()
        self.so_f = nn.Sigmoid()
        self.combine = nn.Linear(128 * 8, 256)
        self.ln = nn.LayerNorm(256)
        self.drop = nn.Dropout(p=dropout)


    def forward(self, LM_fea, Sty_fea):

        Sty_fea_norm = Sty_fea * (abs(torch.mean(LM_fea))/abs(torch.mean(Sty_fea)))
        f_h = torch.cat((LM_fea.unsqueeze(1), Sty_fea_norm.unsqueeze(1)), dim=1)

        f_att = torch.mean(f_h, dim=1)
        f_att = self.so_f(f_att)
        fus_fea = torch.cat((LM_fea, Sty_fea, LM_fea * f_att, Sty_fea * f_att), dim=1)
        fus_fea = self.combine(fus_fea)

        return fus_fea
    