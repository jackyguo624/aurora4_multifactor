import torch.nn as nn
import torch


def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-0.05, 0.05)


def _make_layer(inputdim, outputdim):
    fc = nn.Linear(inputdim, outputdim)
    bn = nn.BatchNorm1d(outputdim)
    activator = nn.Sigmoid()
    return nn.Sequential(fc, bn, activator)



class DNN(nn.Module):
    def __init__(self, inputdim, outputdim, spkdim, phonedim, fbankdim, mode=0):
        super(DNN, self).__init__()
        self.inputdim = inputdim
        self.hidden = 2048
        self.outputdim = outputdim
        self.spkdim = spkdim
        self.phonedim = phonedim
        self.fbankdim = fbankdim
        self.mode = mode
        
        # asr dnn 
        self.asrlayer1 = _make_layer(self.inputdim, self.hidden)
        self.asrlayer2 = _make_layer(self.hidden, self.hidden)
        self.asrlayer3 = _make_layer(self.hidden, self.hidden)
        self.asrlayer4 = _make_layer(self.hidden, self.hidden)
        self.asrlayer5 = _make_layer(self.hidden, self.hidden)
        self.asrlayer6 = _make_layer(self.hidden, self.hidden)
        self.asrlayer_out = nn.Linear(self.hidden, self.outputdim)

        # spk dnn
        self.spklayer1 = _make_layer(self.inputdim+self.hidden, self.hidden)
        self.spklayer2 = _make_layer(self.hidden, self.hidden)
        self.spklayer3 = _make_layer(self.hidden, 100)
        self.spklayer4 = _make_layer(100, self.hidden)
        self.spklayer_out = nn.Linear(self.hidden,self.spkdim)
        
        # phone dnn
        self.phnlayer1 = _make_layer(self.inputdim+self.hidden, self.hidden)
        self.phnlayer2 = _make_layer(self.hidden, self.hidden)
        self.phnlayer3 = _make_layer(self.hidden, 100)
        self.phnlayer4 = _make_layer(100, self.hidden)
        self.phnlayer_out = nn.Linear(self.hidden, self.phonedim)
        
        # fbank dnn
        self.fbklayer1 = _make_layer(self.inputdim+self.hidden, self.hidden)
        self.fbklayer2 = _make_layer(self.hidden, self.hidden)
        self.fbklayer3 = _make_layer(self.hidden, 100)
        self.fbklayer4 = _make_layer(100, self.hidden)
        self.fbklayer_out = nn.Linear(self.hidden, self.fbankdim)
        
        if self.mode == 1:
            self.alllayer_out = nn.Linear(self.outputdim+self.phonedim, self.outputdim)
        if self.mode == 2:
            self.alllayer_out = nn.Linear(self.outputdim+self.spkdim, self.outputdim)
        if self.mode == 3:
            self.alllayer_out = nn.Linear(self.outputdim+self.fbankdim, self.outputdim)
        if self.mode == 4:
            self.alllayer_out = nn.Linear(self.outputdim+self.spkdim+self.phonedim
                                   +self.fbankdim, self.outputdim)
            
    def forward(self, x):
        
        xinput = x.view(-1, self.inputdim)

        # asr trunk
        x = self.asrlayer1(xinput)
        x2layer = self.asrlayer2(x)
        x = self.asrlayer3(x2layer)
        x = self.asrlayer4(x)
        x = self.asrlayer5(x)
        x = self.asrlayer6(x)
        asr_out = self.asrlayer_out(x)
        
        fctinput = torch.cat([xinput, x2layer], dim=1)
        if self.mode == 1 or self.mode == 4:
            # phone branch
            phn_x = self.phnlayer1(fctinput)
            phn_x = self.phnlayer2(phn_x)
            phn_x = self.phnlayer3(phn_x)
            phn_x = self.phnlayer4(phn_x)
            phn_out = self.phnlayer_out(phn_x)

        if self.mode == 2 or self.mode == 4:
            # spk branch
            spk_x = self.spklayer1(fctinput)
            spk_x = self.spklayer2(spk_x)
            spk_x = self.spklayer3(spk_x)
            spk_x = self.spklayer4(spk_x)
            spk_out = self.spklayer_out(spk_x)

        if self.mode == 3 or self.mode == 4:
            # fbank branch
            fbk_x = self.fbklayer1(fctinput)
            fbk_x = self.fbklayer2(fbk_x)
            fbk_x = self.fbklayer3(fbk_x)
            fbk_x = self.fbklayer4(fbk_x)
            fbk_out = self.fbklayer_out(fbk_x)
        
        if self.mode == 0:
            return asr_out, asr_out, None, None, None
        
        if self.mode == 1:
            all_x = torch.cat([asr_out, phn_out], dim=1)
            out = self.alllayer_out(all_x)
            return out, asr_out, phn_out, None, None
        if self.mode == 2:
            # merge and output
            all_x = torch.cat([asr_out, spk_out], dim=1)
            out = self.alllayer_out(all_x)
            return out, asr_out, None, spk_out, None
        if self.mode == 3:
            # merge and output
            all_x = torch.cat([asr_out, fbk_out], dim=1)
            out = self.alllayer_out(all_x)
            return out, asr_out, None, None, fbk_out
        if self.mode == 4:
            # merge and output
            all_x = torch.cat([asr_out, spk_out, phn_out, fbk_out], dim=1)
            out = self.alllayer_out(all_x)
            return out, asr_out, phn_out, spk_out,  fbk_out
