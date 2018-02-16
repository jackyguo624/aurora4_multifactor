import torch
import torch.nn as nn
import argparse
from ..tnt import torchnet as tnt
from tqdm import tqdm
from ..utils import kaldi_io
from ..utils.kaldi_dataloader import KaldiStreamDataloader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .multifactor_dataloader import MultifactorStreamDataloader
import os
import logging
from .DNN import DNN
import numpy as np

torch.manual_seed(7)
torch.cuda.manual_seed_all(7)

def save_checkpoint(state,is_best, savedir):
    filename = os.path.join(savedir, 'checkpoint.th')
    torch.save({'state_dict': state['model'].state_dict(),
                'epoch': state['epoch']}, filename)
    if is_best:
        torch.save(state['model'].state_dict(),
                   os.path.join(savedir, 'model_best.param'))

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-0.05, 0.05)


def load_mean_var(fpath):
    mv = np.loadtxt(fpath)
    return torch.FloatTensor(mv[0]), torch.FloatTensor(mv[1])


def get_logger():
    logging.basicConfig(level=logging.INFO,
                        format="[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger("train")
    return logger


def parse_arg():
    parser = argparse.ArgumentParser()

    # data & labbel 
    parser.add_argument("traindata", type=str)
    parser.add_argument("train_asr_label", type=lambda x:{
        k: v for k, v in kaldi_io.read_ali_ark(x)
    })
    parser.add_argument("cvdata", type=str)
    parser.add_argument("cv_asr_label", type=lambda x:{
        k: v for k, v in kaldi_io.read_ali_ark(x)
    })

    # countsfile, help init iterator
    parser.add_argument("traincounts", type=argparse.FileType("r"))
    parser.add_argument("cvcounts", type=argparse.FileType("r"))


    # factor label
    parser.add_argument("--spk_label", type=lambda x:{
        k: v for k, v in kaldi_io.read_ali_ark(x)
    }, required=True)
    parser.add_argument("--phone_label", type=lambda x:{
        k: v for k, v in kaldi_io.read_ali_ark(x)
    }, required=True)
    parser.add_argument("--fbank64_label", type=str, required=True)



    # inputdim & outputdim, help to init network
    parser.add_argument("--inputdim", type=int, required=True)
    parser.add_argument("--outputdim", type=int, required=True)
    parser.add_argument("--spkdim", type=int, required=True)
    parser.add_argument("--phndim", type=int, required=True)
    parser.add_argument("--fbankdim", type=int, required=True)

    # help to normalize
    parser.add_argument("--meanvar", type=argparse.FileType("r"))

    # about training
    parser.add_argument("--mode", type=int, required=True)
    parser.add_argument("--nocuda", action="store_true", default=False)
    parser.add_argument("--nonorm", action="store_true", default=False)
    parser.add_argument("-ep", "--epochs", default=50, type=int)
    parser.add_argument("-lr", "--learningrate", default=1e-2, type=float)
    parser.add_argument("-bs", "--batchsize", default=256, type=int)

    # output model and checkpoint
    parser.add_argument("-o", "--output", type=str, required=True)

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    return args


def main():

    args = parse_arg()
    logger = get_logger()
    traindataloader = MultifactorStreamDataloader(
        args.traindata, args.train_asr_label, args.spk_label, args.phone_label, 
        args.fbank64_label, countsfile=args.traincounts,
        num_outputs=args.outputdim, batchsize=args.batchsize, 
        cachesize=200, mode=args.mode, shuffle=True)

    epochsize = int(traindataloader.nsamples * 1.0 / args.batchsize)

    cvdataloader = KaldiStreamDataloader(
        args.cvdata, args.cv_asr_label, countsfile=args.cvcounts, num_outputs=args.outputdim,
        batchsize=args.batchsize, cachesize=200, shuffle=False)

    logger.info("train_nsample:"+str(traindataloader.nsamples))
    logger.info("cv_nsample:"+str(cvdataloader.nsamples))

    model = DNN(args.inputdim, args.outputdim, args.spkdim, args.phndim, args.fbankdim,
                mode=args.mode)
    # devide_ids = [0, 1]
    # model = torch.nn.DataParallel(model, device_ids=devide_ids)
    print("CUDA_VISIBLE_DEVICES", str(os.environ['CUDA_VISIBLE_DEVICES']))
    if not args.nocuda:
        model = model.cuda()
        model.apply(init_weights)
    print(model)
    tr_mean, tr_var = load_mean_var(args.meanvar)

    epoch = 0
    epochsize = int(traindataloader.nsamples * 1.0 / args.batchsize)
    cvsize = int(cvdataloader.nsamples * 1. // args.batchsize)
    engine = tnt.engine.Engine()
    ce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learningrate, weight_decay=1e-8)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=1e-8)
    sched = ReduceLROnPlateau(optimizer,mode="min", factor=0.5, patience=0)

    # Statistics
    time_meter = tnt.meter.TimeMeter(False)
    meter_acc = tnt.meter.AverageValueMeter()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_loss_f0 = tnt.meter.AverageValueMeter()
    meter_loss_f1 = tnt.meter.AverageValueMeter()
    meter_loss_f2 = tnt.meter.AverageValueMeter()
    meter_loss_f3 = tnt.meter.AverageValueMeter()

    def lossfunction(sample):
        x, y, y_spk, y_phone, y_clean = sample
        if not args.nonorm:
            x = torch.add(x, -tr_mean)
            x = torch.div(x, tr_var)
        if not args.nocuda:
            x, y = x.cuda(), y.cuda()
            y_spk, y_phone, y_clean = y_spk.cuda(), y_phone.cuda(), y_clean.cuda()
        x, y = Variable(x), Variable(y)
        y_spk, y_phone = Variable(y_spk), Variable(y_phone) 
        y_clean = Variable(y_clean)
        outputs, asr_outs, phn_outs, spk_outs, fbk_outs = model(x)
        
        asr_loss = ce(outputs, y)
        loss = asr_loss
        phn_w = 0.1
        spk_w = 0.1
        fbk_w = 0.01

        meter_loss_f0.add(asr_loss.data[0])
        if phn_outs is not None:
            phn_loss = ce(phn_outs, y_phone) 
            meter_loss_f1.add(phn_loss.data[0])
            loss += phn_w * phn_loss

        if spk_outs is not None:
            spk_loss = ce(spk_outs, y_spk) 
            meter_loss_f2.add(spk_loss.data[0])
            loss += spk_w * spk_loss

        if fbk_outs is not None:
            fbk_loss = mse(fbk_outs,y_clean) 
            meter_loss_f3.add(fbk_loss.data[0])
            loss += fbk_w * fbk_loss

        return loss, outputs

    def evalfunction(sample):
        x, y = sample
        if not args.nonorm:
            x = torch.add(x, -tr_mean)
            x = torch.div(x, tr_var)
        if not args.nocuda:
            x, y = x.cuda(), y.cuda()
        x_var, y_var = Variable(x, volatile=True), Variable(y, volatile=True)
        outputs = model(x_var)
        outputs = outputs[0]
        loss = ce(outputs,y_var)
        _, predicted = torch.max(outputs.data, 1)
        acc = (predicted == y).sum() * 1. / len(y)
        return {'loss': loss, 'acc': acc}, outputs

    def reset_meters():
        meter_acc.reset()
        meter_loss.reset()
        meter_loss_f0.reset()
        meter_loss_f1.reset()
        meter_loss_f2.reset()
        meter_loss_f3.reset()


    def on_start(state):
        state['best_acc'] = state[
            'best_acc'] if 'best_acc' in state else 0.

    def on_end_epoch(state):
        # Output trainmessge
        trainmessage = 'Training Epoch {:>3d}: Time: {:=6.1f}s/{:=4.1f}m Loss: {:=.4f} LR: {:=3.1e}'.format(
            state['epoch'], time_meter.value(), time_meter.value()/60, meter_loss.value()[0],optimizer.param_groups[0]['lr'])
        print(trainmessage)
        loss_message='f0_loss:{:<10.4f} f1_loss:{:<10.4f} f2_loss:{:<10.4f} f3_loss:{:<10.4f}'.format(meter_loss_f0.value()[0],
                meter_loss_f1.value()[0], meter_loss_f2.value()[0], meter_loss_f3.value()[0])
        print(loss_message)
        # evaluate procedure
        reset_meters()
        model.eval()
        engine.hooks['on_forward'] = on_forward_test
        engine.test(evalfunction, tqdm(cvdataloader, total=cvsize))
        engine.hooks['on_forward'] = on_forward
        acc = meter_acc.value()[0]
        loss = meter_loss.value()[0]
        evalmessage = 'CV Epoch {:>3d}: Time: {:=.2f}s/{:=.2f}m, acc:{:=.4f}, loss:{:=.8f}'.format(
            state['epoch'], time_meter.value(), time_meter.value()/60, acc, loss)
        print(evalmessage)
        sched.step(loss)

        # save model
        isbest = acc > state['best_acc']
        state['best_acc'] = acc if isbest else state['best_acc']
        save_checkpoint({
            'model': model,
            'epoch': state['epoch']
        }, isbest, args.output
        )

        # Quit training if learning rate is below 1e-9
        if optimizer.param_groups[0]['lr'] < 1e-9:
            state['epoch'] = 1e30
            print('lr<1e-9 stop training')
            return

    def on_start_epoch(state):
        model.train()
        reset_meters()
        state['iterator'] = tqdm(
            state['iterator'], total=epochsize, unit="batch", leave=False)

    def on_forward(state):
        meter_loss.add(state['loss'].data[0])

    def on_forward_test(state):
        meter_acc.add(state['loss']['acc'])
        meter_loss.add(state['loss']['loss'].data[0])

    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(lossfunction, traindataloader,
                 maxepoch=args.epochs, optimizer=optimizer, epoch=epoch)


if __name__ == '__main__':
    main()
