import argparse
import numpy as np
import torch
from Model.model import  UDFCcon, UDFCdis, FCMI
import os
from Generator import generator
from Utils import utils_os, utils
import random
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

DATASET = "Census"   # MNISTUSPS ReverseMNIST Office HAR MTFL Census Crime
CLASS_NUM = 2
SENSITIVE_NUM = 1
SENSITIVE_DIM = 1
REPRESENTATION_NUM = 4
INPUT_DIM = 32
SENSITIVE_TYPE = "Continuous"
RANDOM_SEED = 4096
INFOMIN_HIDDEN_SIZE = 16
AE_TYPE = 'FC'


if SENSITIVE_TYPE == "Discrete":
    ENCODER_TYPE = 1
    DECODER_TYPE = 1
    ENCODER_LINEAR_TYPE = 0
    DECODER_LINEAR_TYPE = 0
else:
    ENCODER_TYPE = 0
    DECODER_TYPE = 0
    ENCODER_LINEAR_TYPE = 1
    DECODER_LINEAR_TYPE = 1



class InfoMinLayerHyperparams(utils_os.ConfigDict):
    def __init__(self):
        self.estimator = 'CLUB'

        # self.inner_lr = 1e-3 #5e-4
        # self.inner_batch_size = 1000

        self.n_slice = 120
        self.inner_epochs = 50                   # <-- 0 means we don't optimise the slices, but you can also do so
        self.infomin_hidden_size = INFOMIN_HIDDEN_SIZE
        self.dim_learnt = REPRESENTATION_NUM
        self.dim_sensitive = SENSITIVE_DIM
        self.num_sensitive = SENSITIVE_NUM
        self.sensitive_type = SENSITIVE_TYPE
infoMinHyperPara = InfoMinLayerHyperparams()



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=DATASET)


parser.add_argument("--WeightLossBalance", help="", default=0.04, type=float)
parser.add_argument("--WeightLossCompact", help="", default= 0.04, type=float)
parser.add_argument("--WeightLossFair", help="", default=0.18, type=float)



parser.add_argument("--global_infomin_update", help="", default=False)
parser.add_argument("--local_infomin_update", help="", default=True)
parser.add_argument("--is_groupwise_decoder", help="", default=False)


parser.add_argument("--SoftAssignmentTemperatureBalance", help="", default=0.10, type=float)
parser.add_argument("--SoftAssignmentTemperatureCompact", help="", default=0.20, type=float)
parser.add_argument("--SoftAssignmentTemperatureFair", help="", default=0.10, type=float)


parser.add_argument("--sensitive_attr_dim", help="", default=SENSITIVE_DIM, type=float)
parser.add_argument("--sensitive_attr_num", help="", default=SENSITIVE_NUM, type=float)
parser.add_argument("--seed", help="random seed", default=RANDOM_SEED, type=int)

# parser.add_argument("--input_dim", default= 561, type=int)
parser.add_argument("--input_dim", default= INPUT_DIM, type=int)
parser.add_argument("--AE_type", default= AE_TYPE)
parser.add_argument("--sensitive_type", default= SENSITIVE_TYPE)
parser.add_argument("--encoder_out_dim", default= 784)

parser.add_argument("--MI_epochs", default=50, type=int)
parser.add_argument("--CLUB_hidden_size", default=INFOMIN_HIDDEN_SIZE, type=int)



parser.add_argument("--batch_size", help="batch size", default=512, type=int)
parser.add_argument("--train_epoch", help="training epochs", default=300, type=int)
parser.add_argument("--WarmAll", default=20,  type=int)
parser.add_argument("--is_groupwise_decoder_linear", help="", default=False)

parser.add_argument("--LearnRate", help="", default=0.0002, type=float)
parser.add_argument("--LearnRateDecayType", default='None')  # Exp, Cosine
parser.add_argument("--WeightDecay", default=0, type=float)
parser.add_argument("--LearnRateWarm", default=0, type=int)
parser.add_argument("--betas_a", help="", default=0.9, type=float)
parser.add_argument("--betas_v", help="", default=0.999, type=float)
parser.add_argument("--resume", default='')
parser.add_argument("--class_num", default= CLASS_NUM, type=int)
parser.add_argument("--representation_dim", default= REPRESENTATION_NUM, type=int)
parser.add_argument("--decoder_type", default= DECODER_TYPE, type=int)
parser.add_argument("--decoder_linear_type", default= DECODER_LINEAR_TYPE, type=int)
parser.add_argument("--encoder_type", default= ENCODER_TYPE, type=int)
parser.add_argument("--encoder_linear_type", default= ENCODER_LINEAR_TYPE, type=int)
parser.add_argument("--MnistTrain", default=1, type=int)
parser.add_argument('--FeatureType',default='GlT_GaussainlizeAndTanh')
parser.add_argument('--infomin_hyperparams', default=infoMinHyperPara)



args = parser.parse_args()
print('=======Arguments=======')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))


UnDeterministic = False
if not UnDeterministic:
    Torch171 = False
    if Torch171:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.set_deterministic(True)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
else:
    warnings.warn('Not deterministic')


## GPU seeting
device=torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.set_device(device)


## rand seeding
utils.set_seed(args.seed)





##import dataset
train_loader, test_loader, class_num = generator.get_dataloader_continuous(
            dataset=args.dataset, dateset_mode = 1,  batch_size=args.batch_size, path=None, args=args)

dataset_all = generator.get_dataloader_continuous(dataset = args.dataset,  dateset_mode = 2, batch_size=args.batch_size, path=None, args=args)

print("dataset_all: ", dataset_all[0].shape)

net = UDFCcon(args).cuda()
print("Model structure: ", net)
print("*********************************Learn**************************************")
net.run(train_dataloader=train_loader, test_dataloader=test_loader, dataset_all = dataset_all)



