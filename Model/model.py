import numpy as np
import faiss
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from Evaluator import evaluator
from Evaluator.evaluator import evaluate2, evaluate_continuous, evaluate3
import Mi
from Mi.club import CLUBForCategorical
import itertools
from sklearn.manifold import TSNE


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(device)


def inference(net, test_dataloader):
    net.eval()
    feature_vec, label_vec, sensitive_vec, pred_vec = [], [], [], []
    with torch.no_grad():
        for (x, s, s_n, y, idx) in test_dataloader:
            # for iter, (x, s, s_n, y, idx) in enumerate(test_dataloader):
            x = x.to(device)
            # s = s.to(device)
            h = net.encoder(x)
            c = net.clustering(h).detach()
            pred = torch.argmax(c, dim=1)
            feature_vec.extend(h.detach().cpu().numpy())
            label_vec.extend(y.cpu().numpy())
            sensitive_vec.extend(s.cpu().numpy())
            pred_vec.extend(pred.cpu().numpy())

    feature_vec, label_vec, sensitive_vec, pred_vec = (
        np.array(feature_vec),
        np.array(label_vec),
        np.array(sensitive_vec),
        np.array(pred_vec),
    )
    # print(pred_vec[:50])
    d = net.representation_dim
    kmeans = faiss.Clustering(d, net.class_num)
    kmeans.verbose = False
    kmeans.niter = 300
    kmeans.nredo = 10
    kmeans.max_points_per_centroid = 1000
    kmeans.min_points_per_centroid = 10
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = 0
    index = faiss.GpuIndexFlatL2(res, d, cfg)
    # print(feature_vec.shape)
    kmeans.train(feature_vec, index)
    centroids = faiss.vector_to_array(kmeans.centroids).reshape(net.class_num, d)
    net.train()
    # print(centroids.shape)
    return feature_vec, label_vec, sensitive_vec, pred_vec, centroids



##Convolutional encoder
class EncoderConv(nn.Module):
    def __init__(self, encoder_out_dim, representation_dim):
        super(EncoderConv, self).__init__()
        self.encoder_out_dim = encoder_out_dim
        self.representation_dim = representation_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.encoder_linear = nn.Sequential(
            nn.Linear(self.encoder_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.representation_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    #     def reparameterize(self, mu, logvar):
    #         if self.training:
    #             std = logvar.mul(0.5).exp_()
    #             eps = std.data.new(std.size()).normal_()
    #             return eps.mul(std).add_(mu)
    #         else:
    #             return mu

    def forward(self, x):
        hh = self.encoder(x)
        a = self.encoder_linear(hh)
        ## Normalize
        a = F.normalize(a, dim=1)
        return a

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]


##Convolutional decoder
class DecoderConv(nn.Module):
    def __init__(self, encoder_out_dim, representation_dim, is_groupwise_decoder, is_groupwise_decoder_linear,
                 sensitive_attr_num):
        super(DecoderConv, self).__init__()
        self.encoder_out_dim = encoder_out_dim
        self.representation_dim = representation_dim
        self.is_groupwise_decoder = is_groupwise_decoder
        self.sensitive_attr_num = sensitive_attr_num
        self.is_groupwise_decoder_linear = is_groupwise_decoder_linear

        def get_decoder_linear():
            return nn.Sequential(
                nn.Linear(self.representation_dim, 256), nn.ReLU(), nn.Linear(256, self.encoder_out_dim), nn.ReLU(),
            )

        self.decoder_linear_list = nn.ModuleList([
            get_decoder_linear() for _ in range(self.sensitive_attr_num if self.is_groupwise_decoder_linear else 1)
        ])

        def get_decoder():
            return nn.Sequential(
                nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, output_padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
            )

        self.decoder_list = nn.ModuleList([
            get_decoder() for _ in range(self.sensitive_attr_num if self.is_groupwise_decoder else 1)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, z, s_indices):
        if self.is_groupwise_decoder_linear:
            z_ = torch.zeros((len(z), self.representation_dim), device='cuda')
            for s in torch.unique(s_indices):
                ind = s == s_indices
                z_[ind] = self.decoder_linear_list[s](z[ind])
        else:
            z_ = self.decoder_linear_list[0](z)

        z_ = z_.view(-1, 16, 7, 7)
        if self.is_groupwise_decoder:
            x_ = torch.zeros((len(z_), 1, 28, 28), device='cuda')
            for s in torch.unique(s_indices):
                ind = s == s_indices
                x_[ind] = self.decoder_list[s](z_[ind])

        else:
            ## The dim of z needs to be determined
            x_ = self.decoder_list[0](z_)

        return x_

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]

    ##Fully Connected encoder


class EncoderFC(nn.Module):
    def __init__(self, input_dim, encoder_out_dim, representation_dim, encoder_type, encoder_linear_type):
        super(EncoderFC, self).__init__()
        self.input_dim = input_dim
        self.encoder_out_dim = encoder_out_dim
        self.representation_dim = representation_dim
        self.encoder_type = encoder_type
        self.encoder_linear_type = encoder_linear_type
        if self.encoder_type == 1:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, self.encoder_out_dim),
                nn.BatchNorm1d(self.encoder_out_dim),
                nn.ReLU(),
            )
        else:

            # self.encoder = nn.Sequential(
            #     nn.Linear(self.input_dim, 1024),
            #     nn.ReLU(),
            #     nn.Linear(1024, 1024),
            #     nn.ReLU(),
            #     nn.Linear(1024, self.encoder_out_dim),
            #     nn.ReLU(),
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, self.encoder_out_dim),
                nn.BatchNorm1d(self.encoder_out_dim),
                nn.ReLU(),

            )

        if self.encoder_linear_type == 1:
            self.encoder_linear = nn.Sequential(
                nn.Linear(self.encoder_out_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, self.representation_dim),
            )
            # self.encoder_linear = nn.Sequential(
            #     nn.Linear(self.encoder_out_dim, 256),
            #     nn.ReLU(),
            #     nn.Linear(256, self.representation_dim),
            # )
        else:

            self.encoder_linear = nn.Sequential(
                nn.Linear(self.encoder_out_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.representation_dim),
            )

    def forward(self, x):
        hh = self.encoder(x)
        a = self.encoder_linear(hh)
        ## Normalize
        a = F.normalize(a, dim=1)
        return a

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]


## Decoder
class DecoderFC(nn.Module):
    def __init__(self, input_dim, encoder_out_dim, representation_dim, decoder_type, decoder_linear_type,
                 is_groupwise_decoder, is_groupwise_decoder_linear, sensitive_attr_num):
        super(DecoderFC, self).__init__()
        self.input_dim = input_dim
        self.encoder_out_dim = encoder_out_dim
        self.representation_dim = representation_dim
        self.decoder_type = decoder_type
        self.decoder_linear_type = decoder_linear_type
        self.is_groupwise_decoder = is_groupwise_decoder
        self.is_groupwise_decoder_linear = is_groupwise_decoder_linear
        self.sensitive_attr_num = sensitive_attr_num

        def get_decoder_linear():

            if decoder_linear_type == 1:
                return nn.Sequential(
                    nn.Linear(self.representation_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, self.encoder_out_dim),
                    nn.BatchNorm1d(self.encoder_out_dim),
                    nn.ReLU(),
                )
                # return nn.Sequential(
                #     nn.Linear(self.representation_dim, 256),
                #     nn.ReLU(),
                #     nn.Linear(256, self.encoder_out_dim),
                #     nn.ReLU(),
                # )
            else:
                return nn.Sequential(
                    nn.Linear(self.representation_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.encoder_out_dim),
                    nn.ReLU(),
                )

        def get_decoder():
            if decoder_type == 1:
                return nn.Sequential(
                    nn.Linear(self.encoder_out_dim, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Linear(1024, input_dim),
                    ## The function can be changed.
                    nn.Tanh(),
                )
            else:
                # return nn.Sequential(
                #     nn.Linear(self.encoder_out_dim, 1024),
                #     nn.ReLU(),
                #     nn.Linear(1024, 1024),
                #     nn.ReLU(),
                #     nn.Linear(1024, input_dim),
                #     ## The function can be changed.
                #     nn.Tanh(),
                # )
                return nn.Sequential(
                    nn.Linear(self.encoder_out_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, input_dim),
                    ## The function can be changed.
                    nn.Tanh(),
                )

        self.decoder_list = nn.ModuleList([
            get_decoder() for _ in range(self.sensitive_attr_num if self.is_groupwise_decoder else 1)
        ])

        self.decoder_linear_list = nn.ModuleList([
            get_decoder_linear() for _ in range(self.sensitive_attr_num if self.is_groupwise_decoder_linear else 1)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, z, s_indices):
        ## The dim of z needs to be determined

        if self.is_groupwise_decoder_linear:
            z_ = torch.zeros((len(z), self.encoder_out_dim), device='cuda')
            for s in torch.unique(s_indices):
                ind = s == s_indices
                z_[ind] = self.decoder_linear_list[s](z[ind])
        else:
            z_ = self.decoder_linear_list[0](z)

        if self.is_groupwise_decoder:
            x_ = torch.zeros((len(z_), self.input_dim), device='cuda')
            for s in torch.unique(s_indices):
                ind = s == s_indices
                x_[ind] = self.decoder_list[s](z_[ind])
        else:
            x_ = self.decoder_list[0](z_)
        return x_

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]





class UDFCcon(nn.Module):
    def __init__(self, args):
        super(UDFCcon, self).__init__()
        self.infomin_hyperparams = args.infomin_hyperparams
        self.class_num = args.class_num
        self.input_dim = args.input_dim
        self.global_infomin_update = args.global_infomin_update
        self.local_infomin_update = args.local_infomin_update

        if args.representation_dim > 0:
            self.representation_dim = args.representation_dim
        else:
            self.representation_dim = args.class_num
        self.decoder_type = args.decoder_type
        self.decoder_linear_type = args.decoder_linear_type
        self.encoder_type = args.encoder_type
        self.encoder_linear_type = args.encoder_linear_type
        self.sensitive_type = args.sensitive_type
        self.is_groupwise_decoder_linear = args.is_groupwise_decoder_linear
        self.is_groupwise_decoder = args.is_groupwise_decoder
        if self.sensitive_type == "Discrete":
            self.sensitive_attr_num = args.sensitive_attr_num
        else:
            self.sensitive_attr_dim = args.sensitive_attr_dim
            self.sensitive_attr_num = args.sensitive_attr_num

        self.AE_type = args.AE_type
        self.encoder_out_dim = args.encoder_out_dim
        self.args = args

        self.cluster_centers = F.normalize(torch.rand(self.class_num, self.representation_dim), dim=1).cuda()

        if self.AE_type == "Conv":
            self.encoder = EncoderConv(self.encoder_out_dim, self.representation_dim)
            self.decoder = DecoderConv(self.encoder_out_dim, self.representation_dim, self.is_groupwise_decoder,
                                       self.is_groupwise_decoder_linear, self.sensitive_attr_num)

        else:
            self.encoder = EncoderFC(self.input_dim, self.encoder_out_dim, self.representation_dim, self.encoder_type,
                                     self.encoder_linear_type)
            self.decoder = DecoderFC(self.input_dim, self.encoder_out_dim, self.representation_dim, self.decoder_type,
                                     self.decoder_linear_type, self.is_groupwise_decoder,
                                     self.is_groupwise_decoder_linear, self.sensitive_attr_num)

        self.infomin_layer = self.init_infomin_layer(self.infomin_hyperparams)

    def clustering(self, h):
        c = h @ self.cluster_centers.T
        return c

    def update_cluster_center(self, center):
        center = torch.from_numpy(center).cuda()
        center = F.normalize(center, dim=1)
        self.cluster_centers = center


    def init_infomin_layer(self, hyperparms):
        if hyperparms.estimator == 'CLUB':
            return Mi.ClubInfominLayer(hyperparms.dim_learnt, hyperparms.dim_sensitive, hyperparms.infomin_hidden_size, hyperparams=hyperparms)

        if hyperparms.estimator == 'SLICE':
            return Mi.SliceInfominLayer([hyperparms.dim_learnt, hyperparms.n_slice, hyperparms.dim_sensitive],
                                        hyperparms)
        if hyperparms.estimator == 'RENYI':
            return Mi.RenyiInfominLayer([hyperparms.dim_learnt, 128, hyperparms.dim_sensitive], hyperparms)
        if hyperparms.estimator == 'TC':
            return Mi.TCInfominLayer(hyperparms.dim_learnt, hyperparms.dim_sensitive, hyperparams=hyperparms)
        if hyperparms.estimator == 'DC':
            return Mi.DCInfominLayer(hyperparams=hyperparms)
        if hyperparms.estimator == 'PEARSON':
            return Mi.PearsonInfominLayer(hyperparams=hyperparms)
        return infomin_layer

    def train_infomin_layer(self, h, s):

        h, s = h.clone().detach(), s.clone().detach()
        return self.infomin_layer.learn(h, s)

    def run(self, train_dataloader, test_dataloader, dataset_all):

        ## setting optimizer
        optimizer_net = torch.optim.Adam(
            itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters(),

            ),
            lr=self.args.LearnRate,
            weight_decay=self.args.WeightDecay,
            betas=(self.args.betas_a, self.args.betas_v)

        )
        mse_loss = nn.MSELoss().cuda()

        for epoch in range(self.args.train_epoch):
            ## setting learning rate
            if self.args.LearnRateDecayType == 'None':
                lr = self.args.LearnRate
            elif self.args.LearnRateDecayType == 'Exp':
                lr = self.args.LearnRate * ((1 + 10 * (epoch + 1 - self.args.LearnRateWarm) / (
                        self.args.train_epoch - self.args.LearnRateWarm)) ** -0.75)
            elif self.args.LearnRateDecayType == 'Cosine':
                lr = self.args.LearnRate * 0.5 * (1. + math.cos(
                    math.pi * (epoch + 1 - self.args.LearnRateWarm) / (
                            self.args.train_epoch - self.args.LearnRateWarm)))
            else:
                raise NotImplementedError('args.LearnRateDecayType')
            if lr != self.args.LearnRate:
                def adjust_learning_rate(optimizer):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                adjust_learning_rate(optimizer_net)

            ## inference
            feature_vec, type_vec, group_vec, pred_vec, centers = inference(
                self,  test_dataloader
            )

            ## Update center
            if epoch % 1 == 0:
                self.update_cluster_center(centers)

            if self.sensitive_type == "Discrete":
                pred_adjusted = evaluate2(feature_vec, pred_vec, type_vec, group_vec)
            else:
                pred_adjusted = evaluate_continuous(pred_vec, type_vec, group_vec)

            if epoch == 3:
                evaluator.BestBalance = 0.0
                evaluator.BestEntropy = 0.0
                evaluator.BestFairness = 0.0
                evaluator.BestNmiFair = 0.0

            type_vec = torch.from_numpy(type_vec)
            group_vec = torch.from_numpy(group_vec)


            loss_reconstruction_epoch = 0.0
            loss_balance_epoch = 0.0
            loss_fair_epoch = 0.0
            loss_compact_epoch = 0.0
            loss_epoch = 0

            self.train()


            if self.global_infomin_update :

                idx = torch.randperm(len(dataset_all[1]))
                x_for_infomin_all = dataset_all[0][idx]
                x_for_infomin_all = x_for_infomin_all.cuda()
                s_for_infomin_all = dataset_all[2][idx]
                s_for_infomin_all = s_for_infomin_all.cuda()
                h_for_infomin_all = self.encoder(x_for_infomin_all).cuda()
                info_min_loss = self.train_infomin_layer(h_for_infomin_all, s_for_infomin_all)

                # print("info_min_loss: ", info_min_loss)

            ## Start batch training
            for iter, (x, s, s_n, y, idx) in enumerate(train_dataloader):

                if self.local_infomin_update:
                    idx = torch.randperm(len(dataset_all[1]))[0:1000]
                    x_for_infomin_all = dataset_all[0][idx]
                    x_for_infomin_all = x_for_infomin_all.cuda()
                    s_for_infomin_all = dataset_all[2][idx]
                    s_for_infomin_all = s_for_infomin_all.cuda()
                    h_for_infomin_all = self.encoder(x_for_infomin_all).cuda()
                    info_min_loss = self.train_infomin_layer(h_for_infomin_all, s_for_infomin_all)


                self.train()
                optimizer_net.zero_grad()

                x = x.cuda()
                s = s.cuda()
                s_n = torch.unsqueeze(s_n, dim=1)
                s_n = s_n.to(torch.float32).cuda()
                h = self.encoder(x).cuda()
                x_ = self.decoder(h, s).cuda()
                c = self.clustering(h)

                ##calculate loss
                loss = 0
                ## reconstruction loss
                loss_rec = mse_loss(x, x_)
                loss += loss_rec
                loss_reconstruction_epoch += loss_rec.item()

                ## In warmup period
                if epoch > self.args.WarmAll:
                    ## softmax
                    c_balance = F.softmax(c / self.args.SoftAssignmentTemperatureBalance, dim=1)
                    ## Balance loss
                    ck = torch.sum(c_balance, dim=0, keepdim=False) / torch.sum(c_balance)
                    loss_balance = torch.sum(ck * torch.log(ck))
                    loss += loss_balance * self.args.WeightLossBalance
                    loss_balance_epoch += loss_balance.item()

                    ## Compact loss
                    c_compact = F.softmax(c / self.args.SoftAssignmentTemperatureCompact, dim=1)
                    loss_compact = -torch.sum(c_compact * torch.log(c_compact + 1e-8)) / float(len(c_compact))
                    loss += loss_compact * self.args.WeightLossCompact
                    loss_compact_epoch += loss_compact.item()


                    loss_fair = self.infomin_layer.objective_func(h, s_n).squeeze()
                    loss += loss_fair * self.args.WeightLossFair
                    loss_fair_epoch += loss_fair.item()

                    loss_epoch += loss.item()

                loss.backward()
                optimizer_net.step()
                # print("loss: ", loss.item())

            len_train_dataloader = len(train_dataloader)
            loss_reconstruction_epoch /= len_train_dataloader
            loss_balance_epoch /= len_train_dataloader
            loss_fair_epoch /= len_train_dataloader
            loss_compact_epoch /= len_train_dataloader
            loss_epoch /= len_train_dataloader

            print('Epoch [{: 3d}/{: 3d}]'.format(epoch + 1, self.args.train_epoch), end='')
            if loss_reconstruction_epoch != 0:
                print(', Reconstruction:{:04f}'.format(loss_reconstruction_epoch), end='')
            if loss_balance_epoch != 0:
                print(', InfoBalance:{:04f}'.format(loss_balance_epoch), end='')
            if loss_fair_epoch != 0:
                print(', InfoFair:{:04f}'.format(loss_fair_epoch), end='')
            if loss_compact_epoch != 0:
                print(', InfoCompact:{:04f}'.format(loss_compact_epoch), end='')
            if loss_epoch != 0:
                print(', loss_epoch:{:04f}'.format(loss_epoch), end='')

            print()

            if epoch >500:
            # if epoch % 20 == 0:

                self.eval()
                idx = torch.randperm(len(dataset_all[1]))
                x_for_infomin_all = dataset_all[0][idx]
                x_for_infomin_all = x_for_infomin_all.cuda()
                s_for_infomin_all = dataset_all[2][idx]
                s_for_infomin_all = s_for_infomin_all.cuda()
                h_for_infomin_all = self.encoder(x_for_infomin_all)
                h_for_infomin_all, s_for_infomin_all = h_for_infomin_all.clone().detach(), s_for_infomin_all.clone().detach()

                renyi_net = Mi.RenyiInfominLayer([self.infomin_hyperparams.dim_learnt, 128, self.infomin_hyperparams.dim_sensitive], self.infomin_hyperparams)
                renyi_net.max_iteration = 1000
                renyi_net.debug = False
                renyi_net.to(device)
                loss = renyi_net.learn(h_for_infomin_all, s_for_infomin_all)
                print("*****************************I(H,G)*****************************")
                print('[val] rho*(Z;T) =', -loss)


                # renyi_net = Mi.RenyiInfominLayer([1, 128, self.infomin_hyperparams.dim_sensitive], self.infomin_hyperparams)
                # cluster_prop = self.clustering(h_for_infomin_all).detach()
                # Y_label = torch.argmax(cluster_prop, dim=1).unsqueeze(dim=1).to(torch.float32).detach()
                # renyi_net.max_iteration = 1000
                # renyi_net.debug = False
                # renyi_net.to(device)
                # loss = renyi_net.learn(Y_label, s_for_infomin_all)
                # print("*****************************I(Y;H)*****************************")
                # print('[val] rho*(Y;H) =', -loss)

                self.train()



            #Save results and Model
            if (epoch + 1) == self.args.train_epoch:
                ## Model
                dct = {'epoch': epoch, 'state_dict': self.state_dict(),
                       'optimizer': {'optimizer_net': optimizer_net.state_dict()},
                       }
                dct = {**dct, 'self_dic': self.__dict__}
                save_model_dir = './Save/Save_model/'+ 'Ours_' + self.args.dataset + '-res-epoch{:03d}'.format(epoch)
                print('Save check point into {}'.format(save_model_dir))
                torch.save(dct, save_model_dir)


        self.eval()
        idx = torch.randperm(len(dataset_all[1]))
        x_for_infomin_all = dataset_all[0][idx]
        x_for_infomin_all = x_for_infomin_all.cuda()
        s_for_infomin_all = dataset_all[2][idx]
        s_for_infomin_all = s_for_infomin_all.cuda()
        h_for_infomin_all = self.encoder(x_for_infomin_all)
        h_for_infomin_all, s_for_infomin_all = h_for_infomin_all.clone().detach(), s_for_infomin_all.clone().detach()

        # renyi_net = Mi.RenyiInfominLayer(
        #     [self.infomin_hyperparams.dim_learnt, 128, self.infomin_hyperparams.dim_sensitive],
        #     self.infomin_hyperparams)
        # renyi_net.max_iteration = 1000
        # renyi_net.debug = False
        # renyi_net.to(device)
        # loss = renyi_net.learn(h_for_infomin_all, s_for_infomin_all)
        # print("*****************************I(H,G)*****************************")
        # print('[val] rho*(Z;T) =', -loss)

        cluster_prop = self.clustering(h_for_infomin_all).detach()
        Y_label = torch.argmax(cluster_prop, dim=1).unsqueeze(dim=1).to(torch.float32).detach()
        renyi_net = Mi.RenyiInfominLayer([1, 128, self.infomin_hyperparams.dim_sensitive], self.infomin_hyperparams)
        renyi_net.max_iteration = 1000
        renyi_net.debug = False
        renyi_net.to(device)
        loss = renyi_net.learn(Y_label, s_for_infomin_all)
        print("*****************************I(Y;H)*****************************")
        print('[val] rho*(Y;H) =', -loss)




class UDFCdis(nn.Module):
    def __init__(self, args):
        super(UDFCdis, self).__init__()
        self.class_num = args.class_num
        self.input_dim = args.input_dim
        if args.representation_dim > 0:
            self.representation_dim = args.representation_dim
        else:
            self.representation_dim = args.class_num
        self.decoder_type = args.decoder_type
        self.decoder_linear_type = args.decoder_linear_type
        self.encoder_type = args.encoder_type
        self.encoder_linear_type = args.encoder_linear_type
        self.sensitive_type = args.sensitive_type
        self.is_groupwise_decoder_linear = args.is_groupwise_decoder_linear
        self.is_groupwise_decoder = args.is_groupwise_decoder
        self.CLUB_hidden_size = args.CLUB_hidden_size

        self.sensitive_attr_num = args.sensitive_attr_num
        self.MI_estimator = CLUBForCategorical(self.representation_dim, self.sensitive_attr_num,
                                               self.CLUB_hidden_size)


        self.AE_type = args.AE_type
        self.encoder_out_dim = args.encoder_out_dim
        self.args = args

        self.cluster_centers = F.normalize(torch.rand(self.class_num, self.representation_dim), dim=1).cuda()

        if self.AE_type == "Conv":
            self.encoder = EncoderConv(self.encoder_out_dim, self.representation_dim)
            self.decoder = DecoderConv(self.encoder_out_dim, self.representation_dim, self.is_groupwise_decoder,
                                       self.is_groupwise_decoder_linear, self.sensitive_attr_num)



        else:
            self.encoder = EncoderFC(self.input_dim, self.encoder_out_dim, self.representation_dim, self.encoder_type,
                                     self.encoder_linear_type)
            self.decoder = DecoderFC(self.input_dim, self.encoder_out_dim, self.representation_dim, self.decoder_type,
                                     self.decoder_linear_type, self.is_groupwise_decoder,
                                     self.is_groupwise_decoder_linear, self.sensitive_attr_num)

    def clustering(self, h):
        c = h @ self.cluster_centers.T
        return c

    def update_cluster_center(self, center):
        center = torch.from_numpy(center).cuda()
        center = F.normalize(center, dim=1)
        self.cluster_centers = center

    def run(self, train_dataloader, test_dataloader, dataset_all, visual_flag = False):

        ## setting optimizer
        optimizer_net = torch.optim.Adam(
            itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters(),

            ),
            lr=self.args.LearnRate,
            weight_decay=self.args.WeightDecay,
            betas=(self.args.betas_a, self.args.betas_v)

        )
        optimizer_mi = torch.optim.Adam(
            self.MI_estimator.parameters(),
            lr=0.0002,

        )
        mse_loss = nn.MSELoss().cuda()



        fea_all = []
        sensitive_group_all = []
        label_all = []

        for epoch in range(self.args.train_epoch):
            ## setting learning rate
            if self.args.LearnRateDecayType == 'None':
                lr = self.args.LearnRate
            elif self.args.LearnRateDecayType == 'Exp':
                lr = self.args.LearnRate * ((1 + 10 * (epoch + 1 - self.args.LearnRateWarm) / (
                        self.args.train_epoch - self.args.LearnRateWarm)) ** -0.75)
            elif self.args.LearnRateDecayType == 'Cosine':
                lr = self.args.LearnRate * 0.5 * (1. + math.cos(
                    math.pi * (epoch + 1 - self.args.LearnRateWarm) / (
                            self.args.train_epoch - self.args.LearnRateWarm)))
            else:
                raise NotImplementedError('args.LearnRateDecayType')
            if lr != self.args.LearnRate:
                def adjust_learning_rate(optimizer):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                adjust_learning_rate(optimizer_net)
                adjust_learning_rate(optimizer_mi)

            ## inference
            feature_vec, type_vec, group_vec, pred_vec, centers = inference(
                self, test_dataloader
            )

            if visual_flag:
                if epoch == 0 or epoch == 25 or epoch == 50:
                # if epoch == 0 or epoch == 1 or epoch == 2:
                    fea_all.append(feature_vec)
                    sensitive_group_all.append(group_vec)
                    label_all.append(type_vec)


            ## Update center
            if epoch % 1 == 0:
                self.update_cluster_center(centers)


            Acc, Nmi, Balance, NmiFair, Fmeasure = evaluate3(feature_vec, pred_vec, type_vec, group_vec)


            if epoch == 3:
                evaluator.BestBalance = 0.0
                evaluator.BestEntropy = 0.0
                evaluator.BestFairness = 0.0
                evaluator.BestNmiFair = 0.0

            # type_vec = torch.from_numpy(type_vec)
            # group_vec = torch.from_numpy(group_vec)
            # pred_vec = torch.from_numpy(pred_vec).cuda()

            confidence_sum = 0.0
            loss_reconstruction_epoch = 0.0
            loss_balance_epoch = 0.0
            loss_fair_epoch = 0.0
            loss_compact_epoch = 0.0

            self.encoder.train()
            self.decoder.train()
            ## Start batch training





            for iter, (x, s, s_n, y, idx) in enumerate(train_dataloader):

                self.MI_estimator.train()
                loss_train_mi = 0

                for j in range(self.args.MI_epochs):
                    idx = torch.randperm(len(dataset_all[1]))[0:1024]
                    x_for_infomin_all = dataset_all[0][idx]
                    x_for_infomin_all = x_for_infomin_all.cuda()
                    s_for_infomin_all = dataset_all[1][idx]
                    s_for_infomin_all = s_for_infomin_all.cuda()
                    h_for_infomin_all = self.encoder(x_for_infomin_all).cuda()
                    s_for_infomin_all = s_for_infomin_all.squeeze()

                    mi_loss = self.MI_estimator.learning_loss(h_for_infomin_all, s_for_infomin_all).cuda()
                    loss_train_mi += mi_loss.item()

                    optimizer_mi.zero_grad()
                    mi_loss.backward()
                    optimizer_mi.step()

                # loss_train_mi /= self.args.MI_epochs
                # print("loss_train_mi: ", loss_train_mi)

                self.MI_estimator.eval()

                x = x.cuda()
                s = s.cuda()

                h = self.encoder(x).cuda()
                x_ = self.decoder(h, s).cuda()
                c = self.clustering(h)

                ##calculate loss
                loss = 0
                confidence_sum += F.softmax(c / 0.2, dim=1).detach().max(dim=1).values.mean()

                ## reconstruction loss
                loss_rec = mse_loss(x, x_)
                loss += loss_rec
                loss_reconstruction_epoch += loss_rec.item()

                ## In warmup period
                if epoch > self.args.WarmAll:
                    ## softmax
                    c_balance = F.softmax(c / self.args.SoftAssignmentTemperatureBalance, dim=1)
                    ## Balance loss
                    ck = torch.sum(c_balance, dim=0, keepdim=False) / torch.sum(c_balance)
                    loss_balance = torch.sum(ck * torch.log(ck))
                    loss += loss_balance * self.args.WeightLossBalance
                    loss_balance_epoch += loss_balance.item()

                    ## Compact loss
                    c_compact = F.softmax(c / self.args.SoftAssignmentTemperatureCompact, dim=1)
                    loss_compact = -torch.sum(c_compact * torch.log(c_compact + 1e-8)) / float(len(c_compact))
                    loss += loss_compact * self.args.WeightLossCompact
                    loss_compact_epoch += loss_compact.item()

                    # Fair loss: to be continued
                    loss_fair = self.MI_estimator(h, s)
                    loss += loss_fair * self.args.WeightLossFair
                    loss_fair_epoch += loss_fair.item()

                optimizer_net.zero_grad()
                loss.backward()
                optimizer_net.step()
                # print("loss: ", loss.item())

            len_train_dataloader = len(train_dataloader)
            confidence_sum /= len_train_dataloader
            loss_reconstruction_epoch /= len_train_dataloader
            loss_balance_epoch /= len_train_dataloader
            loss_fair_epoch /= len_train_dataloader
            loss_compact_epoch /= len_train_dataloader

            print('Epoch [{: 3d}/{: 3d}]'.format(epoch + 1, self.args.train_epoch), end='')
            if loss_reconstruction_epoch != 0:
                print(', Reconstruction:{:04f}'.format(loss_reconstruction_epoch), end='')
            if loss_balance_epoch != 0:
                print(', InfoBalance:{:04f}'.format(loss_balance_epoch), end='')
            if loss_fair_epoch != 0:
                print(', InfoFair:{:04f}'.format(loss_fair_epoch), end='')
            if loss_compact_epoch != 0:
                print(', InfoCompact:{:04f}'.format(loss_compact_epoch), end='')
            if confidence_sum != 0:
                print(', Confidence:{:04f}'.format(confidence_sum), end='')

            print()



            #Save results and Model
            # if (epoch + 1) == self.args.train_epoch:
            if epoch  == 92:
                ## Model
                dct = {'epoch': epoch, 'state_dict': self.state_dict(),
                       'optimizer': {'optimizer_net': optimizer_net.state_dict()},
                       }
                dct = {**dct, 'self_dic': self.__dict__}
                save_model_dir = './Save/Save_model/'+ 'Ours_' + self.args.dataset + '-res-epoch{:03d}'.format(epoch)
                print('Save check point into {}'.format(save_model_dir))
                torch.save(dct, save_model_dir)


        if visual_flag:
            return fea_all, sensitive_group_all, label_all

        else:
            return Acc, Nmi, Balance, NmiFair, Fmeasure



class FCMI(nn.Module):
    def __init__(self, args):
        super(FCMI, self).__init__()
        self.infomin_hyperparams = args.infomin_hyperparams
        self.class_num = args.class_num
        self.input_dim = args.input_dim
        if args.representation_dim > 0:
            self.representation_dim = args.representation_dim
        else:
            self.representation_dim = args.class_num
        self.decoder_type = args.decoder_type
        self.decoder_linear_type = args.decoder_linear_type
        self.encoder_type = args.encoder_type
        self.encoder_linear_type = args.encoder_linear_type
        self.sensitive_type = args.sensitive_type
        self.is_groupwise_decoder_linear = args.is_groupwise_decoder_linear
        self.CLUB_hidden_size = 16
        if self.sensitive_type == "Discrete":
            self.sensitive_attr_num = args.sensitive_attr_num
        else:
            self.sensitive_attr_dim = args.sensitive_attr_dim
            self.sensitive_attr_num = args.sensitive_attr_num

        self.is_groupwise_decoder = args.is_groupwise_decoder
        self.AE_type = args.AE_type
        self.encoder_out_dim = args.encoder_out_dim
        self.args = args

        self.cluster_centers = F.normalize(torch.rand(self.class_num, self.representation_dim), dim=1).cuda()

        if self.AE_type == "Conv":
            self.encoder = EncoderConv(self.encoder_out_dim, self.representation_dim)
            self.decoder = DecoderConv(self.encoder_out_dim, self.representation_dim, self.is_groupwise_decoder,
                                       self.is_groupwise_decoder_linear, self.sensitive_attr_num)


        else:
            self.encoder = EncoderFC(self.input_dim, self.encoder_out_dim, self.representation_dim, self.encoder_type,
                                     self.encoder_linear_type)
            self.decoder = DecoderFC(self.input_dim, self.encoder_out_dim, self.representation_dim, self.decoder_type,
                                     self.decoder_linear_type, self.is_groupwise_decoder,
                                     self.is_groupwise_decoder_linear, self.sensitive_attr_num)







    def clustering(self, h):
        c = h @ self.cluster_centers.T
        return c

    def update_cluster_center(self, center):
        center = torch.from_numpy(center).cuda()
        center = F.normalize(center, dim=1)
        self.cluster_centers = center

    def run(self, train_dataloader, test_dataloader, dataset_all):

        ## setting optimizer
        optimizer_net = torch.optim.Adam(
            itertools.chain(
                self.encoder.parameters(),
                #                 self.encoder_linear.parameters(),
                #                 self.decoder_linear.parameters(),
                self.decoder.parameters(),

            ),
            lr=self.args.LearnRate,
            betas=(self.args.betas_a, self.args.betas_v),
            weight_decay=self.args.WeightDecay
        )
        mse_loss = nn.MSELoss().cuda()

        for epoch in range(self.args.train_epoch):
            print("Epoch: ", epoch)

            ## setting learning rate
            if self.args.LearnRateDecayType == 'None':
                lr = self.args.LearnRate
            elif self.args.LearnRateDecayType == 'Exp':
                lr = self.args.LearnRate * ((1 + 10 * (epoch + 1 - self.args.LearnRateWarm) / (
                        self.args.train_epoch - self.args.LearnRateWarm)) ** -0.75)
            elif self.args.LearnRateDecayType == 'Cosine':
                lr = self.args.LearnRate * 0.5 * (1. + math.cos(
                    math.pi * (epoch + 1 - self.args.LearnRateWarm) / (
                                self.args.train_epoch - self.args.LearnRateWarm)))
            else:
                raise NotImplementedError('args.LearnRateDecayType')
            if lr != self.args.LearnRate:
                def adjust_learning_rate(optimizer):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                adjust_learning_rate(optimizer_net)

            ## inference
            feature_vec, type_vec, group_vec, pred_vec, centers = inference(
                self, test_dataloader
            )

            ## Update center
            if epoch % 1 == 0:
                self.update_cluster_center(centers)

            pred_adjusted = evaluate2(feature_vec, pred_vec, type_vec, group_vec)

            if epoch == 3:
                evaluator.BestBalance = 0.0
                evaluator.BestEntropy = 0.0
                evaluator.BestFairness = 0.0
                evaluator.BestNmiFair = 0.0

            type_vec = torch.from_numpy(type_vec)
            group_vec = torch.from_numpy(group_vec)
            pred_vec = torch.from_numpy(pred_vec).cuda()
            self.train()

            confidence_sum = 0.0
            loss_reconstruction_epoch = 0.0
            loss_balance_epoch = 0.0
            loss_fair_epoch = 0.0
            loss_compact_epoch = 0.0

            ## Start batch training
            for iter, (x, s, s_n, y, idx) in enumerate(train_dataloader):

                x = x.cuda()
                s = s.cuda()
                h = self.encoder(x).cuda()
                x_ = self.decoder(h, s).cuda()
                c = self.clustering(h)

                self.train()

                ##calculate loss
                confidence_sum += F.softmax(c / 0.2, dim=1).detach().max(dim=1).values.mean()
                loss = 0
                ## reconstruction loss
                loss_rec = mse_loss(x, x_)
                loss += loss_rec
                loss_reconstruction_epoch += loss_rec.item()

                ## In warmup period
                if epoch > self.args.WarmAll:

                    ## softmax
                    c_balance = F.softmax(c / self.args.SoftAssignmentTemperatureBalance, dim=1)
                    # print("c_balance:", c_balance.shape)
                    ## Balance loss
                    O = torch.zeros((self.class_num, self.sensitive_attr_num)).cuda()
                    # E = torch.zeros((self.class_num, self.sensitive_attr_num)).cuda()
                    for b in range(self.sensitive_attr_num):
                        O[:, b] = torch.sum(c_balance[s == b], dim=0)
                        # E[:, b] = (s == b).sum()
                    # E[E <= 0] = torch.min(E[E > 0]) / 10
                    O[O <= 0] = torch.min(O[O > 0]) / 1

                    pcg = O / torch.sum(O)

                    pc = torch.sum(pcg, dim=1, keepdim=False)
                    loss_balance = torch.sum(pc * torch.log(pc))
                    loss += loss_balance * self.args.WeightLossBalance
                    loss_balance_epoch += loss_balance.item()

                    # Fair loss:
                    pc = torch.sum(pcg, dim=1, keepdim=True)
                    pg = torch.sum(pcg, dim=0, keepdim=True)
                    loss_fair = torch.sum(pcg * torch.log(pcg / (pc * pg)))
                    loss += loss_fair * self.args.WeightLossFair
                    loss_fair_epoch += loss_fair.item()

                    ## Compact loss
                    c_compact = F.softmax(c / self.args.SoftAssignmentTemperatureCompact, dim=1)
                    loss_compact = -torch.sum(c_compact * torch.log(c_compact + 1e-8)) / float(len(c_compact))
                    loss += loss_compact * self.args.WeightLossCompact
                    loss_compact_epoch += loss_compact.item()

                optimizer_net.zero_grad()
                loss.backward()
                optimizer_net.step()

            len_train_dataloader = len(train_dataloader)
            confidence_sum /= len_train_dataloader
            loss_reconstruction_epoch /= len_train_dataloader
            loss_balance_epoch /= len_train_dataloader
            loss_fair_epoch /= len_train_dataloader
            loss_compact_epoch /= len_train_dataloader

            print('Epoch [{: 3d}/{: 3d}]'.format(epoch + 1, self.args.train_epoch), end='')
            if loss_reconstruction_epoch != 0:
                print(', Reconstruction:{:04f}'.format(loss_reconstruction_epoch), end='')
            if loss_balance_epoch != 0:
                print(', InfoBalance:{:04f}'.format(loss_balance_epoch), end='')
            if loss_fair_epoch != 0:
                print(', InfoFair:{:04f}'.format(loss_fair_epoch), end='')
            if loss_compact_epoch != 0:
                print(', InfoCompact:{:04f}'.format(loss_compact_epoch), end='')
            if confidence_sum != 0:
                print(', Confidence:{:04f}'.format(confidence_sum), end='')

            print()
            if epoch%20 == 0:


                self.eval()
                idx = torch.randperm(len(dataset_all[1]))
                x_for_infomin_all = dataset_all[0][idx]
                x_for_infomin_all = x_for_infomin_all.cuda()
                s_for_infomin_all = dataset_all[2][idx]
                s_for_infomin_all = s_for_infomin_all.cuda()
                h_for_infomin_all = self.encoder(x_for_infomin_all)
                h_for_infomin_all, s_for_infomin_all = h_for_infomin_all.clone().detach(), s_for_infomin_all.clone().detach()

                renyi_net = Mi.RenyiInfominLayer([self.infomin_hyperparams.dim_learnt, 128, 1], self.infomin_hyperparams)
                renyi_net.max_iteration = 1000
                renyi_net.debug = False
                renyi_net.to(device)
                loss = renyi_net.learn(h_for_infomin_all, s_for_infomin_all)
                print("*****************************I(H,G)*****************************")
                print('[val] rho*(Z;T) =', -loss)



            #Save results and Model
            # if (epoch + 1) == self.args.train_epoch:
            #     ## Model
            #     dct = {'epoch': epoch, 'state_dict': self.state_dict(),
            #            'optimizer': {'optimizer_net': optimizer_net.state_dict()},
            #            }
            #     dct = {**dct, 'self_dic': self.__dict__}
            #     save_model_dir = './Save/Save_model/' + "AE_" + self.args.dataset + '-res-epoch{:03d}'.format(epoch)
            #     print('Save check point into {}'.format(save_model_dir))
            #     torch.save(dct, save_model_dir)

            self.train()

        self.eval()
        idx = torch.randperm(len(dataset_all[1]))
        x_for_infomin_all = dataset_all[0][idx]
        x_for_infomin_all = x_for_infomin_all.cuda()
        s_for_infomin_all = dataset_all[2][idx]
        s_for_infomin_all = s_for_infomin_all.cuda()
        h_for_infomin_all = self.encoder(x_for_infomin_all)
        h_for_infomin_all, s_for_infomin_all = h_for_infomin_all.clone().detach(), s_for_infomin_all.clone().detach()

        cluster_prop = self.clustering(h_for_infomin_all).detach()
        Y_label = torch.argmax(cluster_prop, dim=1).unsqueeze(dim=1).to(torch.float32).detach()
        renyi_net = Mi.RenyiInfominLayer([1, 128, 1], self.infomin_hyperparams)
        renyi_net.max_iteration = 1000
        renyi_net.debug = False
        renyi_net.to(device)
        loss = renyi_net.learn(Y_label, s_for_infomin_all)
        print("*****************************I(Y;H)*****************************")
        print('[val] rho*(Y;H) =', -loss)





