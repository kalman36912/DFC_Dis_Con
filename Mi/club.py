import torch
import torch.nn as nn
import torch.nn.functional as F

from . import base


class ClubInfominLayer(base.ParametricInfoEstimator):
    """ sub-network used in infomin, trained by SGD """

    def __init__(self, dim_z, dim_y, hidden_size=16, hyperparams={}):
        super().__init__(hyperparams=hyperparams)

        self.club = CLUB(dim_z, dim_y, hidden_size)
        self.mode = 'eval'

    def forward(self, x, y):
        return self.club(x, y)

    def objective_func(self, x, y):
        if self.mode == 'learn':  # <-- max step, log p(y|x)
            return -self.club.learning_loss(x, y)
        if self.mode == 'eval':  # <-- min step, use bound
            return self.club(x, y)

    def learn(self, x, y):
        self.to(x.device)
        self.mode = 'learn'
        ret = super().learn(x, y)
        self.mode = 'eval'
        return ret




class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    """This class provides the CLUB estimation to I(X,Y)
    Method:
        forward() :      provides the estimation with input samples
        loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
    Arguments:
        x_dim, y_dim :         the dimensions of samples from X, Y respectively
        hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
        x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim]
    """
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # p_mu outputs mean of q(Y|X)
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 2, y_dim)
        )

        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh()
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = -(mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = -((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def log_likelihood(self, x_samples, y_samples): # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return -self.log_likelihood(x_samples, y_samples)

class CLUBForCategorical(nn.Module):  # Update 04/27/2022
    '''
    This class provide a CLUB estimator to calculate MI upper bound between vector-like embeddings and categorical labels.
    Estimate I(X,Y), where X is continuous vector and Y is discrete label.
    '''

    def __init__(self, input_dim, label_num, hidden_size=None):
        '''
        input_dim : the dimension of input embeddings
        label_num : the number of categorical labels
        '''
        super().__init__()

        if hidden_size is None:
            self.variational_net = nn.Linear(input_dim, label_num)
        else:
            self.variational_net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                # nn.Linear(hidden_size, hidden_size),
                # nn.ReLU(),
                nn.Linear(hidden_size, label_num)
            )

    def forward(self, inputs, labels):
        '''
        inputs : shape [batch_size, input_dim], a batch of embeddings
        labels : shape [batch_size], a batch of label index
        '''
        logits = self.variational_net(inputs)  # [sample_size, label_num]
        # log of conditional probability of positive sample pairs
        # positive = - nn.functional.cross_entropy(logits, labels, reduction='none')
        sample_size, label_num = logits.shape

        logits_extend = logits.unsqueeze(1).repeat(1, sample_size, 1)  # shape [sample_size, sample_size, label_num]
        labels_extend = labels.unsqueeze(0).repeat(sample_size, 1)  # shape [sample_size, sample_size]

        # log of conditional probability of negative sample pairs
        log_mat = - nn.functional.cross_entropy(
            logits_extend.reshape(-1, label_num),
            labels_extend.reshape(-1, ),
            reduction='none'
        )

        log_mat = log_mat.reshape(sample_size, sample_size)
        positive = torch.diag(log_mat).mean()
        negative = log_mat.mean()
        return positive - negative

    def loglikeli(self, inputs, labels):
        logits = self.variational_net(inputs)

        return - nn.functional.cross_entropy(logits, labels)

    def learning_loss(self, inputs, labels):
        return - self.loglikeli(inputs, labels)

def estimate(x, y, club_net):
    club_net.mode = 'eval'
    return club_net.objective_func(x, y)