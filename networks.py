import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    '''
    This encodes the state as a discrete representation, the policy is predicted from this
    '''
    def __init__(self, w, k, c, f):
        super().__init__()

        self.final_im_shape = w - (k - 1) * 3                                                   # Image width after 3 convolutions

        self.enc_conv1 = nn.Conv2d(1, c, k)                                         # In channels, out channels, kernel size
        self.enc_conv2 = nn.Conv2d(c, c, k)
        self.enc_conv3 = nn.Conv2d(c, c, k)
        self.enc_linear = nn.Linear(c * self.final_im_shape * self.final_im_shape, f)           # Input size, output size

        self.dropout2d = nn.Dropout2d(p=0.2)                                                    # Dropout probability
    
    def forward(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = self.enc_conv3(x)
        x = torch.flatten(x, start_dim=1)                                                       # (batch, final_im_shape)
        x = self.enc_linear(x)                                                                  # (batch, f)
        return x



class Decoder(nn.Module):
    '''
    Decodes the state from the discrete representation
    '''
    def __init__(self, w, k, c, f, batch):
        super().__init__()

        self.final_im_shape = w - (k - 1) * 3                                               # Image width after 3 convolutions

        self.dec_linear = nn.Linear(f, c * self.final_im_shape * self.final_im_shape)       # Input size, output size
        self.dec_conv1 = nn.ConvTranspose2d(c, c, k)                                        # In channels, out channels, kernel size                                              # Number of channels
        self.dec_conv2 = nn.ConvTranspose2d(c, c, k)
        self.dec_conv3 = nn.ConvTranspose2d(c, 1, k)

        self.dropout2d = nn.Dropout2d(p=0.2)

        self.batch = batch
        self.c = c

    def forward(self, x):
        x = self.dec_linear(x)
        x = torch.reshape(x, (-1, self.c, self.final_im_shape, self.final_im_shape))
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = self.dec_conv3(x)
        return x



class NextState(nn.Module):
    '''
    Predicts the next state from the current state and action
    '''
    def __init__(self, f, action_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(f + action_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, f)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x



class PrevState(nn.Module):
    '''
    Predicts the previous state from the current state and action which was applied to obtain this state
    '''
    def __init__(self, f, action_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(f + action_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, f)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x



class BinaryConcrete(nn.Module):
    '''
    Activation function that discretizes
    '''
    def __init__(self, device, tau_max, tau_min, epochs):
        super().__init__()
        self.device = device
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.total_epochs = epochs
    
    def forward(self, x, epoch=-1, eps=1e-20):
        # x: (batch, f)
        if epoch == -1:
            tau = self.tau_min
        else:
            tau = self.tau_max * (self.tau_min / self.tau_max) ** (min(epoch, self.total_epochs) / self.total_epochs)
        u = torch.rand(x.shape, device=self.device)
        logistic = torch.log(u + eps) - torch.log(1 - u + eps)
        logits = (x + logistic) / tau
        binary_concrete = torch.sigmoid(logits)
        return binary_concrete
    
    @staticmethod
    def bc_loss(logit_q, logit_p=None, p=None, eps=1e-20):
        '''
        Binary Concrete loss
        logit_p for logits (network outputs before activation), p is for the Bernoulli(p) prior
        '''
        if logit_p is None and p is None:
            raise ValueError("Both logit_p and p cannot be None")
        elif p is None:
            p = torch.sigmoid(logit_p)
        q = torch.sigmoid(logit_q)
        log_q0 = torch.log(q + eps)
        log_q1 = torch.log(1 - q + eps)
        log_p0 = torch.log(p + eps)
        log_p1 = torch.log(1 - p + eps)
        loss = q * (log_q0 - log_p0) + (1 - q) * (log_q1 - log_p1)
        loss_sum = torch.sum(loss)
        return loss_sum



class VAE_Model(nn.Module):
    def __init__(self, img_width, kernel_size, channels, fluents, hidden_size, batch, device, tau_max, tau_min, epochs, **kwargs):
        super().__init__()
        self.encoder = Encoder(img_width, kernel_size, channels, fluents)
        self.decoder = Decoder(img_width, kernel_size, channels, fluents, batch)
        self.predict_next = NextState(fluents, 1, hidden_size)
        self.predict_prev = PrevState(fluents, 1, hidden_size)
        self.binary_concrete = BinaryConcrete(device, tau_max, tau_min, epochs)
    
    def forward(self, prev, succ, action, epoch):

        out = { 'prev' : prev, 'succ' : succ, 'action' : action }

        # Encoder prev and succ images
        out['prev_enc_logits'] = self.encoder(out['prev'])
        out['succ_enc_logits'] = self.encoder(out['succ'])

        # Discretize encoded images
        out['prev_enc_discrete'] = self.binary_concrete(out['prev_enc_logits'], epoch)
        out['succ_enc_discrete'] = self.binary_concrete(out['succ_enc_logits'], epoch)

        # Predict the previous and next states
        out['succ_pred_logits'] = self.predict_next(torch.cat((out['prev_enc_discrete'], out['action']), axis=1))
        out['prev_pred_logits'] = self.predict_prev(torch.cat((out['succ_enc_discrete'], out['action']), axis=1))

        # Discretize them
        out['prev_pred_discrete'] = self.binary_concrete(out['prev_pred_logits'], epoch)
        out['succ_pred_discrete'] = self.binary_concrete(out['succ_pred_logits'], epoch)

        # Decode all the states
        out['prev_enc_dec'] = self.decoder(out['prev_enc_discrete'])
        out['succ_enc_dec'] = self.decoder(out['succ_enc_discrete'])

        out['prev_pred_dec'] = self.decoder(out['prev_pred_discrete'])
        out['succ_pred_dec'] = self.decoder(out['succ_pred_discrete'])

        return out

    def encode(self, x):
        '''
        Simply encodes and discretizes a state (without a batch dimension)
        '''
        x = torch.tensor(x, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        x = self.encoder(x)
        x = self.binary_concrete(x)
        return torch.round(x)                  # Cast to int as we're not backpropagating through here



def vae_loss(out, p, beta_z=0, beta_p=0):

    z0_prior = BinaryConcrete.bc_loss(out['prev_enc_logits'], p=p)
    z1_prior = BinaryConcrete.bc_loss(out['succ_enc_logits'], p=p)
    z0_pred_prior = BinaryConcrete.bc_loss(out['prev_pred_logits'], p=p)
    z1_pred_prior = BinaryConcrete.bc_loss(out['succ_pred_logits'], p=p)

    prior_losses = z0_prior + z1_prior + z0_pred_prior + z1_pred_prior

    l0_l3 = BinaryConcrete.bc_loss(out['prev_enc_logits'], logit_p=out['prev_pred_logits'])
    l1_l2 = BinaryConcrete.bc_loss(out['succ_enc_logits'], logit_p=out['succ_pred_logits'])

    latent_difference = l0_l3 + l1_l2

    criterion = nn.MSELoss()

    prev_recon = criterion(out['prev'], out['prev_enc_dec'])
    succ_recon = criterion(out['succ'], out['succ_enc_dec'])

    prev_pred_recon = criterion(out['prev'], out['prev_pred_dec'])
    succ_pred_recon = criterion(out['succ'], out['succ_pred_dec'])

    recon_loss = prev_recon + succ_recon + prev_pred_recon + succ_pred_recon

    return recon_loss + beta_z * prior_losses + beta_p * latent_difference, recon_loss, prior_losses, latent_difference
