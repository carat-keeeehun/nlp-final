"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
from tkinter import X
import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction


def mean_except_batch(x, num_dims=1):
    return x.mean(dim=tuple(range(num_dims, len(x.shape))))


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def log_categorical(x_start, logits):
    return (F.log_softmax(logits, dim=-1) * F.one_hot(x_start, logits.shape[-1])).sum(dim=-1)


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)

    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def get_diffusion_betas(transition_mat_type, num_timesteps):
    """Get betas from the hyperparameters."""
    if transition_mat_type == 'gaussian':
        return np.linspace(1e-4, 0.02, num_timesteps)
    elif transition_mat_type == 'uniform':
        steps = np.arange(num_timesteps + 1, dtype=np.float64) / num_timesteps
        alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi * 0.5)
        betas = np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
        return betas
    elif transition_mat_type == 'absorbing':
        return 1. / np.linspace(num_timesteps, 1., num_timesteps)
    else:
        raise NotImplementedError(transition_mat_type)


class MultinomialDiffusion(torch.nn.Module):
    def __init__(self, num_classes, shape, denoise_fn, timesteps=1000,
                 loss_type='hybrid', parametrization='x0', hybrid_coeff=0.01, 
                 transition_bands=None, transition_mat_type="uniform"):
        super(MultinomialDiffusion, self).__init__()
        assert loss_type in ('kl', 'cross_entropy_x_start', 'hybrid')
        assert parametrization in ('x0', 'direct')
        assert transition_mat_type in ("uniform", "absorbing")

        self.num_classes = num_classes
        self._denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.shape = shape
        self.num_timesteps = timesteps
        self.parametrization = parametrization

        self.hybrid_coeff = hybrid_coeff
        self.loss_type = loss_type
        self.transition_bands = transition_bands
        self.transition_mat_type = transition_mat_type
        self.eps = 1e-6

        betas = get_diffusion_betas(transition_mat_type, timesteps)
        if not isinstance(betas, np.ndarray):
            raise ValueError('expected betas to be a numpy array')
        if not ((betas > 0).all() and (betas <= 1).all()):
            raise ValueError('betas must be in (0, 1]')
        self.betas = betas = betas.astype(np.float64)

        # Construct transition matrices for q(x_t|x_{t-1}). t goes from {0, ..., T-1}
        if self.transition_mat_type == "uniform":
            q_one_step_mats = [self._get_transition_mat(t) for t in range(self.num_timesteps)]
        elif self.transition_mat_type == "absorbing":
            q_one_step_mats = [self._get_absorbing_transition_mat(t) for t in range(self.num_timesteps)]
        else:
            raise ValueError(f"transition_mat_type must be 'uniform', 'absorbing', but is {self.transition_mat_type}")

        self.q_onestep_mats = torch.stack(q_one_step_mats, dim=0)
        assert self.q_onestep_mats.shape == (self.num_timesteps, self.num_classes, self.num_classes)

        # Construct transition matrices for q(x_t|x_start)
        q_mat_t = self.q_onestep_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
            q_mat_t = torch.tensordot(q_mat_t, self.q_onestep_mats[t], dims=[[1], [0]])
            q_mats.append(q_mat_t)

        self.register_buffer("q_mats", torch.stack(q_mats, dim=0))
        assert self.q_mats.shape == (self.num_timesteps, self.num_classes, self.num_classes), self.q_mats.shape
        self.register_buffer("transpose_q_onestep_mats", torch.permute(self.q_onestep_mats, dims=(0, 2, 1)))
        del self.q_onestep_mats

        alphas = cosine_beta_schedule(timesteps)

        alphas = torch.tensor(alphas.astype('float64'))
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count', torch.zeros(timesteps))

    def _get_full_transition_mat(self, t):
        beta_t = self.betas[t]
        mat = torch.full(size=(self.num_classes, self.num_classes),
                         fill_value=beta_t / float(self.num_classes))
        mat[range(self.num_classes), range(self.num_classes)] = 1. - beta_t * (self.num_classes - 1.) / self.num_classes
        return mat

    def _get_transition_mat(self, t):
        if self.transition_bands is None:
            return self._get_full_transition_mat(t)

        # Assumes num_off_diags < num_pixel_vals
        beta_t = self.betas[t]

        mat = torch.zeros((self.num_classes, self.num_classes))
        off_diag = torch.full(size=(self.num_classes - 1,),
                              fill_value=beta_t / float(self.num_classes))
        for k in range(1, self.transition_bands + 1):
            mat += torch.diag(off_diag, diagonal=k)
            mat += torch.diag(off_diag, diagonal=-k)
            off_diag = off_diag[:-1]

        # Add diagonal values such that rows sum to one.
        diag = 1. - mat.sum(1)
        mat += torch.diag(diag, diagonal=0)
        return mat

    def _get_absorbing_transition_mat(self, t):
        beta_t = self.betas[t]
        diag = torch.full(size=(self.num_classes,), fill_value=1. - beta_t)
        mat = torch.diag(diag, diagonal=0)
        # Add beta_t for the absorbing state.
        mat[:, self.num_classes - 1] += beta_t

        return mat

    def multinomial_kl(self, logits1, logits2, eps=1e-6):
        kl = ((F.softmax(logits1 + eps, dim=-1) * 
              (F.log_softmax(logits1 + eps, dim=-1) - 
              F.log_softmax(logits2 + eps, dim=-1))))
        return kl.sum(dim=-1)

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )

        return log_probs

    def _at(self, a, t, x):
        t_broadcast = t[(...,) + (None,) * len(range(1, x.ndim))]
        return a[t_broadcast, x]

    def _at_onehot(self, a, t, x):
        return torch.matmul(x, a[t, Ellipsis])

    def q_pred(self, x_start, t):
        # x_start.shape = (bs, seq_len)
        # t.shape = (bs,)
        # out.shape = (bs, seq_len, num_classes)
        return self._at(self.q_mats, t, x_start)

    def q_posterior(self, x_start, x_t, t, x_start_logits):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        if x_start_logits:
            assert x_start.shape == x_t.shape + (self.num_classes,), (x_start.shape, x_t.shape)
        else:
            assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)

        fact1 = self._at(self.transpose_q_onestep_mats, t, x_t)
        if x_start_logits:
            fact2 = self._at_onehot(self.q_mats, t - 1, F.softmax(x_start, dim=-1))
            tzero_logits = x_start
        else:
            fact2 = self._at(self.q_mats, t - 1, x_start)
            tzero_logits = torch.log(F.one_hot(x_start, self.num_classes) + self.eps)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        t_broadcast = t[(...,) + (None,) * len(range(1, out.ndim))]
        return torch.where(t_broadcast == 0, tzero_logits, out)

    def p_pred(self, x, t):
        # Compute logits of p(x_{t-1} | x_t).
        assert t.shape == (x.shape[0],)
        model_logits = self._denoise_fn(t, x)

        if self.parametrization == 'x0':
            pred_x_start_logits = model_logits
            t_broadcast = t[(...,) + (None,) * len(range(1, model_logits.ndim))]
            model_logits = torch.where(t_broadcast == 0, 
                                       pred_x_start_logits, 
                                       self.q_posterior(pred_x_start_logits, x, 
                                                        t, x_start_logits=True))
        elif self.parametrization == 'direct':
            pred_x_start_logits = model_logits
            raise NotImplementedError(self.parametrization)

        assert model_logits.shape == pred_x_start_logits.shape == x.shape + (self.num_classes,), \
            (model_logits.shape, pred_x_start_logits.shape, x.shape + (self.num_classes,))
        return model_logits, pred_x_start_logits

    @torch.no_grad()
    def p_sample(self, log_x, t):
        model_log_prob = self.p_pred(log_x=log_x, t=t)
        out = self.log_sample_categorical(model_log_prob)
        return out

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.log_alpha.device

        b = shape[0]
        # start with random normal image.
        img = torch.randn(shape, device=device)

        for i in reversed(range(1, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def _sample(self, image_size, batch_size = 16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, x_start, t):
        log_EV_qxt_x0 = torch.log(self.q_pred(x_start, t) + self.eps)

        uniform = torch.rand_like(log_EV_qxt_x0)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        return (gumbel_noise + log_EV_qxt_x0).argmax(dim=-1)

    # def kl_prior(self, log_x_start):
    #     b = log_x_start.size(0)
    #     device = log_x_start.device
    #     ones = torch.ones(b, device=device).long()

    #     log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
    #     log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))

    #     kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
    #     return sum_except_batch(kl_prior)

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)
            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def compute_Lt(self, x_start, x_t, t, detach_mean=False):
        true_logits = self.q_posterior(x_start=x_start, x_t=x_t, t=t, x_start_logits=False)
        model_logits, pred_x_start_logits = self.p_pred(x=x_t, t=t)

        if detach_mean:
            model_logits = model_logits.detach()

        kl = self.multinomial_kl(logits1=true_logits, logits2=model_logits)
        assert kl.shape == x_start.shape
        kl = mean_except_batch(kl) / np.log(2.)

        decoder_nll = -log_categorical(x_start, model_logits)
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_except_batch(decoder_nll) / np.log(2.)

        assert kl.shape == decoder_nll.shape == t.shape == (x_start.shape[0],)
        return torch.where(t == 0, decoder_nll, kl), pred_x_start_logits

    def cross_entropy_x_start(self, x_start, pred_x_start_logits):
        ce = -log_categorical(x_start, pred_x_start_logits)
        assert ce.shape == x_start.shape
        ce = mean_except_batch(ce) / np.log(2.)
        assert ce.shape == (x_start.shape[0],)
        return ce

    def _train_loss(self, x):
        b, device = x.size(0), x.device

        x_start = x
        t, pt = self.sample_time(b, device, 'uniform')

        x_t = self.q_sample(x_start=x_start, t=t)
        if self.loss_type == 'kl':
            losses, _ = self.compute_Lt(x_start, x_t, t)
        elif self.loss_type == 'cross_entropy_x_start':
            _, pred_x_start_logits = self.p_pred(x=x_t, t=t)
            losses = self.cross_entropy_x_start(x_start=x_start, pred_x_start_logits=pred_x_start_logits)
        elif self.loss_type == 'hybrid':
            vb_losses, pred_x_start_logits = self.compute_Lt(x_start, x_t, t)
            ce_losses = self.cross_entropy_x_start(x_start=x_start, pred_x_start_logits=pred_x_start_logits)
            losses = vb_losses + self.hybrid_coeff * ce_losses
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == t.shape
        return -losses

    def log_prob(self, x):
        b, device = x.size(0), x.device
        if self.training:
            return self._train_loss(x)
        else:
            t, pt = self.sample_time(b, device, 'uniform')

            x_t = self.q_sample(x_start=x, t=t)
            if self.loss_type == 'kl':
                losses, _ = self.compute_Lt(x, x_t, t)
            elif self.loss_type == 'cross_entropy_x_start':
                _, pred_x_start_logits = self.p_pred(x=x_t, t=t)
                losses = self.cross_entropy_x_start(x_start=x, pred_x_start_logits=pred_x_start_logits)
            elif self.loss_type == 'hybrid':
                vb_losses, pred_x_start_logits = self.compute_Lt(x, x_t, t)
                ce_losses = self.cross_entropy_x_start(x_start=x, pred_x_start_logits=pred_x_start_logits)
                losses = vb_losses + self.hybrid_coeff * ce_losses
            else:
                raise NotImplementedError(self.loss_type)

            return -losses

    def sample(self, num_samples):
        b = num_samples
        device = self.log_alpha.device
        uniform_logits = torch.zeros((b, self.num_classes) + self.shape, device=device)
        log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')

            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t)
        print()
        return log_onehot_to_index(log_z)

    def sample_chain(self, num_samples):
        b = num_samples
        device = self.log_alpha.device
        uniform_logits = torch.zeros(
            (b, self.num_classes) + self.shape, device=device)

        zs = torch.zeros((self.num_timesteps, b) + self.shape).long()

        log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t)

            zs[i] = log_onehot_to_index(log_z)
        print()
        return zs
