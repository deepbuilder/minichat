'''Adam Optimizer Implementation for MiniChat

Algorithm:

Initialize parameters: theta_0
Initialize first moment vector: m_0 = 0
Initialize second moment vector: v_0 = 0
Set hyperparameters: learn_rate, beta_1, beta_2, epsilon, weight_decay
For each time step t = 1, 2, ..., T:
    Compute gradient: grad = ∇_theta L(theta_t-1)
    Update biased first moment estimate:
        m_t = beta_1 * (m_t-1) + (1-beta_1) * grad
    Update biased second moment estimate:
        v_t = beta_2 * (v_t-1) + (1-beta_2) * (grad**2)
    Compute bias-corrected first moment estimate:
        m_hat_t = m_t / (1 - beta_1**t)
    Compute bias-corrected second moment estimate:
        v_hat_t = v_t / (1 - beta_2**t)
    Update parameters:
        theta_t = theta_t-1 - learn_rate * (m_hat_t / (sqrt(v_hat_t) + epsilon))
    Apply weight decay:
        theta_t = theta_t - learn_rate * weight_decay * theta_t-1
'''

import torch

class DistAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(DistAdamW, self).__init__(params, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['step'] = 0
                
                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                step = state['step']
                # Decay the first and second moment running average coefficients
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)    
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                # Compute step size
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)
                # Weight decay
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay']) 