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
import torch.distributed as dist

class DistAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(DistAdamW, self).__init__(params, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[torch.Tensor] = group['params']
            for base_i in range(len(params)):
                p = params[base_i]
                if p.grad is None:
                    continue
                grad = p.grad
                rank_size = grad.shape[0] // world_size
                # Slice the gradient for reduce_scatter
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(
                    grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx=0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params: list[torch.Tensor] = group['params']
            for base in len(params):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0]//world_size
                p_slice = p[rank*rank_size:(rank+1)*rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p_slice]
                g_slice = grad_slices[idx]

                # State initialization
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                
                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                step = state['step']
                if wd != 0:
                    eff_weight_decay = wd * lr * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # Decay the first and second moment running average coefficients
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)    
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                # Compute step size
                step_size = lr*(torch.sqrt(bias_correction2) / bias_correction1)
                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx+=1
                all_reduce_futures.append(dist.all_reduce(p_slice, op=dist.ReduceOp.SUM, async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()