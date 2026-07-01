import torch
import torch.nn as nn

class GraphFlowMatching(nn.Module):
    def __init__(self, sigma_min=1e-4):
        super(GraphFlowMatching, self).__init__()
        self.sigma_min = sigma_min
        
    def compute_path(self, x_0, alpha_0, t):
        """
        \psi_t(x_0) = (1 - (1 - \sigma_{min})t)x_0 + t\alpha_0
        """
        if t.dim() < x_0.dim():
            t = t.view(-1, *([1]*(x_0.dim()-1)))
        return (1 - (1 - self.sigma_min) * t) * x_0 + t * alpha_0
    
    def compute_target_velocity(self, x_0, alpha_0):
        """
        u_t = \alpha_0 - (1 - \sigma_{min})x_0
        """
        return alpha_0 - (1 - self.sigma_min) * x_0

    def estimate_alpha_0(self, psi_t, v_t, t):
        """
        \hat{\alpha}_0 = (1-\sigma_{min})\psi_t + (1 - t(1-\sigma_{min})) v_t
        """
        if t.dim() < psi_t.dim():
            t = t.view(-1, *([1]*(psi_t.dim()-1)))
        sigma = self.sigma_min
        return (1 - sigma) * psi_t + (1 - t * (1 - sigma)) * v_t

    def training_losses(self, model, alpha_0, itmEmbeds, batch_index, model_feats):
        """
        alpha_0: Ground truth interaction matrix (batch_size, num_items)
        model: VelocityModel to predict vector field
        itmEmbeds: Item ID embeddings
        model_feats: Multimodal features
        """
        batch_size = alpha_0.size(0)
        device = alpha_0.device

        # 1. Sample t ~ U[0, 1]
        t = torch.rand(batch_size, device=device)
        
        # 2. Sample x_0 ~ N(0, I)
        x_0 = torch.randn_like(alpha_0)
        
        # 3. Compute path and target velocity
        psi_t = self.compute_path(x_0, alpha_0, t)
        v_target = self.compute_target_velocity(x_0, alpha_0)
        
        # 4. Predict velocity
        v_pred = model(psi_t, t)
        
        # 5. Compute Graph-CFM loss
        mse_cfm = torch.mean((v_pred - v_target) ** 2, dim=list(range(1, len(v_pred.shape))))
        cfm_loss = mse_cfm
        
        # 6. Compute MSI loss
        alpha_hat = self.estimate_alpha_0(psi_t, v_pred, t)
        usr_model_embeds = torch.mm(alpha_hat, model_feats)
        usr_id_embeds = torch.mm(alpha_0, itmEmbeds)
        msi_loss = torch.mean((usr_model_embeds - usr_id_embeds) ** 2, dim=list(range(1, len(usr_model_embeds.shape))))
        
        return cfm_loss, msi_loss

    def euler_solve(self, model, x_start, steps=5):
        """
        Solve ODE using Euler method from t=0 to t=1
        """
        device = x_start.device
        batch_size = x_start.size(0)
        
        if steps == 0:
            return x_start

        dt = 1.0 / steps
        x_t = x_start
        
        for i in range(steps):
            t_val = i * dt
            t = torch.full((batch_size,), t_val, device=device)
            v_pred = model(x_t, t)
            x_t = x_t + v_pred * dt
            
        return x_t
