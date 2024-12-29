import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device as CUDA or CPU


class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, device):
        super(CVAE, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        # Encoder layers
        self.fc1 = nn.Linear(input_dim + condition_dim, 512)
        self.fc21 = nn.Linear(512, latent_dim)  # Mu layer
        self.fc22 = nn.Linear(512, latent_dim)  # Log variance layer

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim + condition_dim, 512)
        self.fc4 = nn.Linear(512, input_dim)

        # Ensure all layers are on the correct device
        self.to(device)

    def encode(self, x, condition):
        x = x.to(self.device)
        condition = condition.to(self.device)
        combined = torch.cat([x, condition], dim=1)
        h1 = F.relu(self.fc1(combined))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        z = z.to(self.device)
        condition = condition.to(self.device)
        combined = torch.cat([z, condition], dim=1)
        h3 = F.relu(self.fc3(combined))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, condition):
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, condition), mu, logvar

def train_cvae(cvae, data, condition, epochs=10, batch_size=10, learning_rate=1e-3):
    optimizer = torch.optim.Adam(cvae.parameters(), lr=learning_rate)
    cvae.train()

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size].to(device)
            batch_condition = condition[i:i + batch_size].to(device)

            optimizer.zero_grad()
            mu, logvar = cvae.encode(batch_data, batch_condition)
            z = cvae.reparameterize(mu, logvar)
            recon_data = cvae.decode(z, batch_condition)

            recon_loss = F.mse_loss(recon_data, batch_data, reduction='sum')
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_loss + kl_div
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(data)}')


