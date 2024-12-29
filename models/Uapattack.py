import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import CrossEntropyLoss

class Attack:
    def train(self, target_model, dataloader):
        return self

    def __repr__(self):
        return f"(attack=Attack)"
class UAPAttack(Attack):
    def __init__(self, target_label):
        super(UAPAttack, self).__init__()
        self.target_label = target_label

    def train(self, target_model, dataloader):
        """ overrides parent class train function """
        self.generator = UAP()
        target_model.eval()


        self.generator.cuda()

        uap_train(dataloader,
                  self.generator,
                  target_model)

        return self

    def run(self, inputs, labels):
        for k in range(inputs.size(-4)):
            if labels[k] == self.target_label:
                inputs[k] = torch.squeeze(self.generator(inputs[k]).detach(), axis=0)

        return inputs, labels

    def __repr__(self):
        return f"(attack=UAPAttack, target_label={self.target_label})"

class NegativeCrossEntropy(_WeightedLoss):
    def __init__(self,):
        super(NegativeCrossEntropy, self).__init__()

    def forward(self, x, target, uap, l=1):
        ce = CrossEntropyLoss()
        loss = ce(x, target)
        loss = loss - l * torch.norm(uap)

        return loss
def uap_train(data_loader, generator, target_network, epsilon=1000, num_iterations=1000, targeted=False,
              target_class=4, print_freq=200, use_cuda=False):
    # Define the parameters
    criterion = NegativeCrossEntropy()
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    device = torch.device('cuda:{}'.format(0))
    # switch to train mode
    generator.train()
    target_network.eval()
    data_iterator = iter(data_loader)
    iteration = 0
    while iteration < num_iterations:
        try:
            input, target = next(data_iterator)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iterator = iter(data_loader)
            input, target = next(data_iterator)

            # Class specific UAP
            ind = []
            for k in range(input.size(-4)):
                if target[k] == target_class:
                    ind.append(k)
            input = input[ind]
            target = target[ind]

        if len(input) == 0:
            continue

            #
        input = input.to(device)
        target = torch.tensor(target).to(device)

        # compute output
        output = target_network(generator(input))

        # target = target.float()
        loss = criterion(output, target, generator.uap)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Projection
        generator.uap.data = torch.clamp(generator.uap.data, -epsilon, epsilon)

        iteration += 1
        # if iteration % 100 == 0:  # print every 100 ioteration
        #     print('Optimization Iteration %d of %d' % (iteration, num_iterations))
class UAP(nn.Module):
    def __init__(self, shape=(28, 28), num_channels=1, mean=[0.5], std=[0.5], use_cuda=False):
        super(UAP, self).__init__()
        self.use_cuda = use_cuda
        self.num_channels = num_channels
        self.shape = shape
        self.uap = nn.Parameter(torch.zeros(size=(num_channels, *shape), requires_grad=True))

        self.mean_tensor = torch.ones(1, num_channels, *shape)
        for idx in range(num_channels):
            self.mean_tensor[:,idx] *= mean[idx]

        self.mean_tensor = self.mean_tensor.cuda()

        self.std_tensor = torch.ones(1, num_channels, *shape)
        for idx in range(num_channels):
            self.std_tensor[:,idx] *= std[idx]

        self.std_tensor = self.std_tensor.cuda()

    def forward(self, x):
        uap = self.uap
        orig_img = x * self.std_tensor + self.mean_tensor # Put image into original form
        orig_img = orig_img.cuda()
        uap = uap.cuda()
        adv_orig_img = orig_img + uap # Add uap to input
        adv_x = (adv_orig_img - self.mean_tensor)/self.std_tensor # Put image into normalized form
        return adv_x




    # def show_sample_with_perturbation(self, dataloader):
    #     self.generator.eval()  # Ensure the generator is in evaluation mode
    #
    #     # Fetch a single batch of data
    #     images, labels = next(iter(dataloader))
    #     original_image = images[0:1]  # Take the first image from the batch
    #
    #
    #     original_image = original_image.cuda()
    #
    #     # Generate the perturbed image
    #     perturbed_image = self.generator(original_image)
    #
    #     # Move tensors to CPU for visualization
    #     original_image = original_image.cpu().squeeze().numpy()
    #     perturbed_image = perturbed_image.cpu().squeeze().numpy()
    #
    #     # Plotting the original and the perturbed image
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.title("Original Image")
    #     plt.imshow(original_image, cmap='gray')
    #     plt.axis('off')
    #
    #     plt.subplot(1, 2, 2)
    #     plt.title("Perturbed Image")
    #     plt.imshow(perturbed_image, cmap='gray')
    #     plt.axis('off')
    #
    #     plt.show()
