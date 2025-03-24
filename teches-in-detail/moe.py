import torch
import torch.nn as nn
import torch.optim as optim

class ExpertModel(nn.Module):
    def __init__(self, input_dim):
        super(ExpertModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        print('The parameters of linear:')
        for param in self.fc.parameters():
            print(param)

    def forward(self, x):
        return self.fc(x)

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([
            ExpertModel(input_dim) for _ in range(num_experts)
        ])
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim = 1)
        )
    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim = 1)

        gating_weights = self.gating_network(x)

        final_output = torch.sum(
            expert_outputs * gating_weights.unsqueeze(2), dim = 1
        )
        return final_output

'''
def training(input_data):
    num_epochs = 1000
    for epoch in range(num_epochs):
        output = model(input_data)
        loss = criterion(output, target_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch[{epoch + 1} / {num_epochs}, Loss: {loss.item()}]')
'''

def inference():
    new_data = torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0])
    prediction = model(new_data.unsqueeze(0))
    print(f'Prediction: {prediction.item()}')

if __name__ == '__main__':
    input_dim = 5
    num_experts = 2
    model = MixtureOfExperts(input_dim, num_experts)

    '''
    criterion = nn.MSELoss()
    optimzer = optim.Adam(model.parameters(), lr = 0.001)
    training()
    '''
    
    inference()
