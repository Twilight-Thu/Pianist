import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

steps = np.linspace(0, 2 * np.pi, 100, dtype=np.float32)  # x axes -> time
x_np = np.sin(steps)  # input
y_np = np.cos(steps)  # predicted output

# plt.plot(steps, y_np, 'r-', label="y (cos)")
# plt.plot(steps, x_np, 'b-', label="x (sin)")
# plt.legend(loc='best')
# plt.show()

time_step = 10  # the length of sequence
input_size = 1  # input size of data
hidden_size = 32  # the features of hidden state
number_of_layers = 1  # the number of reccurent layers

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=number_of_layers,
            batch_first=True,
        )

        self.out = nn.Linear(hidden_size, 1)

    def forward(self, X, h_state):
        r_out, h_state = self.rnn(X, h_state)
        outs = []
        for i in range(r_out.size(1)):   # reserve the output of every time step
            outs.append(self.out(r_out[:, i, :]))
        #print(len(outs))
        return torch.stack(outs, dim=1), h_state

rnn = RNN()
print(rnn)

lr = 0.02
epochs = 100

optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
loss_func = nn.MSELoss()

h_state = None
entire_preds = []

for epoch in range(epochs):
    # prepare the data
    start, end = epoch * np.pi, (epoch + 1) * np.pi
    steps = np.linspace(start, end, time_step, dtype=np.float32) # the length of the sequence
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    # input data : x & y -> [batch_size, time_step, input_size/output_size]
    # where the batch_size = 1 and input_size/output_size = 1
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    # preds, h_state = rnn(x, None)
    preds, h_state = rnn(x, h_state)
    h_state = h_state.data

    # print(preds.shape)
    # print(y.shape)

    loss = loss_func(preds, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    preds_plot = preds.detach().clone()
    entire_preds.append(preds_plot)

entire_preds = torch.stack(entire_preds, dim=0)
print("entire_preds.shape: ", entire_preds.shape)
entire_preds = entire_preds.reshape(epochs * time_step)
entire_steps = np.linspace(0, epochs * np.pi, epochs * time_step, dtype=np.float32)
entire_x = np.sin(entire_steps)
entire_y = np.cos(entire_steps)

seg = int(epochs * time_step / 4)

print("The first segment : the iteration of 1/4")  # 第一段图，epoch 0-24，250个样本点
plt.figure(figsize=(60, 15))
plt.plot(entire_steps[0:seg], entire_x[0:seg], 'g-', label='input')
plt.plot(entire_steps[0:seg], entire_y[0:seg], 'r-', label='target output')
plt.plot(entire_steps[0:seg], entire_preds[0:seg], 'b-', label='real output')
plt.show()

print("The second segment : the iteration of 2/4")  # 第二段图，epoch 25-49，250个样本点
plt.figure(figsize=(60, 15))
plt.plot(entire_steps[seg:seg*2], entire_x[seg:seg*2], 'g-', label='input')
plt.plot(entire_steps[seg:seg*2], entire_y[seg:seg*2], 'r-', label='target output')
plt.plot(entire_steps[seg:seg*2], entire_preds[seg:seg*2], 'b-', label='real output')
plt.show()

print("The third segment : the iteration of 3/4")  # 第三段图，epoch 50-74，250个样本点
plt.figure(figsize=(60, 15))
plt.plot(entire_steps[seg*2:seg*3], entire_x[seg*2:seg*3], 'g-', label='input')
plt.plot(entire_steps[seg*2:seg*3], entire_y[seg*2:seg*3], 'r-', label='target output')
plt.plot(entire_steps[seg*2:seg*3], entire_preds[seg*2:seg*3], 'b-', label='real output')
plt.show()

print("The fourth segment : the iteration of 4/4")  # 第四段图，epoch 0-24，250个样本点
plt.figure(figsize=(60, 15))
plt.plot(entire_steps[seg*3:seg*4], entire_x[seg*3:seg*4], 'g-', label='input')
plt.plot(entire_steps[seg*3:seg*4], entire_y[seg*3:seg*4], 'r-', label='target output')
plt.plot(entire_steps[seg*3:seg*4], entire_preds[seg*3:seg*4], 'b-', label='real output')
plt.show()




