import torch

def find_highest(task, model, n=1, optc=torch.optim.Adam, lr=1e-1, steps=1000, randstart=False, restriction_fn=lambda x:x, search_for_error=False):
    if randstart: x = torch.autograd.Variable(torch.rand([n] + task.x_shape)*(task.x_max - task.x_min) + task.x_min, requires_grad=True)
    else: x = torch.autograd.Variable(torch.ones([n] + task.x_shape)*(task.x_max - task.x_min)/2., requires_grad=True)
    opt = optc([x], lr)
    for step_i in range(steps):
        if search_for_error:
            loss = -torch.nn.functional.mse_loss(model.predict(x), task.score_fn(x).detach())
        else:
            loss = -model.predict(x).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        x.data = x.data.clamp(task.x_min, task.x_max)
        x.data = restriction_fn(x.data)
    return x