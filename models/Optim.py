'''A wrapper class for optimizer '''
import numpy as np

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps,init_lr = 1e-4):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        # self.init_lr = np.power(d_model, -0.5)
        self.init_lr = init_lr
        self.cur_lr = self.init_lr

    def step(self):
        "Step with the inner optimizer"
        # self._update_learning_rate()
        #self._update_learning_rate_normal()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    """
    update learning rate using the normal operation like Caffe Training
    """
    def _update_learning_rate_normal(self, epoch):
        self.n_current_steps = epoch
        if self.n_current_steps % self.n_warmup_steps == 0:
            if self.n_current_steps == 0:
                lr = self.cur_lr
                print('Learning Rate: {:4f}'.format(lr))
            else:
                lr = self.cur_lr * 0.1
                print('Learning Rate Drop to {:4f}'.format(lr))
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr
            # update the current learning rate here
            self.cur_lr = lr
        else:
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = self.cur_lr

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        # set all the parameter learning rate in the optimizer parameters groups
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        self.cur_lr = lr

