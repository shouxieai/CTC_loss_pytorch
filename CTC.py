import torch
import torch.nn.functional as F
import numpy as np
# ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html
# https://github.com/pytorch/pytorch/blob/master/torch/testing/_internal/common_nn.py#L3918

# def one_func_set_all_random_seed(seed=0):
#     # different random seeds
#     import torch
#     torch.manual_seed(seed)

#     import random
#     random.seed(seed)

#     import numpy as np
#     np.random.seed(seed)

#     torch.use_deterministic_algorithms(True)

#     # for dataloader
#     g = torch.Generator()
#     g.manual_seed(seed)

#     return g

# def seed_worker(worker_id):
#     import random
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# _ = one_func_set_all_random_seed(3)


def ctcloss_reference(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean'):
    input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long)
    dt = log_probs.dtype
    log_probs = log_probs.double()  # we need the accuracy as we are not in logspace
    targets = targets.long()
    cum_target_lengths = target_lengths.cumsum(0)
    losses = []
    for i in range(log_probs.size(1)): # for loop the batchsize
        input_length = input_lengths[i].item()
        target_length = target_lengths[i].item()
        cum_target_length = cum_target_lengths[i].item()
        target_prime = targets.new_full((2 * target_length + 1,), blank) # create a target_prime full of zero
        if targets.dim() == 2:
            target_prime[1::2] = targets[i, :target_length] # equivalent to insert blanks in targets. e.g. targets = "dog" --> "-d-o-g-"
        else:
            target_prime[1::2] = targets[cum_target_length -
                                          target_length:cum_target_length]
        #! y
        probs = log_probs[:input_length, i].exp() # Also we have to convert original inputs from log-space like this
        
        #! alpha
        alpha_col = log_probs.new_zeros((target_length * 2 + 1,))
        alpha_col[0] = probs[0, blank] # 
        alpha_col[1] = probs[0, target_prime[1]]

        mask_third = (target_prime[:-2] != target_prime[2:]) # for condition 3 (a.k.a otherwise) 
        """ 
            this mask is only true when target_prime[current] != target_prime[current - 2]. i.e. apple's pp is not the case
            plz note that every odd element is blank
            so the condition never holds for them.

            cond1 --> token s is blank
            cond2 --> token s == token s -2
            cond2 --> others (a.k.a  token s != token s -2)

            mask_thid means mask for third condition. We also have first condition and second condition.
                1st, 2nd and 3rd condition visualization
                refer to cond1.jpg cond2.jpg, cond3.jpg and all_cond.jpg
         """
        
        for t in range(1, input_length): # traverse alph_col one by one (along the axis of input_length)
            alpha_col_next = alpha_col.clone()
            """ 
                alpha_col is the alpha_{t-1} in recursion.jpg
                alpha_col_next is the alpha_{t}   in recursion.jpg
             """

            # we always add alpha[current-1] to alpha[current] adhering to eq7 in recursion.jpg
            #!for cond 1 or 2 
            alpha_col_next[1:] += alpha_col[:-1] # code->paper   alpha_next->alpha_hat_t    alpha->alpha_{t-1}
            """ 
                stagger the two vectors by one postion/unit then add them
                1234567
                  1234567
                  s-2 s 
                
                l : 1212121212   l[s -2] == l[s]
                    1212121212


                  12121212
                  12121212

                alpha_next is the staggered-vectors sum.(a.k.a the one generated from the last moment )
             """

            # but we add a[current-2] to a[current] only when mask condition is true
            #! for cond3 due to mask_third 
            alpha_col_next[2:] += torch.where(mask_third, alpha_col[:-2], alpha_col.new_zeros(1)) # if condition return x, else y
            """ 
                Different from paper, in paper, we calculate them in scalor, but in code here
                we are using matrix. The conditions hold for different position of the vector at the same time.
             """

            #! regardless of conditions, all needs 
            alpha_col = alpha_col_next * probs[t, target_prime] # the alpha here is a new alpha
        
        losses.append(-alpha_col[-2:].sum().log()[None]) # refer to sum_of_loss.jpg. Get the total loss only w.r.t the last token and blank

    output = torch.cat(losses, 0)


    if reduction == 'mean':
        return (output / target_lengths.to(dtype=output.dtype, device=output.device)).mean() # .mean() is the mean over batchsize.
    elif reduction == 'sum':
        return output.sum()
    output = output.to(dt)
    return output


# ------------- general example --------------------
# log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
# targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
# input_lengths = torch.full((16,), 50, dtype=torch.long)
# target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)

# ------------ a dummy input ----------------
log_probs = torch.randn(15, 3, 27).log_softmax(2).detach().requires_grad_()
targets = torch.randint(0, 27, (3, 10), dtype=torch.long)
targets = torch.tensor([[15, 18, 1,  14, 7, 5,  0, 0,  0,  0],
                        [1,  16, 16, 12, 5, 0,  0, 0,  0,  0],
                        [23, 1,  20, 5, 18, 13, 5, 12, 15, 14]]
                        )
input_lengths = torch.full((3,), 15, dtype=torch.long)
target_lengths = torch.tensor([6,5,10], dtype = torch.long)

# ------------ compute loss --------------------------  
loss1 = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
loss2 = ctcloss_reference(log_probs, targets, input_lengths, target_lengths)
""" 
    log_probs.shape
    torch.Size([50, 16, 20])  T N C    input_length, batch_size, class_num

    targets.shape
    torch.Size([16, 30])

    input_lengths 50
    target_lengths tensor([27, 18, 27, 20, 15, 13, 26, 16, 24, 21, 20, 18, 18, 26, 27, 11])

 """
print(loss1)
print(loss2)

