import torch

def get_epsilon(args, curr_step):
    return max(args.max_epsilon - (args.max_epsilon - args.min_epsilon) / args.epsilon_decay_steps * curr_step, args.min_epsilon)

def normalize_reward(reward):
    return torch.sign(reward)

def get_optimizer(args, model):
    if args.optimizer == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=args.weight_decay, momentum=0, centered=False)
    elif args.optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {args.optimizer} has not been implemented yet!')
