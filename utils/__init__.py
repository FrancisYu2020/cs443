def get_epsilon(args, curr_step):
    return max(args.max_epsilon - (args.max_epsilon - args.min_epsilon) / args.epsilon_decay_steps * curr_step, args.min_epsilon)