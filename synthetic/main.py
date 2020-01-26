import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

from model import Generator, Discriminator
from datasets import NoiseDataset, ToyDataset
from utils import draw_sample, draw_kde

# arguments
def parse_args():
	parser = argparse.ArgumentParser()
	# network
	parser.add_argument('--nhidden', type=int, default=64, help='number of hidden neurons')
	parser.add_argument('--nlayers', type=int, default=6, help='number of hidden layers')
	# training
	parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
	parser.add_argument('--lrg', type=float, default=5e-3, help='lr for G')
	parser.add_argument('--lrd', type=float, default=1e-2, help='lr for D')
	parser.add_argument('--niter', type=int, default=10001, help='number of iterations')
	# dataset
	parser.add_argument('--dataset', type=str, default='Imbal-8Gaussians',
						help='dataset to use: 8Gaussians | 25Gaussians | swissroll | mnist')
	parser.add_argument('--scale', type=float, default=10., help='data scaling')
	parser.add_argument('--ratio', type=float, default=0.9, help='ratio of imbalance')
	# stage
	parser.add_argument('--mode', type=str, default='train',
						help='type of running: train, shape, calibrate, test')
	parser.add_argument('--method', type=str, default='standard',
						help='type of running: standard, refinement, rejection, hastings, benchmark')
	# sampling
	parser.add_argument('--ckpt_num', type=int, default=0, help='ckpt number')
	parser.add_argument('--rollout_rate', type=float, default=0.1, help='rollout rate')
	parser.add_argument('--rollout_method', type=str, default='ladam')
	parser.add_argument('--rollout_steps', type=int, default=50)
	# misc
	parser.add_argument('--seed', type=int, help='manual seed')
	parser.add_argument('--out_dir', default='./out', help='folder to output')
	return parser.parse_args()

args = parse_args()
print(args)

# folders
try:
	os.makedirs(args.out_dir)
except OSError:
	pass

# seeds
if args.seed is None:
	args.seed = random.randint(1, 10000)
print("Random Seed: ", args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# gpu
cudnn.benchmark = True
cuda = True if torch.cuda.is_available() else False
if not cuda:
	print("WARNING: You have no CUDA devices")

device = torch.device("cuda" if cuda else "cpu")

# model
generator = Generator(args.nhidden).to(device)
discriminator = Discriminator(args.nhidden).to(device)

# TODO: load
print(generator)
print(discriminator)

noise = NoiseDataset()
data = ToyDataset(distr=args.dataset, scale=args.scale, ratio=args.ratio)

real_batch = data.next_batch(args.batch_size).to(device)
draw_sample(None, real_batch.numpy(), args.scale, os.path.join(args.out_dir, 'batch_real.png'))

# training 
criterion = nn.BCELoss()
optim_g = optim.SGD(generator.parameters(), lr=args.lrg)
optim_d = optim.SGD(discriminator.parameters(), lr=args.lrd)
# optim_g = optim.Adam(generator.parameters(), lr=args.lrg, betas=(0.5, 0.999))
# optim_d = optim.Adam(discriminator.parameters(), lr=args.lrd, betas=(0.5, 0.999))

for i in range(args.niter):

	############################
	# Update D network: maximize log(D(x)) + log(1 - D(G(z)))
	###########################
	discriminator.zero_grad()

	# train with real
	real_batch = data.next_batch(args.batch_size).to(device)
	label = torch.full((args.batch_size,), 1, device=device)
	output = discriminator(real_batch)
	loss_d_real = criterion(output, label)
	loss_d_real.backward()

	# train with fake
	noise_batch = noise.next_batch(args.batch_size).to(device)
	fake_batch = generator(noise_batch)
	label.fill_(0)

	output = discriminator(fake_batch.detach())
	loss_d_fake = criterion(output, label)
	loss_d_fake.backward()

	loss_d = loss_d_real + loss_d_fake
	optim_d.step()

	############################
	# (2) Update G network: maximize log(D(G(z)))
	###########################
	generator.zero_grad()

	label.fill_(1)  # fake labels are real for generator cost
	output = discriminator(fake_batch)
	loss_g = criterion(output, label)
	loss_g.backward()

	optim_g.step()

	if i % 1000 == 0:
		print('[%d/%d] Loss_D: %.4f Loss_G: %.4f'
			% (i, args.niter, loss_d.item(), loss_g.item()))
		draw_sample(fake_batch.detach().numpy(), real_batch.numpy(), args.scale, os.path.join(args.out_dir, 'batch_fake_{:05d}.png'.format(i)))
		# import pdb; pdb.set_trace()
