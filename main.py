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
	parser.add_argument('--dataset', type=str, default="Imbal-8Gaussians",
						help='dataset to use: 8Gaussians | 25Gaussians | swissroll | mnist')
	parser.add_argument('--scale', type=float, default=10., help='data scaling')
	parser.add_argument('--ratio', type=float, default=0.9, help='ratio of imbalance')
	# stage
	parser.add_argument('--mode', type=str, default="train",
						help='type of running: train, collab, refine')
	parser.add_argument('--method', type=str, default="standard",
						help='type of running: standard, refinement, rejection, hastings, benchmark')
	# sampling
	parser.add_argument('--ckpt_num', type=int, default=0, help='ckpt number')
	parser.add_argument('--rollout_rate', type=float, default=0.1, help='rollout rate')
	parser.add_argument('--rollout_steps', type=int, default=50)
	# misc
	parser.add_argument('--seed', type=int, default=2020, help='manual seed')
	parser.add_argument('--out_dir', type=str, default="./out", help='folder to output')
	parser.add_argument('--ckpt_dir', type=str, default="./ckpt", help='folder to output')
	return parser.parse_args()

args = parse_args()
print(args)

# folders
try:
	os.makedirs(os.path.join(args.out_dir, args.mode))
except OSError as err:
	pass
try:
	os.makedirs(args.ckpt_dir)
except OSError as err:
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

# load
try:
	generator.load_state_dict(torch.load("%s/generator_%d.pth" % (args.ckpt_dir, args.ckpt_num)))
	discriminator.load_state_dict(torch.load("%s/discriminator_%d.pth" % (args.ckpt_dir, args.ckpt_num)))
except Exception as e:
	print('Failed to load ckpt. ',e)
	args.ckpt_num = 0

# data
noise = NoiseDataset()
data = ToyDataset(distr=args.dataset, scale=args.scale, ratio=args.ratio)

# ground truth
real_batch = data.next_batch(args.batch_size).to(device)
draw_sample(None, real_batch.cpu().numpy(), args.scale, os.path.join(args.out_dir, 'batch_real.png'))
draw_kde(real_batch.cpu().numpy(), args.scale, os.path.join(args.out_dir, 'kde_real.png'))

criterion = nn.BCEWithLogitsLoss()

########################################################
# 		Train GANs
########################################################

if args.mode == "train":
	
	optim_g = optim.SGD(generator.parameters(), lr=args.lrg)
	optim_d = optim.SGD(discriminator.parameters(), lr=args.lrd)

	for i in range(args.ckpt_num, args.niter):

		############################
		# Update D network: maximize log(D(x)) + log(1 - D(G(z)))
		###########################
		discriminator.zero_grad()

		# train with real
		real_batch = data.next_batch(args.batch_size).to(device)
		label = torch.full((args.batch_size,), 1, dtype=torch.float, device=device)
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
		# Update G network: maximize log(D(G(z)))
		###########################
		generator.zero_grad()

		label.fill_(1)  # fake labels are real for generator cost
		output = discriminator(fake_batch)
		loss_g = criterion(output, label)
		loss_g.backward()

		optim_g.step()

		if i % 1000 == 0:
			# Display samples
			print('[%d/%d] Loss_D: %.4f Loss_G: %.4f'
				% (i, args.niter, loss_d.item(), loss_g.item()))
			draw_sample(fake_batch.detach().cpu().numpy(), real_batch.cpu().numpy(), args.scale, os.path.join(args.out_dir, 'train', 'batch_fake_{:05d}.png'.format(i)))
			draw_kde(fake_batch.detach().cpu().numpy(), args.scale, os.path.join(args.out_dir, 'train', 'kde_fake_{:05d}.png'.format(i)))
			
			# Save model checkpoints
			torch.save(generator.state_dict(), "%s/generator_%d.pth" % (args.ckpt_dir, i))
			torch.save(discriminator.state_dict(), "%s/discriminator_%d.pth" % (args.ckpt_dir, i))

########################################################
# Refine Samples w/ the Vanilla Discriminator 
########################################################

if args.mode == "refine":

	noise_batch = noise.next_batch(args.batch_size).to(device)
	fake_batch = generator(noise_batch)

	delta_refine = torch.zeros([args.batch_size, 2], dtype=torch.float32, requires_grad=True, device=device)
	optim_r = optim.Adam([delta_refine], lr=args.rollout_rate)
	label = torch.full((args.batch_size,), 1, dtype=torch.float, device=device)
	for k in range(args.rollout_steps):
		optim_r.zero_grad()
		output = discriminator(fake_batch.detach() + delta_refine)
		loss_r = criterion(output, label)
		loss_r.backward()
		optim_r.step()
	
	draw_sample(fake_batch.detach().cpu().numpy(), real_batch.cpu().numpy(), args.scale, os.path.join(args.out_dir, 'refine', 'batch_propose_{:05d}.png'.format(args.ckpt_num)))
	draw_kde(fake_batch.detach().cpu().numpy(), args.scale, os.path.join(args.out_dir, 'refine', 'kde_propose_{:05d}.png'.format(args.ckpt_num)))
	
	draw_sample((fake_batch+delta_refine).detach().cpu().numpy(), real_batch.cpu().numpy(), args.scale, os.path.join(args.out_dir, 'refine', 'batch_refine_{:05d}.png'.format(args.ckpt_num)))
	draw_kde((fake_batch+delta_refine).detach().cpu().numpy(), args.scale, os.path.join(args.out_dir, 'refine', 'kde_refine_{:05d}.png'.format(args.ckpt_num)))

########################################################
# Collaborative Sampling w/ the Shaped Discriminator 
########################################################

if args.mode == "collab":
	
	optim_d = optim.SGD(discriminator.parameters(), lr=args.lrd)

	for i in range(args.niter):

		# synthesize refined samples
		noise_batch = noise.next_batch(args.batch_size).to(device)
		fake_batch = generator(noise_batch)

		# probabilistic refinement
		proba_refine = torch.zeros([args.batch_size, 2], dtype=torch.float32, requires_grad=False, device=device)
		proba_steps = torch.LongTensor(args.batch_size,1).random_() % args.rollout_steps
		proba_steps_one_hot = torch.LongTensor(args.batch_size, args.rollout_steps)
		proba_steps_one_hot.zero_()
		proba_steps_one_hot.scatter_(1, proba_steps, 1)

		delta_refine = torch.zeros([args.batch_size, 2], dtype=torch.float32, requires_grad=True, device=device)
		optim_r = optim.Adam([delta_refine], lr=args.rollout_rate)
		label = torch.full((args.batch_size,), 1, dtype=torch.float, device=device)
		for k in range(args.rollout_steps):
			optim_r.zero_grad()
			output = discriminator(fake_batch.detach() + delta_refine)
			loss_r = criterion(output, label)
			loss_r.backward()
			optim_r.step()

			# probabilistic assignment
			proba_refine[proba_steps_one_hot[:,k] == 1, :] = delta_refine[proba_steps_one_hot[:,k] == 1, :]

		############################
		# Shape D network: maximize log(D(x)) + log(1 - D(R(G(z))))
		###########################
		optim_d.zero_grad()

		# train with real
		real_batch = data.next_batch(args.batch_size).to(device)
		output = discriminator(real_batch)
		loss_d_real = criterion(output, label)
		loss_d_real.backward()

		# train with refined
		label.fill_(0)
		output = discriminator((fake_batch+proba_refine).detach())
		loss_d_fake = criterion(output, label)
		loss_d_fake.backward()

		loss_d = loss_d_real + loss_d_fake
		optim_d.step()

		if i % 500 == 0:
			# Display samples
			print('[%d/%d] Loss_D: %.4f' % (i, args.niter, loss_d.item()))
			# refined samples
			draw_sample((fake_batch+delta_refine).detach().cpu().numpy(), real_batch.cpu().numpy(), args.scale, os.path.join(args.out_dir, 'collab', 'batch_refine_{:04d}_collab_{:04d}.png'.format(args.ckpt_num, i)))
			draw_kde((fake_batch+delta_refine).detach().cpu().numpy(), args.scale, os.path.join(args.out_dir, 'collab', 'kde_refine_{:04d}_collab_{:04d}.png'.format(args.ckpt_num, i)))
			# shaping samples
			draw_sample((fake_batch+proba_refine).detach().cpu().numpy(), real_batch.cpu().numpy(), args.scale, os.path.join(args.out_dir, 'collab', 'batch_proba_{:04d}_collab_{:04d}.png'.format(args.ckpt_num, i)))
			draw_kde((fake_batch+proba_refine).detach().cpu().numpy(), args.scale, os.path.join(args.out_dir, 'collab', 'kde_proba_{:04d}_collab_{:04d}.png'.format(args.ckpt_num, i)))
