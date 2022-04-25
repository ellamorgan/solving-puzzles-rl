import sys
import torch
import numpy as np
import wandb
from PIL import Image

from data import SlidingTiles
from networks import VAE_Model, vae_loss
from rl_networks import DoubleDQN


'''
I consulted references when learning to implement this kind of reinforcement learning agent
The most useful were PyTorch's tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
And the following resource on Double Deep Q Learning: https://medium.com/@leosimmons/double-dqn-implementation-to-solve-openai-gyms-cartpole-v-0-df554cd0614d
'''


def load_parameters():
    args = dict()

    # Double DQN parameters
    args['num_episodes'] = 1000
    args['experience_batch'] = 200
    args['max_memory_size'] = 2000
    args['gamma'] = 0.99
    args['epsilon'] = 0.1
    args['start_steps'] = 10
    args['vae_model_path'] = 'vae-state-dict.pt'
    args['update_target_every'] = 10

    # Environment Model parameters
    args['epochs'] = 2000
    args['n_data'] = 2000
    args['train_split'] = 0.8
    args['val_split'] = 0.1
    args['batch'] = 40
    args['usecuda'] = False         # Warning: may not work with this set to true as I finalized the code on a laptop without a GPU
    args['lr'] = 1e-5
    args['weight_decay'] = 1e-6
    args['save_every'] = 50
    args['no_wandb'] = False
    args['tau_max'] = 0.95
    args['tau_min'] = 0.05
    args['img_width'] = 15
    args['kernel_size'] = 3
    args['channels'] = 4
    args['fluents'] = 50
    args['n_actions'] = 4
    args['hidden_size'] = 100

    return args



def save_as_gif(result, filepath):
    '''
    Saves a list of images (represented as arrays) as a gif
    '''
    gif = []
    for arr in result:
        gif.append(Image.fromarray((arr + 1) * 127.5))

    gif[0].save(filepath, save_all=True, append_images=gif[1:], duration=1000, loop=0)



def train_ddqn_agent(args):
    '''
    Trains the Double DQN agent
    Expects that the Environment Model weights are stored in a file called vae-state-dict.pt
    '''
    env = SlidingTiles(size=3)
    device = torch.device("cuda") if args['usecuda'] else torch.device("cpu")
    agent = DoubleDQN(args, device)
    
    episode_durations = []

    '''
    Training loop
    '''
    for episode in range(args['num_episodes']):

        # Starts start_steps away from the goal
        env.reset(args['start_steps'])
        # The image is rendered from the environment then encoded as a discrete representation
        state = agent.encode_state(env.render(normalize=True))

        finished = False
        frames = []
        steps_done = 0

        while not finished:
            action = agent.choose_action(state)
            steps_done += 1
            reward, done = env.step(int(action))
            reward = torch.tensor([reward], device=device)

            if not done:
                next_state = agent.encode_state(env.render(normalize=True))
            else:
                next_state = None
            
            agent.add_to_memory(state, next_state, action, reward, done)
            frames.append(env.render(normalize=True))

            state = next_state

            # Performs a online network training update by sampling from experience
            agent.experience_replay()

            # Terminate epsiode if it lasts longer than 299 steps
            if done or steps_done >= 298:
                episode_durations.append(steps_done + 1)
                print("Episode %d complete after %d steps" % (episode, steps_done + 1))
                sys.stdout.flush()
                finished = True
            
        save_as_gif(frames, "gifs/ep%d.gif" % (episode))
        
        if episode % args['update_target_every']:
            agent.match_networks()


def train_vae(args):
    '''
    Trains the environment model. The Environment Model is trained before the Double DQN agent
    Running this code stores the model weights, which are then used when running the RL agent
    A copy of the weights are saved in vae-state-dict.pt, unnecessary to run this portion again unless
    wanting to retrain the Environment Model.
    '''
    env = SlidingTiles(size=3)

    device = torch.device("cuda") if args['usecuda'] else torch.device("cpu")
    model = VAE_Model(w=args['img_width'], k=args['kernel_size'], c=args['channels'], f=args['fluents'], hidden_state=args['hidden_size'], batch=args['batch'], device=device, tau_max=0.95, tau_min=0.05, epochs=args['epochs']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    loaders = env.generate_vae_dataset(args['n_data'], args['train_split'], args['val_split'], args['batch'], args['usecuda'])
    p = torch.Tensor([0.1]).to(device)

    if not args['no_wandb']:
        import wandb
        run = wandb.init(project='rl-vae',
            reinit=True)
    else:
        wandb = None

    # Training loop
    for epoch in range(args['epochs']):

        train_loss = 0
        losses = {'recon_loss' : 0, 'prior_losses' : 0, 'latent_difference' : 0}

        for data in loaders['train']:

            prev = data['prev'].to(device, dtype=torch.float)
            succ = data['succ'].to(device, dtype=torch.float)
            action = data['action'].to(device, dtype=torch.float)
            optimizer.zero_grad()
            out = model(prev, succ, action, epoch)
            loss, recon_loss, prior_losses, latent_difference = vae_loss(out, p)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Track individual components of the losses - they vary a lot in magnitude, useful to display to understand how to normalize
            losses['recon_loss'] += recon_loss.item()
            losses['prior_losses'] += prior_losses.item()
            losses['latent_difference'] += latent_difference.item()
            

        train_loss /= (len(loaders['train']) * args['batch'])
        print("epoch: {}, train loss = {:.6f}, ".format(epoch + 1, train_loss), end="")

        losses['recon_loss'] /= (len(loaders['train']) * args['batch'])
        losses['prior_losses'] /= (len(loaders['train']) * args['batch'])
        losses['latent_difference'] /= (len(loaders['train']) * args['batch'])

        with torch.no_grad():

            model.eval()
            val_loss = 0

            for data in loaders['val']:

                prev = data['prev'].to(device, dtype=torch.float)
                succ = data['succ'].to(device, dtype=torch.float)
                action = data['action'].to(device, dtype=torch.float)
                out = model(prev, succ, action, epoch)
                loss, _, _, _ = vae_loss(out, p)
                val_loss += loss.item()
            
            val_loss /= (len(loaders['val']) * args['batch'])
            print("val loss = {:.6f}".format(val_loss))
        
            model.train()
        

        # Save results as gif
        if (epoch + 1) % args['save_every'] == 0:
            
            pres_dec = out['prev_enc_dec'].to('cpu').numpy()
            sucs_dec = out['succ_enc_dec'].to('cpu').numpy()
            pres_aae = out['prev_pred_dec'].to('cpu').numpy()
            sucs_aae = out['succ_pred_dec'].to('cpu').numpy()

            dec_joint = np.concatenate((pres_dec, sucs_dec), axis=3)
            aae_joint = np.concatenate((pres_aae, sucs_aae), axis=3)
            joint = np.concatenate((dec_joint, aae_joint), axis=2)
            
            save_as_gif(joint, 'saved_gifs/epoch_' + str(epoch + 1) + '_out' + '.gif')

            pres_in = out['prev'].to('cpu').numpy()
            sucs_in = out['succ'].to('cpu').numpy()

            joint = np.concatenate((pres_in, sucs_in), axis=3)
            
            save_as_gif(joint[:, 0], 'saved_gifs/epoch_' + str(epoch + 1) + '_in' + '.gif')
        

        # Log results in wandb
        if wandb is not None:
            wandb.log({"train-loss": train_loss, "val-loss": val_loss, 'recon-loss' : losses['recon_loss'], 'prior-losses' : losses['prior_losses'], 'latent-difference' : losses['latent_difference']})

    # Finish wandb run
    if wandb is not None:
        run.finish()

    # Save model weights after training - these are used when running the reinforcement learning agent
    torch.save(model.state_dict(), 'vae-state-dict.pt')

    # Save final images after training finished
    # (Ensure epochs >= save_every)
    for i in range(len(pres_dec)):
        save_image(pres_dec[i, 0], "out_images/pres_dec_%d.png" % (i))
        save_image(sucs_dec[i, 0], "out_images/sucs_dec_%d.png" % (i))
        save_image(pres_aae[i, 0], "out_images/pres_aae_%d.png" % (i))
        save_image(sucs_aae[i, 0], "out_images/sucs_aae_%d.png" % (i))


if __name__ == '__main__':
    args = load_parameters()
    train_ddqn_agent(args)