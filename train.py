import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import ClassiModel,ImCNN,CapCNN
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(args):
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    train_loss = []
    val_loss = []

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    train_dataloader = get_loader(args.train_img_dir,args.train_cap_path,vocab,
                                  transform, args.batch_size,
                                  shuffle=True,num_workers=args.num_workers)
    val_dataloader = get_loader(args.val_img_dir,args.val_cap_path,vocab,
                                  transform, args.batch_size,
                                  shuffle=True,num_workers=args.num_workers)

    im_encoder = ImCNN().to(device)
    tex_encoder = CapCNN(args.embed_size,len(vocab)).to(device)
    classifier = ClassiModel(12256,2).to(device)

    if args.pretrained == True:
        im_encoder.load_state_dict(torch.load(args.im_model_path))
        tex_encoder.load_state_dict(torch.load(args.tex_model_path))
        classifier.load_state_dict(torch.load(args.cla_model_path))

    criterion = nn.CrossEntropyLoss()
    params = list(im_encoder.parameters()) + list(tex_encoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam(params,lr=args.learning_rate)


    #train model
    total_step = len(train_dataloader)
    for epoch in range(args.num_epochs):
        for ii, (images,captions,targets) in enumerate(train_dataloader):

            images = images.to(device)
            captions = captions.to(device)
            targets =targets.to(device)

            im_features = im_encoder(images)
            cap_features = tex_encoder(captions)
            outputs = classifier(im_features, cap_features)
            loss = criterion(outputs,targets)
            im_encoder.zero_grad()
            tex_encoder.zero_grad()
            classifier.zero_grad()
            loss.backward()
            optimizer.step()

            if (ii) % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, ii, total_step, loss.item(), np.exp(loss.item())))

                train_loss.append(loss.item())
                v_loss = val(im_encoder, tex_encoder,classifier,val_dataloader)
                val_loss.append(v_loss)
                print('Epoch [{}/{}], Step [{}/{}], Val_Loss: {:.4f}, Perplexity: {:5.4f}'
                .format(epoch, args.num_epochs, ii, total_step,v_loss,np.exp(v_loss)))

            if (ii+1) % args.save_step == 0:
                torch.save(im_encoder.state_dict(), os.path.join(
                    args.model_path, 'im_encoder-{}-{}.ckpt'.format(epoch+6, ii+1)))
                torch.save(tex_encoder.state_dict(), os.path.join(
                    args.model_path, 'tex_encoder-{}-{}.ckpt'.format(epoch+6, ii+1)))
                torch.save(classifier.state_dict(), os.path.join(
                    args.model_path, 'classifier-{}-{}.ckpt'.format(epoch+6, ii+1)))

    with open('train_loss.txt', 'w') as f:
        f.write(str(train_loss))
        f.close()
    with open('val_loss.txt', 'w') as f:
        f.write(str(val_loss))
        f.close()

criterion = nn.CrossEntropyLoss()
def val(im_encoder,tex_encoder,classifier,val_dataloader):
    im_encoder.eval()
    tex_encoder.eval()
    classifier.eval()
    for ii, (images, captions, targets) in enumerate(val_dataloader):
        images = images.to(device)
        captions = captions.to(device)
        targets = targets.to(device)

        im_features = im_encoder(images)
        cap_features = tex_encoder(captions)
        outputs = classifier(im_features, cap_features)
        loss = criterion(outputs, targets)
        return loss.item()






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--train_img_dir', type=str, default='/media/theo/data/MVSO/images/resized_train/',
                        help='directory for resized train images')
    parser.add_argument('--val_img_dir', type=str, default='/media/theo/data/MVSO/images/resized_val/',
                        help='directory for resized val images')
    parser.add_argument('--train_cap_path', type=str,
                        default='/home/theo/Xli/train.json',
                        help='path for train annotation json file')
    parser.add_argument('--val_cap_path', type=str,
                        default='/home/theo/Xli/val.json',
                        help='path for val annotation json file')
    parser.add_argument('--log_step', type=int, default=100, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=100, help='dimension of word embedding vectors')
    #parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    #parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--im_model_path', type=str, default='./models/im_encoder-5-3000.ckpt')
    parser.add_argument('--tex_model_path', type=str, default='./models/tex_encoder-5-3000.ckpt')
    parser.add_argument('--cla_model_path', type=str, default='./models/classifier-5-3000.ckpt')


    args = parser.parse_args()
    print(args)
    main(args)