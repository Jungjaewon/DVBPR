import os
import random
import time
import argparse
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch
import torchvision
import torchvision.models as models
from cStringIO import StringIO
import torch.multiprocessing as mp
import subprocess
'''
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
'''
mp.Process()

dataset_name = 'AmazonFashion6ImgPartitioned.npy'

#Hyper-prameters
use_cuda = True
K = 100 # Latent dimensionality
lambda1 = 0.001 # Weight decay
lambda2 = 1.0 # Regularizer for theta_u
learning_rate = 1e-4
training_epoch = 20
batch_size = 128
dropout = 0.5 # Dropout, probability to keep units
numldprocess=4 # multi-threading for loading images

dataset = np.load('/home/woodcook486/DVBPR/'+dataset_name,encoding='latin1')
#random.seed(13)
[user_train, user_validation, user_test, Item, usernum, itemnum] = dataset

print 'Data load is completed!!'

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', dest='gpu', help='the number of gpu to use', default=0, type=int)
    parser.add_argument('--K', dest='K', help='dimension of latent code', default=100, type=int)
    parser.add_argument('--lambda1', dest='lambda1', help='lambda1 value', default=0.001, type=float)
    parser.add_argument('--lambda2', dest='lambda2', help='lambda2 value', default=1.0, type=float)
    parser.add_argument('--lr', dest='lr', help='learning rate value', default=0.0001, type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size', default=256, type=int)
    parser.add_argument('--drop_out', dest='drop_out', help='dropout rate', default=0.5, type=float)
    parser.add_argument('--num_of_thread', dest='num_of_thread', help='num_of_thread', default=4, type=int)
    parser.add_argument('--training_epoch', dest='training_epoch', help='training_epoch', default=30, type=int)
    parser.add_argument('--mode', dest='mode', help='mode', default=3, type=int)
    parser.add_argument('--dissimilarity', dest='dissimilarity', help='dissimilarity between two feature', default=1200.0, type=float)

    '''
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    '''

    args = parser.parse_args()
    return args

class customDataset(Dataset):
    def __init__(self, mode, dissimilarity, gpu):
        self.mode = mode
        self.dissimilarity = dissimilarity
        self.gpu = gpu
        if self.mode in [3,4,5]:
            self.vgg16 = models.vgg16(pretrained=True).cuda(self.gpu)
    def __len__(self):
        return -1
    def __getitem__(self, user_dataset):

        u = random.randrange(usernum)
        numu = len(user_dataset[u])
        i = user_dataset[u][random.randrange(numu)]['productid']
        img1 = np.uint8(np.asarray(Image.open(StringIO(Item[i]['imgs'])).convert('RGB').resize((224, 224))))
        img1 = torch.from_numpy(img1).float().cuda(self.gpu)
        img1 = torch.unsqueeze(img1, 0)
        img1 = img1.permute(0, 3, 1, 2)

        M = set()
        for item in user_dataset[u]:
            M.add(item['productid'])
        while True:
            j = random.randrange(itemnum)

            if j not in M:
                if self.mode in [3,4,5]:
                    img2 = np.uint8(np.asarray(Image.open(StringIO(Item[j]['imgs'])).convert('RGB').resize((224, 224))))
                    img2 = torch.from_numpy(img2).float().cuda(self.gpu)
                    img2 = torch.unsqueeze(img2,0)
                    img2 = img2.permute(0,3,1,2)

                    with torch.no_grad():
                        out1 = self.vgg16(img1)
                        out2 = self.vgg16(img2)
                    dist = torch.norm(out1-out2,2).item()
                    #print dist
                    if dist > self.dissimilarity:
                        break
                    else:
                        continue
                else:
                    break

        return u, i, j

'''
class Lookup_Matrix(nn.Module):
    def __init__(self, usernum, K):
        super(Lookup_Matrix, self).__init__()
        self.a = torch.rand((usernum,K))/100
        self.Matrix = nn.Parameter(self.a, requires_grad=True)

    def forward(self, u):
        output = torch.index_select(self.Matrix, 0, u)

        return output
'''


class Lookup_Matrix(nn.Module):
    def __init__(self, usernum, K):
        super(Lookup_Matrix, self).__init__()
        self.linear = nn.Linear(usernum, K,bias=False)
        self.usernum = usernum
        #self.Matrix = nn.Parameter(self.a, requires_grad=True)

    def forward(self, u):

        input = u.float()
        output = self.linear(input)

        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 11, 4,padding=1),nn.ReLU(True),nn.MaxPool2d(kernel_size=2,padding=1))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 256, 5,padding=1), nn.ReLU(True), nn.MaxPool2d(kernel_size=2,padding=1))
        self.layer3 = nn.Sequential(nn.Conv2d(256, 256, 3,padding=1), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 256, 3,padding=2), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Conv2d(256, 256, 3), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.layer6 = nn.Sequential(nn.Linear(7*7*256, 4096), nn.ReLU(), nn.Dropout2d(dropout))
        self.layer7 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout2d(dropout))
        self.layer8 = nn.Sequential(nn.Linear(4096, K), nn.ReLU(), nn.Dropout2d(dropout))

    def forward(self, input1):
        #print 'input : ',input.size()
        #print 'imagesize',input.size()
        out = self.layer1(input1)
        out = self.layer2(out)
        out = self.layer3(out)
        #print 'layer 3 :',out.size()
        out = self.layer4(out)
        #print 'layer 4 :',out.size()
        out = self.layer5(out)
        out = out.view(-1, out.size()[1] * out.size()[2] * out.size()[3])
        #print 'layer 5 :', out.size()
        out = self.layer6(out)
        #print 'layer 6 :', out.size()
        out = self.layer7(out)
        out1 = self.layer8(out)

        return out1

def calculate_Loss(feature1, feature2, theta, u):

    feature_diff = feature1 - feature2
    theta_u = theta(u)
    #print 'feature diff',feature_diff.size()
    #print 'theta u',theta_u.size()
    #print (feature_diff * theta_u).size()
    l2_reg_theta = theta_u.norm(2).sum()
    #logit = (feature_diff * theta_u).cuda()
    logit = torch.mul(feature_diff,theta_u)
    #print 'logit size : ', logit.size()
    #print 'logit sum size : ',logit.sum(dim=1).size()

    Log_sigmoid = nn.LogSigmoid()
    loss = Log_sigmoid(logit.sum(dim=1,keepdim=True)).sum() - lambda2 * l2_reg_theta


    #Sigmoid = nn.Sigmoid()
    #out = Sigmoid(logit)
    #loss = torch.log(out + 1e-8)

    #print 'loss in function :',loss
    return loss

def get_batch(batch_size, gpu):
    u_list = []
    i_list = []
    j_list = []
    img1_list = []
    img2_list = []

    for b in xrange(batch_size):
        u, i, j = fashion_dataset.__getitem__(user_train)
        img1 = np.uint8(np.asarray(Image.open(StringIO(Item[i]['imgs'])).convert('RGB').resize((224, 224))))
        img2 = np.uint8(np.asarray(Image.open(StringIO(Item[j]['imgs'])).convert('RGB').resize((224, 224))))
        u_list.append(u)
        i_list.append(i)
        j_list.append(j)
        img1 = (img1 - 127.5) / 127.5
        img2 = (img2 - 127.5) / 127.5
        img1_list.append(img1)
        img2_list.append(img2)

    a = np.zeros((batch_size,usernum))

    for i in range(batch_size):
        a[i,u_list[i]] = 1
    u_list = torch.from_numpy(np.array(a)).cuda(gpu)
    i_list = torch.from_numpy(np.array(i_list)).cuda(gpu)
    j_list = torch.from_numpy(np.array(j_list)).cuda(gpu)
    img1_list = torch.from_numpy(np.array(img1_list)).permute(0,3,1,2).float().cuda(gpu)
    img2_list = torch.from_numpy(np.array(img2_list)).permute(0,3,1,2).float().cuda(gpu)

    return u_list, i_list, j_list, img1_list, img2_list

def AUC(train, test, theta, I, gpu):
    ans = 0
    cc = 0
    for u in train:
        i = test[u][0]['productid']
        #T = np.dot(U[u, :], I.T)
        #theta_u = torch.index_select(U, 0, torch.Tensor([u]).long().cuda())
        tensor_u = torch.zeros(1,usernum)
        tensor_u[0,u] = 1
        tensor_u = Variable(tensor_u.float().cuda(gpu),requires_grad=True)
        theta_u = theta(tensor_u)
        T = torch.mm(theta_u,I.t())
        T = torch.squeeze(T,0)
        cc += 1
        M = set()
        for item in train[u]:
            M.add(item['productid'])
        M.add(i)

        count = 0
        tmpans = 0
        # for j in xrange(itemnum):
        #print 'test list in AUC : ',test_list
        for j in random.sample(xrange(itemnum), 100):  # sample
            if j in M:
                continue
            if T[i] > T[j]:
                tmpans += 1
            count += 1

        tmpans /= float(count)
        ans += tmpans
    ans /= float(cc)
    return ans

def Evaltuation(save_folder, epoch, theta, cnn, gpu):

    #print next(cnn1.parameters()).is_cuda
    I = torch.zeros(itemnum, K).float().cuda(gpu)
    #I = torch.from_numpy(I).cuda()
    test_size = batch_size
    idx = np.array_split(range(itemnum), (itemnum + test_size - 1) / test_size)
    for i in range(len(idx)):
        cc = 0
        input_images = np.zeros([test_size, 224, 224, 3], dtype=np.uint8)
        for j in idx[i]:
            test_im = np.asarray(Image.open(StringIO(Item[j]['imgs'])).convert('RGB').resize((224,224)))
            #test_im = (test_im - 127.5) / 127.5
            input_images[cc] = test_im
            cc += 1
        #print i, idx[i][0],(idx[i][-1]+1)
        test_image = torch.from_numpy(input_images).permute(0,3,1,2).float().cuda(gpu)
        feature_vector = cnn(test_image)
        I.data[idx[i][0]:(idx[i][-1]+1)] = feature_vector[:(idx[i][-1]-idx[i][0]+1)]

    if epoch >= 25:
        np.save(save_folder + '/' + 'UI_' + str(K) + '_' + str(epoch) + '.npy', [theta, I])
    return AUC(user_train, user_validation, theta, I, gpu), AUC(user_train, user_test, theta, I, gpu)

def model_save(save_folder, cnn, theta, optimizer, epoch, mode):
    state = {'epoch': epoch + 1,
             'state_dict': cnn.state_dict(),
             'optim_dict': optimizer.state_dict()}
    torch.save(state, save_folder + '/epoch_' + str(epoch + 1) + 'cnn_mode' + str(mode) + '.ckpt')

    state = {'epoch': epoch + 1,
             'state_dict': theta.state_dict(),
             'optim_dict': optimizer.state_dict()}
    torch.save(state, save_folder + '/epoch_' + str(epoch + 1) + 'theta_mode' + str(mode) + '.ckpt')

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        if m.weight is not None:
            torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Linear):
        if m.weight is not None:
            torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def weights_init_theta(m):
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            torch.nn.init.uniform_(m.weight,a=0,b=1)
            m.weight.data = m.weight.data * 0.01
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input :', input[0])
    print('input sum :', input[0].sum())
    print('input size:', input[0].size())
    print('output     :', output.data)
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())

def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad : ',grad_input)
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].norm())

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    # https://stackoverflow.com/questions/49595663/find-a-gpu-with-enough-memory
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def make_dir(file_path):
    if not os.path.exists(file_path):
        os.mkdir(file_path)

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    gpu = args.gpu
    K = args.K
    lambda1 = args.lambda1
    lambda2 = args.lambda2
    learning_rate = args.lr
    training_epoch = args.training_epoch
    batch_size = args.batch_size
    dropout = args.drop_out
    num_of_thread = args.num_of_thread
    mode = args.mode
    dissimilarity = args.dissimilarity

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    pid = os.getpid()
    save_folder = 'pid_' + str(pid)+ '_mode' + str(mode)
    make_dir(save_folder)

    fashion_dataset = customDataset(mode, dissimilarity, gpu)
    f = open(save_folder + '/result.log', 'w')

    device = torch.device("cuda:" + str(gpu))

    print 'before model set up', get_gpu_memory_map()
    if mode == 0:
        cnn = Net().to(device)
        cnn.apply(weights_init)
        print 'mode is ',mode
    elif mode == 1:
        cnn = models.vgg16(pretrained=False,**{'num_classes' : 100})
        cnn = cnn.to(device)
        print 'mode is ', mode
    elif mode == 2:
        cnn = models.vgg16(pretrained=True)

        classifier = cnn.classifier
        new_classifier = []

        for i in range(len(classifier) - 1):
            new_classifier.append(classifier[i])

        linear = nn.Linear(4096, 100)
        torch.nn.init.xavier_uniform_(linear.weight)
        linear.bias.data.fill_(0.0)
        #https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
        new_classifier.append(linear)
        new_classifier = nn.Sequential(*new_classifier)
        cnn.classifier = new_classifier
        #https://discuss.pytorch.org/t/append-for-nn-sequential-or-directly-converting-nn-modulelist-to-nn-sequential/7104
        #https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/2
        #https://discuss.pytorch.org/t/how-to-replace-last-layer-in-sequential/14422
        #https://discuss.pytorch.org/t/how-to-modify-the-final-fc-layer-based-on-the-torch-model/766
        cnn = cnn.to(device)
        print 'mode is ', mode
    elif mode == 3:
        cnn = Net().to(device)
        cnn.apply(weights_init)
        print 'mode is ', mode
    elif mode == 4:
        cnn = models.vgg16(pretrained=False, **{'num_classes': 100})
        cnn = cnn.to(device)
        print 'mode is ', mode
    elif mode == 5:
        cnn = models.vgg16(pretrained=True, **{'num_classes': 100})
        cnn = cnn.to(device)
        print 'mode is ', mode
    elif mode == 6:
        pass
        print 'mode is ', mode
        #not implemented

    print 'After model set up',  get_gpu_memory_map()
    #cnn1.share_memory()
    #print cnn1
    #cnn2 = Net().to(device)
    #cnn2.share_memory()
    #cnn_test = Net().to(device)
    #theta = Variable(torch.rand((usernum,K)),requires_grad=True).cuda()
    #theta = theta / 100
    #theta = nn.Parameter(theta)

    theta = Lookup_Matrix(usernum, K).to(device)
    theta.apply(weights_init_theta)
    #theta.share_memory()
    #print(type(theta))
    oneiteration = 0

    l2_reg_nn = torch.FloatTensor(1).cuda(gpu)
    l2_reg_theta = torch.FloatTensor(1).cuda(gpu)

    for weight in cnn.parameters():
        l2_reg_nn = l2_reg_nn + weight.norm(2)
    #for weight in theta.parameters():
    #    l2_reg_theta = l2_reg_theta + weight.norm(2)

    l2_reg_nn = l2_reg_nn.view([])
    l2_reg_theta = l2_reg_theta.view([])
    #for W in theta.parameters():
    #    l2_reg_theta = l2_reg_theta + W.norm(2)

    for item in user_train:
        oneiteration += len(user_train[item])

    oneiteration = oneiteration // batch_size + 1

    #num_processes = 3
    optimizer = optim.Adam([{'params': cnn.parameters()},
                            {'params': theta.parameters()}], lr= learning_rate)

    #I = torch.zeros(itemnum, K).cuda()
    #AUC(user_train, user_validation, theta, I)

    '''
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(cnn1,cnn2,theta,oneiteration))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    '''
    #cnn1.layer1.register_backward_hook(printgradnorm)
    #cnn2.layer1.register_backward_hook(printgradnorm)
    #theta.linear.register_backward_hook(printgradnorm)
    #theta.linear.register_forward_hook(printnorm)
    #val_score, test_score = Evaltuation(theta, cnn1)
    #print('val_score : {}, test_score : {}\n'.format(val_score, test_score))
    #f.write('val_score : {}, test_score : {}\n'.format(val_score, test_score))
    #f.flush()
    #print 'Before training ',get_gpu_memory_map()
    for epoch in xrange(training_epoch):
        print 'epoch : ', epoch + 1, '....'
        for iter in xrange(oneiteration):
            optimizer.zero_grad()
            u, i, j, img1, img2 = get_batch(batch_size, gpu)
            #print 'After get batch : ',get_gpu_memory_map()
            feature_vector1 = cnn(img1)
            feature_vector2 = cnn(img2)
            #print 'After extract feature : ',get_gpu_memory_map()
            u = Variable(u,requires_grad=True)
            #print l2_reg_nn
            loss = -calculate_Loss(feature_vector1,feature_vector2,theta,u) + lambda1 * l2_reg_nn
            #loss -= lambda1 * l2_reg_nn + lambda2 * l2_reg_theta
            #print 'l2_reg_nn : ',l2_reg_nn
            #print 'l2_reg_theta : ',l2_reg_theta
            loss.backward(retain_graph=True)
            optimizer.step()
            print 'epoch : ', epoch + 1, ' loss : ', np.around(loss.item(),2), ' ', iter + 1, ' / ', oneiteration
            #RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time
            #https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795
        cnn.eval()
        theta.eval()
        val_score, test_score = Evaltuation(save_folder, epoch + 1, theta, cnn, gpu)
        cnn.train()
        theta.train()
        print('val_score : {}, test_score : {}\n'.format(val_score,test_score))
        f.write('val_score : {}, test_score : {}\n'.format(val_score,test_score))
        f.flush()
        if epoch >= 25:
            model_save(save_folder, cnn, theta, optimizer, epoch, mode)

    f.close()







